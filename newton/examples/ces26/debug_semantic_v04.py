# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Render semantic segmentation AOV using preprocessed cache metadata.

This script tests the preprocessing cache by:
1. Loading metadata from the v04 preprocessed cache (no geometry)
2. Using cached group info to know which USD paths to load and their semantic colors
3. Loading geometry from USD for only those paths
4. Rendering the semantic AOV for frames 2920 and 3130

The semantic color for each mesh is now determined by:
- Group category (terrain/bg -> cool, unsafe -> hot, safe -> happy)
- Distance from camera eye to group's bounding ellipsoid center
- Normalized by the furthest group's distance at frame 2920

Usage: uv run python newton/examples/ces26/debug_semantic_v04.py
"""

from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

from ces26_utils import (
    CameraData,
    Diegetic,
    RenderConfig,
    get_camera_from_stage,
    make_diegetic_names_unique,
    open_usd_stage,
    setup_render_context,
    triangulate,
    transform_points_to_world,
)
from dynamic_segmentation import (
    load_preprocessing_cache,
    DiageticGroupMetadata,
    DiageticMetadata,
    CameraCurve,
    GroupCategory,
)
from dynamic_segmentation.gradients_b import (
    gradient_cool,
    gradient_hot,
    gradient_happy,
)

# =============================================================================
# Configuration
# =============================================================================

# V04 USD file
USD_FILE = r"C:\Users\chorvath\Downloads\20251223_iv060_flat_04\Collected_20251223_iv060_flat_04\20251223_iv060_flat_04.usd"

# Preprocessed cache directory
CACHE_DIR = Path(r"D:\ces26_data\td060\v04")
CACHE_FILE = CACHE_DIR / "preprocess_v2.npz"

# Camera and frames
CAMERA_PATH = "/World/TD060"
FRAMES = [2920, 3130]

# Output configuration
OUTPUT_DIR = CACHE_DIR
RENDER_CONFIG = RenderConfig(
    width=3840,
    height=2160,
    output_dir=OUTPUT_DIR,
    filename_pattern="semantic_gradient.{frame}.png",
)


# =============================================================================
# Transform groups: assign colors by category + distance
# =============================================================================

RGB = tuple[float, float, float]


def compute_distance_colored_groups(
    groups: dict[str, DiageticGroupMetadata],
    camera_curve: CameraCurve,
    reference_frame: int = 2920,
    verbose: bool = False,
) -> dict[str, DiageticGroupMetadata]:
    """
    Transform groups to use gradient-based colors determined by category and distance.
    
    For each group:
    1. Get its category (GROUND_TERRAIN, UNSAFE, or SAFE)
    2. Compute distance from camera eye (at reference_frame) to group's ellipsoid center
    3. Normalize distance by the max distance across all groups
    4. Use the normalized distance as t to sample the appropriate gradient:
       - GROUND_TERRAIN -> gradient_cool (blues/purples)
       - UNSAFE -> gradient_hot (reds/pinks)
       - SAFE -> gradient_happy (greens/yellows)
    
    Args:
        groups: Dictionary of DiageticGroupMetadata
        camera_curve: Pre-baked camera animation
        reference_frame: Frame number to use for camera position (default 2920)
        verbose: Print detailed progress
        
    Returns:
        New dictionary with groups having updated objectid_color based on gradients
    """
    # Step 1: Get camera eye position at reference frame
    frame_idx = np.where(camera_curve.frames == reference_frame)[0]
    if len(frame_idx) == 0:
        raise ValueError(f"Frame {reference_frame} not found in camera curve")
    frame_idx = frame_idx[0]
    camera_eye = camera_curve.positions[frame_idx]
    
    print(f"  Camera eye at frame {reference_frame}: "
          f"[{camera_eye[0]:.1f}, {camera_eye[1]:.1f}, {camera_eye[2]:.1f}]")
    
    # Step 2: Compute distance for each group
    group_distances: dict[str, float] = {}
    for group_id, group in groups.items():
        center = group.bounding_ellipsoid.center_world
        distance = float(np.linalg.norm(camera_eye - center))
        group_distances[group_id] = distance
    
    # Step 3: Find max distance for normalization
    max_distance = max(group_distances.values())
    print(f"  Max distance from camera: {max_distance:.1f}")
    
    # Step 4: Transform each group with gradient-based color
    result: dict[str, DiageticGroupMetadata] = {}
    category_counts: dict[GroupCategory, int] = {c: 0 for c in GroupCategory}
    
    for group_id, group in groups.items():
        distance = group_distances[group_id]
        t = np.clip(distance / max_distance, 0.0, 1.0)
        
        # Select gradient based on category
        if group.category == GroupCategory.GROUND_TERRAIN:
            # Keep original objectid_color for terrain/bg
            new_color = group.objectid_color
        elif group.category == GroupCategory.UNSAFE:
            color_rgb = gradient_hot(t)
            new_color = (float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))
        else:  # SAFE
            color_rgb = gradient_happy(t)
            new_color = (float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))
        category_counts[group.category] += 1
        
        # Create new group with updated color
        updated_group = DiageticGroupMetadata(
            group_id=group.group_id,
            unique_name=group.unique_name,
            common_ancestor_path=group.common_ancestor_path,
            objectid_color=new_color,
            member_paths=group.member_paths,
            member_count=group.member_count,
            total_vertex_count=group.total_vertex_count,
            total_triangle_count=group.total_triangle_count,
            bbox_min=group.bbox_min,
            bbox_max=group.bbox_max,
            bounding_ellipsoid=group.bounding_ellipsoid,
            path_danger=group.path_danger,
            category=group.category,
        )
        result[group_id] = updated_group
        
        if verbose:
            print(f"  {group.unique_name}: {group.category.value} @ dist={distance:.1f} "
                  f"-> t={t:.3f} -> color=({new_color[0]:.2f}, {new_color[1]:.2f}, {new_color[2]:.2f})")
    
    # Report category counts
    print(f"  Categories: "
          f"terrain={category_counts[GroupCategory.GROUND_TERRAIN]}, "
          f"unsafe={category_counts[GroupCategory.UNSAFE]}, "
          f"safe={category_counts[GroupCategory.SAFE]}")
    
    return result


# =============================================================================
# Build path-to-group mapping from cache
# =============================================================================

def build_path_to_group_map(
    groups: dict[str, DiageticGroupMetadata],
) -> dict[str, DiageticGroupMetadata]:
    """
    Build a mapping from mesh path to its parent group.
    
    Each mesh path in member_paths gets mapped to its containing group,
    so we can look up the semantic color (objectid_color) for any mesh.
    """
    path_to_group: dict[str, DiageticGroupMetadata] = {}
    
    for group in groups.values():
        for member_path in group.member_paths:
            path_to_group[member_path] = group
    
    return path_to_group


# =============================================================================
# Load geometry from USD using cached metadata as filter
# =============================================================================

def load_diegetics_from_usd_with_cache(
    stage: Usd.Stage,
    time_code: Usd.TimeCode,
    path_to_group: dict[str, DiageticGroupMetadata],
    verbose: bool = False,
    show_progress: bool = True,
    progress_interval: int = 500,
) -> list[Diegetic]:
    """
    Load geometry from USD for paths specified in the cache.
    
    Directly fetches prims by path from the cache - no stage traversal needed.
    Semantic color is determined by the group's objectid_color.
    
    Args:
        stage: Opened USD stage
        time_code: Time code for geometry sampling
        path_to_group: Mapping from mesh path to group (from cache)
        verbose: Print detailed progress
        show_progress: Print periodic progress updates
        progress_interval: Update interval for progress messages
        
    Returns:
        List of Diegetic objects with geometry and semantic colors from cache
    """
    xcache = UsdGeom.XformCache(time_code)
    
    diegetics: list[Diegetic] = []
    paths_processed = 0
    paths_skipped = 0
    
    cached_paths = list(path_to_group.keys())
    
    if show_progress:
        print(f"Loading {len(cached_paths)} meshes from USD (direct path lookup)...", flush=True)
    
    # Iterate over cached paths and fetch each prim directly
    for path_str in cached_paths:
        paths_processed += 1
        
        if show_progress and paths_processed % progress_interval == 0:
            print(f"  Loaded {paths_processed}/{len(cached_paths)} meshes...", flush=True)
        
        # Directly get the prim by path
        prim = stage.GetPrimAtPath(path_str)
        
        if not prim or not prim.IsValid():
            if verbose:
                print(f"  SKIP (prim not found): {path_str}")
            paths_skipped += 1
            continue
        
        if not prim.IsA(UsdGeom.Mesh):
            if verbose:
                print(f"  SKIP (not a mesh): {path_str}")
            paths_skipped += 1
            continue
        
        # Get the group for this mesh (for semantic color)
        group = path_to_group[path_str]
        semantic_color = group.objectid_color
        
        # Extract geometry - handle instanced geometry
        if prim.IsInstanceProxy():
            geom_prim = prim.GetPrimInPrototype()
        else:
            geom_prim = prim
        
        mesh = UsdGeom.Mesh(geom_prim)
        
        points = mesh.GetPointsAttr().Get(time_code)
        indices = mesh.GetFaceVertexIndicesAttr().Get(time_code)
        counts = mesh.GetFaceVertexCountsAttr().Get(time_code)
        
        if not points or not indices or not counts:
            if verbose:
                print(f"  SKIP (no geometry): {path_str}")
            paths_skipped += 1
            continue
        
        # Transform to world space
        world_mat = xcache.GetLocalToWorldTransform(prim)
        vertices = transform_points_to_world(points, world_mat).astype(np.float32)
        
        # Triangulate
        faces = triangulate(np.array(indices), np.array(counts))
        
        name = path_str.split("/")[-1]
        
        # Create Diegetic with semantic color from group
        diegetic = Diegetic(
            name=name,
            path=path_str,
            vertices=vertices,
            faces=faces,
            diffuse_albedo=semantic_color,
            object_id=semantic_color,
            semantic=semantic_color,
        )
        
        diegetics.append(diegetic)
        
        if verbose:
            print(f"  LOADED: {path_str} -> group {group.unique_name} "
                  f"(color=({semantic_color[0]:.2f}, {semantic_color[1]:.2f}, {semantic_color[2]:.2f}))")
    
    if show_progress:
        print(f"  Done: {len(diegetics)} meshes loaded, {paths_skipped} skipped", flush=True)
    
    return diegetics


# =============================================================================
# Custom kernel with configurable background color
# =============================================================================

@wp.kernel
def shape_index_to_color_lut_with_bg(
    shape_indices: wp.array(dtype=wp.uint32, ndim=3),
    color_lut: wp.array(dtype=wp.uint32),
    bg_color: wp.uint32,
    out_rgba: wp.array(dtype=wp.uint32, ndim=3),
):
    """Map shape indices to colors using a lookup table, with custom background color."""
    world_id, camera_id, pixel_id = wp.tid()
    shape_index = shape_indices[world_id, camera_id, pixel_id]
    if shape_index < wp.uint32(color_lut.shape[0]):
        out_rgba[world_id, camera_id, pixel_id] = color_lut[wp.int32(shape_index)]
    else:
        # Background or invalid - use provided background color
        out_rgba[world_id, camera_id, pixel_id] = bg_color


# =============================================================================
# Render semantic AOV only
# =============================================================================

def render_semantic_aov(
    ctx,
    color_luts,
    camera: CameraData,
    config: RenderConfig,
    frame_num: int,
    base_name: str,
) -> None:
    """
    Render only the semantic AOV and save as PNG.
    
    This is a simplified version of render_and_save_all_aovs that only
    outputs the semantic pass.
    """
    from PIL import Image
    
    from ces26_utils import (
        transforms_and_rays_from_camera_data,
        packed_uint32_to_rgb,
    )
    from newton._src.sensors.warp_raytrace import ClearData
    
    # Use gradient_cool(0) as background color instead of gray
    # Packed format is 0xAABBGGRR (blue in bits 16-23, red in bits 0-7)
    bg_rgb = gradient_cool(0.0)
    bg_r = int(bg_rgb[0] * 255)
    bg_g = int(bg_rgb[1] * 255)
    bg_b = int(bg_rgb[2] * 255)
    bg_color_packed = 0xFF000000 | (bg_b << 16) | (bg_g << 8) | bg_r
    
    print(f"Rendering frame {frame_num} (semantic AOV only)...")
    
    # Generate camera rays
    camera_transforms, camera_rays = transforms_and_rays_from_camera_data(
        camera, config.width, config.height
    )
    
    # Create output arrays - we only need shape_index for semantic
    shape_index_image = ctx.create_shape_index_image_output()
    
    # Render (we need shape indices to map to semantic colors)
    ctx.render(
        camera_transforms=camera_transforms,
        camera_rays=camera_rays,
        shape_index_image=shape_index_image,
        refit_bvh=True,
        clear_data=ClearData(clear_color=bg_color_packed),
    )
    
    # Map shape indices to semantic colors using LUT with custom background
    semantic_rgba = wp.zeros_like(shape_index_image)
    wp.launch(
        shape_index_to_color_lut_with_bg,
        shape_index_image.shape,
        [shape_index_image, color_luts.semantic, wp.uint32(bg_color_packed), semantic_rgba],
    )
    
    # Convert to RGB
    semantic_rgb = packed_uint32_to_rgb(
        semantic_rgba.numpy()[0, 0], config.height, config.width
    )
    
    # Save
    output_path = config.output_dir / f"{base_name}_semantic.{frame_num:04d}.png"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    img = Image.fromarray(semantic_rgb, mode="RGB")
    img.save(output_path)
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Debug Semantic V04 - Distance-Based Gradient Colors")
    print("=" * 70)
    
    # Step 1: Load preprocessing cache (metadata only, no geometry)
    print(f"\nStep 1: Loading preprocessing cache from {CACHE_FILE}")
    metadata_list, groups, camera_curve = load_preprocessing_cache(CACHE_FILE, verbose=False)
    
    print(f"  Loaded {len(metadata_list)} mesh metadata entries")
    print(f"  Loaded {len(groups)} groups")
    print(f"  Camera path: {camera_curve.camera_path}")
    print(f"  Camera frames: {camera_curve.frames[0]} to {camera_curve.frames[-1]}")
    
    # Step 2: Transform groups to use gradient-based colors
    print("\nStep 2: Computing distance-based gradient colors...")
    groups = compute_distance_colored_groups(
        groups=groups,
        camera_curve=camera_curve,
        reference_frame=FRAMES[0],  # Use frame 2920 for normalization
        verbose=False,
    )
    
    # Step 3: Build path-to-group mapping (with new colors)
    print("\nStep 3: Building path-to-group mapping...")
    path_to_group = build_path_to_group_map(groups)
    print(f"  Mapped {len(path_to_group)} mesh paths to groups")
    
    # Step 4: Open USD stage
    print(f"\nStep 4: Opening USD stage...")
    stage = open_usd_stage(USD_FILE)
    time_code = Usd.TimeCode(FRAMES[0])
    
    # Step 5: Load diegetics from USD using cache metadata
    print("\nStep 5: Loading geometry from USD (filtered by cache)...")
    diegetics = load_diegetics_from_usd_with_cache(
        stage=stage,
        time_code=time_code,
        path_to_group=path_to_group,
        verbose=False,
        show_progress=True,
    )
    
    if not diegetics:
        print("ERROR: No diegetics loaded!")
        return
    
    # Make names unique
    diegetics = make_diegetic_names_unique(diegetics)
    print(f"  Final: {len(diegetics)} diegetics ready for rendering")
    
    # Step 6: Setup render context
    print("\nStep 6: Setting up render context...")
    ctx, color_luts, _ = setup_render_context(diegetics, RENDER_CONFIG, lights=None)
    
    # Step 7: Render semantic AOV for each frame
    print(f"\nStep 7: Rendering semantic AOV for frames {FRAMES}...")
    
    for frame in FRAMES:
        time_code = Usd.TimeCode(frame)
        camera = get_camera_from_stage(stage, CAMERA_PATH, time_code, verbose=True)
        
        render_semantic_aov(
            ctx=ctx,
            color_luts=color_luts,
            camera=camera,
            config=RENDER_CONFIG,
            frame_num=frame,
            base_name="semantic_gradient",
        )
    
    print("\n" + "=" * 70)
    print("Done! Output saved to:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()

