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

# Distance thresholds for object classification
UNSAFE_MAX_DIST = 2.5    # Unsafe objects with path_danger > this are "inert" (meters)
SAFE_MAX_DIST = 5.0      # Safe objects with path_danger > this are "inert" (meters)
INERT_NORM_DIST = 100.0  # Normalization distance for inert objects (meters)


# =============================================================================
# Transform groups: assign colors by category + distance
# =============================================================================

from dataclasses import dataclass
from enum import Enum

RGB = tuple[float, float, float]


class DynamicCategory(Enum):
    """Category for dynamic per-frame color updates."""
    UNSAFE = "unsafe"      # gradient_hot, normalize by UNSAFE_MAX_DIST
    SAFE = "safe"          # gradient_happy, normalize by SAFE_MAX_DIST
    INERT = "inert"        # gradient_cool, normalize by INERT_NORM_DIST


@dataclass
class DynamicGroupInfo:
    """Information about a group requiring per-frame color updates."""
    group_id: str
    center_world: np.ndarray  # (3,) ellipsoid center in world space
    shape_indices: list[int]  # Shape indices in the render context
    dynamic_category: DynamicCategory  # Which gradient/normalization to use


def compute_distance_colored_groups(
    groups: dict[str, DiageticGroupMetadata],
    camera_curve: CameraCurve,
    reference_frame: int = 2920,
    verbose: bool = False,
) -> tuple[dict[str, DiageticGroupMetadata], dict[str, DynamicCategory]]:
    """
    Transform groups and classify them for dynamic per-frame coloring.
    
    Classification:
    - GROUND_TERRAIN -> keep original objectid_color (static)
    - UNSAFE with path_danger <= UNSAFE_MAX_DIST -> truly unsafe (dynamic, gradient_hot)
    - UNSAFE with path_danger > UNSAFE_MAX_DIST -> inert (dynamic, gradient_cool)
    - SAFE with path_danger <= SAFE_MAX_DIST -> truly safe (dynamic, gradient_happy)
    - SAFE with path_danger > SAFE_MAX_DIST -> inert (dynamic, gradient_cool)
    
    All dynamic groups get placeholder colors here; actual colors are computed per-frame.
    Gradients use (1-t) for reversed polarity (closer = brighter).
    
    Args:
        groups: Dictionary of DiageticGroupMetadata
        camera_curve: Pre-baked camera animation
        reference_frame: Frame number for initial classification
        verbose: Print detailed progress
        
    Returns:
        Tuple of (groups dict with placeholder colors, dict mapping group_id to DynamicCategory)
    """
    # Step 1: Get camera eye position at reference frame
    frame_idx = np.where(camera_curve.frames == reference_frame)[0]
    if len(frame_idx) == 0:
        raise ValueError(f"Frame {reference_frame} not found in camera curve")
    frame_idx = frame_idx[0]
    camera_eye = camera_curve.positions[frame_idx]
    
    print(f"  Camera eye at frame {reference_frame}: "
          f"[{camera_eye[0]:.1f}, {camera_eye[1]:.1f}, {camera_eye[2]:.1f}]")
    
    # Step 2: Classify all groups
    dynamic_groups: dict[str, DynamicCategory] = {}
    category_counts = {"terrain": 0, "unsafe": 0, "safe": 0, "inert": 0}
    
    for group_id, group in groups.items():
        if group.category == GroupCategory.GROUND_TERRAIN:
            # Terrain is static, no dynamic category
            category_counts["terrain"] += 1
        elif group.category == GroupCategory.UNSAFE:
            if group.path_danger <= UNSAFE_MAX_DIST:
                dynamic_groups[group_id] = DynamicCategory.UNSAFE
                category_counts["unsafe"] += 1
            else:
                dynamic_groups[group_id] = DynamicCategory.INERT
                category_counts["inert"] += 1
        else:  # SAFE
            if group.path_danger <= SAFE_MAX_DIST:
                dynamic_groups[group_id] = DynamicCategory.SAFE
                category_counts["safe"] += 1
            else:
                dynamic_groups[group_id] = DynamicCategory.INERT
                category_counts["inert"] += 1
    
    print(f"  Classification: {category_counts['unsafe']} truly unsafe (path_danger <= {UNSAFE_MAX_DIST}m), "
          f"{category_counts['safe']} truly safe (path_danger <= {SAFE_MAX_DIST}m), "
          f"{category_counts['inert']} inert")
    
    # Step 3: Create placeholder colors for all groups
    result: dict[str, DiageticGroupMetadata] = {}
    
    for group_id, group in groups.items():
        if group.category == GroupCategory.GROUND_TERRAIN:
            # Keep original objectid_color for terrain/bg
            new_color = group.objectid_color
        else:
            # All dynamic groups get a placeholder (will be updated per-frame)
            # Use mid-gradient as placeholder
            dyn_cat = dynamic_groups.get(group_id, DynamicCategory.INERT)
            if dyn_cat == DynamicCategory.UNSAFE:
                color_rgb = gradient_hot(0.5)
            elif dyn_cat == DynamicCategory.SAFE:
                color_rgb = gradient_happy(0.5)
            else:  # INERT
                color_rgb = gradient_cool(0.5)
            new_color = (float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))
        
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
            dyn_cat = dynamic_groups.get(group_id)
            cat_str = dyn_cat.value if dyn_cat else "terrain"
            print(f"  {group.unique_name}: {cat_str} -> placeholder color")
    
    # Report category counts
    print(f"  Categories: terrain={category_counts['terrain']}, "
          f"unsafe={category_counts['unsafe']}, safe={category_counts['safe']}, "
          f"inert={category_counts['inert']}")
    
    return result, dynamic_groups


def build_dynamic_group_infos(
    dynamic_groups: dict[str, DynamicCategory],
    groups: dict[str, DiageticGroupMetadata],
    path_to_shape_index: dict[str, int],
) -> list[DynamicGroupInfo]:
    """
    Build a list of DynamicGroupInfo for per-frame color updates.
    
    Args:
        dynamic_groups: Dict mapping group_id to DynamicCategory
        groups: Dictionary of DiageticGroupMetadata
        path_to_shape_index: Mapping from mesh path to shape index
        
    Returns:
        List of DynamicGroupInfo with shape indices and category for each dynamic group
    """
    dynamic_infos: list[DynamicGroupInfo] = []
    
    for group_id, dyn_category in dynamic_groups.items():
        group = groups[group_id]
        
        # Find all shape indices for this group's member paths
        shape_indices = []
        for member_path in group.member_paths:
            if member_path in path_to_shape_index:
                shape_indices.append(path_to_shape_index[member_path])
        
        if shape_indices:
            dynamic_infos.append(DynamicGroupInfo(
                group_id=group_id,
                center_world=group.bounding_ellipsoid.center_world.copy(),
                shape_indices=shape_indices,
                dynamic_category=dyn_category,
            ))
    
    return dynamic_infos


def rgb_to_packed_uint32_bgr(r: float, g: float, b: float) -> int:
    """Convert RGB floats (0-1) to packed uint32 in 0xAABBGGRR format."""
    return (
        0xFF000000 |
        (int(b * 255) << 16) |
        (int(g * 255) << 8) |
        int(r * 255)
    )


def update_dynamic_colors_for_frame(
    semantic_lut: wp.array,
    dynamic_infos: list[DynamicGroupInfo],
    camera_eye: np.ndarray,
) -> None:
    """
    Update the semantic LUT with per-frame colors for all dynamic objects.
    
    For each dynamic group:
    1. Compute distance from camera_eye to group's ellipsoid center
    2. Normalize by the appropriate distance (UNSAFE_MAX_DIST, SAFE_MAX_DIST, or INERT_NORM_DIST)
    3. Use the appropriate gradient with reversed polarity (1-t) for closer=brighter
    4. Update all shape indices belonging to this group
    
    Args:
        semantic_lut: Warp array of packed uint32 colors (modified in-place)
        dynamic_infos: List of DynamicGroupInfo with shape indices and category
        camera_eye: Camera position at this frame (3,)
    """
    # Get LUT as numpy for modification
    lut_np = semantic_lut.numpy()
    
    for info in dynamic_infos:
        # Compute distance from camera to group center
        distance = float(np.linalg.norm(camera_eye - info.center_world))
        
        # Select normalization distance and gradient based on category
        if info.dynamic_category == DynamicCategory.UNSAFE:
            norm_dist = UNSAFE_MAX_DIST
            t = np.clip(distance / norm_dist, 0.0, 1.0)
            color_rgb = gradient_hot(1.0 - t)  # Reversed: closer = brighter (higher t)
        elif info.dynamic_category == DynamicCategory.SAFE:
            norm_dist = SAFE_MAX_DIST
            t = np.clip(distance / norm_dist, 0.0, 1.0)
            color_rgb = gradient_happy(1.0 - t)  # Reversed: closer = brighter
        else:  # INERT
            norm_dist = INERT_NORM_DIST
            t = np.clip(distance / norm_dist, 0.0, 1.0)
            color_rgb = gradient_cool(1.0 - t)  # Reversed: closer = brighter
        
        packed_color = rgb_to_packed_uint32_bgr(
            float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2])
        )
        
        # Update all shape indices for this group
        for shape_idx in info.shape_indices:
            lut_np[shape_idx] = packed_color
    
    # Copy back to GPU
    semantic_lut.assign(lut_np)


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
    print(f"  UNSAFE_MAX_DIST = {UNSAFE_MAX_DIST}m (unsafe threshold)")
    print(f"  SAFE_MAX_DIST = {SAFE_MAX_DIST}m (safe threshold)")
    print(f"  INERT_NORM_DIST = {INERT_NORM_DIST}m (normalization for inert objects)")
    
    # Step 1: Load preprocessing cache (metadata only, no geometry)
    print(f"\nStep 1: Loading preprocessing cache from {CACHE_FILE}")
    metadata_list, groups, camera_curve = load_preprocessing_cache(CACHE_FILE, verbose=False)
    
    print(f"  Loaded {len(metadata_list)} mesh metadata entries")
    print(f"  Loaded {len(groups)} groups")
    print(f"  Camera path: {camera_curve.camera_path}")
    print(f"  Camera frames: {camera_curve.frames[0]} to {camera_curve.frames[-1]}")
    
    # Step 2: Classify groups for dynamic coloring
    print("\nStep 2: Classifying groups for dynamic coloring...")
    groups, dynamic_groups = compute_distance_colored_groups(
        groups=groups,
        camera_curve=camera_curve,
        reference_frame=FRAMES[0],
        verbose=False,
    )
    
    # Step 3: Build path-to-group mapping (with placeholder colors)
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
    
    # Step 6: Build path-to-shape-index mapping for dynamic object updates
    print("\nStep 6: Building path-to-shape-index mapping...")
    path_to_shape_index: dict[str, int] = {}
    for shape_idx, d in enumerate(diegetics):
        path_to_shape_index[d.path] = shape_idx
    print(f"  Mapped {len(path_to_shape_index)} paths to shape indices")
    
    # Build dynamic group info for per-frame updates
    dynamic_infos = build_dynamic_group_infos(dynamic_groups, groups, path_to_shape_index)
    
    # Count by category
    unsafe_count = sum(1 for i in dynamic_infos if i.dynamic_category == DynamicCategory.UNSAFE)
    safe_count = sum(1 for i in dynamic_infos if i.dynamic_category == DynamicCategory.SAFE)
    inert_count = sum(1 for i in dynamic_infos if i.dynamic_category == DynamicCategory.INERT)
    total_dynamic_shapes = sum(len(info.shape_indices) for info in dynamic_infos)
    print(f"  Dynamic groups: {unsafe_count} unsafe, {safe_count} safe, {inert_count} inert")
    print(f"  Total dynamic shapes: {total_dynamic_shapes}")
    
    # Step 7: Setup render context
    print("\nStep 7: Setting up render context...")
    ctx, color_luts, _ = setup_render_context(diegetics, RENDER_CONFIG, lights=None)
    
    # Step 8: Render semantic AOV for each frame
    print(f"\nStep 8: Rendering semantic AOV for frames {FRAMES}...")
    
    for frame in FRAMES:
        time_code = Usd.TimeCode(frame)
        camera = get_camera_from_stage(stage, CAMERA_PATH, time_code, verbose=True)
        
        # Update ALL dynamic object colors based on distance at THIS frame
        if dynamic_infos:
            update_dynamic_colors_for_frame(
                semantic_lut=color_luts.semantic,
                dynamic_infos=dynamic_infos,
                camera_eye=camera.position,
            )
        
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

