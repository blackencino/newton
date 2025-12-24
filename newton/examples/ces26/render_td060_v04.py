# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Render all visible meshes in USD scene using the TD060 camera (v04).

Uses the preprocessing cache with dynamic semantic coloring:
1. Load preprocessing cache (metadata only, no geometry)
2. Classify groups for dynamic per-frame coloring
3. Load geometry from USD using cached metadata as filter
4. Setup render context with headlight lighting
5. Render all AOVs to EXR format, saving each AOV to a separate subdirectory

EXR output (saved to subdirectories):
- color/{base}_color.{frame:04d}.exr: Lit diffuse render (half-float RGB, [0, 1])
- depth/{base}_depth.{frame:04d}.exr: Normalized depth (half-float Y, [0, 1], 0=near 1=far)
- depth_heat/{base}_depth_heat.{frame:04d}.exr: Depth heat map (half-float RGB)
- normal/{base}_normal.{frame:04d}.exr: Surface normals (half-float RGB, [0, 1] mapped from [-1, 1])
- object_id/{base}_object_id.{frame:04d}.exr: Object ID colors from cache (half-float RGB)
- semantic/{base}_semantic.{frame:04d}.exr: Semantic colors with distance-based gradients (half-float RGB)

Renders at 4K resolution (3840x2160), frames 2920-3130.

Usage: uv run python newton/examples/ces26/render_td060_v04.py
"""

from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

from ces26_utils import (
    CameraData,
    DepthColormap,
    Diegetic,
    ExrOutputs,
    RenderConfig,
    RenderOutputs,
    ColorLUTs,
    get_colormap_lut,
    get_camera_from_stage,
    make_diegetic_names_unique,
    open_usd_stage,
    packed_uint32_to_float_rgb,
    render_all_aovs,
    save_exr_depth,
    save_exr_rgb,
    setup_render_context,
    triangulate,
    transform_points_to_world,
)
from dynamic_segmentation import (
    CameraCurve,
    DiageticGroupMetadata,
    GroupCategory,
    load_preprocessing_cache,
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
FRAMES = list(range(2920, 3131))  # 2920 to 3130 inclusive

# Output configuration
OUTPUT_DIR = CACHE_DIR
OUTPUT_FORMAT = "exr"
DEPTH_COLORMAP = DepthColormap.MAGMA

RENDER_CONFIG = RenderConfig(
    width=3840,
    height=2160,
    output_dir=OUTPUT_DIR,  # Base directory, subdirs created per AOV
    filename_pattern="frame.{frame}.exr",
)

# Distance thresholds for object classification
UNSAFE_MAX_DIST = 2.5    # Unsafe objects with path_danger > this are "inert" (meters)
SAFE_MAX_DIST = 5.0      # Safe objects with path_danger > this are "inert" (meters)
INERT_NORM_DIST = 100.0   # Normalization distance for inert objects (meters)


# =============================================================================
# Dynamic Category Classification
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
    
    if verbose:
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
    
    if verbose:
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
        objectid_color = group.objectid_color
        
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
        
        # Create Diegetic with colors from group
        # Use objectid_color for both object_id and semantic (semantic will be updated per-frame)
        # Use a neutral gray for diffuse_albedo (will be lit by headlight)
        diegetic = Diegetic(
            name=name,
            path=path_str,
            vertices=vertices,
            faces=faces,
            diffuse_albedo=(0.7, 0.7, 0.7),  # Neutral gray for headlight lighting
            object_id=objectid_color,  # From cache
            semantic=objectid_color,  # Placeholder, will be updated per-frame
        )
        
        diegetics.append(diegetic)
        
        if verbose:
            print(f"  LOADED: {path_str} -> group {group.unique_name} "
                  f"(objectid=({objectid_color[0]:.2f}, {objectid_color[1]:.2f}, {objectid_color[2]:.2f}))")
    
    if show_progress:
        print(f"  Done: {len(diegetics)} meshes loaded, {paths_skipped} skipped", flush=True)
    
    return diegetics


# =============================================================================
# Custom Kernel for Background Color
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
# Headlight Lighting
# =============================================================================

def update_lights_for_headlight(ctx, camera_forward: np.ndarray) -> None:
    """
    Update render context lights to use headlight pointing along camera forward.
    
    Shadows are disabled for headlight mode since a camera-following light
    would create unnatural shadows that shift with camera movement.
    
    Args:
        ctx: RenderContext to update
        camera_forward: Camera forward direction (normalized)
    """
    ctx.lights_active = wp.array([True], dtype=wp.bool)
    ctx.lights_type = wp.array([1], dtype=wp.int32)  # directional
    ctx.lights_cast_shadow = wp.array([False], dtype=wp.bool)  # No shadows for headlight
    ctx.lights_position = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f)
    ctx.lights_orientation = wp.array([camera_forward.tolist()], dtype=wp.vec3f)


# =============================================================================
# Custom EXR Conversion with Dark Purple Background
# =============================================================================

def convert_aovs_to_exr_data_with_bg(
    outputs: RenderOutputs,
    color_luts: ColorLUTs,
    config: RenderConfig,
    depth_colormap: str = DepthColormap.MAGMA,
) -> ExrOutputs:
    """
    Convert raw GPU render outputs to float32 arrays for OpenEXR saving.
    
    Uses gradient_cool(0) as background color for semantic and object_id passes.
    This is a custom version of convert_aovs_to_exr_data that uses a custom background.
    
    Args:
        outputs: Raw RenderOutputs from render_all_aovs
        color_luts: ColorLUTs for object_id and semantic mapping
        config: Render configuration (for dimensions)
        depth_colormap: Colormap for depth_heat (DepthColormap.VIRIDIS or .MAGMA)
        
    Returns:
        ExrOutputs with all passes as float32 arrays in [0, 1]
    """
    import ces26_utils
    
    height, width = config.height, config.width
    
    # Color pass - convert packed uint32 to float RGB
    color_rgb = packed_uint32_to_float_rgb(
        outputs.color_image.numpy()[0, 0], height, width
    )
    
    # Compute depth range once (used by both depth passes)
    depth_range = wp.array([1e10, 0.0], dtype=wp.float32)
    wp.launch(
        ces26_utils.find_depth_range,
        outputs.depth_image.shape,
        [outputs.depth_image, depth_range],
    )
    depth_range_np = depth_range.numpy()
    min_depth, max_depth = depth_range_np[0], depth_range_np[1]
    
    # Depth pass - normalize to 0-1
    depth_raw = outputs.depth_image.numpy()[0, 0].reshape(height, width)
    valid_mask = depth_raw > 0
    if valid_mask.any():
        denom = max(max_depth - min_depth, 0.001)
        # Normalize: 0=near, 1=far
        depth = np.where(valid_mask, (depth_raw - min_depth) / denom, 1.0)
    else:
        depth = np.ones_like(depth_raw)
    
    # Depth heat pass - interpolated colormap for smooth gradients
    colormap_lut = get_colormap_lut(depth_colormap)
    depth_heat_rgb = wp.zeros((1, 1, width * height), dtype=wp.vec3f)
    wp.launch(
        ces26_utils.depth_to_colormap_float,
        outputs.depth_image.shape,
        [outputs.depth_image, min_depth, max_depth, colormap_lut, depth_heat_rgb],
    )
    depth_heat = depth_heat_rgb.numpy()[0, 0].reshape(height, width, 3)
    
    # Normal pass - map from [-1, 1] to [0, 1]
    normal_raw = outputs.normal_image.numpy()[0, 0].reshape(height, width, 3)
    normal = normal_raw * 0.5 + 0.5
    
    # Background color: gradient_cool(0) - dark purple
    bg_rgb = gradient_cool(0.0)
    bg_r = int(bg_rgb[0] * 255)
    bg_g = int(bg_rgb[1] * 255)
    bg_b = int(bg_rgb[2] * 255)
    bg_color_packed = 0xFF000000 | (bg_b << 16) | (bg_g << 8) | bg_r
    
    # Object ID pass - use LUT on shape indices with custom background
    object_id_rgba = wp.zeros_like(outputs.color_image)
    wp.launch(
        shape_index_to_color_lut_with_bg,
        outputs.shape_index_image.shape,
        [outputs.shape_index_image, color_luts.object_id, wp.uint32(bg_color_packed), object_id_rgba],
    )
    object_id_rgb = packed_uint32_to_float_rgb(
        object_id_rgba.numpy()[0, 0], height, width
    )
    
    # Semantic pass - use LUT on shape indices with custom background
    semantic_rgba = wp.zeros_like(outputs.color_image)
    wp.launch(
        shape_index_to_color_lut_with_bg,
        outputs.shape_index_image.shape,
        [outputs.shape_index_image, color_luts.semantic, wp.uint32(bg_color_packed), semantic_rgba],
    )
    semantic_rgb = packed_uint32_to_float_rgb(
        semantic_rgba.numpy()[0, 0], height, width
    )
    
    return ExrOutputs(
        color=color_rgb.astype(np.float32),
        depth=depth.astype(np.float32),
        depth_heat=depth_heat.astype(np.float32),
        normal=normal.astype(np.float32),
        object_id=object_id_rgb.astype(np.float32),
        semantic=semantic_rgb.astype(np.float32),
    )


# =============================================================================
# Save AOVs to separate subdirectories
# =============================================================================

def save_all_aovs_exr_to_subdirs(
    exr_outputs: ExrOutputs,
    output_base_dir: Path,
    frame_num: int,
    base_name: str = "td060_v04",
) -> None:
    """
    Save all AOV passes to disk as OpenEXR files in separate subdirectories.
    
    Creates subdirectories for each AOV:
    - color/{base_name}_color.{frame:04d}.exr
    - depth/{base_name}_depth.{frame:04d}.exr
    - depth_heat/{base_name}_depth_heat.{frame:04d}.exr
    - normal/{base_name}_normal.{frame:04d}.exr
    - object_id/{base_name}_object_id.{frame:04d}.exr
    - semantic/{base_name}_semantic.{frame:04d}.exr
    
    Args:
        exr_outputs: ExrOutputs from convert_aovs_to_exr_data
        output_base_dir: Base directory (subdirectories created inside)
        frame_num: Frame number for filename (formatted as 4-digit zero-padded)
        base_name: Base filename (before the AOV suffix)
    """
    aov_subdirs = {
        "color": "color",
        "depth": "depth",
        "depth_heat": "depth_heat",
        "normal": "normal",
        "object_id": "object_id",
        "semantic": "semantic",
    }
    
    for aov_name, subdir_name in aov_subdirs.items():
        subdir = output_base_dir / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{base_name}_{aov_name}.{frame_num:04d}.exr"
        output_path = subdir / filename
        
        if aov_name == "depth":
            # Single-channel depth
            save_exr_depth(getattr(exr_outputs, aov_name), output_path)
        else:
            # RGB channels
            save_exr_rgb(getattr(exr_outputs, aov_name), output_path)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Render TD060 V04 - Preprocessing Cache + Dynamic Semantic Colors")
    print("=" * 70)
    print(f"  USD file: {USD_FILE}")
    print(f"  Cache file: {CACHE_FILE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Frames: {FRAMES[0]} to {FRAMES[-1]} ({len(FRAMES)} frames)")
    print(f"  Resolution: {RENDER_CONFIG.width}x{RENDER_CONFIG.height}")
    print(f"  UNSAFE_MAX_DIST = {UNSAFE_MAX_DIST}m")
    print(f"  SAFE_MAX_DIST = {SAFE_MAX_DIST}m")
    print(f"  INERT_NORM_DIST = {INERT_NORM_DIST}m")
    
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
    
    # Step 7: Setup render context with headlight lighting
    print("\nStep 7: Setting up render context with headlight lighting...")
    ctx, color_luts, _ = setup_render_context(diegetics, RENDER_CONFIG, lights=None)
    
    # Step 8: Render all AOVs for each frame
    print(f"\nStep 8: Rendering all AOVs for frames {FRAMES[0]} to {FRAMES[-1]}...")
    
    for i, frame in enumerate(FRAMES):
        time_code = Usd.TimeCode(frame)
        camera = get_camera_from_stage(stage, CAMERA_PATH, time_code, verbose=False)
        
        # Update headlight direction to follow camera
        update_lights_for_headlight(ctx, camera.forward)
        
        # Update ALL dynamic object colors based on distance at THIS frame
        if dynamic_infos:
            update_dynamic_colors_for_frame(
                semantic_lut=color_luts.semantic,
                dynamic_infos=dynamic_infos,
                camera_eye=camera.position,
            )
        
        # Single render pass
        outputs = render_all_aovs(ctx, camera, RENDER_CONFIG)
        
        # Convert to EXR data (with custom dark purple background for semantic/object_id)
        exr_outputs = convert_aovs_to_exr_data_with_bg(outputs, color_luts, RENDER_CONFIG, DEPTH_COLORMAP)
        
        # Save to separate subdirectories
        save_all_aovs_exr_to_subdirs(
            exr_outputs=exr_outputs,
            output_base_dir=OUTPUT_DIR,
            frame_num=frame,
            base_name="td060_v04",
        )
        
        # Progress update
        if (i + 1) % 10 == 0 or i == 0 or i == len(FRAMES) - 1:
            print(f"Rendered frame {frame} ({i + 1}/{len(FRAMES)})")
    
    print("\n" + "=" * 70)
    print("Done! Output saved to subdirectories in:", OUTPUT_DIR)
    print("  - color/")
    print("  - depth/")
    print("  - depth_heat/")
    print("  - normal/")
    print("  - object_id/")
    print("  - semantic/")
    print("=" * 70)


if __name__ == "__main__":
    main()

