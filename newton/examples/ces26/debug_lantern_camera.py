# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Render USD scene using the TD060 camera from the USD file.

Uses ces26_utils for mesh loading (known working transforms).
Handles USD camera coordinate system: X-right, Y-up, -Z forward (camera looks down -Z).

Usage: uv run python newton/examples/ces26/debug_lantern_camera.py
"""

import math
from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image
from pxr import Gf, Usd, UsdGeom

from newton._src.sensors.warp_raytrace import ClearData, RenderContext, RenderShapeType

from ces26_utils import (
    MeshData,
    MeshLoadOptions,
    load_meshes_from_stage,
    make_mesh_names_unique,
    transform_points_to_world,
)

# =============================================================================
# Configuration
# =============================================================================

# Path to USD file
USD_FILE = r"C:\Users\chorvath\Downloads\20251219_iv060_flat_01\Collected_iv060_flat_01\iv060_flat_01.usd"

# Output directory (same as this script)
OUTPUT_DIR = Path(__file__).parent

# Camera path in USD
CAMERA_PATH = "/World/TD060"

# Frames to render
FRAMES = [2920, 3130]

# Render settings (16:9 aspect ratio)
WIDTH, HEIGHT = 960, 540


# =============================================================================
# Warp Kernels
# =============================================================================

@wp.kernel
def generate_world_rays_kernel(
    width: int,
    height: int,
    fov_y: float,
    cam_pos: wp.vec3f,
    cam_right: wp.vec3f,   # Camera +X axis in world space (normalized)
    cam_up: wp.vec3f,      # Camera +Y axis in world space (normalized)
    cam_forward: wp.vec3f, # Camera -Z axis in world space (normalized, view direction)
    out_rays: wp.array(dtype=wp.vec3, ndim=4),  # Shape: (1, height, width, 2)
):
    """
    Generate rays directly in world space using camera basis vectors.
    
    USD/RenderMan camera coordinate system:
    - X: right
    - Y: up  
    - Z: toward the sensor ("into the lens")
    
    The view direction is -Z, same as OpenGL convention.
    """
    x, y = wp.tid()
    
    if x >= width or y >= height:
        return

    aspect_ratio = float(width) / float(height)
    
    # Normalized device coordinates: center of pixel
    # u: -0.5 (left) to +0.5 (right)
    # v: -0.5 (top) to +0.5 (bottom) in image space
    u = (float(x) + 0.5) / float(width) - 0.5
    v = (float(y) + 0.5) / float(height) - 0.5
    
    # Half-height of the view frustum at z=1
    h = wp.tan(fov_y / 2.0)
    
    # Local ray direction (camera looks down -Z)
    # x_local = u * 2 * h * aspect
    # y_local = -v * 2 * h  (flip v for Y-up)
    # z_local = -1 (forward)
    x_local = u * 2.0 * h * aspect_ratio
    y_local = -v * 2.0 * h
    z_local = -1.0
    
    # Transform to world space using camera basis vectors
    # world_dir = x_local * right + y_local * up + z_local * (-forward)
    # But forward is already -Z, so z_local * forward gives us the -Z contribution
    ray_dir_world = x_local * cam_right + y_local * cam_up + z_local * (-cam_forward)
    
    out_rays[0, y, x, 0] = cam_pos
    out_rays[0, y, x, 1] = wp.normalize(ray_dir_world)


# =============================================================================
# Camera Utilities
# =============================================================================

@wp.struct
class CameraData:
    """Camera data for ray generation."""
    position: wp.vec3f
    right: wp.vec3f    # +X axis in world (normalized)
    up: wp.vec3f       # +Y axis in world (normalized)
    forward: wp.vec3f  # -Z axis in world (normalized, view direction)
    fov_v: float


def get_camera_data(
    stage: Usd.Stage,
    camera_path: str,
    time_code: Usd.TimeCode
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Extract camera world-space basis and FOV from USD camera.
    
    Uses the same transform extraction approach as ces26_utils mesh loading.
    
    Args:
        stage: USD stage
        camera_path: Path to camera prim
        time_code: Time code for sampling
        
    Returns:
        Tuple of (position, rotation_matrix_3x3, vertical_fov_radians)
        rotation_matrix columns are [right, up, forward] where forward is the view direction (-Z)
    """
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Camera not found at {camera_path}")

    usd_camera = UsdGeom.Camera(camera_prim)
    
    # --- Intrinsics from Gf.Camera ---
    gf_camera = usd_camera.GetCamera(time_code)
    fov_v = gf_camera.GetFieldOfView(Gf.Camera.FOVVertical)

    print(f"Camera at frame {time_code}: FOV={fov_v:.2f}°")

    # --- Extrinsics: Use XformCache like ces26_utils does for meshes ---
    xcache = UsdGeom.XformCache(time_code)
    world_mat = xcache.GetLocalToWorldTransform(camera_prim)

    # Transform basis points to world space
    pts_camera = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    pts_world = transform_points_to_world(pts_camera, world_mat)

    cam_origin, x_pt, y_pt, z_pt = pts_world

    # Extract and normalize basis vectors
    right = x_pt - cam_origin
    up = y_pt - cam_origin
    z_axis = z_pt - cam_origin  # +Z points toward sensor
    
    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)
    forward = -z_axis / np.linalg.norm(z_axis)  # View direction is -Z

    print(f"  Position: [{cam_origin[0]:.3f}, {cam_origin[1]:.3f}, {cam_origin[2]:.3f}]")
    print(f"  Forward:  [{forward[0]:.3f}, {forward[1]:.3f}, {forward[2]:.3f}]")

    # Build rotation matrix: columns are [right, up, forward]
    # This is an orthonormal basis representing camera orientation
    rot_mat = np.column_stack([right, up, forward]).astype(np.float32)
    position = cam_origin.astype(np.float32)

    return position, rot_mat, math.radians(fov_v)


# =============================================================================
# Mesh Loading (using ces26_utils)
# =============================================================================

def load_lantern_meshes(stage: Usd.Stage, usd_path: str, time_code: Usd.TimeCode) -> list[MeshData]:
    """Load lantern meshes using ces26_utils (known working transforms)."""
    
    def lantern_filter(path: str) -> bool:
        return "Hanging" in path
    
    options = MeshLoadOptions(
        time_code=time_code,
        load_material_colors=False,
        load_texture_colors=False,
        path_filter=lantern_filter,
        skip_invisible=True,
        skip_proxy=True
    )
    
    meshes = load_meshes_from_stage(stage, usd_path, options, verbose=False)
    make_mesh_names_unique(meshes)
    
    print(f"Loaded {len(meshes)} lantern meshes")
    return meshes


# =============================================================================
# Rendering
# =============================================================================

def setup_render_context(meshes: list[MeshData]) -> tuple[RenderContext, list]:
    """Create and configure RenderContext with mesh data."""
    num_shapes = len(meshes)
    print(f"Setting up render context with {num_shapes} shapes...")

    warp_meshes = []
    mesh_bounds = np.zeros((num_shapes, 2, 3), dtype=np.float32)
    
    scene_min = np.array([np.inf, np.inf, np.inf])
    scene_max = np.array([-np.inf, -np.inf, -np.inf])

    for i, mesh_data in enumerate(meshes):
        verts = mesh_data.vertices.astype(np.float32)
        tris = mesh_data.faces
        
        mesh = wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3f),
            indices=wp.array(tris.flatten(), dtype=wp.int32),
        )
        warp_meshes.append(mesh)
        mesh_bounds[i, 0] = verts.min(axis=0)
        mesh_bounds[i, 1] = verts.max(axis=0)
        
        scene_min = np.minimum(scene_min, mesh_bounds[i, 0])
        scene_max = np.maximum(scene_max, mesh_bounds[i, 1])
    
    print(f"Scene bounds: [{scene_min[0]:.1f}, {scene_min[1]:.1f}, {scene_min[2]:.1f}] to "
          f"[{scene_max[0]:.1f}, {scene_max[1]:.1f}, {scene_max[2]:.1f}]")

    ctx = RenderContext(
        width=WIDTH,
        height=HEIGHT,
        enable_textures=False,
        enable_shadows=True,
        enable_ambient_lighting=True,
        enable_particles=False,
        num_worlds=1,
        num_cameras=1,
    )

    ctx.num_shapes = num_shapes
    ctx.mesh_ids = wp.array([m.id for m in warp_meshes], dtype=wp.uint64)
    ctx.mesh_bounds = wp.array(mesh_bounds, dtype=wp.vec3f)
    ctx.shape_types = wp.array([RenderShapeType.MESH] * num_shapes, dtype=wp.int32)
    ctx.shape_enabled = wp.array(list(range(num_shapes)), dtype=wp.uint32)
    ctx.shape_world_index = wp.array([0] * num_shapes, dtype=wp.int32)
    ctx.shape_mesh_indices = wp.array(list(range(num_shapes)), dtype=wp.int32)
    ctx.shape_materials = wp.array([-1] * num_shapes, dtype=wp.int32)

    # Identity transforms (vertices already in world space)
    transforms = [wp.transformf(wp.vec3f(0, 0, 0), wp.quatf(0, 0, 0, 1))] * num_shapes
    ctx.shape_transforms = wp.array(transforms, dtype=wp.transformf)
    ctx.shape_sizes = wp.array([[1, 1, 1]] * num_shapes, dtype=wp.vec3f)

    # Random colors per mesh
    np.random.seed(42)
    colors = np.random.rand(num_shapes, 4).astype(np.float32) * 0.5 + 0.5
    colors[:, 3] = 1.0
    ctx.shape_colors = wp.array(colors, dtype=wp.vec4f)

    # Directional light
    ctx.lights_active = wp.array([True], dtype=wp.bool)
    ctx.lights_type = wp.array([1], dtype=wp.int32)
    ctx.lights_cast_shadow = wp.array([True], dtype=wp.bool)
    ctx.lights_position = wp.array([[0, 0, 0]], dtype=wp.vec3f)
    ctx.lights_orientation = wp.array([[-0.577, 0.577, -0.577]], dtype=wp.vec3f)

    return ctx, warp_meshes


def render_frame(
    ctx: RenderContext,
    position: np.ndarray,
    rot_mat: np.ndarray,
    fov_v: float,
    frame_num: int
) -> Image.Image:
    """Render a single frame and save to PNG."""
    
    # Extract camera basis vectors from rotation matrix columns
    cam_right = wp.vec3f(rot_mat[0, 0], rot_mat[1, 0], rot_mat[2, 0])
    cam_up = wp.vec3f(rot_mat[0, 1], rot_mat[1, 1], rot_mat[2, 1])
    cam_forward = wp.vec3f(rot_mat[0, 2], rot_mat[1, 2], rot_mat[2, 2])
    cam_pos = wp.vec3f(position[0], position[1], position[2])
    
    # Allocate rays buffer
    camera_rays = wp.zeros((1, HEIGHT, WIDTH, 2), dtype=wp.vec3f)
    
    # Generate rays directly in world space
    wp.launch(
        kernel=generate_world_rays_kernel,
        dim=(WIDTH, HEIGHT),
        inputs=[WIDTH, HEIGHT, fov_v, cam_pos, cam_right, cam_up, cam_forward, camera_rays]
    )

    # Identity camera transform (rays are already in world space)
    identity_transform = wp.transformf(wp.vec3f(0, 0, 0), wp.quatf(0, 0, 0, 1))
    camera_transforms = wp.array([[identity_transform]], dtype=wp.transformf)

    color_image = ctx.create_color_image_output()

    print(f"Rendering frame {frame_num}...")
    ctx.render(
        camera_transforms=camera_transforms,
        camera_rays=camera_rays,
        color_image=color_image,
        refit_bvh=True,
        clear_data=ClearData(clear_color=0xFF404040),
    )

    # Save output
    pixels = color_image.numpy()[0, 0].reshape(HEIGHT, WIDTH)
    r = (pixels >> 0) & 0xFF
    g = (pixels >> 8) & 0xFF
    b = (pixels >> 16) & 0xFF
    rgba = np.stack([r, g, b], axis=-1).astype(np.uint8)

    img = Image.fromarray(rgba, mode="RGB")
    output_path = OUTPUT_DIR / f"debug_lantern_camera.{frame_num}.png"
    img.save(output_path)
    print(f"Saved: {output_path}")

    return img


def main():
    print(f"Loading USD: {USD_FILE}")
    stage = Usd.Stage.Open(USD_FILE)
    if not stage:
        raise RuntimeError("Failed to open USD file")

    # Load meshes
    time_code = Usd.TimeCode(FRAMES[0])
    meshes = load_lantern_meshes(stage, USD_FILE, time_code)
    
    if not meshes:
        print("No meshes found!")
        return

    ctx, warp_meshes = setup_render_context(meshes)

    # Render each frame
    for frame in FRAMES:
        time_code = Usd.TimeCode(frame)
        position, rot_mat, fov_v = get_camera_data(stage, CAMERA_PATH, time_code)
        render_frame(ctx, position, rot_mat, fov_v, frame)

    print("\nDone!")


if __name__ == "__main__":
    main()
