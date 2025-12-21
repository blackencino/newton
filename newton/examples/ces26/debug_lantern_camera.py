# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Render USD scene using the TD060 camera from the USD file.

Uses ces26_utils for mesh and camera loading.
Handles USD camera coordinate system: X-right, Y-up, -Z forward (camera looks down -Z).

Usage: uv run python newton/examples/ces26/debug_lantern_camera.py
"""

from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image
from pxr import Usd

from newton._src.sensors.warp_raytrace import ClearData, RenderContext, RenderShapeType

from ces26_utils import (
    CameraData,
    MeshData,
    MeshLoadOptions,
    get_camera_from_stage,
    load_meshes_from_stage,
    make_mesh_names_unique,
    transforms_and_rays_from_camera_data,
)

# =============================================================================
# Configuration
# =============================================================================

#USD_FILE = r"C:\Users\chorvath\Downloads\20251219_iv060_flat_01\Collected_iv060_flat_01\iv060_flat_01.usd"
USD_FILE = r"C:\Users\chorvath\Downloads\20251220_iv060_flat_02\Collected_20251220_iv060_flat_02\20251220_iv060_flat_02.usd"
OUTPUT_DIR = Path(__file__).parent
CAMERA_PATH = "/World/TD060"
FRAMES = [2920, 3130]
WIDTH, HEIGHT = 960, 540



# =============================================================================
# Mesh Loading
# =============================================================================

def load_lantern_meshes(stage: Usd.Stage, usd_path: str, time_code: Usd.TimeCode) -> list[MeshData]:
    """Load lantern meshes using ces26_utils."""
    options = MeshLoadOptions(
        time_code=time_code,
        load_material_colors=False,
        load_texture_colors=False,
        path_filter=lambda path: "Hanging" in path,
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
    scene_min = np.full(3, np.inf)
    scene_max = np.full(3, -np.inf)

    for i, mesh_data in enumerate(meshes):
        verts = mesh_data.vertices.astype(np.float32)
        mesh = wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3f),
            indices=wp.array(mesh_data.faces.flatten(), dtype=wp.int32),
        )
        warp_meshes.append(mesh)
        mesh_bounds[i, 0] = verts.min(axis=0)
        mesh_bounds[i, 1] = verts.max(axis=0)
        scene_min = np.minimum(scene_min, mesh_bounds[i, 0])
        scene_max = np.maximum(scene_max, mesh_bounds[i, 1])
    
    print(f"Scene bounds: [{scene_min[0]:.1f}, {scene_min[1]:.1f}, {scene_min[2]:.1f}] to "
          f"[{scene_max[0]:.1f}, {scene_max[1]:.1f}, {scene_max[2]:.1f}]")

    ctx = RenderContext(
        width=WIDTH, height=HEIGHT,
        enable_textures=False, enable_shadows=True,
        enable_ambient_lighting=True, enable_particles=False,
        num_worlds=1, num_cameras=1,
    )

    ctx.num_shapes = num_shapes
    ctx.mesh_ids = wp.array([m.id for m in warp_meshes], dtype=wp.uint64)
    ctx.mesh_bounds = wp.array(mesh_bounds, dtype=wp.vec3f)
    ctx.shape_types = wp.array([RenderShapeType.MESH] * num_shapes, dtype=wp.int32)
    ctx.shape_enabled = wp.array(list(range(num_shapes)), dtype=wp.uint32)
    ctx.shape_world_index = wp.array([0] * num_shapes, dtype=wp.int32)
    ctx.shape_mesh_indices = wp.array(list(range(num_shapes)), dtype=wp.int32)
    ctx.shape_materials = wp.array([-1] * num_shapes, dtype=wp.int32)

    transforms = [wp.transformf(wp.vec3f(0, 0, 0), wp.quatf(0, 0, 0, 1))] * num_shapes
    ctx.shape_transforms = wp.array(transforms, dtype=wp.transformf)
    ctx.shape_sizes = wp.array([[1, 1, 1]] * num_shapes, dtype=wp.vec3f)

    np.random.seed(42)
    colors = np.random.rand(num_shapes, 4).astype(np.float32) * 0.5 + 0.5
    colors[:, 3] = 1.0
    ctx.shape_colors = wp.array(colors, dtype=wp.vec4f)

    ctx.lights_active = wp.array([True], dtype=wp.bool)
    ctx.lights_type = wp.array([1], dtype=wp.int32)
    ctx.lights_cast_shadow = wp.array([True], dtype=wp.bool)
    ctx.lights_position = wp.array([[0, 0, 0]], dtype=wp.vec3f)
    ctx.lights_orientation = wp.array([[-0.577, 0.577, -0.577]], dtype=wp.vec3f)

    return ctx, warp_meshes

def render_frame(ctx: RenderContext, camera: CameraData, frame_num: int) -> Image.Image:
    """Render a single frame and save to PNG."""

    camera_transforms, camera_rays = transforms_and_rays_from_camera_data(camera, WIDTH, HEIGHT)

    color_image = ctx.create_color_image_output()

    print(f"Rendering frame {frame_num}...")
    ctx.render(
        camera_transforms=camera_transforms,
        camera_rays=camera_rays,
        color_image=color_image,
        refit_bvh=True,
        clear_data=ClearData(clear_color=0xFF404040),
    )

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

    time_code = Usd.TimeCode(FRAMES[0])
    meshes = load_lantern_meshes(stage, USD_FILE, time_code)
    
    if not meshes:
        print("No meshes found!")
        return

    ctx, _ = setup_render_context(meshes)

    for frame in FRAMES:
        time_code = Usd.TimeCode(frame)
        camera = get_camera_from_stage(stage, CAMERA_PATH, time_code, verbose=True)
        render_frame(ctx, camera, frame)

    print("\nDone!")


if __name__ == "__main__":
    main()
