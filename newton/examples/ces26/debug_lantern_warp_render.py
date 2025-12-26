# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Minimal Warp raytracer render of USD meshes.

Same mesh loading as debug_lantern.py, but renders using RenderContext
instead of displaying with polyscope.

Usage: python newton/examples/ces26/debug_lantern_warp_render.py
"""

import math
from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image
from pxr import Usd, UsdGeom

# Output directory (same as this script)
OUTPUT_DIR = Path(__file__).parent

from newton._src.sensors.warp_raytrace import ClearData, RenderContext, RenderShapeType

# Path to USD file
USD_FILE = r"C:\Users\chorvath\Downloads\20251219_iv060_flat_01\Collected_iv060_flat_01\iv060_flat_01.usd"

# Target prim to search under
TARGET_PRIM = "HangingLanternE_01"

# Render settings
WIDTH, HEIGHT = 512, 512


def triangulate(indices, counts):
    """Convert polygon mesh to triangles via fan triangulation."""
    tris = []
    idx = 0
    for count in counts:
        if count >= 3:
            for i in range(1, count - 1):
                tris.append([indices[idx], indices[idx + i], indices[idx + i + 1]])
        idx += count
    return np.array(tris, dtype=np.int32)


def load_meshes_from_usd():
    """Load meshes from USD file, returning list of (vertices, triangles) tuples."""
    print(f"Loading: {USD_FILE}")
    stage = Usd.Stage.Open(USD_FILE)
    if not stage:
        raise RuntimeError("Failed to open USD file")

    print(f"Searching for '{TARGET_PRIM}' meshes (skipping proxy)...")
    meshes = []

    for prim in stage.Traverse():
        path_str = str(prim.GetPath())

        if TARGET_PRIM not in path_str:
            continue
        if "/proxy/" in path_str.lower():
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()

        if not points or not indices or not counts:
            continue

        # Get world transform and apply to vertices
        xform = UsdGeom.Xformable(prim)
        world_mat = np.array(xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

        pts = np.array(points, dtype=np.float32)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_world = (world_mat @ pts_h.T).T[:, :3].astype(np.float32)

        tris = triangulate(np.array(indices), np.array(counts))

        print(f"  {path_str.split('/')[-1]}: {len(pts)} verts, {len(tris)} tris")
        meshes.append((pts_world, tris))

    print(f"Loaded {len(meshes)} meshes")
    return meshes


def compute_camera_rays(width, height, fov_radians):
    """Compute pinhole camera rays. Returns array shape (1, height, width, 2)."""
    rays = np.zeros((1, height, width, 2, 3), dtype=np.float32)
    aspect = width / height
    h = math.tan(fov_radians / 2.0)

    for py in range(height):
        for px in range(width):
            u = (px + 0.5) / width - 0.5
            v = (py + 0.5) / height - 0.5
            direction = np.array([u * 2.0 * h * aspect, -v * 2.0 * h, -1.0], dtype=np.float32)
            direction /= np.linalg.norm(direction)
            rays[0, py, px, 0] = [0, 0, 0]  # origin
            rays[0, py, px, 1] = direction

    return wp.array(rays, dtype=wp.vec3f)


def main():
    # 1. Load meshes from USD
    mesh_data = load_meshes_from_usd()
    if not mesh_data:
        print("No meshes found!")
        return

    num_shapes = len(mesh_data)

    # 2. Create Warp meshes and compute bounds
    warp_meshes = []
    mesh_bounds = np.zeros((num_shapes, 2, 3), dtype=np.float32)

    for i, (verts, tris) in enumerate(mesh_data):
        mesh = wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3f),
            indices=wp.array(tris.flatten(), dtype=wp.int32),
        )
        warp_meshes.append(mesh)
        mesh_bounds[i, 0] = verts.min(axis=0)  # min bounds
        mesh_bounds[i, 1] = verts.max(axis=0)  # max bounds

    # Compute scene bounds for camera positioning
    scene_min = mesh_bounds[:, 0, :].min(axis=0)
    scene_max = mesh_bounds[:, 1, :].max(axis=0)
    scene_center = (scene_min + scene_max) / 2
    scene_extent = scene_max - scene_min
    print(f"Scene center: {scene_center}, extent: {scene_extent}")

    # 3. Create RenderContext
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

    # 4. Populate shape data
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

    # Add a directional light
    ctx.lights_active = wp.array([True], dtype=wp.bool)
    ctx.lights_type = wp.array([1], dtype=wp.int32)  # DIRECTIONAL
    ctx.lights_cast_shadow = wp.array([True], dtype=wp.bool)
    ctx.lights_position = wp.array([[0, 0, 0]], dtype=wp.vec3f)
    ctx.lights_orientation = wp.array([[-0.577, 0.577, -0.577]], dtype=wp.vec3f)

    # 5. Set up camera
    # Position camera to look at scene from the front
    cam_distance = max(scene_extent) * 2.0
    cam_pos = scene_center + np.array([0, -cam_distance * 0.7, cam_distance * 0.3])

    # Compute camera orientation (look-at)
    forward = scene_center - cam_pos
    forward /= np.linalg.norm(forward)
    up = np.array([0, 0, 1], dtype=np.float32)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # Build rotation matrix and convert to quaternion
    rot_mat = np.array([right, up, -forward]).T
    # Convert rotation matrix to quaternion (simplified)
    trace = rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot_mat[2, 1] - rot_mat[1, 2]) * s
        y = (rot_mat[0, 2] - rot_mat[2, 0]) * s
        z = (rot_mat[1, 0] - rot_mat[0, 1]) * s
    else:
        if rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
            s = 2.0 * math.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
            w = (rot_mat[2, 1] - rot_mat[1, 2]) / s
            x = 0.25 * s
            y = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            z = (rot_mat[0, 2] + rot_mat[2, 0]) / s
        elif rot_mat[1, 1] > rot_mat[2, 2]:
            s = 2.0 * math.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
            w = (rot_mat[0, 2] - rot_mat[2, 0]) / s
            x = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            y = 0.25 * s
            z = (rot_mat[1, 2] + rot_mat[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
            w = (rot_mat[1, 0] - rot_mat[0, 1]) / s
            x = (rot_mat[0, 2] + rot_mat[2, 0]) / s
            y = (rot_mat[1, 2] + rot_mat[2, 1]) / s
            z = 0.25 * s

    cam_quat = wp.quatf(x, y, z, w)
    cam_transform = wp.transformf(wp.vec3f(*cam_pos), cam_quat)

    # Camera transforms: shape (num_cameras, num_worlds) = (1, 1)
    camera_transforms = wp.array([[cam_transform]], dtype=wp.transformf)

    # Camera rays
    fov = math.radians(60)
    camera_rays = compute_camera_rays(WIDTH, HEIGHT, fov)

    print(f"Camera at: {cam_pos}")

    # 6. Create output buffers and render
    color_image = ctx.create_color_image_output()
    depth_image = ctx.create_depth_image_output()

    print("Rendering...")
    ctx.render(
        camera_transforms=camera_transforms,
        camera_rays=camera_rays,
        color_image=color_image,
        depth_image=depth_image,
        refit_bvh=True,
        clear_data=ClearData(clear_color=0xFF404040),
    )

    # 7. Convert to image and display
    pixels = color_image.numpy()[0, 0]  # shape: (width*height,) uint32
    pixels = pixels.reshape(HEIGHT, WIDTH)

    # Extract RGBA channels from packed uint32
    r = (pixels >> 0) & 0xFF
    g = (pixels >> 8) & 0xFF
    b = (pixels >> 16) & 0xFF
    rgba = np.stack([r, g, b], axis=-1).astype(np.uint8)

    img = Image.fromarray(rgba, mode="RGB")
    output_path = OUTPUT_DIR / "debug_lantern_render.png"
    img.save(output_path)
    print(f"Saved: {output_path}")
    img.show()


if __name__ == "__main__":
    main()

