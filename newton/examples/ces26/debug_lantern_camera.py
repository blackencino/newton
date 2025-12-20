# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Render USD scene using the TD060 camera from the USD file.

Refactored to use robust Gf math for transforms and Warp kernels for ray generation.
"""

import math
from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image
from pxr import Gf, Usd, UsdGeom

from newton._src.sensors.warp_raytrace import ClearData, RenderContext, RenderShapeType

# Path to USD file
USD_FILE = r"C:\Users\chorvath\Downloads\20251219_iv060_flat_01\Collected_iv060_flat_01\iv060_flat_01.usd"

# Output directory (same as this script)
OUTPUT_DIR = Path(__file__).parent

# Target prim to search under (None = load all meshes)
TARGET_PRIM = "Hanging"  # Only load prims with "Hanging" in the name

# Camera path in USD
CAMERA_PATH = "/World/TD060"

# Frames to render
FRAMES = [2920, 3130]

# Render settings (16:9 aspect ratio)
WIDTH, HEIGHT = 960, 540

# -----------------------------------------------------------------------------
# Warp Kernels
# -----------------------------------------------------------------------------

@wp.kernel
def generate_rays_kernel(
    width: int,
    height: int,
    fov_y: float,
    out_rays: wp.array(dtype=wp.vec3, ndim=4),  # Shape: (1, height, width, 2)
):
    """Generate rays in LOCAL camera space (Y-up, -Z forward). Matches tiled_camera_sensor convention."""
    x, y = wp.tid()
    
    if x >= width or y >= height:
        return

    # Match exactly what tiled_camera_sensor.compute_pinhole_camera_rays does
    aspect_ratio = float(width) / float(height)
    u = (float(x) + 0.5) / float(width) - 0.5
    v = (float(y) + 0.5) / float(height) - 0.5
    h = wp.tan(fov_y / 2.0)
    
    # Y-up, -Z forward convention (standard OpenGL/USD camera)
    ray_dir = wp.vec3f(u * 2.0 * h * aspect_ratio, -v * 2.0 * h, -1.0)
    
    out_rays[0, y, x, 0] = wp.vec3f(0.0)
    out_rays[0, y, x, 1] = wp.normalize(ray_dir)

# -----------------------------------------------------------------------------
# USD Utilities
# -----------------------------------------------------------------------------

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


def load_meshes_from_usd(stage, time_code, max_meshes=None):
    """Load meshes from USD file at given time code."""
    if TARGET_PRIM:
        print(f"Loading meshes for '{TARGET_PRIM}' (skipping proxy)...")
    else:
        print("Loading all scene meshes (skipping proxy)...")
    meshes = []

    for prim in stage.Traverse():
        path_str = str(prim.GetPath())

        if TARGET_PRIM and TARGET_PRIM not in path_str:
            continue
        if "/proxy/" in path_str.lower():
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        
        # Get attributes
        points_attr = mesh.GetPointsAttr()
        indices_attr = mesh.GetFaceVertexIndicesAttr()
        counts_attr = mesh.GetFaceVertexCountsAttr()
        
        points = points_attr.Get(time_code)
        indices = indices_attr.Get(time_code)
        counts = counts_attr.Get(time_code)

        if not points or not indices or not counts:
            continue

        # Get world transform at this time code
        # FIX: Ensure we get the full transform stack for the prim
        xform = UsdGeom.Xformable(prim)
        world_mat_gf = xform.ComputeLocalToWorldTransform(time_code)
        world_mat = np.array(world_mat_gf)

        pts = np.array(points, dtype=np.float32)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        
        # FIX: USD uses Row-Vector convention: v' = v * M
        # Previous code did: (M @ v.T).T which assumes M acts on columns.
        # Correct for USD: points @ matrix
        pts_world = (pts_h @ world_mat)[:, :3].astype(np.float32)

        tris = triangulate(np.array(indices), np.array(counts))

        meshes.append((pts_world, tris, path_str.split('/')[-1]))
        
        if max_meshes and len(meshes) >= max_meshes:
            print(f"  ... stopping at {max_meshes} meshes")
            break

    print(f"Loaded {len(meshes)} meshes")
    return meshes


def get_camera_data(stage, camera_path, time_code):
    """Extract camera data robustly using Gf transforms."""
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Camera not found at {camera_path}")

    usd_camera = UsdGeom.Camera(camera_prim)
    
    # 1. Intrinsics (from GfCamera value object)
    gf_camera = usd_camera.GetCamera(time_code)
    
    # Use Gf to calculate FOV (handles aperture/focal length math)
    fov_v = gf_camera.GetFieldOfView(Gf.Camera.FOVVertical)
    fov_h = gf_camera.GetFieldOfView(Gf.Camera.FOVHorizontal)

    print(f"Camera intrinsics at frame {time_code}:")
    print(f"  Vertical FOV: {fov_v:.2f} degrees")
    print(f"  Horizontal FOV: {fov_h:.2f} degrees")

    # 2. Extrinsics (from Prim Transform)
    xform_mat = usd_camera.ComputeLocalToWorldTransform(time_code)
    
    # Extract translation
    translate = xform_mat.ExtractTranslation()
    
    # Extract rotation - USD uses same convention as Newton (Y-up, -Z forward for cameras)
    rotation = xform_mat.ExtractRotation().GetQuat()
    rot_real = rotation.GetReal()      # w
    rot_imag = rotation.GetImaginary() # (x, y, z)

    print(f"Camera world transform at frame {time_code}:")
    print(f"  Position: {translate}")
    print(f"  Rotation (quat): {rotation}")

    return {
        "vertical_fov": math.radians(fov_v),
        "position": np.array(translate, dtype=np.float32),
        # Warp expects quat as (x, y, z, w)
        "rotation": np.array([rot_imag[0], rot_imag[1], rot_imag[2], rot_real], dtype=np.float32),
    }

# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def setup_render_context(mesh_data):
    """Create and configure RenderContext with mesh data."""
    num_shapes = len(mesh_data)
    print(f"Setting up render context with {num_shapes} shapes...")

    # Create Warp meshes and compute bounds
    warp_meshes = []
    mesh_bounds = np.zeros((num_shapes, 2, 3), dtype=np.float32)

    for i, (verts, tris, name) in enumerate(mesh_data):
        mesh = wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3f),
            indices=wp.array(tris.flatten(), dtype=wp.int32),
        )
        warp_meshes.append(mesh)
        mesh_bounds[i, 0] = verts.min(axis=0)
        mesh_bounds[i, 1] = verts.max(axis=0)

    # Create RenderContext
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

    # Populate shape data
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
    ctx.lights_type = wp.array([1], dtype=wp.int32)  # DIRECTIONAL
    ctx.lights_cast_shadow = wp.array([True], dtype=wp.bool)
    ctx.lights_position = wp.array([[0, 0, 0]], dtype=wp.vec3f)
    ctx.lights_orientation = wp.array([[-0.577, 0.577, -0.577]], dtype=wp.vec3f)

    return ctx, warp_meshes


def render_frame(ctx, camera_data, frame_num):
    """Render a single frame and save to PNG."""
    
    # 1. Allocate rays buffer - shape: (num_cameras, height, width, 2)
    camera_rays = wp.zeros((1, HEIGHT, WIDTH, 2), dtype=wp.vec3f)
    
    # 2. Generate rays in LOCAL camera space
    wp.launch(
        kernel=generate_rays_kernel,
        dim=(WIDTH, HEIGHT),
        inputs=[
            WIDTH,
            HEIGHT,
            camera_data["vertical_fov"],
            camera_rays,
        ]
    )

    # 3. Create camera transform (camera-to-world)
    # The render kernel uses this to transform local rays to world space
    cam_transform = wp.transformf(
        wp.vec3f(*camera_data["position"]),
        wp.quatf(*camera_data["rotation"])
    )
    # Shape: (num_cameras, num_worlds) = (1, 1)
    camera_transforms = wp.array([[cam_transform]], dtype=wp.transformf)

    # 4. Render
    color_image = ctx.create_color_image_output()

    print(f"Rendering frame {frame_num}...")
    ctx.render(
        camera_transforms=camera_transforms,
        camera_rays=camera_rays,
        color_image=color_image,
        refit_bvh=True,
        clear_data=ClearData(clear_color=0xFF404040),
    )

    # 5. Save Output
    pixels = color_image.numpy()[0, 0]
    pixels = pixels.reshape(HEIGHT, WIDTH)

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

    # Load meshes (use first frame's time code for geometry)
    time_code = Usd.TimeCode(FRAMES[0])
    mesh_data = load_meshes_from_usd(stage, time_code, max_meshes=None)
    if not mesh_data:
        print("No meshes found!")
        return

    # Set up render context (only once, meshes don't change)
    ctx, warp_meshes = setup_render_context(mesh_data)

    # Render each frame
    for frame in FRAMES:
        time_code = Usd.TimeCode(frame)
        camera_data = get_camera_data(stage, CAMERA_PATH, time_code)
        render_frame(ctx, camera_data, frame)

    print("\nDone!")


if __name__ == "__main__":
    main()

# # SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# # SPDX-License-Identifier: Apache-2.0
# """
# Render USD scene using the TD060 camera from the USD file.

# All camera intrinsics (FOV from focal length + aperture) and extrinsics
# (animated world transform) come directly from the USD.
# Renders frames 2920 and 3130 at 16:9 aspect ratio.

# Usage: uv run python newton/examples/ces26/debug_lantern_camera.py
# """

# import math
# from pathlib import Path

# import numpy as np
# import warp as wp
# from PIL import Image
# from pxr import Gf, Usd, UsdGeom

# from newton._src.sensors.warp_raytrace import ClearData, RenderContext, RenderShapeType

# # Path to USD file
# USD_FILE = r"C:\Users\chorvath\Downloads\20251219_iv060_flat_01\Collected_iv060_flat_01\iv060_flat_01.usd"

# # Output directory (same as this script)
# OUTPUT_DIR = Path(__file__).parent

# # Target prim to search under (None = load all meshes)
# TARGET_PRIM = None  # Load entire scene for TD060 camera view

# # Camera path in USD
# CAMERA_PATH = "/World/TD060"

# # Frames to render
# FRAMES = [2920, 3130]

# # Render settings (16:9 aspect ratio)
# WIDTH, HEIGHT = 960, 540


# def triangulate(indices, counts):
#     """Convert polygon mesh to triangles via fan triangulation."""
#     tris = []
#     idx = 0
#     for count in counts:
#         if count >= 3:
#             for i in range(1, count - 1):
#                 tris.append([indices[idx], indices[idx + i], indices[idx + i + 1]])
#         idx += count
#     return np.array(tris, dtype=np.int32)


# def load_meshes_from_usd(stage, time_code, max_meshes=None):
#     """Load meshes from USD file at given time code."""
#     if TARGET_PRIM:
#         print(f"Loading meshes for '{TARGET_PRIM}' (skipping proxy)...")
#     else:
#         print("Loading all scene meshes (skipping proxy)...")
#     meshes = []

#     for prim in stage.Traverse():
#         path_str = str(prim.GetPath())

#         if TARGET_PRIM and TARGET_PRIM not in path_str:
#             continue
#         if "/proxy/" in path_str.lower():
#             continue
#         if not prim.IsA(UsdGeom.Mesh):
#             continue

#         mesh = UsdGeom.Mesh(prim)
#         points = mesh.GetPointsAttr().Get(time_code)
#         indices = mesh.GetFaceVertexIndicesAttr().Get(time_code)
#         counts = mesh.GetFaceVertexCountsAttr().Get(time_code)

#         if not points or not indices or not counts:
#             continue

#         # Get world transform at this time code
#         xform = UsdGeom.Xformable(prim)
#         world_mat = np.array(xform.ComputeLocalToWorldTransform(time_code))

#         pts = np.array(points, dtype=np.float32)
#         pts_h = np.hstack([pts, np.ones((len(pts), 1))])
#         pts_world = (world_mat @ pts_h.T).T[:, :3].astype(np.float32)

#         tris = triangulate(np.array(indices), np.array(counts))

#         meshes.append((pts_world, tris, path_str.split('/')[-1]))
        
#         if max_meshes and len(meshes) >= max_meshes:
#             print(f"  ... stopping at {max_meshes} meshes")
#             break

#     print(f"Loaded {len(meshes)} meshes")
#     return meshes





# def get_camera_data(stage, camera_path, time_code):
#     """Extract camera intrinsics and extrinsics from USD camera using Gf.Camera."""
#     camera_prim = stage.GetPrimAtPath(camera_path)
#     if not camera_prim or not camera_prim.IsA(UsdGeom.Camera):
#         raise RuntimeError(f"Camera not found at {camera_path}")

#     usd_camera = UsdGeom.Camera(camera_prim)
    
#     # Get the Gf.Camera which properly computes the camera frustum
#     gf_camera = usd_camera.GetCamera(time_code)
#     frustum = gf_camera.frustum  # Property, not method
    
#     # Get intrinsics
#     focal_length = usd_camera.GetFocalLengthAttr().Get(time_code)
#     horizontal_aperture = usd_camera.GetHorizontalApertureAttr().Get(time_code)
#     vertical_aperture = usd_camera.GetVerticalApertureAttr().Get(time_code)

#     print(f"Camera intrinsics at frame {time_code}:")
#     print(f"  Focal length: {focal_length} mm")
#     print(f"  Horizontal aperture: {horizontal_aperture} mm")
#     print(f"  Vertical aperture: {vertical_aperture} mm")

#     # Compute FOV from aperture and focal length
#     vertical_fov = 2.0 * math.atan(vertical_aperture / (2.0 * focal_length))
#     horizontal_fov = 2.0 * math.atan(horizontal_aperture / (2.0 * focal_length))
#     print(f"  Vertical FOV: {math.degrees(vertical_fov):.2f} degrees")
#     print(f"  Horizontal FOV: {math.degrees(horizontal_fov):.2f} degrees")

#     # Get the camera-to-world transform from gf_camera
#     # This is the proper way to get the camera's world space orientation
#     cam_to_world = gf_camera.transform
    
#     # Extract position (translation component)
#     cam_pos = Gf.Vec3d(cam_to_world[3][0], cam_to_world[3][1], cam_to_world[3][2])
    
#     # Extract camera axes from the rotation part of the transform
#     # In USD, the camera looks down -Z in local space
#     # The transform columns give us where local axes point in world space
#     # Matrix is row-major: m[row][col], so column i is m[0][i], m[1][i], m[2][i]
#     cam_right = Gf.Vec3d(cam_to_world[0][0], cam_to_world[1][0], cam_to_world[2][0])
#     cam_up = Gf.Vec3d(cam_to_world[0][1], cam_to_world[1][1], cam_to_world[2][1])
#     cam_back = Gf.Vec3d(cam_to_world[0][2], cam_to_world[1][2], cam_to_world[2][2])
#     cam_forward = -cam_back
    
#     # Normalize (in case of scale)
#     cam_right = cam_right.GetNormalized()
#     cam_up = cam_up.GetNormalized()
#     cam_forward = cam_forward.GetNormalized()

#     print(f"Camera world transform at frame {time_code}:")
#     print(f"  Position: {cam_pos}")
#     print(f"  Forward (look) direction: {cam_forward}")
#     print(f"  Up direction: {cam_up}")
#     print(f"  Right direction: {cam_right}")

#     return {
#         "focal_length": focal_length,
#         "horizontal_aperture": horizontal_aperture,
#         "vertical_aperture": vertical_aperture,
#         "vertical_fov": vertical_fov,
#         "horizontal_fov": horizontal_fov,
#         "position": np.array([cam_pos[0], cam_pos[1], cam_pos[2]]),
#         "right": np.array([cam_right[0], cam_right[1], cam_right[2]]),
#         "up": np.array([cam_up[0], cam_up[1], cam_up[2]]),
#         "forward": np.array([cam_forward[0], cam_forward[1], cam_forward[2]]),
#     }


# def camera_axes_to_transform(position, right, up, forward):
#     """Convert camera position and axes to warp transform (position, quaternion).
    
#     The camera looks down -Z in local space, so:
#     - right = local +X in world space
#     - up = local +Y in world space  
#     - forward = local -Z in world space (i.e., -forward = local +Z)
#     """
#     # Build rotation matrix from camera axes
#     # Column 0 = right (local +X), Column 1 = up (local +Y), Column 2 = -forward (local +Z)
#     rot_mat = np.column_stack([right, up, -forward])
    
#     # Ensure orthonormal (normalize columns)
#     for i in range(3):
#         rot_mat[:, i] /= np.linalg.norm(rot_mat[:, i])

#     # Convert rotation matrix to quaternion
#     trace = rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]
#     if trace > 0:
#         s = 0.5 / math.sqrt(trace + 1.0)
#         w = 0.25 / s
#         x = (rot_mat[2, 1] - rot_mat[1, 2]) * s
#         y = (rot_mat[0, 2] - rot_mat[2, 0]) * s
#         z = (rot_mat[1, 0] - rot_mat[0, 1]) * s
#     else:
#         if rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
#             s = 2.0 * math.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
#             w = (rot_mat[2, 1] - rot_mat[1, 2]) / s
#             x = 0.25 * s
#             y = (rot_mat[0, 1] + rot_mat[1, 0]) / s
#             z = (rot_mat[0, 2] + rot_mat[2, 0]) / s
#         elif rot_mat[1, 1] > rot_mat[2, 2]:
#             s = 2.0 * math.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
#             w = (rot_mat[0, 2] - rot_mat[2, 0]) / s
#             x = (rot_mat[0, 1] + rot_mat[1, 0]) / s
#             y = 0.25 * s
#             z = (rot_mat[1, 2] + rot_mat[2, 1]) / s
#         else:
#             s = 2.0 * math.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
#             w = (rot_mat[1, 0] - rot_mat[0, 1]) / s
#             x = (rot_mat[0, 2] + rot_mat[2, 0]) / s
#             y = (rot_mat[1, 2] + rot_mat[2, 1]) / s
#             z = 0.25 * s

#     return wp.transformf(wp.vec3f(*position), wp.quatf(x, y, z, w))


# def compute_camera_rays(width, height, vertical_fov):
#     """Compute pinhole camera rays using vertical FOV."""
#     rays = np.zeros((1, height, width, 2, 3), dtype=np.float32)
#     aspect = width / height
#     h = math.tan(vertical_fov / 2.0)

#     for py in range(height):
#         for px in range(width):
#             u = (px + 0.5) / width - 0.5
#             v = (py + 0.5) / height - 0.5
#             # Camera looks down -Z, with Y up in camera space
#             direction = np.array([u * 2.0 * h * aspect, -v * 2.0 * h, -1.0], dtype=np.float32)
#             direction /= np.linalg.norm(direction)
#             rays[0, py, px, 0] = [0, 0, 0]  # origin
#             rays[0, py, px, 1] = direction

#     return wp.array(rays, dtype=wp.vec3f)


# def setup_render_context(mesh_data):
#     """Create and configure RenderContext with mesh data."""
#     num_shapes = len(mesh_data)
#     print(f"Setting up render context with {num_shapes} shapes...")

#     # Create Warp meshes and compute bounds
#     warp_meshes = []
#     mesh_bounds = np.zeros((num_shapes, 2, 3), dtype=np.float32)

#     for i, (verts, tris, name) in enumerate(mesh_data):
#         mesh = wp.Mesh(
#             points=wp.array(verts, dtype=wp.vec3f),
#             indices=wp.array(tris.flatten(), dtype=wp.int32),
#         )
#         warp_meshes.append(mesh)
#         mesh_bounds[i, 0] = verts.min(axis=0)
#         mesh_bounds[i, 1] = verts.max(axis=0)

#     # Create RenderContext
#     ctx = RenderContext(
#         width=WIDTH,
#         height=HEIGHT,
#         enable_textures=False,
#         enable_shadows=True,
#         enable_ambient_lighting=True,
#         enable_particles=False,
#         num_worlds=1,
#         num_cameras=1,
#     )

#     # Populate shape data
#     ctx.num_shapes = num_shapes
#     ctx.mesh_ids = wp.array([m.id for m in warp_meshes], dtype=wp.uint64)
#     ctx.mesh_bounds = wp.array(mesh_bounds, dtype=wp.vec3f)
#     ctx.shape_types = wp.array([RenderShapeType.MESH] * num_shapes, dtype=wp.int32)
#     ctx.shape_enabled = wp.array(list(range(num_shapes)), dtype=wp.uint32)
#     ctx.shape_world_index = wp.array([0] * num_shapes, dtype=wp.int32)
#     ctx.shape_mesh_indices = wp.array(list(range(num_shapes)), dtype=wp.int32)
#     ctx.shape_materials = wp.array([-1] * num_shapes, dtype=wp.int32)

#     # Identity transforms (vertices already in world space)
#     transforms = [wp.transformf(wp.vec3f(0, 0, 0), wp.quatf(0, 0, 0, 1))] * num_shapes
#     ctx.shape_transforms = wp.array(transforms, dtype=wp.transformf)
#     ctx.shape_sizes = wp.array([[1, 1, 1]] * num_shapes, dtype=wp.vec3f)

#     # Random colors per mesh
#     np.random.seed(42)
#     colors = np.random.rand(num_shapes, 4).astype(np.float32) * 0.5 + 0.5
#     colors[:, 3] = 1.0
#     ctx.shape_colors = wp.array(colors, dtype=wp.vec4f)

#     # Directional light
#     ctx.lights_active = wp.array([True], dtype=wp.bool)
#     ctx.lights_type = wp.array([1], dtype=wp.int32)  # DIRECTIONAL
#     ctx.lights_cast_shadow = wp.array([True], dtype=wp.bool)
#     ctx.lights_position = wp.array([[0, 0, 0]], dtype=wp.vec3f)
#     ctx.lights_orientation = wp.array([[-0.577, 0.577, -0.577]], dtype=wp.vec3f)

#     return ctx, warp_meshes


# def render_frame(ctx, camera_data, frame_num):
#     """Render a single frame and save to PNG."""
#     # Get camera transform from camera axes
#     cam_transform = camera_axes_to_transform(
#         camera_data["position"],
#         camera_data["right"],
#         camera_data["up"],
#         camera_data["forward"],
#     )

#     # Camera transforms: shape (num_cameras, num_worlds) = (1, 1)
#     camera_transforms = wp.array([[cam_transform]], dtype=wp.transformf)

#     # Compute camera rays using the camera's vertical FOV
#     camera_rays = compute_camera_rays(WIDTH, HEIGHT, camera_data["vertical_fov"])

#     # Create output buffers
#     color_image = ctx.create_color_image_output()

#     print(f"Rendering frame {frame_num}...")
#     ctx.render(
#         camera_transforms=camera_transforms,
#         camera_rays=camera_rays,
#         color_image=color_image,
#         refit_bvh=True,
#         clear_data=ClearData(clear_color=0xFF404040),
#     )

#     # Convert to image
#     pixels = color_image.numpy()[0, 0]
#     pixels = pixels.reshape(HEIGHT, WIDTH)

#     r = (pixels >> 0) & 0xFF
#     g = (pixels >> 8) & 0xFF
#     b = (pixels >> 16) & 0xFF
#     rgba = np.stack([r, g, b], axis=-1).astype(np.uint8)

#     img = Image.fromarray(rgba, mode="RGB")
#     output_path = OUTPUT_DIR / f"debug_lantern_camera.{frame_num}.png"
#     img.save(output_path)
#     print(f"Saved: {output_path}")

#     return img


# def main():
#     print(f"Loading USD: {USD_FILE}")
#     stage = Usd.Stage.Open(USD_FILE)
#     if not stage:
#         raise RuntimeError("Failed to open USD file")

#     # Load meshes (use first frame's time code for geometry)
#     time_code = Usd.TimeCode(FRAMES[0])
#     mesh_data = load_meshes_from_usd(stage, time_code, max_meshes=None)  # Load ALL meshes
#     if not mesh_data:
#         print("No meshes found!")
#         return

#     # Set up render context (only once, meshes don't change)
#     ctx, warp_meshes = setup_render_context(mesh_data)

#     # Render each frame
#     for frame in FRAMES:
#         time_code = Usd.TimeCode(frame)
#         camera_data = get_camera_data(stage, CAMERA_PATH, time_code)
#         render_frame(ctx, camera_data, frame)

#     print("\nDone!")


# if __name__ == "__main__":
#     main()

