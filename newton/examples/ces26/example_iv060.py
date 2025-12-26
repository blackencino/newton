# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example IV060 USD Scene
#
# Loads a USD scene and renders it using the TiledCameraSensor class.
# The scene is loaded from a USD file and meshes are extracted and added
# as visual shapes to the Newton model.
#
# Command: python -m newton.examples ces26/example_iv060
# Headless debug: python newton/examples/ces26/example_iv060.py --headless-debug
#
###########################################################################

import argparse
import ctypes
import math
import sys

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.usd
from newton.sensors import TiledCameraSensor
from newton.viewer import ViewerGL

# Path to the USD file
USD_FILE_PATH = r"C:\Users\chorvath\Downloads\20251219_iv060_flat_01\Collected_iv060_flat_01\iv060_flat_01.usd"


def find_all_meshes(stage: Usd.Stage, verbose: bool = True, proxy_mode: str = "skip", max_meshes: int = None):
    """Find all mesh prims in the USD stage and extract their geometry.
    
    Args:
        stage: The USD stage to search
        verbose: Print debug info
        proxy_mode: How to handle proxy meshes:
            - "skip": Skip proxy meshes (use high-poly geo meshes)
            - "only": Only include proxy meshes (use low-poly versions)
            - "all": Include all meshes
        max_meshes: Maximum number of meshes to return (None for all)
    """
    meshes = []
    
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            path_str = str(prim.GetPath())
            is_proxy = "/proxy/" in path_str
            
            # Filter based on proxy_mode
            if proxy_mode == "skip" and is_proxy:
                if verbose:
                    print(f"  SKIP (proxy) {path_str}")
                continue
            elif proxy_mode == "only" and not is_proxy:
                if verbose:
                    print(f"  SKIP (not proxy) {path_str}")
                continue
            
            try:
                mesh_geom = UsdGeom.Mesh(prim)
                points = mesh_geom.GetPointsAttr().Get()
                indices = mesh_geom.GetFaceVertexIndicesAttr().Get()
                counts = mesh_geom.GetFaceVertexCountsAttr().Get()
                
                if points is None or len(points) == 0:
                    if verbose:
                        print(f"  SKIP {path_str}: No points")
                    continue
                    
                if indices is None or len(indices) == 0:
                    if verbose:
                        print(f"  SKIP {path_str}: No indices")
                    continue
                
                # Get world transform
                xformable = UsdGeom.Xformable(prim)
                world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                
                if verbose:
                    print(f"  FOUND {path_str}: {len(points)} vertices, {len(indices)} indices, {len(counts)} faces")
                
                meshes.append({
                    "path": path_str,
                    "prim": prim,
                    "points": np.array(points, dtype=np.float32),
                    "indices": np.array(indices, dtype=np.int32),
                    "counts": np.array(counts, dtype=np.int32),
                    "transform": np.array(world_transform, dtype=np.float32),
                })
                
                # Check if we've hit the limit
                if max_meshes is not None and len(meshes) >= max_meshes:
                    if verbose:
                        print(f"  ... stopping at {max_meshes} meshes")
                    break
                    
            except Exception as e:
                if verbose:
                    print(f"  ERROR {path_str}: {e}")
                continue
    
    return meshes


def triangulate_mesh(points, indices, counts):
    """Convert polygonal mesh to triangles using fan triangulation."""
    triangles = []
    idx = 0
    for count in counts:
        if count >= 3:
            # Fan triangulation: first vertex connects to all others
            for i in range(1, count - 1):
                triangles.append(indices[idx])
                triangles.append(indices[idx + i])
                triangles.append(indices[idx + i + 1])
        idx += count
    return np.array(triangles, dtype=np.int32)


def run_headless_debug(proxy_mode: str = "only", max_meshes: int = 100):
    """Run in headless debug mode to inspect the USD file without viewer."""
    print("=" * 60)
    print("HEADLESS DEBUG MODE")
    print("=" * 60)
    print(f"\nLoading USD file: {USD_FILE_PATH}")
    print(f"Options: proxy_mode={proxy_mode}, max_meshes={max_meshes}")
    
    try:
        stage = Usd.Stage.Open(USD_FILE_PATH)
        if stage is None:
            print("ERROR: Failed to open USD stage")
            return
        print("SUCCESS: USD stage opened")
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    # Check for cameras
    print("\n--- Cameras ---")
    cameras = [p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
    if cameras:
        for cam in cameras:
            print(f"  Camera: {cam.GetPath()}")
    else:
        print("  No cameras found")
    
    # Find all meshes
    print("\n--- Meshes ---")
    meshes = find_all_meshes(stage, verbose=True, proxy_mode=proxy_mode, max_meshes=max_meshes)
    print(f"\nTotal meshes found: {len(meshes)}")
    
    if len(meshes) == 0:
        print("\nNo meshes found in USD file. Nothing to render.")
        return
    
    # Summary stats
    total_vertices = sum(len(m["points"]) for m in meshes)
    total_faces = sum(len(m["counts"]) for m in meshes)
    print(f"Total vertices: {total_vertices}")
    print(f"Total faces: {total_faces}")
    
    # Try building a Newton model with the meshes
    print("\n--- Building Newton Model ---")
    builder = newton.ModelBuilder()
    
    mesh_count = 0
    for mesh_data in meshes:
        try:
            # Get the mesh using newton.usd.get_mesh
            mesh = newton.usd.get_mesh(mesh_data["prim"])
            
            # Add a body at the mesh's world position
            mat = mesh_data["transform"]
            pos = wp.vec3(float(mat[3, 0]), float(mat[3, 1]), float(mat[3, 2]))
            
            # Extract rotation from the transform matrix
            rot_mat = mat[:3, :3]
            rot = wp.quat_from_matrix(wp.mat33(rot_mat.T.flatten().tolist()))
            
            body = builder.add_body(xform=wp.transform(p=pos, q=rot), key=mesh_data["path"])
            builder.add_shape_mesh(body, mesh=mesh)
            mesh_count += 1
            print(f"  Added mesh: {mesh_data['path']}")
        except Exception as e:
            print(f"  ERROR adding {mesh_data['path']}: {e}")
    
    print(f"\nMeshes added to builder: {mesh_count}")
    
    if mesh_count > 0:
        model = builder.finalize()
        print(f"Model finalized: {model.body_count} bodies, {model.shape_count} shapes")
        
        # Check shape flags
        shape_flags = model.shape_flags.numpy()
        from newton._src.geometry import ShapeFlags
        visible_count = sum(1 for f in shape_flags if f & ShapeFlags.VISIBLE)
        print(f"Visible shapes: {visible_count} / {model.shape_count}")
        
        # Print scene bounds
        state = model.state()
        body_q = state.body_q.numpy()
        if len(body_q) > 0:
            positions = body_q[:, :3]
            min_pos = positions.min(axis=0)
            max_pos = positions.max(axis=0)
            print(f"Scene bounds: min={min_pos}, max={max_pos}")
            center = (min_pos + max_pos) / 2
            extent = max_pos - min_pos
            print(f"Scene center: {center}")
            print(f"Scene extent: {extent}")
    else:
        print("No meshes could be added to the model")
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


class Example:
    def __init__(self, viewer: ViewerGL):
        # Just 1 world for now while debugging
        self.num_worlds_per_row = 1
        self.num_worlds_per_col = 1
        self.num_worlds_total = self.num_worlds_per_row * self.num_worlds_per_col

        self.time = 0.0
        self.time_delta = 0.005
        self.image_output = 0
        self.texture_id = 0

        self.viewer = viewer
        if isinstance(self.viewer, ViewerGL):
            self.viewer.register_ui_callback(self.display, "free")

        # Open the USD stage
        print(f"Loading USD file: {USD_FILE_PATH}")
        self.usd_stage = Usd.Stage.Open(USD_FILE_PATH)
        
        # Find all meshes in the USD
        # Use proxy meshes (low-poly versions) for better performance
        print("\nSearching for proxy meshes in USD (low-poly versions)...")
        meshes = find_all_meshes(self.usd_stage, verbose=False, proxy_mode="only", max_meshes=200)
        print(f"Found {len(meshes)} proxy meshes to load")

        # Try to get camera from USD
        self.usd_camera = None
        self.usd_camera_fov = 45.0
        camera_prim = self.usd_stage.GetPrimAtPath("/World/TD060")
        if camera_prim and camera_prim.IsA(UsdGeom.Camera):
            self.usd_camera = UsdGeom.Camera(camera_prim)
            focal_length = self.usd_camera.GetFocalLengthAttr().Get()
            vertical_aperture = self.usd_camera.GetVerticalApertureAttr().Get()
            if focal_length and vertical_aperture:
                self.usd_camera_fov = 2.0 * math.atan(vertical_aperture / (2.0 * focal_length))
            print(f"Found USD camera at /World/TD060 with FOV: {math.degrees(self.usd_camera_fov):.1f} degrees")

        # Build the model by manually adding meshes
        builder = newton.ModelBuilder()

        for world_idx in range(self.num_worlds_total):
            builder.begin_world()
            
            mesh_count = 0
            for mesh_data in meshes:
                try:
                    # Get the mesh using newton.usd.get_mesh
                    mesh = newton.usd.get_mesh(mesh_data["prim"])
                    
                    # Get world transform from the mesh data
                    mat = mesh_data["transform"]
                    pos = wp.vec3(float(mat[3, 0]), float(mat[3, 1]), float(mat[3, 2]))
                    
                    # Extract rotation from the transform matrix
                    rot_mat = mat[:3, :3]
                    rot = wp.quat_from_matrix(wp.mat33(rot_mat.T.flatten().tolist()))
                    
                    # Add body and mesh shape
                    body = builder.add_body(xform=wp.transform(p=pos, q=rot), key=f"world{world_idx}_{mesh_data['path']}")
                    builder.add_shape_mesh(body, mesh=mesh)
                    mesh_count += 1
                except Exception as e:
                    if world_idx == 0:  # Only print errors for first world
                        print(f"  Could not add mesh {mesh_data['path']}: {e}")
            
            if world_idx == 0:
                print(f"Added {mesh_count} meshes to world {world_idx}")
            
            builder.end_world()

        print(f"\nFinalizing model...")
        self.model = builder.finalize()
        print(f"Model created: {self.model.body_count} bodies, {self.model.shape_count} shapes")
        
        # Debug: Check shape visibility flags
        from newton._src.geometry import ShapeFlags
        shape_flags = self.model.shape_flags.numpy()
        visible_count = sum(1 for f in shape_flags if f & ShapeFlags.VISIBLE)
        print(f"Visible shapes: {visible_count} / {self.model.shape_count}")
        
        self.state = self.model.state()
        
        # Debug: Print bounding box of all bodies and position camera
        body_q = self.state.body_q.numpy()
        if len(body_q) > 0:
            positions = body_q[:, :3]  # Extract position from transforms
            min_pos = positions.min(axis=0)
            max_pos = positions.max(axis=0)
            print(f"Scene bounds: min={min_pos}, max={max_pos}")
            center = (min_pos + max_pos) / 2
            extent = max_pos - min_pos
            print(f"Scene center: {center}")
            print(f"Scene extent: {extent}")
            
            # Position camera to view the scene
            if isinstance(self.viewer, ViewerGL):
                from pyglet.math import Vec3 as PyVec3
                
                # Calculate camera distance based on scene size
                max_extent = max(extent[0], extent[1], extent[2])
                cam_distance = max_extent * 1.5
                
                # Position camera above and behind the scene center
                # For Z-up: camera at (center_x, center_y - distance, center_z + distance/2)
                cam_pos = PyVec3(
                    float(center[0]),
                    float(center[1] - cam_distance * 0.5),
                    float(center[2] + cam_distance * 0.3)
                )
                self.viewer.camera.pos = cam_pos
                
                # Calculate yaw and pitch to look at center
                # Direction from camera to center
                dx = float(center[0]) - cam_pos.x
                dy = float(center[1]) - cam_pos.y
                dz = float(center[2]) - cam_pos.z
                
                # Yaw is the horizontal angle (around Z for Z-up)
                self.viewer.camera.yaw = np.degrees(np.arctan2(dy, dx))
                
                # Pitch is the vertical angle
                horizontal_dist = np.sqrt(dx*dx + dy*dy)
                self.viewer.camera.pitch = np.degrees(np.arctan2(dz, horizontal_dist))
                
                print(f"Camera positioned at: {self.viewer.camera.pos}")
                print(f"Camera yaw: {self.viewer.camera.yaw:.1f}, pitch: {self.viewer.camera.pitch:.1f}")

        self.viewer.set_model(self.model)

        self.ui_padding = 10
        self.ui_side_panel_width = 300

        sensor_render_width = 64
        sensor_render_height = 64

        if isinstance(self.viewer, ViewerGL):
            display_width = self.viewer.ui.io.display_size[0] - self.ui_side_panel_width - self.ui_padding * 4
            display_height = self.viewer.ui.io.display_size[1] - self.ui_padding * 2

            sensor_render_width = int(display_width // self.num_worlds_per_row)
            sensor_render_height = int(display_height // self.num_worlds_per_col)

        # Setup Tiled Camera Sensor
        self.tiled_camera_sensor = TiledCameraSensor(
            model=self.model,
            num_cameras=1,
            width=sensor_render_width,
            height=sensor_render_height,
            options=TiledCameraSensor.Options(
                default_light=True, default_light_shadows=True, colors_per_shape=True, checkerboard_texture=True
            ),
        )

        fov = 45.0
        if isinstance(self.viewer, ViewerGL):
            fov = self.viewer.camera.fov

        self.camera_rays = self.tiled_camera_sensor.compute_pinhole_camera_rays(math.radians(fov))
        self.tiled_camera_sensor_color_image = self.tiled_camera_sensor.create_color_image_output()
        self.tiled_camera_sensor_depth_image = self.tiled_camera_sensor.create_depth_image_output()
        self.tiled_camera_sensor_normal_image = self.tiled_camera_sensor.create_normal_image_output()
        self.tiled_camera_sensor_shape_index_image = self.tiled_camera_sensor.create_shape_index_image_output()

        if isinstance(self.viewer, ViewerGL):
            self.create_texture()

    def step(self):
        self.time += self.time_delta

    def render(self):
        self.render_sensors()
        self.viewer.begin_frame(0.0)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def render_sensors(self):
        self.tiled_camera_sensor.render(
            self.state,
            self.get_camera_transforms(),
            self.camera_rays,
            color_image=self.tiled_camera_sensor_color_image,
            depth_image=self.tiled_camera_sensor_depth_image,
            normal_image=self.tiled_camera_sensor_normal_image,
            shape_index_image=self.tiled_camera_sensor_shape_index_image,
        )
        self.update_texture()

    def get_camera_transforms(self) -> wp.array(dtype=wp.transformf):
        if isinstance(self.viewer, ViewerGL):
            return wp.array(
                [
                    [
                        wp.transformf(
                            self.viewer.camera.pos,
                            wp.quat_from_matrix(wp.mat33f(self.viewer.camera.get_view_matrix().reshape(4, 4)[:3, :3])),
                        )
                    ]
                    * self.num_worlds_total
                ],
                dtype=wp.transformf,
            )
        return wp.array(
            [[wp.transformf(wp.vec3f(10.0, 0.0, 2.0), wp.quatf(0.5, 0.5, 0.5, 0.5))] * self.num_worlds_total],
            dtype=wp.transformf,
        )

    def create_texture(self):
        import OpenGL.GL as gl
        
        width = self.tiled_camera_sensor.render_context.width * self.num_worlds_per_row
        height = self.tiled_camera_sensor.render_context.height * self.num_worlds_per_col

        self.texture_id = gl.glGenTextures(1)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self.pixel_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer)
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, width * height * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

        self.texture_buffer = wp.RegisteredGLBuffer(self.pixel_buffer)

    def update_texture(self):
        import OpenGL.GL as gl
        
        if not self.texture_id:
            return

        texture_buffer = self.texture_buffer.map(
            dtype=wp.uint8,
            shape=(
                self.num_worlds_per_col * self.tiled_camera_sensor.render_context.height,
                self.num_worlds_per_row * self.tiled_camera_sensor.render_context.width,
                4,
            ),
        )
        if self.image_output == 0:
            self.tiled_camera_sensor.flatten_color_image_to_rgba(
                self.tiled_camera_sensor_color_image, texture_buffer, self.num_worlds_per_row
            )
        elif self.image_output == 1:
            self.tiled_camera_sensor.flatten_depth_image_to_rgba(
                self.tiled_camera_sensor_depth_image, texture_buffer, self.num_worlds_per_row
            )
        elif self.image_output == 2:
            self.tiled_camera_sensor.flatten_normal_image_to_rgba(
                self.tiled_camera_sensor_normal_image, texture_buffer, self.num_worlds_per_row
            )
        elif self.image_output == 3:
            self.tiled_camera_sensor.flatten_color_image_to_rgba(
                self.tiled_camera_sensor_shape_index_image, texture_buffer, self.num_worlds_per_row
            )
        self.texture_buffer.unmap()

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.tiled_camera_sensor.render_context.width * self.num_worlds_per_row,
            self.tiled_camera_sensor.render_context.height * self.num_worlds_per_col,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def test_final(self):
        self.render_sensors()

        color_image = self.tiled_camera_sensor_color_image.numpy()
        assert color_image.shape == (self.num_worlds_total, 1, 64 * 64)
        assert color_image.min() < color_image.max()

        depth_image = self.tiled_camera_sensor_depth_image.numpy()
        assert depth_image.shape == (self.num_worlds_total, 1, 64 * 64)
        assert depth_image.min() < depth_image.max()

    def gui(self, ui):
        if ui.radio_button("Show Color Output", self.image_output == 0):
            self.image_output = 0
        if ui.radio_button("Show Depth Output", self.image_output == 1):
            self.image_output = 1
        if ui.radio_button("Show Normal Output", self.image_output == 2):
            self.image_output = 2
        if ui.radio_button("Show Shape Index Output", self.image_output == 3):
            self.image_output = 3

    def display(self, imgui):
        line_color = imgui.get_color_u32(imgui.Col_.window_bg)

        width = self.viewer.ui.io.display_size[0] - self.ui_side_panel_width - self.ui_padding * 4
        height = self.viewer.ui.io.display_size[1] - self.ui_padding * 2

        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(self.viewer.ui.io.display_size)

        flags = (
            imgui.WindowFlags_.no_title_bar.value
            | imgui.WindowFlags_.no_mouse_inputs.value
            | imgui.WindowFlags_.no_bring_to_front_on_focus.value
            | imgui.WindowFlags_.no_scrollbar.value
        )

        if imgui.begin("Sensors", flags=flags):
            pos_x = self.ui_side_panel_width + self.ui_padding * 2
            pos_y = self.ui_padding

            if self.texture_id > 0:
                imgui.set_cursor_pos(imgui.ImVec2(pos_x, pos_y))
                imgui.image(imgui.ImTextureRef(self.texture_id), imgui.ImVec2(width, height))

            draw_list = imgui.get_window_draw_list()
            for x in range(1, self.num_worlds_per_row):
                draw_list.add_line(
                    imgui.ImVec2(pos_x + x * (width / self.num_worlds_per_row), pos_y),
                    imgui.ImVec2(pos_x + x * (width / self.num_worlds_per_row), pos_y + height),
                    line_color,
                    2.0,
                )
            for y in range(1, self.num_worlds_per_col):
                draw_list.add_line(
                    imgui.ImVec2(pos_x, pos_y + y * (height / self.num_worlds_per_col)),
                    imgui.ImVec2(pos_x + width, pos_y + y * (height / self.num_worlds_per_col)),
                    line_color,
                    2.0,
                )

        imgui.end()


if __name__ == "__main__":
    # Check for headless debug mode first
    if "--headless-debug" in sys.argv:
        run_headless_debug()
        sys.exit(0)
    
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer)

    newton.examples.run(example, args)
