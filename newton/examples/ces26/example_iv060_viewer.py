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
# Example IV060 USD Scene Viewer
#
# Loads a USD scene and displays it using the standard Newton viewer.
# This is a simplified version of example_iv060.py that focuses on
# loading USD meshes into Newton and viewing them with the standard
# OpenGL viewer.
#
# Command: python -m newton.examples ces26/example_iv060_viewer
# Direct: python newton/examples/ces26/example_iv060_viewer.py
#
###########################################################################

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.usd

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


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0

        # Get settings from args
        proxy_mode = args.proxy_mode if hasattr(args, "proxy_mode") else "only"
        max_meshes = args.max_meshes if hasattr(args, "max_meshes") else None

        # Open the USD stage
        print(f"Loading USD file: {USD_FILE_PATH}")
        self.usd_stage = Usd.Stage.Open(USD_FILE_PATH)
        
        if self.usd_stage is None:
            raise RuntimeError(f"Failed to open USD stage: {USD_FILE_PATH}")
        
        # Find all meshes in the USD
        print(f"\nSearching for meshes (proxy_mode={proxy_mode}, max_meshes={max_meshes})...")
        meshes = find_all_meshes(self.usd_stage, verbose=False, proxy_mode=proxy_mode, max_meshes=max_meshes)
        print(f"Found {len(meshes)} meshes to load")

        if len(meshes) == 0:
            raise RuntimeError("No meshes found in USD file")

        # Build the Newton model
        builder = newton.ModelBuilder()

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
                body = builder.add_body(xform=wp.transform(p=pos, q=rot), key=mesh_data["path"])
                builder.add_shape_mesh(body, mesh=mesh)
                mesh_count += 1
            except Exception as e:
                print(f"  Could not add mesh {mesh_data['path']}: {e}")

        print(f"Added {mesh_count} meshes to Newton model")

        if mesh_count == 0:
            raise RuntimeError("No meshes could be added to the Newton model")

        # Finalize the model
        print("Finalizing model...")
        self.model = builder.finalize()
        print(f"Model created: {self.model.body_count} bodies, {self.model.shape_count} shapes")

        # Create state
        self.state = self.model.state()

        # Set model on viewer
        self.viewer.set_model(self.model)

        # Position camera based on scene bounds
        self._setup_camera()

    def _setup_camera(self):
        """Position the camera to view the entire scene."""
        body_q = self.state.body_q.numpy()
        if len(body_q) == 0:
            return

        # Calculate scene bounds from body positions
        positions = body_q[:, :3]
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        center = (min_pos + max_pos) / 2
        extent = max_pos - min_pos

        print(f"Scene bounds: min={min_pos}, max={max_pos}")
        print(f"Scene center: {center}")
        print(f"Scene extent: {extent}")

        # Calculate camera distance based on scene size
        max_extent = max(extent[0], extent[1], extent[2])
        cam_distance = max_extent * 1.5

        # Position camera to view the scene
        # For Z-up: camera at (center_x, center_y - distance, center_z + distance/3)
        cam_pos = wp.vec3(
            float(center[0]),
            float(center[1] - cam_distance * 0.5),
            float(center[2] + cam_distance * 0.3)
        )

        # Calculate pitch and yaw to look at center
        dx = float(center[0]) - float(cam_pos[0])
        dy = float(center[1]) - float(cam_pos[1])
        dz = float(center[2]) - float(cam_pos[2])

        yaw = np.degrees(np.arctan2(dy, dx))
        horizontal_dist = np.sqrt(dx * dx + dy * dy)
        pitch = np.degrees(np.arctan2(dz, horizontal_dist))

        self.viewer.set_camera(pos=cam_pos, pitch=pitch, yaw=yaw)
        print(f"Camera positioned at: {cam_pos}")
        print(f"Camera yaw: {yaw:.1f}, pitch: {pitch:.1f}")

    def step(self):
        # No simulation - this is just a static scene viewer
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()
        self.sim_time += self.frame_dt

    def gui(self, ui):
        ui.text(f"Bodies: {self.model.body_count}")
        ui.text(f"Shapes: {self.model.shape_count}")

    def test_final(self):
        # Basic test: verify we have a valid model
        assert self.model.body_count > 0, "Model should have at least one body"
        assert self.model.shape_count > 0, "Model should have at least one shape"


if __name__ == "__main__":
    # Create parser with example-specific arguments
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--proxy-mode",
        type=str,
        default="only",
        choices=["skip", "only", "all"],
        help="How to handle proxy meshes: 'skip' (high-poly only), 'only' (low-poly proxies only), 'all' (all meshes)",
    )
    parser.add_argument(
        "--max-meshes",
        type=int,
        default=200,
        help="Maximum number of meshes to load (for performance)",
    )

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create and run example
    example = Example(viewer, args)
    newton.examples.run(example, args)
