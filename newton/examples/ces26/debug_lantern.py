# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Simple USD mesh debug script using polyscope.

Finds all meshes under "HangingLanternE_01" and displays them.
Shows only "beauty" geometry (skips proxy meshes).

Usage: python newton/examples/ces26/debug_lantern.py
"""

import numpy as np
import polyscope as ps
from pxr import Usd, UsdGeom

# Path to USD file
USD_FILE = r"C:\Users\chorvath\Downloads\20251219_iv060_flat_01\Collected_iv060_flat_01\iv060_flat_01.usd"

# Target prim to search under
TARGET_PRIM = "HangingLanternE_01"


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


def main():
    print(f"Loading: {USD_FILE}")
    stage = Usd.Stage.Open(USD_FILE)
    if not stage:
        print("Failed to open USD file")
        return

    # Find all prims containing our target name
    print(f"\nSearching for prims containing '{TARGET_PRIM}'...")
    
    meshes_found = []
    
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        
        # Check if this prim is under our target
        if TARGET_PRIM not in path_str:
            continue
            
        # Skip proxy meshes - we want beauty geo
        if "/proxy/" in path_str.lower():
            print(f"  SKIP (proxy): {path_str}")
            continue
        
        if not prim.IsA(UsdGeom.Mesh):
            continue
            
        try:
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            
            if not points or not indices or not counts:
                print(f"  SKIP (no data): {path_str}")
                continue
            
            # Get world transform
            xform = UsdGeom.Xformable(prim)
            world_mat = np.array(xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            
            # Transform points to world space
            pts = np.array(points, dtype=np.float32)
            pts_h = np.hstack([pts, np.ones((len(pts), 1))])  # homogeneous
            pts_world = (world_mat @ pts_h.T).T[:, :3]
            
            # Triangulate
            tris = triangulate(np.array(indices), np.array(counts))
            
            print(f"  FOUND: {path_str}")
            print(f"         {len(pts)} verts, {len(tris)} triangles")
            
            meshes_found.append({
                "name": path_str.split("/")[-1],
                "path": path_str,
                "vertices": pts_world.astype(np.float64),
                "faces": tris,
            })
            
        except Exception as e:
            print(f"  ERROR: {path_str}: {e}")
    
    print(f"\nFound {len(meshes_found)} meshes")
    
    if not meshes_found:
        print("No meshes found. Listing all prims with target name...")
        for prim in stage.Traverse():
            if TARGET_PRIM in str(prim.GetPath()):
                print(f"  {prim.GetPath()} [{prim.GetTypeName()}]")
        return
    
    # Display with polyscope
    ps.init()
    ps.set_up_dir("z_up")
    
    for m in meshes_found:
        ps.register_surface_mesh(m["name"], m["vertices"], m["faces"])
    
    print("\nPolyscope viewer opened. Close window to exit.")
    ps.show()


if __name__ == "__main__":
    main()

