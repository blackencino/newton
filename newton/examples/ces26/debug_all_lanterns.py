# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Simple USD mesh debug script using polyscope.

Finds all meshes with "Hanging" in their path and displays them.
Shows only "beauty" geometry (skips proxy meshes).

Usage: python newton/examples/ces26/debug_all_lanterns.py
"""

import numpy as np
import polyscope as ps
from pxr import Gf, Usd, UsdGeom

# Path to USD file
USD_FILE = r"C:\Users\chorvath\Downloads\20251219_iv060_flat_01\Collected_iv060_flat_01\iv060_flat_01.usd"

# Target pattern to search for (case-insensitive)
TARGET_PATTERN = "Hanging"

# Time code for sampling transforms
TIME_CODE = Usd.TimeCode(2920)


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


def rotate_points_by_quaternion(points: np.ndarray, quat_wxyz: tuple) -> np.ndarray:
    """
    Rotate points by a quaternion using vectorized operations.
    
    Args:
        points: (N, 3) array of points
        quat_wxyz: quaternion as (w, x, y, z) - USD/GfQuaternion convention
        
    Returns:
        (N, 3) array of rotated points
    """
    w, x, y, z = quat_wxyz
    q_vec = np.array([x, y, z], dtype=np.float64)
    
    # Rodrigues rotation formula via quaternion:
    # t = 2 * (q_xyz × v)
    # v' = v + w*t + (q_xyz × t)
    t = 2.0 * np.cross(q_vec, points)
    rotated = points + w * t + np.cross(q_vec, t)
    return rotated


def main():
    print(f"Loading: {USD_FILE}")
    stage = Usd.Stage.Open(USD_FILE)
    if not stage:
        print("Failed to open USD file")
        return
    
    # Flatten the stage to ensure all layers, references, and overrides are composed
    # print("Flattening stage to resolve all composition arcs...")
    # flattened_layer = stage.Flatten()
    # stage = Usd.Stage.Open(flattened_layer)

    xcache = UsdGeom.XformCache(TIME_CODE)

    # Find all prims containing our target pattern
    print(f"\nSearching for prims containing '{TARGET_PATTERN}'...")
    
    meshes_found = []

    # Traverse including instance proxies
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        path_str = str(prim.GetPath())
        
        # Check if this prim path contains our target pattern (case-insensitive)
        if TARGET_PATTERN.lower() not in path_str.lower():
            continue

        
        meshPrim = prim 
        # If this is an instance proxy, pull geometry from the prototype prim 
        if meshPrim.IsInstanceProxy(): 
            geomPrim = meshPrim.GetPrimInPrototype() 
        else: 
            geomPrim = meshPrim
        mesh = UsdGeom.Mesh(geomPrim)

        if mesh.ComputeEffectiveVisibility(UsdGeom.Tokens.render, TIME_CODE) == UsdGeom.Tokens.invisible:
            print(f"  SKIPPING effectively invisible: {path_str}")
            continue
            
        # Skip proxy meshes - we want beauty geo
        if "/proxy/" in path_str.lower():
            print(f"  SKIP (proxy): {path_str}")
            continue

            
        try:
            mesh = UsdGeom.Mesh(geomPrim)
            points = mesh.GetPointsAttr().Get(TIME_CODE)
            indices = mesh.GetFaceVertexIndicesAttr().Get(TIME_CODE)
            counts = mesh.GetFaceVertexCountsAttr().Get(TIME_CODE)
            
            if not points or not indices or not counts:
                print(f"  SKIP (no data): {path_str}")
                continue
            
            # IMPORTANT: world transform must be computed on the *proxy prim* (meshPrim) 
            # not on geomPrim (prototype)
            world_mat = xcache.GetLocalToWorldTransform(meshPrim)
            M = np.array(world_mat, dtype=np.float64)
            pts = np.array(points, dtype=np.float64)
            pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)

            # Row-vectors * matrix
            pts_world_h = pts_h @ M 
            pts_world = pts_world_h[:, :3]
            
            # Triangulate
            tris = triangulate(np.array(indices), np.array(counts))
            
            print(f"  FOUND: {path_str}")
            print(f"         {len(pts)} verts, {len(tris)} triangles")
            
            meshes_found.append({
                "name": path_str.split("/")[-1],
                "path": path_str,
                "vertices": pts_world,
                "faces": tris,
            })
            
        except Exception as e:
            print(f"  ERROR: {path_str}: {e}")
    
    print(f"\nFound {len(meshes_found)} meshes")
    
    if not meshes_found:
        print("No meshes found. Listing all prims with target pattern...")
        for prim in stage.Traverse():
            if TARGET_PATTERN.lower() in str(prim.GetPath()).lower():
                print(f"  {prim.GetPath()} [{prim.GetTypeName()}]")
        return

    # Check that all mesh names are unique. If not, warn and make unique.
    names = [m["name"] for m in meshes_found]
    if len(names) != len(set(names)):
        print("\nWARNING: Mesh names are not unique. Making names unique.")
        name_counts = {}
        for m in meshes_found:
            name = m["name"]
            if name in name_counts:
                name_counts[name] += 1
                m["name"] = f"{name}_{name_counts[name]}"
            else:
                name_counts[name] = 1  # first instance keeps name as is
    
    # Display with polyscope
    ps.init()
    ps.set_up_dir("z_up")
    
    for m in meshes_found:
        ps.register_surface_mesh(m["name"], m["vertices"], m["faces"])
    
    print("\nPolyscope viewer opened. Close window to exit.")
    ps.show()


if __name__ == "__main__":
    main()

