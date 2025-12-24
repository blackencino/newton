# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Test loading the geometry cache and examining groups.

Usage: uv run python newton/examples/ces26/run_preprocess_test2.py
"""

import time
from pathlib import Path

import numpy as np

# Paths
OUTPUT_DIR = Path(r"D:\ces26_data\td060\v04")
CACHE_WITH_GEOMETRY = OUTPUT_DIR / "preprocess_with_geometry.npz"
CACHE_WITHOUT_GEOMETRY = OUTPUT_DIR / "preprocess_metadata_only.npz"


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def main():
    from dynamic_segmentation.preprocess import (
        GroupCategory,
        load_preprocessing_cache,
    )
    
    print("=" * 70)
    print("Loading and Examining Preprocessing Cache")
    print("=" * 70)
    print()
    
    # =========================================================================
    # Test 1: Load metadata-only cache
    # =========================================================================
    print("Test 1: Loading metadata-only cache...")
    start_time = time.time()
    metadata_list, groups, camera_curve = load_preprocessing_cache(CACHE_WITHOUT_GEOMETRY)
    load_time = time.time() - start_time
    print(f"  Loaded in {load_time:.3f}s")
    print(f"  {len(metadata_list)} diegetics, {len(groups)} groups")
    print()
    
    # =========================================================================
    # Test 2: Load geometry cache (may be slow)
    # =========================================================================
    print("Test 2: Loading geometry cache (may be slow)...")
    start_time = time.time()
    data = np.load(CACHE_WITH_GEOMETRY, allow_pickle=True)
    load_time = time.time() - start_time
    
    # Check if geometry is present
    has_geometry = "mesh_vertices" in data.files
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Has geometry: {has_geometry}")
    
    if has_geometry:
        vertices = data["mesh_vertices"]
        faces = data["mesh_faces"]
        vertex_offsets = data["mesh_vertex_offsets"]
        face_offsets = data["mesh_face_offsets"]
        
        print(f"  Vertex array shape: {vertices.shape} ({format_size(vertices.nbytes)})")
        print(f"  Face array shape: {faces.shape} ({format_size(faces.nbytes)})")
        print(f"  Number of meshes: {len(vertex_offsets) - 1}")
        
        # Verify we can extract individual meshes
        mesh_idx = 0
        v_start, v_end = vertex_offsets[mesh_idx], vertex_offsets[mesh_idx + 1]
        f_start, f_end = face_offsets[mesh_idx], face_offsets[mesh_idx + 1]
        
        print(f"  First mesh vertices: {v_end - v_start:,}")
        print(f"  First mesh triangles: {f_end - f_start:,}")
    print()
    
    # =========================================================================
    # Examine Groups
    # =========================================================================
    print("=" * 70)
    print("Examining Groups")
    print("=" * 70)
    print()
    
    # Sort by member count (largest first)
    sorted_groups = sorted(groups.values(), key=lambda g: -g.member_count)
    
    print("Top 10 groups by member count:")
    print("-" * 70)
    for i, g in enumerate(sorted_groups[:10]):
        cat_str = g.category.value[:6]  # Short category name
        print(f"{i+1:2}. {g.unique_name:<40} {g.member_count:4} meshes  [{cat_str}]")
        print(f"    Ancestor: {g.common_ancestor_path}")
        print(f"    Color: RGB({g.objectid_color[0]:.3f}, {g.objectid_color[1]:.3f}, {g.objectid_color[2]:.3f})")
        print(f"    Path danger: {g.path_danger:.1f}")
        print()
    
    # =========================================================================
    # Look at the big StarWarsSet_01 group
    # =========================================================================
    print("=" * 70)
    print("Examining StarWarsSet_01 Group (GROUND_TERRAIN)")
    print("=" * 70)
    print()
    
    terrain_groups = [g for g in groups.values() if g.category == GroupCategory.GROUND_TERRAIN]
    for g in terrain_groups:
        print(f"Group: {g.unique_name}")
        print(f"  Member count: {g.member_count}")
        print(f"  Total vertices: {g.total_vertex_count:,}")
        print(f"  Total triangles: {g.total_triangle_count:,}")
        print(f"  Common ancestor: {g.common_ancestor_path}")
        print()
        
        # Count how many mesh paths contain specific patterns
        terrain_meshes = [p for p in g.member_paths if "/terrain" in p.lower()]
        cloth_meshes = [p for p in g.member_paths if "/cloth" in p.lower()]
        crate_meshes = [p for p in g.member_paths if "/crate" in p.lower()]
        pottery_meshes = [p for p in g.member_paths if "/pottery" in p.lower()]
        tent_meshes = [p for p in g.member_paths if "/tent" in p.lower()]
        
        print("  Mesh pattern breakdown:")
        print(f"    Terrain: {len(terrain_meshes)}")
        print(f"    Cloth: {len(cloth_meshes)}")
        print(f"    Crate: {len(crate_meshes)}")
        print(f"    Pottery: {len(pottery_meshes)}")
        print(f"    Tent: {len(tent_meshes)}")
        print(f"    Other: {g.member_count - len(terrain_meshes) - len(cloth_meshes) - len(crate_meshes) - len(pottery_meshes) - len(tent_meshes)}")
        print()
        
        # Show sample terrain paths
        if terrain_meshes:
            print("  Sample terrain paths:")
            for p in terrain_meshes[:5]:
                print(f"    {p}")
        print()
        
        # Show sample other paths 
        other_paths = [p for p in g.member_paths if "/terrain" not in p.lower()]
        if other_paths:
            print("  Sample other paths in this group:")
            for p in other_paths[:5]:
                print(f"    {p}")
        print()
    
    # =========================================================================
    # Look at path danger values
    # =========================================================================
    print("=" * 70)
    print("Path Danger Analysis")
    print("=" * 70)
    print()
    
    sorted_by_danger = sorted(groups.values(), key=lambda g: g.path_danger)
    
    print("10 closest groups to camera path:")
    print("-" * 70)
    for i, g in enumerate(sorted_by_danger[:10]):
        cat_str = g.category.value[:6]
        danger_str = f"{g.path_danger:8.1f}" if g.path_danger < 1e6 else "      inf"
        print(f"{i+1:2}. {g.unique_name:<35} danger={danger_str}  [{cat_str}]")
    print()
    
    # Check for negative distances (camera inside ellipsoid)
    inside_groups = [g for g in groups.values() if g.path_danger < 0]
    if inside_groups:
        print(f"WARNING: Camera passes through {len(inside_groups)} group ellipsoids:")
        for g in inside_groups:
            print(f"  - {g.unique_name}: {g.path_danger:.1f}")
    print()
    
    print("Done!")


if __name__ == "__main__":
    main()

