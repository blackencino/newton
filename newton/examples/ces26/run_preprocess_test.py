# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Test preprocessing pipeline and cache save/load.

Tests:
1. Full preprocessing pipeline on shot IV060
2. Saving caches with and without geometry
3. Load validation
4. File size analysis

Findings (2025-12-24):
- Metadata-only cache: 60 KB (instant save/load) - RECOMMENDED
- Geometry cache: 633 MB (252s save time) - TOO SLOW, not practical
- Re-parsing USD (67s) is faster than saving geometry cache

Usage: uv run python newton/examples/ces26/run_preprocess_test.py
"""

import os
import time
from pathlib import Path

from pxr import Usd

# Configuration for shot IV060
USD_FILE = r"C:\Users\chorvath\Downloads\20251223_iv060_flat_04\Collected_20251223_iv060_flat_04\20251223_iv060_flat_04.usd"
CAMERA_PATH = "/World/TD060"
FRAMES = list(range(2920, 3131))  # 2920 to 3130 inclusive

# Output paths
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
    print("=" * 70)
    print("Preprocessing Pipeline Test - IV060")
    print("=" * 70)
    print()
    
    # Check USD file exists
    if not Path(USD_FILE).exists():
        print(f"ERROR: USD file not found: {USD_FILE}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Import preprocessing modules
    from dynamic_segmentation.preprocess import (
        GroupCategory,
        extract_camera_curve,
        group_diagetics_by_color_and_ancestor,
        load_preprocessing_cache,
        make_group_names_unique,
        parse_diagetic_metadata,
        save_preprocessing_cache,
        update_groups_with_categories,
        update_groups_with_path_danger,
    )
    
    # =========================================================================
    # Phase 1: Open USD Stage
    # =========================================================================
    print("Phase 1: Opening USD Stage...")
    start_time = time.time()
    
    stage = Usd.Stage.Open(USD_FILE)
    if not stage:
        print(f"ERROR: Could not open {USD_FILE}")
        return
    
    open_time = time.time() - start_time
    print(f"  Stage opened in {open_time:.1f}s")
    print()
    
    # =========================================================================
    # Phase 2: Parse Diegetic Metadata
    # =========================================================================
    print("Phase 2: Parsing Diegetic Metadata...")
    start_time = time.time()
    
    time_code = Usd.TimeCode(FRAMES[0])
    metadata_list, mesh_data_list = parse_diagetic_metadata(
        stage=stage,
        time_code=time_code,
        show_progress=True,
    )
    
    parse_time = time.time() - start_time
    print(f"  Parsed {len(metadata_list)} diegetics in {parse_time:.1f}s")
    
    # Report geometry size
    total_verts = sum(m.vertex_count for m in metadata_list)
    total_tris = sum(m.triangle_count for m in metadata_list)
    # Each vertex is 3 float32 (12 bytes), each triangle is 3 int32 (12 bytes)
    estimated_geom_size = total_verts * 12 + total_tris * 12
    print(f"  Total vertices: {total_verts:,}")
    print(f"  Total triangles: {total_tris:,}")
    print(f"  Estimated uncompressed geometry size: {format_size(estimated_geom_size)}")
    print()
    
    # =========================================================================
    # Phase 3: Group by Color and Ancestor
    # =========================================================================
    print("Phase 3: Grouping by Color and Ancestor...")
    start_time = time.time()
    
    groups = group_diagetics_by_color_and_ancestor(
        stage=stage,
        mesh_data_list=mesh_data_list,
        time_code=time_code,
        show_progress=True,
    )
    
    groups = make_group_names_unique(groups)
    
    group_time = time.time() - start_time
    print(f"  Created {len(groups)} groups in {group_time:.1f}s")
    print()
    
    # =========================================================================
    # Phase 4: Extract Camera Curve
    # =========================================================================
    print("Phase 4: Extracting Camera Curve...")
    start_time = time.time()
    
    camera_curve = extract_camera_curve(
        stage=stage,
        camera_path=CAMERA_PATH,
        frames=FRAMES,
        show_progress=True,
    )
    
    camera_time = time.time() - start_time
    print(f"  Extracted {len(camera_curve.frames)} camera positions in {camera_time:.1f}s")
    print()
    
    # =========================================================================
    # Phase 5: Compute Path Danger
    # =========================================================================
    print("Phase 5: Computing Path Danger...")
    start_time = time.time()
    
    groups = update_groups_with_path_danger(
        groups=groups,
        camera_curve=camera_curve,
        show_progress=True,
    )
    
    danger_time = time.time() - start_time
    print(f"  Computed path danger in {danger_time:.1f}s")
    print()
    
    # =========================================================================
    # Phase 6: Categorize Groups
    # =========================================================================
    print("Phase 6: Categorizing Groups...")
    start_time = time.time()
    
    groups = update_groups_with_categories(
        groups=groups,
        show_progress=True,
    )
    
    categorize_time = time.time() - start_time
    print(f"  Categorized groups in {categorize_time:.1f}s")
    print()
    
    # Print category summary
    ground_groups = [g for g in groups.values() if g.category == GroupCategory.GROUND_TERRAIN]
    safe_groups = [g for g in groups.values() if g.category == GroupCategory.SAFE]
    unsafe_groups = [g for g in groups.values() if g.category == GroupCategory.UNSAFE]
    
    print("Category Summary:")
    print(f"  GROUND_TERRAIN: {len(ground_groups)} groups")
    for g in ground_groups:
        print(f"    - {g.unique_name} ({g.member_count} meshes)")
    print(f"  SAFE: {len(safe_groups)} groups (lanterns/chains)")
    print(f"  UNSAFE: {len(unsafe_groups)} groups (props)")
    print()
    
    # =========================================================================
    # Phase 7: Save WITHOUT Geometry
    # =========================================================================
    print("Phase 7: Saving WITHOUT geometry...")
    start_time = time.time()
    
    save_preprocessing_cache(
        output_path=CACHE_WITHOUT_GEOMETRY,
        metadata_list=metadata_list,
        groups=groups,
        camera_curve=camera_curve,
        mesh_data_list=None,  # No geometry
    )
    
    save_time_no_geom = time.time() - start_time
    size_no_geom = os.path.getsize(CACHE_WITHOUT_GEOMETRY)
    print(f"  Saved in {save_time_no_geom:.1f}s")
    print(f"  File size (no geometry): {format_size(size_no_geom)}")
    print()
    
    # =========================================================================
    # Phase 8: Save WITH Geometry
    # =========================================================================
    print("Phase 8: Saving WITH geometry...")
    start_time = time.time()
    
    save_preprocessing_cache(
        output_path=CACHE_WITH_GEOMETRY,
        metadata_list=metadata_list,
        groups=groups,
        camera_curve=camera_curve,
        mesh_data_list=mesh_data_list,  # Include geometry!
    )
    
    save_time_with_geom = time.time() - start_time
    size_with_geom = os.path.getsize(CACHE_WITH_GEOMETRY)
    print(f"  Saved in {save_time_with_geom:.1f}s")
    print(f"  File size (with geometry): {format_size(size_with_geom)}")
    print()
    
    # =========================================================================
    # Phase 9: Test Loading
    # =========================================================================
    print("Phase 9: Testing Load (without geometry cache)...")
    start_time = time.time()
    
    loaded_metadata, loaded_groups, loaded_camera = load_preprocessing_cache(
        input_path=CACHE_WITHOUT_GEOMETRY
    )
    
    load_time = time.time() - start_time
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  {len(loaded_metadata)} diegetics, {len(loaded_groups)} groups")
    
    # Validate loaded data
    assert len(loaded_metadata) == len(metadata_list), "Metadata count mismatch"
    assert len(loaded_groups) == len(groups), "Group count mismatch"
    assert len(loaded_camera.frames) == len(camera_curve.frames), "Camera frame count mismatch"
    print("  Validation passed!")
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"USD File: {USD_FILE}")
    print(f"Total diegetics: {len(metadata_list)}")
    print(f"Total groups: {len(groups)}")
    print(f"Camera frames: {len(camera_curve.frames)}")
    print()
    print("Geometry Statistics:")
    print(f"  Total vertices: {total_verts:,}")
    print(f"  Total triangles: {total_tris:,}")
    print(f"  Estimated uncompressed: {format_size(estimated_geom_size)}")
    print()
    print("Cache File Sizes:")
    print(f"  Without geometry: {format_size(size_no_geom)}")
    print(f"  With geometry:    {format_size(size_with_geom)}")
    print(f"  Geometry overhead: {format_size(size_with_geom - size_no_geom)}")
    compression_ratio = estimated_geom_size / (size_with_geom - size_no_geom) if size_with_geom > size_no_geom else 0
    print(f"  Compression ratio: {compression_ratio:.1f}x")
    print()
    print("Is storing geometry crazy?")
    if size_with_geom > 500 * 1024 * 1024:  # > 500 MB
        print("  -> YES! File is over 500 MB. Consider skipping geometry.")
    elif size_with_geom > 100 * 1024 * 1024:  # > 100 MB
        print("  -> Maybe. File is over 100 MB. Geometry adds significant overhead.")
    else:
        print("  -> NO! File is reasonable. Storing geometry is fine.")
    print()
    print("Output files:")
    print(f"  {CACHE_WITHOUT_GEOMETRY}")
    print(f"  {CACHE_WITH_GEOMETRY}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()

