# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Run the updated preprocessing pipeline with hierarchy-based grouping.

Uses the "parent of geo" heuristic to group meshes by asset root instead of
by objectid_color. This gives more granular, semantically-correct groups.

Outputs:
- D:\ces26_data\td060\v04\preprocess_v2.npz (metadata-only cache)
- Console summary of groups

Usage: uv run python newton/examples/ces26/run_preprocess_v2.py
"""

import time
from pathlib import Path

from pxr import Usd

# Configuration
USD_FILE = r"C:\Users\chorvath\Downloads\20251223_iv060_flat_04\Collected_20251223_iv060_flat_04\20251223_iv060_flat_04.usd"
CAMERA_PATH = "/World/TD060"
FRAMES = list(range(2920, 3131))  # 2920 to 3130 inclusive

OUTPUT_DIR = Path(r"D:\ces26_data\td060\v04")
CACHE_FILE = OUTPUT_DIR / "preprocess_v2.npz"


def main():
    print("=" * 70)
    print("Preprocessing Pipeline V2 - Hierarchy-Based Grouping")
    print("=" * 70)
    print()
    
    if not Path(USD_FILE).exists():
        print(f"ERROR: USD file not found: {USD_FILE}")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    from dynamic_segmentation import (
        GroupCategory,
        run_preprocessing_pipeline,
        load_preprocessing_cache,
    )
    
    # Open USD stage
    print(f"Opening USD: {USD_FILE}")
    start_time = time.time()
    stage = Usd.Stage.Open(USD_FILE)
    if not stage:
        print("ERROR: Could not open USD file")
        return
    print(f"  Stage opened in {time.time() - start_time:.1f}s")
    print()
    
    # Run preprocessing pipeline
    print("Running preprocessing pipeline...")
    start_time = time.time()
    
    metadata_list, groups, camera_curve = run_preprocessing_pipeline(
        stage=stage,
        camera_path=CAMERA_PATH,
        frames=FRAMES,
        output_cache_path=CACHE_FILE,
        include_geometry=False,  # Don't include geometry (too large)
        verbose=False,
        show_progress=True,
    )
    
    total_time = time.time() - start_time
    print(f"\nTotal preprocessing time: {total_time:.1f}s")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    print(f"Diegetic meshes: {len(metadata_list)}")
    print(f"Asset groups: {len(groups)}")
    print(f"Camera frames: {len(camera_curve.frames)}")
    print()
    
    # Category breakdown
    ground = [g for g in groups.values() if g.category == GroupCategory.GROUND_TERRAIN]
    safe = [g for g in groups.values() if g.category == GroupCategory.SAFE]
    unsafe = [g for g in groups.values() if g.category == GroupCategory.UNSAFE]
    
    print("Category breakdown:")
    print(f"  GROUND_TERRAIN: {len(ground)}")
    print(f"  SAFE: {len(safe)}")
    print(f"  UNSAFE: {len(unsafe)}")
    print()
    
    # Top groups by mesh count
    print("Top 15 groups by mesh count:")
    print("-" * 70)
    sorted_groups = sorted(groups.values(), key=lambda g: -g.member_count)
    for i, g in enumerate(sorted_groups[:15]):
        cat_str = g.category.value[:6]
        print(f"{i+1:2}. {g.unique_name:<35} {g.member_count:4} meshes  [{cat_str}]")
    print()
    
    # Lantern groups
    lantern_groups = [g for g in groups.values() if "lantern" in g.unique_name.lower()]
    print(f"Lantern groups: {len(lantern_groups)}")
    for g in sorted(lantern_groups, key=lambda x: x.unique_name)[:10]:
        print(f"  - {g.unique_name} ({g.member_count} meshes)")
    if len(lantern_groups) > 10:
        print(f"  ... and {len(lantern_groups) - 10} more")
    print()
    
    # Terrain groups
    terrain_groups = [g for g in groups.values() if g.category == GroupCategory.GROUND_TERRAIN]
    print(f"Terrain groups: {len(terrain_groups)}")
    for g in terrain_groups:
        print(f"  - {g.unique_name} ({g.member_count} meshes)")
        # Show member paths
        for p in g.member_paths[:3]:
            print(f"      {p.split('/')[-1]}")
        if len(g.member_paths) > 3:
            print(f"      ... and {len(g.member_paths) - 3} more")
    print()
    
    # Test loading
    print("=" * 70)
    print("Testing cache load...")
    print("=" * 70)
    
    loaded_metadata, loaded_groups, loaded_camera = load_preprocessing_cache(CACHE_FILE)
    
    assert len(loaded_metadata) == len(metadata_list), "Metadata count mismatch"
    assert len(loaded_groups) == len(groups), "Group count mismatch"
    print("  Cache validation passed!")
    print()
    
    print(f"Cache saved to: {CACHE_FILE}")
    import os
    cache_size = os.path.getsize(CACHE_FILE)
    print(f"Cache size: {cache_size / 1024:.1f} KB")
    print()
    print("Done!")


if __name__ == "__main__":
    main()

