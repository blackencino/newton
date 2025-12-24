# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Run the dynamic segmentation preprocessing pipeline on the IV060 USD scene.

This script:
1. Parses the USD file for diegetic meshes
2. Groups them by objectid_color and scene graph ancestry
3. Extracts camera animation over the frame range
4. Computes path danger (distance from camera to each group)
5. Categorizes groups (Ground/Terrain, Unsafe, Safe)
6. Saves results to an NPZ cache file

Usage: uv run python newton/examples/ces26/run_preprocessing.py
"""

from pathlib import Path

from pxr import Usd

from ces26_utils import open_usd_stage
from dynamic_segmentation import run_preprocessing_pipeline

# =============================================================================
# Configuration
# =============================================================================

USD_FILE = r"C:\Users\chorvath\Downloads\20251223_iv060_flat_04\Collected_20251223_iv060_flat_04\20251223_iv060_flat_04.usd"
CAMERA_PATH = "/World/TD060"
FRAMES = list(range(2920, 3131))  # 2920 to 3130 inclusive

# Output cache file
OUTPUT_CACHE = Path(r"D:\ces26_data\preprocessing_cache.npz")

# Whether to include geometry in cache (can be very large!)
INCLUDE_GEOMETRY = False


def main():
    print("=" * 70)
    print("Dynamic Segmentation Preprocessing Pipeline")
    print("=" * 70)
    print()
    print(f"USD File: {USD_FILE}")
    print(f"Camera: {CAMERA_PATH}")
    print(f"Frames: {FRAMES[0]} to {FRAMES[-1]} ({len(FRAMES)} frames)")
    print(f"Output: {OUTPUT_CACHE}")
    print()

    # Open USD stage
    stage = open_usd_stage(USD_FILE)

    # Run preprocessing pipeline
    metadata_list, groups, camera_curve = run_preprocessing_pipeline(
        stage=stage,
        camera_path=CAMERA_PATH,
        frames=FRAMES,
        output_cache_path=OUTPUT_CACHE,
        include_geometry=INCLUDE_GEOMETRY,
        verbose=False,
        show_progress=True,
    )

    # Print summary
    print()
    print("=" * 70)
    print("Preprocessing Complete!")
    print("=" * 70)
    print()
    print(f"Total diegetics: {len(metadata_list)}")
    print(f"Total groups: {len(groups)}")
    print(f"Camera frames: {len(camera_curve.frames)}")
    print()

    # Print top 10 closest groups (most dangerous)
    print("Top 10 groups closest to camera path (highest path danger):")
    sorted_groups = sorted(groups.values(), key=lambda g: g.path_danger)
    for i, g in enumerate(sorted_groups[:10]):
        print(f"  {i+1}. {g.unique_name}")
        print(f"      path_danger={g.path_danger:.2f}, category={g.category.value}")
        print(f"      members={g.member_count}, verts={g.total_vertex_count}")
        print()

    # Print category breakdown
    print("Category breakdown:")
    from dynamic_segmentation import GroupCategory

    for cat in GroupCategory:
        count = sum(1 for g in groups.values() if g.category == cat)
        print(f"  {cat.value}: {count} groups")


if __name__ == "__main__":
    main()

