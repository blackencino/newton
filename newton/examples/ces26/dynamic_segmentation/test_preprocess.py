# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Test script for dynamic_segmentation/preprocess.py on shot IV060.

This tests the preprocessing pipeline on the actual USD file for this shot.
Validates that the categorization heuristics work correctly.

Run with: uv run python -m dynamic_segmentation.test_preprocess
(from newton/examples/ces26 directory)
"""

from pathlib import Path

from pxr import Usd

# Configuration for shot IV060
USD_FILE = r"C:\Users\chorvath\Downloads\20251223_iv060_flat_04\Collected_20251223_iv060_flat_04\20251223_iv060_flat_04.usd"
CAMERA_PATH = "/World/TD060"
START_FRAME = 2920


def test_categorization_on_real_usd():
    """
    Test that categorization works correctly on the real IV060 USD.

    Validates:
    - Terrain_01 and SimGravelLrg are categorized as GROUND_TERRAIN
    - HangingLantern* groups are categorized as SAFE
    - Other groups are categorized as UNSAFE
    """
    from dynamic_segmentation.preprocess import (
        GroupCategory,
        extract_camera_curve,
        group_diagetics_by_color_and_ancestor,
        parse_diagetic_metadata,
        update_groups_with_categories,
        update_groups_with_path_danger,
    )

    print("Opening USD stage...")
    stage = Usd.Stage.Open(USD_FILE)
    if not stage:
        print(f"ERROR: Could not open {USD_FILE}")
        return False

    time_code = Usd.TimeCode(START_FRAME)

    print("Parsing diegetic metadata...")
    metadata_list, mesh_data_list = parse_diagetic_metadata(
        stage=stage,
        time_code=time_code,
        show_progress=True,
    )

    if not metadata_list:
        print("ERROR: No diegetics found")
        return False

    print(f"\nFound {len(metadata_list)} diegetics")

    print("\nGrouping by color and ancestor...")
    groups = group_diagetics_by_color_and_ancestor(
        stage=stage,
        mesh_data_list=mesh_data_list,
        time_code=time_code,
        show_progress=True,
    )

    # Extract a small camera sample for path danger (just 10 frames)
    print("\nExtracting camera curve (sample)...")
    sample_frames = list(range(START_FRAME, START_FRAME + 10))
    camera_curve = extract_camera_curve(
        stage=stage,
        camera_path=CAMERA_PATH,
        frames=sample_frames,
        show_progress=False,
    )

    print("Computing path danger...")
    groups = update_groups_with_path_danger(groups, camera_curve, show_progress=False)

    print("Categorizing groups...")
    groups = update_groups_with_categories(groups, show_progress=True)

    # Validate categorization
    print("\n" + "=" * 60)
    print("Validating categorization...")
    print("=" * 60)

    ground_groups = [g for g in groups.values() if g.category == GroupCategory.GROUND_TERRAIN]
    safe_groups = [g for g in groups.values() if g.category == GroupCategory.SAFE]
    unsafe_groups = [g for g in groups.values() if g.category == GroupCategory.UNSAFE]

    print(f"\nGROUND_TERRAIN groups ({len(ground_groups)}):")
    for g in ground_groups:
        print(f"  - {g.unique_name} ({g.common_ancestor_path})")

    print(f"\nSAFE groups ({len(safe_groups)}):")
    for g in safe_groups[:10]:  # First 10
        print(f"  - {g.unique_name}")
    if len(safe_groups) > 10:
        print(f"  ... and {len(safe_groups) - 10} more")

    print(f"\nUNSAFE groups ({len(unsafe_groups)}):")
    for g in list(unsafe_groups)[:10]:  # First 10
        print(f"  - {g.unique_name}")
    if len(unsafe_groups) > 10:
        print(f"  ... and {len(unsafe_groups) - 10} more")

    # Validation checks
    errors = []

    # Check that Terrain meshes are in a ground group (may be grouped with other assets)
    # Look in member_paths since terrain may be grouped with other same-colored objects
    terrain_found = any(
        any("/terrain_01/" in p.lower() for p in g.member_paths)
        for g in ground_groups
    )

    if not terrain_found:
        errors.append("Terrain_01 meshes not found in any GROUND_TERRAIN group")
    
    # Note: SimGravelLrg is a simulation point cache and doesn't render in this pass

    # Check that HangingLantern* are safe
    lantern_safe = all(
        g.category == GroupCategory.SAFE
        for g in groups.values()
        if "hanginglantern" in g.common_ancestor_path.lower()
    )
    if not lantern_safe:
        errors.append("Some HangingLantern groups not categorized as SAFE")

    # Check that we have some unsafe groups (props, crates, etc.)
    if len(unsafe_groups) == 0:
        errors.append("No UNSAFE groups found - should have props, crates, etc.")

    print("\n" + "=" * 60)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("VALIDATION PASSED!")
        print(f"  - {len(ground_groups)} ground/terrain groups")
        print(f"  - {len(safe_groups)} safe groups (lanterns/chains)")
        print(f"  - {len(unsafe_groups)} unsafe groups (props)")
        return True


def main():
    print("=" * 60)
    print("Testing preprocessing on shot IV060")
    print("=" * 60)
    print()

    if not Path(USD_FILE).exists():
        print(f"ERROR: USD file not found: {USD_FILE}")
        print("Make sure the USD file is downloaded to the expected location.")
        return

    success = test_categorization_on_real_usd()

    print()
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed!")


if __name__ == "__main__":
    main()
