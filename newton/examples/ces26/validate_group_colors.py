# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Validate that all meshes within each DiageticGroup share the same objectid_color.

This script checks whether the asset-root-based grouping accidentally combines
meshes that have different artist-assigned objectid_color primvars.
"""

from pathlib import Path
from collections import defaultdict

from pxr import Usd

# Import preprocessing functions
from dynamic_segmentation.preprocess import (
    parse_diagetic_metadata,
    _find_asset_root,
)

# =============================================================================
# Configuration
# =============================================================================

USD_FILE = r"C:\Users\chorvath\Downloads\20251223_iv060_flat_04\Collected_20251223_iv060_flat_04\20251223_iv060_flat_04.usd"
START_FRAME = 2920


def _colors_match(c1, c2, tolerance: float = 1e-6) -> bool:
    """Check if two colors are equal within tolerance."""
    if c1 is None and c2 is None:
        return True
    if c1 is None or c2 is None:
        return False
    return all(abs(a - b) < tolerance for a, b in zip(c1, c2))


def _color_to_str(color) -> str:
    """Format color for display."""
    if color is None:
        return "None"
    return f"({color[0]:.4f}, {color[1]:.4f}, {color[2]:.4f})"


def main():
    print("=" * 70)
    print("Validating objectid_color consistency within asset groups")
    print("=" * 70)
    print(f"\nUSD: {USD_FILE}")
    print(f"Frame: {START_FRAME}\n")

    # Open stage
    print("Opening USD stage...")
    stage = Usd.Stage.Open(USD_FILE)
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {USD_FILE}")
    print("  Stage opened.\n")

    # Parse all diegetic metadata
    time_code = Usd.TimeCode(START_FRAME)
    metadata_list, mesh_data_list = parse_diagetic_metadata(
        stage=stage,
        time_code=time_code,
        verbose=False,
        show_progress=True,
    )

    print(f"\nTotal diegetics parsed: {len(metadata_list)}\n")

    # Group by asset root
    asset_groups: dict[str, list] = defaultdict(list)
    orphan_count = 0

    for mesh in mesh_data_list:
        asset_root = _find_asset_root(mesh.metadata.path)
        if asset_root:
            asset_groups[asset_root].append(mesh.metadata)
        else:
            orphan_count += 1

    print(f"Asset groups: {len(asset_groups)}")
    print(f"Orphan meshes (no 'geo' in path): {orphan_count}\n")

    # Validate each group
    print("-" * 70)
    print("Checking color consistency within each group...")
    print("-" * 70)

    groups_with_mixed_colors = []
    groups_with_uniform_colors = 0
    groups_all_none_colors = 0

    for asset_root in sorted(asset_groups.keys()):
        members = asset_groups[asset_root]
        
        # Collect all unique colors in this group
        color_to_paths: dict[str, list[str]] = defaultdict(list)
        
        for m in members:
            color_key = _color_to_str(m.objectid_color)
            color_to_paths[color_key].append(m.path)
        
        unique_colors = list(color_to_paths.keys())
        
        if len(unique_colors) == 1:
            if unique_colors[0] == "None":
                groups_all_none_colors += 1
            else:
                groups_with_uniform_colors += 1
        else:
            # Multiple different colors in this group!
            groups_with_mixed_colors.append({
                "asset_root": asset_root,
                "members": members,
                "color_breakdown": dict(color_to_paths),
            })

    print(f"\nResults:")
    print(f"  Groups with uniform colors: {groups_with_uniform_colors}")
    print(f"  Groups with all None colors: {groups_all_none_colors}")
    print(f"  Groups with MIXED colors: {len(groups_with_mixed_colors)}")

    if groups_with_mixed_colors:
        print("\n" + "=" * 70)
        print("GROUPS WITH MIXED objectid_color VALUES")
        print("=" * 70)
        
        for info in groups_with_mixed_colors:
            print(f"\n  Asset Root: {info['asset_root']}")
            print(f"  Total meshes: {len(info['members'])}")
            print(f"  Color breakdown:")
            
            for color_str, paths in sorted(info['color_breakdown'].items()):
                print(f"    {color_str}: {len(paths)} meshes")
                # Show first few paths as examples
                for p in paths[:3]:
                    print(f"      - {p}")
                if len(paths) > 3:
                    print(f"      ... and {len(paths) - 3} more")
    else:
        print("\n" + "=" * 70)
        print("VALIDATION PASSED: All groups have consistent objectid_color values!")
        print("=" * 70)

    print("\nDone.")


if __name__ == "__main__":
    main()

