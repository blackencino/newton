# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Validate that all meshes within each DiageticGroup share the same objectid_color.

This version uses the saved NPZ cache - no USD parsing required.
"""

from pathlib import Path
from collections import defaultdict

from dynamic_segmentation.preprocess import load_preprocessing_cache

# =============================================================================
# Configuration
# =============================================================================

CACHE_PATH = Path(r"D:\ces26_data\td060\v04\preprocess_v2.npz")


def _color_to_str(color) -> str:
    """Format color for display."""
    if color is None:
        return "None"
    return f"({color[0]:.4f}, {color[1]:.4f}, {color[2]:.4f})"


def _colors_match(c1, c2, tolerance: float = 1e-6) -> bool:
    """Check if two colors are equal within tolerance."""
    if c1 is None and c2 is None:
        return True
    if c1 is None or c2 is None:
        return False
    return all(abs(a - b) < tolerance for a, b in zip(c1, c2))


def main():
    print("=" * 70)
    print("Validating objectid_color consistency within asset groups")
    print("(Using cached metadata - no USD parsing)")
    print("=" * 70)
    print(f"\nCache: {CACHE_PATH}\n")

    # Load from cache
    metadata_list, groups, camera_curve = load_preprocessing_cache(CACHE_PATH)

    print(f"\nTotal diegetics: {len(metadata_list)}")
    print(f"Total groups: {len(groups)}\n")

    # Build path -> metadata lookup
    path_to_metadata = {m.path: m for m in metadata_list}

    # Validate each group
    print("-" * 70)
    print("Checking color consistency within each group...")
    print("-" * 70)

    groups_with_mixed_colors = []
    groups_with_uniform_colors = 0
    groups_all_none_colors = 0

    for group_id, group in sorted(groups.items(), key=lambda x: x[1].unique_name):
        # Collect all unique colors in this group's members
        color_to_paths: dict[str, list[str]] = defaultdict(list)
        
        for member_path in group.member_paths:
            metadata = path_to_metadata.get(member_path)
            if metadata is None:
                # Member path not found in metadata list (shouldn't happen)
                color_key = "MISSING"
            else:
                color_key = _color_to_str(metadata.objectid_color)
            color_to_paths[color_key].append(member_path)
        
        unique_colors = list(color_to_paths.keys())
        
        if len(unique_colors) == 1:
            if unique_colors[0] == "None":
                groups_all_none_colors += 1
            else:
                groups_with_uniform_colors += 1
        else:
            # Multiple different colors in this group!
            groups_with_mixed_colors.append({
                "group_id": group_id,
                "unique_name": group.unique_name,
                "common_ancestor_path": group.common_ancestor_path,
                "assigned_color": group.objectid_color,
                "member_count": group.member_count,
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
            print(f"\n  Group: {info['unique_name']} ({info['group_id']})")
            print(f"  Ancestor: {info['common_ancestor_path']}")
            print(f"  Assigned color: {_color_to_str(info['assigned_color'])}")
            print(f"  Total meshes: {info['member_count']}")
            print(f"  Actual colors found ({len(info['color_breakdown'])} unique):")
            
            for color_str, paths in sorted(info['color_breakdown'].items()):
                print(f"    {color_str}: {len(paths)} meshes")
                # Show first few paths as examples
                for p in paths[:2]:
                    print(f"      - {p}")
                if len(paths) > 2:
                    print(f"      ... and {len(paths) - 2} more")
    else:
        print("\n" + "=" * 70)
        print("VALIDATION PASSED: All groups have consistent objectid_color values!")
        print("=" * 70)

    print("\nDone.")


if __name__ == "__main__":
    main()

