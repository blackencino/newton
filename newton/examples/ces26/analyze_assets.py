# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Analyze USD scene to identify asset boundaries using the "parent of geo" heuristic.

The key insight: in this flattened USD, each asset has a structure like:
  /World/{asset_name}/geo/{mesh1, mesh2, ...}
or
  /World/StarWarsSet_01/assembly/{asset_name}/geo/{mesh1, mesh2, ...}

The asset root is the prim that is the PARENT of the "geo" prim.

Usage: uv run python newton/examples/ces26/analyze_assets.py
"""

from pathlib import Path
from collections import defaultdict

from pxr import Usd, UsdGeom

# Configuration
USD_FILE = r"C:\Users\chorvath\Downloads\20251223_iv060_flat_04\Collected_20251223_iv060_flat_04\20251223_iv060_flat_04.usd"
OUTPUT_DIR = Path(r"D:\ces26_data\td060\v04")
START_FRAME = 2920


def is_proxy_path(path: str) -> bool:
    """Check if a prim path appears to be proxy geometry."""
    return "/proxy/" in path.lower()


def is_visible_mesh(prim: Usd.Prim, time_code: Usd.TimeCode) -> bool:
    """Check if a prim is a visible mesh."""
    if not prim.IsA(UsdGeom.Mesh):
        return False
    
    if prim.IsInstanceProxy():
        geom_prim = prim.GetPrimInPrototype()
    else:
        geom_prim = prim
    
    mesh = UsdGeom.Mesh(geom_prim)
    visibility = mesh.ComputeEffectiveVisibility(UsdGeom.Tokens.render, time_code)
    return visibility != UsdGeom.Tokens.invisible


def find_geo_parent(mesh_path: str) -> str | None:
    """
    Find the asset root for a mesh by looking for the parent of 'geo' in the path.
    
    Examples:
      /World/HangingLanternA_01/geo/part001 -> /World/HangingLanternA_01
      /World/StarWarsSet_01/assembly/Terrain_01/geo/mesh -> /World/StarWarsSet_01/assembly/Terrain_01
    
    Returns None if no 'geo' is found in the path.
    """
    parts = mesh_path.split("/")
    
    # Find the 'geo' component
    for i, part in enumerate(parts):
        if part == "geo":
            # Return everything up to (but not including) 'geo'
            if i > 0:
                return "/".join(parts[:i])
    
    return None


def main():
    print("=" * 70)
    print("Asset Analysis - Finding asset boundaries via 'geo' parent")
    print("=" * 70)
    print()
    
    if not Path(USD_FILE).exists():
        print(f"ERROR: USD file not found: {USD_FILE}")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening USD: {USD_FILE}")
    stage = Usd.Stage.Open(USD_FILE)
    if not stage:
        print("ERROR: Could not open USD file")
        return
    
    time_code = Usd.TimeCode(START_FRAME)
    
    # Map asset roots to their meshes
    asset_to_meshes: dict[str, list[str]] = defaultdict(list)
    orphan_meshes: list[str] = []  # Meshes without a geo parent
    
    print("Scanning for diegetic meshes...")
    mesh_count = 0
    
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path_str = str(prim.GetPath())
        
        if not prim.IsA(UsdGeom.Mesh):
            continue
        
        if is_proxy_path(path_str):
            continue
        
        if not is_visible_mesh(prim, time_code):
            continue
        
        mesh_count += 1
        
        # Find asset root
        asset_root = find_geo_parent(path_str)
        if asset_root:
            asset_to_meshes[asset_root].append(path_str)
        else:
            orphan_meshes.append(path_str)
        
        if mesh_count % 500 == 0:
            print(f"  Found {mesh_count} diegetic meshes...")
    
    print(f"  Total diegetic meshes: {mesh_count}")
    print(f"  Unique asset roots: {len(asset_to_meshes)}")
    print(f"  Orphan meshes (no geo parent): {len(orphan_meshes)}")
    print()
    
    # Write analysis to file
    output_file = OUTPUT_DIR / "asset_roots.txt"
    print(f"Writing to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ASSET ROOT ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"USD File: {USD_FILE}\n")
        f.write(f"Total diegetic meshes: {mesh_count}\n")
        f.write(f"Unique asset roots: {len(asset_to_meshes)}\n")
        f.write(f"Orphan meshes: {len(orphan_meshes)}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("GROUPING HEURISTIC:\n")
        f.write("  Asset root = parent of 'geo' prim in mesh path\n")
        f.write("  e.g., /World/LanternA_01/geo/mesh -> asset root = /World/LanternA_01\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Group by depth
        depth_groups: dict[int, list[tuple[str, list[str]]]] = defaultdict(list)
        for root, meshes in asset_to_meshes.items():
            depth = root.count("/")
            depth_groups[depth].append((root, meshes))
        
        for depth in sorted(depth_groups.keys()):
            items = sorted(depth_groups[depth], key=lambda x: x[0])
            f.write(f"\n{'='*80}\n")
            f.write(f"DEPTH {depth} - {len(items)} assets\n")
            f.write(f"{'='*80}\n\n")
            
            for root, meshes in items:
                # Extract short name
                short_name = root.split("/")[-1]
                f.write(f"{root}\n")
                f.write(f"  Name: {short_name}\n")
                f.write(f"  Meshes: {len(meshes)}\n")
                
                # Show first few mesh names
                mesh_names = [m.split("/")[-1] for m in meshes[:5]]
                f.write(f"  Sample: {', '.join(mesh_names)}")
                if len(meshes) > 5:
                    f.write(f" ... (+{len(meshes)-5} more)")
                f.write("\n\n")
        
        if orphan_meshes:
            f.write("\n" + "=" * 80 + "\n")
            f.write("ORPHAN MESHES (no 'geo' in path)\n")
            f.write("=" * 80 + "\n\n")
            for path in sorted(orphan_meshes):
                f.write(f"  {path}\n")
    
    print(f"  Written {output_file}")
    
    # Also write a simple CSV for easy processing
    csv_file = OUTPUT_DIR / "asset_roots.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("asset_root,mesh_count,short_name,depth\n")
        for root, meshes in sorted(asset_to_meshes.items()):
            short_name = root.split("/")[-1]
            depth = root.count("/")
            f.write(f'"{root}",{len(meshes)},"{short_name}",{depth}\n')
    
    print(f"  Written {csv_file}")
    
    # Print summary by asset type
    print()
    print("=" * 70)
    print("Asset Type Summary")
    print("=" * 70)
    
    # Group by asset type (first word of name)
    type_counts: dict[str, int] = defaultdict(int)
    type_meshes: dict[str, int] = defaultdict(int)
    
    for root, meshes in asset_to_meshes.items():
        short_name = root.split("/")[-1]
        # Extract type (everything before the underscore or number)
        import re
        match = re.match(r"([A-Za-z]+)", short_name)
        if match:
            asset_type = match.group(1)
            type_counts[asset_type] += 1
            type_meshes[asset_type] += len(meshes)
    
    sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
    print(f"{'Asset Type':<30} {'Count':>8} {'Meshes':>10}")
    print("-" * 50)
    for asset_type, count in sorted_types[:20]:
        meshes = type_meshes[asset_type]
        print(f"{asset_type:<30} {count:>8} {meshes:>10}")
    if len(sorted_types) > 20:
        print(f"... and {len(sorted_types) - 20} more types")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()

