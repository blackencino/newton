# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Dump the USD scene hierarchy showing only diegetic-relevant prims.

This script outputs:
1. hierarchy.txt - Full scene hierarchy with diegetic meshes
2. references.txt - Prims that have references/payloads (potential asset boundaries)

Usage: uv run python newton/examples/ces26/dump_hierarchy.py
"""

from pathlib import Path
from collections import defaultdict

from pxr import Usd, UsdGeom, Sdf

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


def get_prim_references(prim: Usd.Prim) -> list[str]:
    """Get list of reference/payload asset paths on a prim."""
    refs = []
    
    # Get the prim's metadata about references
    prim_spec = prim.GetPrimStack()
    if prim_spec:
        for spec in prim_spec:
            # Check references
            ref_list = spec.referenceList
            if ref_list:
                for ref in ref_list.prependedItems:
                    if ref.assetPath:
                        refs.append(f"ref:{ref.assetPath}")
                for ref in ref_list.appendedItems:
                    if ref.assetPath:
                        refs.append(f"ref:{ref.assetPath}")
            
            # Check payloads
            payload_list = spec.payloadList
            if payload_list:
                for payload in payload_list.prependedItems:
                    if payload.assetPath:
                        refs.append(f"payload:{payload.assetPath}")
                for payload in payload_list.appendedItems:
                    if payload.assetPath:
                        refs.append(f"payload:{payload.assetPath}")
    
    return refs


def main():
    print("=" * 70)
    print("USD Hierarchy Dump - Diegetics Only")
    print("=" * 70)
    print()
    
    # Check USD file exists
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
    
    # Track which paths have visible meshes (diegetics)
    diegetic_mesh_paths: set[str] = set()
    
    # Track paths that have references/payloads
    referenced_prims: dict[str, list[str]] = {}
    
    print("Scanning for diegetic meshes...")
    mesh_count = 0
    
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path_str = str(prim.GetPath())
        
        # Check for references on ALL prims (not just meshes)
        refs = get_prim_references(prim)
        if refs:
            referenced_prims[path_str] = refs
        
        # Filter for visible meshes
        if not prim.IsA(UsdGeom.Mesh):
            continue
        
        if is_proxy_path(path_str):
            continue
        
        if not is_visible_mesh(prim, time_code):
            continue
        
        diegetic_mesh_paths.add(path_str)
        mesh_count += 1
        
        if mesh_count % 500 == 0:
            print(f"  Found {mesh_count} diegetic meshes...")
    
    print(f"  Total diegetic meshes: {mesh_count}")
    print(f"  Prims with references/payloads: {len(referenced_prims)}")
    print()
    
    # Build set of all ancestor paths that contain diegetic meshes
    ancestor_paths: set[str] = set()
    for mesh_path in diegetic_mesh_paths:
        parts = mesh_path.split("/")
        for i in range(1, len(parts)):
            ancestor_paths.add("/".join(parts[:i+1]))
    
    ancestor_paths.add("/")  # Include root
    
    # Now output the hierarchy
    hierarchy_file = OUTPUT_DIR / "hierarchy.txt"
    print(f"Writing hierarchy to {hierarchy_file}...")
    
    with open(hierarchy_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("USD SCENE HIERARCHY - Diegetic Paths Only\n")
        f.write("=" * 80 + "\n")
        f.write(f"USD File: {USD_FILE}\n")
        f.write(f"Frame: {START_FRAME}\n")
        f.write(f"Total diegetic meshes: {mesh_count}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Legend:\n")
        f.write("  [M] = Mesh (diegetic geometry)\n")
        f.write("  [R] = Has reference to sub-USD\n")
        f.write("  [P] = Has payload to sub-USD\n")
        f.write("  [X] = Xform or other grouping prim\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Traverse and output
        for prim in stage.Traverse():
            path_str = str(prim.GetPath())
            
            # Skip if this path is not an ancestor of any diegetic mesh
            if path_str not in ancestor_paths and path_str not in diegetic_mesh_paths:
                continue
            
            depth = path_str.count("/") - 1
            indent = "  " * depth
            name = prim.GetName()
            
            # Determine type marker
            markers = []
            if path_str in diegetic_mesh_paths:
                markers.append("M")
            if path_str in referenced_prims:
                refs = referenced_prims[path_str]
                if any(r.startswith("ref:") for r in refs):
                    markers.append("R")
                if any(r.startswith("payload:") for r in refs):
                    markers.append("P")
            if not markers:
                markers.append("X")
            
            marker_str = "[" + "".join(markers) + "]"
            
            # Add reference info if present
            ref_info = ""
            if path_str in referenced_prims:
                refs = referenced_prims[path_str]
                # Just show the filename, not full path
                short_refs = []
                for r in refs[:3]:  # Limit to first 3
                    parts = r.split("/")
                    short_refs.append(parts[-1] if "/" in r else r)
                ref_info = " <- " + ", ".join(short_refs)
                if len(refs) > 3:
                    ref_info += f" (+{len(refs)-3} more)"
            
            f.write(f"{indent}{marker_str} {name}{ref_info}\n")
    
    print(f"  Written {hierarchy_file}")
    
    # Write references file
    refs_file = OUTPUT_DIR / "references.txt"
    print(f"Writing references to {refs_file}...")
    
    # Group by referenced asset
    assets_to_prims: dict[str, list[str]] = defaultdict(list)
    for path, refs in referenced_prims.items():
        for ref in refs:
            assets_to_prims[ref].append(path)
    
    with open(refs_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("USD REFERENCE/PAYLOAD ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"USD File: {USD_FILE}\n")
        f.write(f"Total prims with references: {len(referenced_prims)}\n")
        f.write(f"Unique referenced assets: {len(assets_to_prims)}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("PRIMS WITH REFERENCES (sorted by path)\n")
        f.write("-" * 80 + "\n\n")
        
        for path in sorted(referenced_prims.keys()):
            refs = referenced_prims[path]
            # Check if this path is an ancestor of any diegetic
            has_diegetic = path in ancestor_paths or path in diegetic_mesh_paths
            diegetic_marker = " [DIEGETIC]" if has_diegetic else ""
            
            f.write(f"{path}{diegetic_marker}\n")
            for ref in refs:
                f.write(f"    {ref}\n")
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("REFERENCED ASSETS (by usage count)\n")
        f.write("-" * 80 + "\n\n")
        
        # Sort by usage count
        sorted_assets = sorted(assets_to_prims.items(), key=lambda x: -len(x[1]))
        for asset, paths in sorted_assets[:50]:  # Top 50
            f.write(f"{asset}\n")
            f.write(f"  Used by {len(paths)} prims:\n")
            for p in paths[:5]:
                f.write(f"    {p}\n")
            if len(paths) > 5:
                f.write(f"    ... and {len(paths)-5} more\n")
            f.write("\n")
    
    print(f"  Written {refs_file}")
    
    # Also create a simplified "asset boundaries" file
    boundaries_file = OUTPUT_DIR / "asset_boundaries.txt"
    print(f"Writing asset boundaries to {boundaries_file}...")
    
    # Find prims that have references AND contain diegetic meshes
    diegetic_asset_roots: list[tuple[str, list[str]]] = []
    for path in sorted(referenced_prims.keys()):
        # Check if this path is ancestor of any diegetic mesh
        is_diegetic_root = any(
            mesh_path.startswith(path + "/") or mesh_path == path
            for mesh_path in diegetic_mesh_paths
        )
        if is_diegetic_root:
            diegetic_asset_roots.append((path, referenced_prims[path]))
    
    with open(boundaries_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("POTENTIAL ASSET BOUNDARIES (prims with refs that contain diegetic meshes)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Found {len(diegetic_asset_roots)} potential asset boundaries\n")
        f.write("=" * 80 + "\n\n")
        
        # Group by depth
        depth_groups: dict[int, list[tuple[str, list[str]]]] = defaultdict(list)
        for path, refs in diegetic_asset_roots:
            depth = path.count("/")
            depth_groups[depth].append((path, refs))
        
        for depth in sorted(depth_groups.keys()):
            f.write(f"\n--- Depth {depth} ({len(depth_groups[depth])} prims) ---\n\n")
            for path, refs in depth_groups[depth]:
                # Count diegetic meshes under this path
                mesh_count = sum(
                    1 for mp in diegetic_mesh_paths
                    if mp.startswith(path + "/") or mp == path
                )
                f.write(f"{path}\n")
                f.write(f"  Diegetic meshes: {mesh_count}\n")
                for ref in refs[:2]:
                    # Extract just filename
                    if "/" in ref:
                        short = ref.split("/")[-1]
                    else:
                        short = ref
                    f.write(f"  {short}\n")
                f.write("\n")
    
    print(f"  Written {boundaries_file}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()

