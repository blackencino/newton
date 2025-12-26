# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Debug script for loading and visualizing lantern meshes from USD.

This script demonstrates using ces26_utils to load scene geometry with:
- Proper handling of instanced geometry and transforms
- Visibility checking (skips invisible prims)
- Material color extraction (optionally from textures)
- Polyscope visualization

Usage: uv run python newton/examples/ces26/debug_all_lanterns_from_utils.py
"""

from pxr import Usd

from ces26_utils import (
    MeshLoadOptions,
    load_and_visualize_usd,
    load_meshes_from_stage,
    make_mesh_names_unique,
    register_meshes_with_polyscope,
    setup_polyscope,
    show_polyscope,
)

# =============================================================================
# Configuration
# =============================================================================

# Path to USD file
USD_FILE = r"C:\Users\chorvath\Downloads\20251219_iv060_flat_01\Collected_iv060_flat_01\iv060_flat_01.usd"

# Time code for sampling transforms and attributes
TIME_CODE = Usd.TimeCode(2920)

# Whether to load texture files to sample average colors
# (requires PIL/Pillow, slower but more accurate colors)
LOAD_TEXTURES = True


def main():
    """Load and visualize all lantern meshes from the USD file."""
    
    # Option 1: Use the simple high-level function
    # This is the easiest way if you don't need much customization
    # 
    # meshes = load_and_visualize_usd(
    #     USD_FILE,
    #     time_code=TIME_CODE,
    #     load_textures=LOAD_TEXTURES,
    #     verbose=True
    # )
    
    # Option 2: Use lower-level functions for more control
    # This demonstrates the modular approach
    
    print(f"Loading: {USD_FILE}")
    
    stage = Usd.Stage.Open(USD_FILE)
    if not stage:
        print("Failed to open USD file")
        return
    
    # Configure loading options
    options = MeshLoadOptions(
        time_code=TIME_CODE,
        load_material_colors=True,
        load_texture_colors=LOAD_TEXTURES,
        path_filter=None,  # Load all meshes (no filtering)
        skip_invisible=True,
        skip_proxy=True
    )
    
    # Load meshes
    meshes = load_meshes_from_stage(
        stage,
        USD_FILE,
        options,
        verbose=True
    )
    
    print(f"\nFound {len(meshes)} meshes")
    
    if not meshes:
        print("No meshes found matching criteria.")
        # List all mesh prims for debugging
        print("\nAll Mesh prims in stage:")
        for prim in stage.Traverse():
            from pxr import UsdGeom
            if prim.IsA(UsdGeom.Mesh):
                print(f"  {prim.GetPath()}")
        return
    
    # Ensure unique names for polyscope
    make_mesh_names_unique(meshes)
    
    # Visualize
    setup_polyscope(up_direction="z_up")
    register_meshes_with_polyscope(meshes, verbose=True)
    
    print("\nPolyscope viewer opened. Close window to exit.")
    show_polyscope()


if __name__ == "__main__":
    main()

