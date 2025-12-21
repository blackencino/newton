# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Render USD scene using the TD060 camera from the USD file.

Uses ces26_utils for mesh and camera loading.
Handles USD camera coordinate system: X-right, Y-up, -Z forward (camera looks down -Z).

Usage: uv run python newton/examples/ces26/debug_lantern_camera.py
"""

from pathlib import Path

from pxr import Usd

from ces26_utils import (
    MeshLoadOptions,
    RenderConfig,
    get_camera_from_stage,
    load_meshes_from_stage,
    make_mesh_names_unique,
    render_and_save_frame,
    setup_render_context,
    use_object_id_colors,
)

# =============================================================================
# Configuration
# =============================================================================

#USD_FILE = r"C:\Users\chorvath\Downloads\20251219_iv060_flat_01\Collected_iv060_flat_01\iv060_flat_01.usd"
USD_FILE = r"C:\Users\chorvath\Downloads\20251220_iv060_flat_02\Collected_20251220_iv060_flat_02\20251220_iv060_flat_02.usd"
CAMERA_PATH = "/World/TD060"
FRAMES = [2920, 3130]

RENDER_CONFIG = RenderConfig(
    width=960,
    height=540,
    output_dir=Path(__file__).parent,
    filename_pattern="debug_lantern_camera.{frame}.png",
)


# =============================================================================
# Scene Building
# =============================================================================

def load_lantern_meshes(stage: Usd.Stage, usd_path: str, time_code: Usd.TimeCode):
    """Load lantern meshes using ces26_utils."""
    options = MeshLoadOptions(
        time_code=time_code,
        load_material_colors=False,
        load_texture_colors=False,
        path_filter=lambda path: "Hanging" in path,
        skip_invisible=True,
        skip_proxy=True
    )
    
    meshes = load_meshes_from_stage(stage, usd_path, options, verbose=False)
    make_mesh_names_unique(meshes)
    
    print(f"Loaded {len(meshes)} lantern meshes")
    return meshes


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Loading USD: {USD_FILE}")
    stage = Usd.Stage.Open(USD_FILE)
    if not stage:
        raise RuntimeError("Failed to open USD file")

    # Phase 1: Build scene representation (data loading, color assignment)
    meshes = load_lantern_meshes(stage, USD_FILE, Usd.TimeCode(FRAMES[0]))
    
    if not meshes:
        print("No meshes found!")
        return

    # Convert meshes to scene shapes with object ID colors from primvars
    # Meshes missing primvars:objectid_color will appear in bright orange (error color)
    shapes = use_object_id_colors(meshes)

    # Phase 2: Convert scene representation to render context
    ctx, _ = setup_render_context(shapes, RENDER_CONFIG)

    # Phase 3: Render frames
    for frame in FRAMES:
        time_code = Usd.TimeCode(frame)
        camera = get_camera_from_stage(stage, CAMERA_PATH, time_code, verbose=True)
        render_and_save_frame(ctx, camera, RENDER_CONFIG, frame)

    print("\nDone!")


if __name__ == "__main__":
    main()
