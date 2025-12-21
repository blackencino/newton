# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Render all visible meshes in USD scene using the TD060 camera.

Uses ces26_utils for mesh and camera loading.
Renders at 4K resolution (3840x2160) with object ID colors.

Usage: uv run python newton/examples/ces26/debug_all_visible_camera.py
"""

from pathlib import Path

from pxr import Usd

from ces26_utils import (
    MeshLoadOptions,
    RenderConfig,
    get_camera_from_stage,
    load_meshes_from_stage,
    make_mesh_names_unique,
    open_usd_stage,
    render_and_save_frame,
    setup_render_context,
    use_object_id_colors,
)

# =============================================================================
# Configuration
# =============================================================================

USD_FILE = r"C:\Users\chorvath\Downloads\20251220_iv060_flat_02\Collected_20251220_iv060_flat_02\20251220_iv060_flat_02.usd"
CAMERA_PATH = "/World/TD060"
FRAMES = [2920, 3130]

RENDER_CONFIG = RenderConfig(
    width=3840,
    height=2160,
    output_dir=Path(__file__).parent,
    filename_pattern="debug_all_visible_camera.{frame}.png",
)


# =============================================================================
# Scene Building
# =============================================================================

def load_all_visible_meshes(stage: Usd.Stage, usd_path: str, time_code: Usd.TimeCode):
    """Load all visible meshes using ces26_utils (no name filtering)."""
    options = MeshLoadOptions(
        time_code=time_code,
        load_material_colors=False,
        load_texture_colors=False,
        path_filter=None,  # No filtering - load all visible meshes
        skip_invisible=True,
        skip_proxy=True
    )
    
    meshes = load_meshes_from_stage(stage, usd_path, options, verbose=False)
    make_mesh_names_unique(meshes)
    
    print(f"Loaded {len(meshes)} visible meshes")
    return meshes


# =============================================================================
# Main
# =============================================================================

def main():
    stage = open_usd_stage(USD_FILE)

    # Phase 1: Build scene representation (data loading, color assignment)
    meshes = load_all_visible_meshes(stage, USD_FILE, Usd.TimeCode(FRAMES[0]))
    
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

