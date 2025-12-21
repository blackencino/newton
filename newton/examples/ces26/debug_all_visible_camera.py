# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Render all visible meshes in USD scene using the TD060 camera.

Uses the Diegetic functional pipeline from ces26_utils with multi-AOV output:
1. parse_diegetics: Folds geometry + color extractors over USD stage
2. setup_render_context: Converts Diegetics to GPU-ready structures + ColorLUTs
3. render_and_save_all_aovs: Single render pass outputs all AOVs to disk

Outputs files named {base}_{AOV}.{frame:04d}.png:
- debug_all_visible_color.2920.png: Lit diffuse render
- debug_all_visible_depth.2920.png: Depth visualization (closer = brighter)
- debug_all_visible_normal.2920.png: Surface normal visualization
- debug_all_visible_object_id.2920.png: Object ID colors (from USD primvar)
- debug_all_visible_semantic.2920.png: Semantic segmentation colors (random per-object)

Renders at 4K resolution (3840x2160).

Usage: uv run python newton/examples/ces26/debug_all_visible_camera.py
"""

from pathlib import Path

from pxr import Usd

from ces26_utils import (
    ParseOptions,
    RenderConfig,
    get_camera_from_stage,
    make_diegetic_names_unique,
    material_diffuse_extractor,
    open_usd_stage,
    parse_diegetics,
    primvar_color,
    random_color,
    render_and_save_all_aovs,
    setup_render_context,
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
    filename_pattern="frame.{frame}.png",  # Pattern not used with multi-AOV output
)

# =============================================================================
# Color Extractor Configuration
# =============================================================================

# Define extractors for each color channel
DIFFUSE_EXTRACTOR = material_diffuse_extractor(
    fallback=(0.7, 0.7, 0.7),
    load_textures=False,
)

OBJECT_ID_EXTRACTOR = primvar_color(
    primvar_name="objectid_color",
    fallback=(1.0, 0.5, 0.0),  # Bright orange for missing object IDs
)

SEMANTIC_EXTRACTOR = random_color(seed=42)


# =============================================================================
# Main
# =============================================================================

def main():
    stage = open_usd_stage(USD_FILE)

    # Phase 1: Parse USD into Diegetics (geometry + all color channels)
    options = ParseOptions(
        time_code=Usd.TimeCode(FRAMES[0]),
        path_filter=None,  # Load all visible meshes
        skip_invisible=True,
        skip_proxy=True,
    )
    
    diegetics = parse_diegetics(
        stage=stage,
        usd_file_path=USD_FILE,
        options=options,
        diffuse_extractor=DIFFUSE_EXTRACTOR,
        object_id_extractor=OBJECT_ID_EXTRACTOR,
        semantic_extractor=SEMANTIC_EXTRACTOR,
        verbose=False,
    )
    
    if not diegetics:
        print("No diegetics found!")
        return
    
    # Make names unique (returns new list since Diegetic is frozen)
    diegetics = make_diegetic_names_unique(diegetics)
    print(f"Loaded {len(diegetics)} diegetics")

    # Phase 2: Setup render context (uses diffuse_albedo for lit pass) + ColorLUTs
    ctx, color_luts, _ = setup_render_context(diegetics, RENDER_CONFIG)

    # Phase 3: Render all AOVs for each frame
    for frame in FRAMES:
        time_code = Usd.TimeCode(frame)
        camera = get_camera_from_stage(stage, CAMERA_PATH, time_code, verbose=True)
        
        # Single render pass outputs all AOVs
        render_and_save_all_aovs(
            ctx=ctx,
            color_luts=color_luts,
            camera=camera,
            config=RENDER_CONFIG,
            frame_num=frame,
            base_name="debug_all_visible",
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
