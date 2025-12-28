# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Render all visible meshes in USD scene using the TD060 camera.

Uses the Diegetic functional pipeline from ces26_utils with multi-AOV output:
1. parse_diegetics: Folds geometry + color extractors over USD stage
2. setup_render_context: Converts Diegetics to GPU-ready structures + ColorLUTs
3. render_and_save_all_aovs: Single render pass outputs all AOVs to disk

EXR output ({base}_{AOV}.{frame:04d}.exr):
- td060_v03_color.2920.exr: Lit diffuse render (half-float RGB, [0, 1])
- td060_v03_depth.2920.exr: Normalized depth (half-float Y, [0, 1], 0=near 1=far)
- td060_v03_depth_heat.2920.exr: Depth heat map with interpolated colormap (half-float RGB)
- td060_v03_normal.2920.exr: Surface normals (half-float RGB, [0, 1] mapped from [-1, 1])
- td060_v03_object_id.2920.exr: Object ID colors (half-float RGB)
- td060_v03_semantic.2920.exr: Semantic colors (half-float RGB)

Renders at 4K resolution (3840x2160), frames 2920-3130.

Usage: uv run python newton/examples/ces26/render_td060_v03.py
"""

from pathlib import Path

from pxr import Usd

from ces26_utils import (
    DepthColormap,
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

USD_FILE = r"C:\Users\chorvath\Downloads\20251221_iv060_flat_03\20251221_iv060_flat_03\20251221_iv060_flat_03.usd"
CAMERA_PATH = "/World/TD060"
FRAMES = list(range(2920, 3131))  # 2920 to 3130 inclusive

# Output format: EXR for half-float OpenEXR
OUTPUT_FORMAT = "exr"

# Depth heat map colormap
DEPTH_COLORMAP = DepthColormap.MAGMA

RENDER_CONFIG = RenderConfig(
    width=3840,
    height=2160,
    output_dir=Path(r"D:\ces26_data"),
    filename_pattern="frame.{frame}.exr",
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
    # Use first frame for geometry parsing (geometry is static, only camera moves)
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
    print(f"Rendering {len(FRAMES)} frames ({FRAMES[0]} to {FRAMES[-1]})")
    print(f"Output directory: {RENDER_CONFIG.output_dir}")

    # Phase 2: Setup render context (uses diffuse_albedo for lit pass) + ColorLUTs
    ctx, color_luts, _ = setup_render_context(diegetics, RENDER_CONFIG)

    # Phase 3: Render all AOVs for each frame
    for i, frame in enumerate(FRAMES):
        time_code = Usd.TimeCode(frame)
        camera = get_camera_from_stage(stage, CAMERA_PATH, time_code, verbose=False)
        
        # Single render pass outputs all AOVs
        render_and_save_all_aovs(
            ctx=ctx,
            color_luts=color_luts,
            camera=camera,
            config=RENDER_CONFIG,
            frame_num=frame,
            base_name="td060_v03",
            output_format=OUTPUT_FORMAT,
            depth_colormap=DEPTH_COLORMAP,
        )
        
        # Progress update
        if (i + 1) % 10 == 0 or i == 0 or i == len(FRAMES) - 1:
            print(f"Rendered frame {frame} ({i + 1}/{len(FRAMES)})")

    print("\nDone!")


if __name__ == "__main__":
    main()




