# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Render only the object_id AOV for TD060 v04 sequence.

Uses the Diegetic functional pipeline from ces26_utils:
1. parse_diegetics: Folds geometry + color extractors over USD stage
2. setup_render_context: Converts Diegetics to GPU-ready structures + ColorLUTs
3. render_all_aovs: Single render pass outputs all AOVs
4. Extract and save only object_id AOV as EXR

EXR output:
- td060_v04_fixed_object_id.{frame:04d}.exr: Object ID colors (half-float RGB)

Renders at 4K resolution (3840x2160), frames 2920-3130.

Usage: uv run python newton/examples/ces26/render_td060_fixed_object_id_v04.py
"""

from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd

from ces26_utils import (
    ParseOptions,
    RenderConfig,
    convert_aovs_to_exr_data,
    get_camera_from_stage,
    make_diegetic_names_unique,
    material_diffuse_extractor,
    open_usd_stage,
    parse_diegetics,
    primvar_color,
    random_color,
    render_all_aovs,
    save_exr_rgb,
    setup_render_context,
)

# =============================================================================
# Configuration
# =============================================================================

USD_FILE = r"C:\Users\chorvath\Downloads\20251223_iv060_flat_04\Collected_20251223_iv060_flat_04\20251223_iv060_flat_04.usd"
CAMERA_PATH = "/World/TD060"
FRAMES = list(range(2920, 3131))  # 2920 to 3130 inclusive

OUTPUT_DIR = Path(r"D:\ces26_data\td060\v04\fixed_object_id")

RENDER_CONFIG = RenderConfig(
    width=3840,
    height=2160,
    output_dir=OUTPUT_DIR,  # Base directory
    filename_pattern="frame.{frame}.exr",  # Not used, but required
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
# Headlight Helper
# =============================================================================


def update_lights_for_headlight(ctx, camera_forward: np.ndarray) -> None:
    """
    Update render context lights to use headlight pointing along camera forward.
    
    Shadows are disabled for headlight mode since a camera-following light
    would create unnatural shadows that shift with camera movement.
    
    Args:
        ctx: RenderContext to update
        camera_forward: Camera forward direction (normalized)
    """
    ctx.lights_active = wp.array([True], dtype=wp.bool)
    ctx.lights_type = wp.array([1], dtype=wp.int32)  # directional
    ctx.lights_cast_shadow = wp.array([False], dtype=wp.bool)  # No shadows for headlight
    ctx.lights_position = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f)
    ctx.lights_orientation = wp.array([camera_forward.tolist()], dtype=wp.vec3f)


# =============================================================================
# Main
# =============================================================================

def main():
    stage = open_usd_stage(USD_FILE)
    time_code = Usd.TimeCode(FRAMES[0])

    # Headlight mode: simple camera-following light + ambient
    # Lights will be updated per-frame based on camera direction
    print("Using HEADLIGHT mode (camera-following light + ambient)")
    lights = None  # Will be set per-frame

    # Phase 1: Parse USD into Diegetics (geometry + all color channels)
    # Use first frame for geometry parsing (geometry is static, only camera moves)
    options = ParseOptions(
        time_code=time_code,
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
    print(f"Output directory: {OUTPUT_DIR}")

    # Phase 2: Setup render context with headlight lighting + ColorLUTs
    ctx, color_luts, _ = setup_render_context(diegetics, RENDER_CONFIG, lights=lights)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 3: Render object_id AOV for each frame
    print(f"\nRendering object_id AOV for frames {FRAMES[0]} to {FRAMES[-1]}...")
    
    for i, frame in enumerate(FRAMES):
        time_code = Usd.TimeCode(frame)
        camera = get_camera_from_stage(stage, CAMERA_PATH, time_code, verbose=False)
        
        # Update headlight direction to follow camera
        update_lights_for_headlight(ctx, camera.forward)
        
        # Single render pass outputs all AOVs (same as debug_all_visible_camera.py)
        outputs = render_all_aovs(ctx, camera, RENDER_CONFIG)
        
        # Convert to EXR data (same methodology as debug_all_visible_camera.py)
        exr_outputs = convert_aovs_to_exr_data(outputs, color_luts, RENDER_CONFIG)
        
        # Save only the object_id AOV
        output_path = OUTPUT_DIR / f"td060_v04_fixed_object_id.{frame:04d}.exr"
        save_exr_rgb(exr_outputs.object_id, output_path)
        
        # Progress update
        if (i + 1) % 10 == 0 or i == 0 or i == len(FRAMES) - 1:
            print(f"Rendered frame {frame} ({i + 1}/{len(FRAMES)})")

    print("\nDone!")
    print(f"Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

