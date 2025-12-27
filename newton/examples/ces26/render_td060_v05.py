# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Render all visible meshes in USD scene using the TD060 camera (v05).

Uses the Diegetic functional pipeline from ces26_utils with multi-AOV output:
1. parse_diegetics: Folds geometry + color extractors over USD stage
2. setup_render_context: Converts Diegetics to GPU-ready structures + ColorLUTs
3. render_all_aovs: Single render pass outputs all AOVs

EXR output saved to subdirectories (D:\\ces26_data\\td060\\v05\\):
- color/td060_v05_color.{frame:04d}.exr: Lit diffuse render (half-float RGB)
- depth/td060_v05_depth.{frame:04d}.exr: Normalized depth (half-float Y, [0, 1])
- depth_heat/td060_v05_depth_heat.{frame:04d}.exr: Depth heat map (half-float RGB)
- normal/td060_v05_normal.{frame:04d}.exr: Surface normals (half-float RGB)
- object_id/td060_v05_object_id.{frame:04d}.exr: Object ID colors (half-float RGB)
- semantic/td060_v05_semantic.{frame:04d}.exr: Semantic colors (half-float RGB)

Renders at 4K resolution (3840x2160), frames 2920-3130.

Usage: uv run python newton/examples/ces26/render_td060_v05.py
"""

from pathlib import Path

from pxr import Usd

from ces26_utils import (
    ColorLUTs,
    DepthColormap,
    ExrOutputs,
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
    save_exr_depth,
    save_exr_rgb,
    setup_render_context,
)

# =============================================================================
# Configuration
# =============================================================================

USD_FILE = r"C:\Users\chorvath\Downloads\Collected_20251227_iv060_unflat_05\Collected_20251227_iv060_unflat_05\iv060_base.usda"
CAMERA_PATH = "/World/TD060"
FRAMES = list(range(2920, 3131))  # 2920 to 3130 inclusive

# Output format: EXR for half-float OpenEXR
OUTPUT_FORMAT = "exr"

# Depth heat map colormap
DEPTH_COLORMAP = DepthColormap.MAGMA

RENDER_CONFIG = RenderConfig(
    width=3840,
    height=2160,
    output_dir=Path(r"D:\ces26_data\td060\v05"),
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
# Save AOVs to separate subdirectories
# =============================================================================

def save_all_aovs_exr_to_subdirs(
    exr_outputs: ExrOutputs,
    output_base_dir: Path,
    frame_num: int,
    base_name: str = "td060_v05",
) -> None:
    """
    Save all AOV passes to disk as OpenEXR files in separate subdirectories.
    
    Creates subdirectories for each AOV:
    - color/{base_name}_color.{frame:04d}.exr
    - depth/{base_name}_depth.{frame:04d}.exr
    - depth_heat/{base_name}_depth_heat.{frame:04d}.exr
    - normal/{base_name}_normal.{frame:04d}.exr
    - object_id/{base_name}_object_id.{frame:04d}.exr
    - semantic/{base_name}_semantic.{frame:04d}.exr
    
    Args:
        exr_outputs: ExrOutputs from convert_aovs_to_exr_data
        output_base_dir: Base directory (subdirectories created inside)
        frame_num: Frame number for filename (formatted as 4-digit zero-padded)
        base_name: Base filename (before the AOV suffix)
    """
    aov_subdirs = {
        "color": "color",
        "depth": "depth",
        "depth_heat": "depth_heat",
        "normal": "normal",
        "object_id": "object_id",
        "semantic": "semantic",
    }
    
    for aov_name, subdir_name in aov_subdirs.items():
        subdir = output_base_dir / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{base_name}_{aov_name}.{frame_num:04d}.exr"
        output_path = subdir / filename
        
        if aov_name == "depth":
            # Single-channel depth
            save_exr_depth(getattr(exr_outputs, aov_name), output_path)
        else:
            # RGB channels
            save_exr_rgb(getattr(exr_outputs, aov_name), output_path)


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
        
        # Single render pass
        outputs = render_all_aovs(ctx, camera, RENDER_CONFIG)
        
        # Convert to EXR data
        exr_outputs = convert_aovs_to_exr_data(outputs, color_luts, RENDER_CONFIG, DEPTH_COLORMAP)
        
        # Save to separate subdirectories
        save_all_aovs_exr_to_subdirs(
            exr_outputs=exr_outputs,
            output_base_dir=RENDER_CONFIG.output_dir,
            frame_num=frame,
            base_name="td060_v05",
        )
        
        # Progress update
        if (i + 1) % 10 == 0 or i == 0 or i == len(FRAMES) - 1:
            print(f"Rendered frame {frame} ({i + 1}/{len(FRAMES)})")

    print("\n" + "=" * 70)
    print("Done! Output saved to subdirectories in:", RENDER_CONFIG.output_dir)
    print("  - color/")
    print("  - depth/")
    print("  - depth_heat/")
    print("  - normal/")
    print("  - object_id/")
    print("  - semantic/")
    print("=" * 70)


if __name__ == "__main__":
    main()
