# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Render all visible meshes in USD scene using the TD060 camera.

Uses the Diegetic functional pipeline from ces26_utils with multi-AOV output:
1. parse_diegetics: Folds geometry + color extractors over USD stage
2. setup_render_context: Converts Diegetics to GPU-ready structures + ColorLUTs
3. render_and_save_all_aovs: Single render pass outputs all AOVs to disk

Supports two output formats (set OUTPUT_FORMAT in config):

PNG output ({base}_{AOV}.{frame:04d}.png):
- debug_all_visible_color.2920.png: Lit diffuse render (uint8 RGB)
- debug_all_visible_depth.2920.png: Depth visualization (grayscale, closer = brighter)
- debug_all_visible_depth_heat.2920.png: Depth heat map (viridis or magma colormap)
- debug_all_visible_normal.2920.png: Surface normal visualization ((n+1)/2 mapped)
- debug_all_visible_object_id.2920.png: Object ID colors (from USD primvar)
- debug_all_visible_semantic.2920.png: Semantic segmentation colors (random per-object)

EXR output ({base}_{AOV}.{frame:04d}.exr):
- debug_all_visible_color.2920.exr: Lit diffuse render (half-float RGB, [0, 1])
- debug_all_visible_depth.2920.exr: Normalized depth (half-float Y, [0, 1], 0=near 1=far)
- debug_all_visible_depth_heat.2920.exr: Depth heat map with interpolated colormap (half-float RGB)
- debug_all_visible_normal.2920.exr: Surface normals (half-float RGB, [0, 1] mapped from [-1, 1])
- debug_all_visible_object_id.2920.exr: Object ID colors (half-float RGB)
- debug_all_visible_semantic.2920.exr: Semantic colors (half-float RGB)

Renders at 4K resolution (3840x2160).

Usage: uv run python newton/examples/ces26/debug_all_visible_camera.py
"""

from pathlib import Path

from pxr import Usd

from ces26_utils import (
    DepthColormap,
    ExrOutputs,
    LightData,
    ParseOptions,
    PixelOutputs,
    RenderConfig,
    create_fill_light_from_ambient,
    find_ambient_light,
    find_lights,
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

#USD_FILE = r"C:\Users\chorvath\Downloads\20251220_iv060_flat_02\Collected_20251220_iv060_flat_02\20251220_iv060_flat_02.usd"
USD_FILE = r"C:\Users\chorvath\Downloads\20251221_iv060_flat_03\20251221_iv060_flat_03\20251221_iv060_flat_03.usd"
CAMERA_PATH = "/World/TD060"
FRAMES = [2920, 3130]

# Output format: "png" for uint8 PNG, "exr" for half-float OpenEXR
OUTPUT_FORMAT = "png"  # or "exr"

# Depth heat map colormap: try MAGMA (warm) or VIRIDIS (cool-to-warm)
# Note: Only used for PNG output (depth_heat pass)
DEPTH_COLORMAP = DepthColormap.MAGMA  # or DepthColormap.VIRIDIS

# Lighting mode:
#   "scene" - Use lights from USD scene (DistantLight + fill from DomeLight)
#   "headlight" - Simple headlight (follows camera) + constant ambient
#LIGHTING_MODE = "scene"  # or "headlight"
LIGHTING_MODE = "headlight"

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
# Headlight Helper
# =============================================================================

import numpy as np
import warp as wp


def create_headlight(camera_forward: np.ndarray) -> LightData:
    """
    Create a headlight that points in the camera's view direction.
    
    A headlight provides simple, consistent lighting for debugging.
    It follows the camera so all surfaces facing the camera are lit.
    Shadows are disabled since they would shift unnaturally with camera movement.
    
    Args:
        camera_forward: Camera forward direction (normalized)
        
    Returns:
        LightData for a directional light pointing along the camera view
    """
    return LightData(
        path="/Headlight",
        light_type=1,  # directional
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        direction=camera_forward.astype(np.float32),
        color=(1.0, 1.0, 1.0),  # Neutral white
        intensity=1.0,
        cast_shadows=False,  # No shadows for headlight
    )


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

    # Lighting setup depends on mode
    lights = None
    use_headlight = (LIGHTING_MODE == "headlight")
    
    if use_headlight:
        # Headlight mode: simple camera-following light + ambient
        # Lights will be updated per-frame based on camera direction
        print("Using HEADLIGHT mode (camera-following light + ambient)")
        lights = None  # Will be set per-frame
    else:
        # Scene mode: use lights from USD
        print("Using SCENE LIGHTING mode")
        
        # Phase 1a: Find lights in the USD scene
        # Filter to use only /Environment lights (skip duplicates in BarbieColors)
        def environment_light_filter(path: str) -> bool:
            return path.startswith("/Environment/")
        
        lights = find_lights(
            stage=stage,
            time_code=time_code,
            path_filter=environment_light_filter,
            verbose=True,
        )
        print(f"Found {len(lights)} directional/positional lights from USD scene")
        
        # Phase 1a.2: Find ambient (DomeLight) and create a fill light
        # Since the renderer has hardcoded ambient, we add a fill light to brighten shadows
        ambient = find_ambient_light(
            stage=stage,
            time_code=time_code,
            path_filter=environment_light_filter,
            verbose=True,
        )
        
        if ambient:
            # Get key light direction if available
            key_direction = lights[0].direction if lights else None
            fill_light = create_fill_light_from_ambient(ambient, key_direction)
            
            # Artistic decision: drop fill light by 2 stops (multiply by 0.25)
            # The renderer treats all lights equally (no per-light intensity),
            # so we bake the reduction into the color
            fill_exposure_reduction = 0.25  # 2 stops down
            dimmed_color = (
                fill_light.color[0] * fill_exposure_reduction,
                fill_light.color[1] * fill_exposure_reduction,
                fill_light.color[2] * fill_exposure_reduction,
            )
            # Create a new LightData with the dimmed color
            fill_light = LightData(
                path=fill_light.path,
                light_type=fill_light.light_type,
                position=fill_light.position,
                direction=fill_light.direction,
                color=dimmed_color,
                intensity=fill_light.intensity,
                cast_shadows=fill_light.cast_shadows,
            )
            
            lights.append(fill_light)
            print(f"Added fill light from DomeLight ({ambient.path})")
            print(f"  Fill direction: [{fill_light.direction[0]:.3f}, {fill_light.direction[1]:.3f}, {fill_light.direction[2]:.3f}]")
            print(f"  Fill color (after -2 stops): RGB({fill_light.color[0]:.3f}, {fill_light.color[1]:.3f}, {fill_light.color[2]:.3f})")

    # Phase 1b: Parse USD into Diegetics (geometry + all color channels)
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

    # Phase 2: Setup render context with USD lights + ColorLUTs
    ctx, color_luts, _ = setup_render_context(diegetics, RENDER_CONFIG, lights=lights)

    # Phase 3: Render all AOVs for each frame
    for frame in FRAMES:
        time_code = Usd.TimeCode(frame)
        camera = get_camera_from_stage(stage, CAMERA_PATH, time_code, verbose=True)
        
        # In headlight mode, update light direction to follow camera each frame
        if use_headlight:
            update_lights_for_headlight(ctx, camera.forward)
            print(f"  Headlight direction: [{camera.forward[0]:.3f}, {camera.forward[1]:.3f}, {camera.forward[2]:.3f}]")
        
        # Single render pass outputs all AOVs
        render_and_save_all_aovs(
            ctx=ctx,
            color_luts=color_luts,
            camera=camera,
            config=RENDER_CONFIG,
            frame_num=frame,
            base_name="debug_all_visible",
            output_format=OUTPUT_FORMAT,
            depth_colormap=DEPTH_COLORMAP,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
