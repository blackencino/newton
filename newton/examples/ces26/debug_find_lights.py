# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Debug script to find and enumerate all lights in a USD scene.

Discovers lights using UsdLux API and prints their properties:
- Light type (DirectionalLight, DistantLight, SphereLight, etc.)
- Position and orientation (from Xform)
- Intensity and color
- Shadow settings

Usage: uv run python newton/examples/ces26/debug_find_lights.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdLux

if TYPE_CHECKING:
    from collections.abc import Iterator

# =============================================================================
# Configuration
# =============================================================================

USD_FILE = r"C:\Users\chorvath\Downloads\20251221_iv060_flat_03\20251221_iv060_flat_03\20251221_iv060_flat_03.usd"
TIME_CODE = Usd.TimeCode(2920)

# =============================================================================
# Light Data
# =============================================================================

@dataclass
class LightInfo:
    """Container for extracted light information."""
    path: str
    light_type: str
    position: np.ndarray  # World-space position (3,)
    direction: np.ndarray  # Light direction in world space (3,), normalized
    color: tuple[float, float, float]  # RGB color
    intensity: float
    exposure: float  # Log2 exposure bias
    cast_shadows: bool
    is_distant: bool  # True for directional/distant lights


def get_light_direction_from_xform(
    prim: Usd.Prim,
    xcache: UsdGeom.XformCache,
) -> np.ndarray:
    """
    Get the light direction from its transform.
    
    USD lights point down their local -Z axis by convention.
    We transform the local -Z vector to world space.
    """
    world_mat = xcache.GetLocalToWorldTransform(prim)
    
    # Transform origin and -Z direction point
    pts_local = np.array([[0, 0, 0], [0, 0, -1]], dtype=np.float64)
    
    # Transform to world space
    M = np.array(world_mat, dtype=np.float64)
    pts_h = np.concatenate([pts_local, np.ones((pts_local.shape[0], 1), dtype=np.float64)], axis=1)
    pts_world_h = pts_h @ M
    pts_world = pts_world_h[:, :3]
    
    origin, z_neg_pt = pts_world
    direction = z_neg_pt - origin
    direction = direction / np.linalg.norm(direction)
    
    return direction.astype(np.float32)


def get_light_position_from_xform(
    prim: Usd.Prim,
    xcache: UsdGeom.XformCache,
) -> np.ndarray:
    """Get the world-space position of a light from its transform."""
    world_mat = xcache.GetLocalToWorldTransform(prim)
    
    # Transform origin
    M = np.array(world_mat, dtype=np.float64)
    origin = np.array([0, 0, 0, 1], dtype=np.float64)
    origin_world = origin @ M
    
    return origin_world[:3].astype(np.float32)


def extract_light_info(
    prim: Usd.Prim,
    xcache: UsdGeom.XformCache,
    time_code: Usd.TimeCode,
) -> LightInfo | None:
    """
    Extract light information from a USD light prim.
    
    Returns None if the prim is not a recognized light type.
    """
    # Determine light type
    light_type = None
    is_distant = False
    
    if prim.IsA(UsdLux.DistantLight):
        light_type = "DistantLight"
        is_distant = True
    elif prim.IsA(UsdLux.DomeLight):
        light_type = "DomeLight"
        is_distant = True  # Environment light
    elif prim.IsA(UsdLux.SphereLight):
        light_type = "SphereLight"
    elif prim.IsA(UsdLux.RectLight):
        light_type = "RectLight"
    elif prim.IsA(UsdLux.DiskLight):
        light_type = "DiskLight"
    elif prim.IsA(UsdLux.CylinderLight):
        light_type = "CylinderLight"
    elif prim.IsA(UsdLux.PluginLight):
        light_type = "PluginLight"
    else:
        # Check if it's a generic light-like prim
        # (some prims may have light attributes without being typed)
        return None
    
    # Get the base light API
    light = UsdLux.LightAPI(prim)
    
    # Extract common properties
    color = (1.0, 1.0, 1.0)
    color_attr = light.GetColorAttr()
    if color_attr and color_attr.HasValue():
        c = color_attr.Get(time_code)
        if c:
            color = (float(c[0]), float(c[1]), float(c[2]))
    
    intensity = 1.0
    intensity_attr = light.GetIntensityAttr()
    if intensity_attr and intensity_attr.HasValue():
        intensity = float(intensity_attr.Get(time_code))
    
    exposure = 0.0
    exposure_attr = light.GetExposureAttr()
    if exposure_attr and exposure_attr.HasValue():
        exposure = float(exposure_attr.Get(time_code))
    
    # Shadow settings - use the ShadowAPI if available
    cast_shadows = True
    shadow_api = UsdLux.ShadowAPI(prim)
    if shadow_api:
        enable_attr = shadow_api.GetShadowEnableAttr()
        if enable_attr and enable_attr.HasValue():
            cast_shadows = bool(enable_attr.Get(time_code))
    
    # Get position and direction
    position = get_light_position_from_xform(prim, xcache)
    direction = get_light_direction_from_xform(prim, xcache)
    
    return LightInfo(
        path=str(prim.GetPath()),
        light_type=light_type,
        position=position,
        direction=direction,
        color=color,
        intensity=intensity,
        exposure=exposure,
        cast_shadows=cast_shadows,
        is_distant=is_distant,
    )


def find_all_lights(
    stage: Usd.Stage,
    time_code: Usd.TimeCode,
    verbose: bool = True,
) -> list[LightInfo]:
    """
    Find all lights in a USD stage.
    
    Uses UsdLux.LightListAPI for comprehensive light discovery,
    falling back to traversal if needed.
    """
    xcache = UsdGeom.XformCache(time_code)
    lights: list[LightInfo] = []
    
    if verbose:
        print("Searching for lights in USD stage...")
    
    # Method 1: Try LightListAPI (comprehensive, handles light linking)
    try:
        light_list_api = UsdLux.ListAPI(stage.GetPseudoRoot())
        light_paths = light_list_api.ComputeLightList(UsdLux.ListAPI.ComputeModeIgnoreCache)
        
        if light_paths:
            if verbose:
                print(f"  LightListAPI found {len(light_paths)} lights")
            
            for path in light_paths:
                prim = stage.GetPrimAtPath(path)
                if prim and prim.IsValid():
                    light_info = extract_light_info(prim, xcache, time_code)
                    if light_info:
                        lights.append(light_info)
            
            return lights
    except AttributeError:
        # UsdLux.ListAPI might be named differently in older USD versions
        pass
    
    # Method 2: Traverse stage looking for light prims
    if verbose:
        print("  Using stage traversal to find lights...")
    
    for prim in stage.Traverse():
        # Check if this prim is any type of light
        if prim.IsA(UsdLux.BoundableLightBase) or prim.IsA(UsdLux.NonboundableLightBase):
            light_info = extract_light_info(prim, xcache, time_code)
            if light_info:
                lights.append(light_info)
    
    return lights


def print_light_info(light: LightInfo) -> None:
    """Print detailed information about a light."""
    print(f"\n  {light.path}")
    print(f"    Type: {light.light_type}")
    print(f"    Position: [{light.position[0]:.3f}, {light.position[1]:.3f}, {light.position[2]:.3f}]")
    print(f"    Direction: [{light.direction[0]:.3f}, {light.direction[1]:.3f}, {light.direction[2]:.3f}]")
    print(f"    Color: RGB({light.color[0]:.3f}, {light.color[1]:.3f}, {light.color[2]:.3f})")
    print(f"    Intensity: {light.intensity}")
    print(f"    Exposure: {light.exposure}")
    print(f"    Cast Shadows: {light.cast_shadows}")
    print(f"    Distant/Directional: {light.is_distant}")


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Loading USD: {USD_FILE}")
    stage = Usd.Stage.Open(USD_FILE)
    
    if not stage:
        print("Failed to open USD file!")
        return
    
    print(f"Stage opened successfully")
    print(f"Time code: {TIME_CODE}")
    
    # Find all lights
    lights = find_all_lights(stage, TIME_CODE)
    
    print(f"\n{'='*60}")
    print(f"Found {len(lights)} lights:")
    print(f"{'='*60}")
    
    if not lights:
        print("\n  No lights found in the scene!")
        print("\n  This USD file may not contain any light prims.")
        print("  The renderer will use default ambient + directional lighting.")
    else:
        for light in lights:
            print_light_info(light)
        
        # Summary
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"{'='*60}")
        
        distant_lights = [l for l in lights if l.is_distant]
        local_lights = [l for l in lights if not l.is_distant]
        
        print(f"  Distant/Directional lights: {len(distant_lights)}")
        for l in distant_lights:
            print(f"    - {l.path} ({l.light_type})")
        
        print(f"  Local lights (point/area): {len(local_lights)}")
        for l in local_lights:
            print(f"    - {l.path} ({l.light_type})")


if __name__ == "__main__":
    main()

