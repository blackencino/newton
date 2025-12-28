# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Functional utilities for loading and rendering USD scene geometry.

Core concepts:
- Diegetic: An immutable scene element with geometry and multiple color channels
- ColorExtractor: A function that extracts a color from a USD prim
- parse_diegetics: A fold that maps geometry + color extractors over a USD stage

The pipeline is: USD Stage → list[Diegetic] → RenderContext → pixels
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import warp as wp
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade

from newton._src.sensors.warp_raytrace import ClearData, RenderContext, RenderShapeType

# Optional PIL import for texture loading
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Optional OpenEXR import for EXR output
try:
    import OpenEXR
    _OPENEXR_AVAILABLE = True
except ImportError:
    _OPENEXR_AVAILABLE = False


# =============================================================================
# Multi-AOV Render Outputs
# =============================================================================

@dataclass
class RenderOutputs:
    """
    Container for all raw GPU render outputs from a single render pass.
    
    The raytracer outputs multiple AOVs (Arbitrary Output Variables) in one pass:
    - color_image: Lit diffuse render (packed RGBA as uint32)
    - depth_image: Ray hit distance (float32)
    - shape_index_image: Shape index at each pixel (uint32)
    - normal_image: Surface normal at hit point (vec3f)
    
    The shape_index_image is key for generating additional passes like object_id
    and semantic - we use lookup tables to map shape indices to colors in post.
    """
    color_image: wp.array  # (num_worlds, num_cameras, width*height), uint32
    depth_image: wp.array  # (num_worlds, num_cameras, width*height), float32
    shape_index_image: wp.array  # (num_worlds, num_cameras, width*height), uint32
    normal_image: wp.array  # (num_worlds, num_cameras, width*height), vec3f


@dataclass
class ColorLUTs:
    """
    Lookup tables for mapping shape indices to different color channels.
    
    Each LUT is a warp array of packed uint32 RGBA colors, indexed by shape index.
    Used to post-process shape_index_image into different color passes.
    """
    object_id: wp.array  # (num_shapes,), uint32 packed RGBA
    semantic: wp.array   # (num_shapes,), uint32 packed RGBA


@dataclass
class PixelOutputs:
    """
    Container for converted numpy RGB arrays, ready for saving to disk.
    
    All arrays are (height, width, 3) uint8 RGB format.
    """
    color: np.ndarray       # Lit diffuse render
    depth: np.ndarray       # Depth visualization (grayscale as RGB)
    depth_heat: np.ndarray  # Depth visualization (colormap false-color)
    normal: np.ndarray      # Normal visualization ((n+1)/2 mapped to 0-255)
    object_id: np.ndarray   # Object ID colors
    semantic: np.ndarray    # Semantic colors


@dataclass
class ExrOutputs:
    """
    Container for float arrays, ready for saving as OpenEXR files.
    
    All color arrays are (height, width, 3) float32.
    Depth is (height, width) float32, normalized to [0, 1].
    Normal is (height, width, 3) float32, mapped to [0, 1] from [-1, 1].
    Shape index is (height, width) float32, raw integer indices stored as floats.
    """
    color: np.ndarray       # Lit diffuse render, float32 RGB in [0, 1]
    depth: np.ndarray       # Normalized depth values, float32 in [0, 1] (0=near, 1=far)
    depth_heat: np.ndarray  # Depth heat map visualization, float32 RGB in [0, 1]
    normal: np.ndarray      # Surface normals mapped to [0, 1], float32
    object_id: np.ndarray   # Object ID colors, float32 RGB in [0, 1]
    semantic: np.ndarray    # Semantic colors, float32 RGB in [0, 1]
    shape_index: np.ndarray # Raw shape index integers as float32, (height, width)


# =============================================================================
# Colormaps for Depth Visualization
# =============================================================================

class DepthColormap:
    """Enum-like class for depth colormap options."""
    VIRIDIS = "viridis"
    MAGMA = "magma"


# Viridis colormap - 256 entries, RGB 0-255
# Source: matplotlib viridis
_VIRIDIS_DATA = [
    (68, 1, 84), (68, 2, 86), (69, 4, 87), (69, 5, 89), (70, 7, 90),
    (70, 8, 92), (70, 10, 93), (70, 11, 94), (71, 13, 96), (71, 14, 97),
    (71, 16, 99), (71, 17, 100), (71, 19, 101), (72, 20, 103), (72, 22, 104),
    (72, 23, 105), (72, 24, 106), (72, 26, 108), (72, 27, 109), (72, 28, 110),
    (72, 29, 111), (72, 31, 112), (72, 32, 113), (72, 33, 115), (72, 35, 116),
    (72, 36, 117), (72, 37, 118), (72, 38, 119), (72, 40, 120), (72, 41, 121),
    (71, 42, 122), (71, 44, 122), (71, 45, 123), (71, 46, 124), (71, 47, 125),
    (70, 48, 126), (70, 50, 126), (70, 51, 127), (69, 52, 128), (69, 53, 129),
    (69, 55, 129), (68, 56, 130), (68, 57, 131), (68, 58, 131), (67, 60, 132),
    (67, 61, 132), (66, 62, 133), (66, 63, 133), (66, 64, 134), (65, 66, 134),
    (65, 67, 135), (64, 68, 135), (64, 69, 136), (63, 71, 136), (63, 72, 137),
    (62, 73, 137), (62, 74, 137), (62, 76, 138), (61, 77, 138), (61, 78, 138),
    (60, 79, 139), (60, 80, 139), (59, 82, 139), (59, 83, 140), (58, 84, 140),
    (58, 85, 140), (57, 86, 141), (57, 88, 141), (56, 89, 141), (56, 90, 141),
    (55, 91, 142), (55, 92, 142), (54, 94, 142), (54, 95, 142), (53, 96, 142),
    (53, 97, 143), (52, 98, 143), (52, 100, 143), (51, 101, 143), (51, 102, 143),
    (50, 103, 143), (50, 104, 144), (49, 106, 144), (49, 107, 144), (49, 108, 144),
    (48, 109, 144), (48, 110, 144), (47, 111, 144), (47, 113, 144), (46, 114, 144),
    (46, 115, 144), (45, 116, 144), (45, 117, 144), (45, 118, 144), (44, 120, 144),
    (44, 121, 144), (43, 122, 144), (43, 123, 144), (43, 124, 144), (42, 125, 144),
    (42, 126, 144), (42, 128, 144), (41, 129, 144), (41, 130, 143), (40, 131, 143),
    (40, 132, 143), (40, 133, 143), (39, 134, 143), (39, 136, 143), (39, 137, 142),
    (38, 138, 142), (38, 139, 142), (38, 140, 142), (37, 141, 141), (37, 142, 141),
    (37, 144, 141), (36, 145, 140), (36, 146, 140), (36, 147, 140), (36, 148, 139),
    (35, 149, 139), (35, 150, 138), (35, 152, 138), (35, 153, 137), (35, 154, 137),
    (34, 155, 136), (34, 156, 136), (34, 157, 135), (34, 158, 135), (34, 160, 134),
    (34, 161, 133), (34, 162, 133), (34, 163, 132), (34, 164, 131), (34, 165, 131),
    (34, 166, 130), (34, 168, 129), (34, 169, 128), (35, 170, 128), (35, 171, 127),
    (35, 172, 126), (36, 173, 125), (36, 174, 124), (37, 176, 124), (37, 177, 123),
    (38, 178, 122), (38, 179, 121), (39, 180, 120), (40, 181, 119), (40, 182, 118),
    (41, 183, 117), (42, 184, 116), (43, 186, 115), (44, 187, 114), (45, 188, 113),
    (46, 189, 112), (47, 190, 111), (48, 191, 110), (49, 192, 109), (50, 193, 108),
    (52, 194, 106), (53, 195, 105), (54, 196, 104), (56, 197, 103), (57, 198, 101),
    (59, 199, 100), (60, 200, 99), (62, 201, 97), (63, 202, 96), (65, 203, 95),
    (67, 203, 93), (69, 204, 92), (70, 205, 90), (72, 206, 89), (74, 207, 87),
    (76, 208, 86), (78, 209, 84), (80, 209, 83), (82, 210, 81), (84, 211, 79),
    (86, 212, 78), (88, 212, 76), (90, 213, 74), (92, 214, 73), (94, 214, 71),
    (97, 215, 69), (99, 216, 68), (101, 216, 66), (103, 217, 64), (105, 218, 62),
    (108, 218, 60), (110, 219, 59), (112, 219, 57), (115, 220, 55), (117, 221, 53),
    (119, 221, 51), (122, 222, 49), (124, 222, 47), (127, 223, 45), (129, 223, 43),
    (132, 224, 41), (134, 224, 39), (137, 225, 37), (139, 225, 35), (142, 225, 33),
    (144, 226, 31), (147, 226, 29), (149, 227, 27), (152, 227, 25), (155, 227, 23),
    (157, 228, 21), (160, 228, 19), (162, 228, 18), (165, 229, 16), (168, 229, 14),
    (170, 229, 13), (173, 229, 11), (176, 230, 10), (178, 230, 9), (181, 230, 8),
    (184, 230, 7), (186, 230, 7), (189, 231, 6), (192, 231, 6), (194, 231, 6),
    (197, 231, 6), (200, 231, 6), (202, 231, 7), (205, 231, 7), (208, 231, 8),
    (210, 231, 9), (213, 231, 10), (215, 231, 11), (218, 231, 12), (221, 231, 14),
    (223, 230, 15), (226, 230, 17), (228, 230, 18), (231, 230, 20), (233, 229, 22),
    (236, 229, 24), (238, 229, 26), (240, 228, 28), (243, 228, 30), (245, 227, 32),
    (247, 227, 34), (249, 226, 37), (251, 226, 39), (253, 225, 41),
]

# Magma colormap - 256 entries, RGB 0-255
# Source: matplotlib magma
_MAGMA_DATA = [
    (0, 0, 4), (1, 0, 5), (1, 1, 6), (1, 1, 8), (2, 1, 9),
    (2, 2, 11), (2, 2, 13), (3, 3, 15), (3, 3, 18), (4, 4, 20),
    (5, 4, 22), (6, 5, 24), (6, 5, 26), (7, 6, 28), (8, 7, 30),
    (9, 7, 32), (10, 8, 34), (11, 9, 36), (12, 9, 38), (13, 10, 41),
    (14, 11, 43), (16, 11, 45), (17, 12, 47), (18, 13, 49), (19, 13, 52),
    (20, 14, 54), (21, 14, 56), (22, 15, 59), (24, 15, 61), (25, 16, 63),
    (26, 16, 66), (28, 16, 68), (29, 17, 71), (30, 17, 73), (32, 17, 75),
    (33, 17, 78), (34, 17, 80), (36, 18, 83), (37, 18, 85), (39, 18, 88),
    (41, 17, 90), (42, 17, 92), (44, 17, 95), (45, 17, 97), (47, 17, 99),
    (49, 17, 101), (51, 16, 103), (52, 16, 105), (54, 16, 107), (56, 16, 108),
    (57, 15, 110), (59, 15, 112), (61, 15, 113), (63, 15, 114), (64, 15, 116),
    (66, 15, 117), (68, 15, 118), (69, 16, 119), (71, 16, 120), (73, 16, 120),
    (74, 16, 121), (76, 17, 122), (78, 17, 123), (79, 18, 123), (81, 18, 124),
    (82, 19, 124), (84, 19, 125), (86, 20, 125), (87, 21, 126), (89, 21, 126),
    (90, 22, 126), (92, 22, 127), (93, 23, 127), (95, 24, 127), (96, 24, 128),
    (98, 25, 128), (100, 26, 128), (101, 26, 128), (103, 27, 128), (104, 28, 129),
    (106, 28, 129), (107, 29, 129), (109, 30, 129), (110, 30, 129), (112, 31, 129),
    (114, 32, 129), (115, 32, 129), (117, 33, 129), (118, 34, 129), (120, 34, 129),
    (121, 35, 129), (123, 36, 129), (124, 36, 129), (126, 37, 129), (127, 38, 129),
    (129, 38, 129), (130, 39, 129), (132, 40, 129), (133, 40, 129), (135, 41, 128),
    (136, 42, 128), (138, 42, 128), (140, 43, 128), (141, 44, 127), (143, 44, 127),
    (144, 45, 127), (146, 46, 126), (147, 46, 126), (149, 47, 126), (150, 48, 125),
    (152, 48, 125), (153, 49, 124), (155, 50, 124), (156, 51, 123), (158, 51, 123),
    (160, 52, 122), (161, 53, 122), (163, 53, 121), (164, 54, 121), (166, 55, 120),
    (167, 56, 119), (169, 56, 119), (170, 57, 118), (172, 58, 117), (173, 58, 117),
    (175, 59, 116), (176, 60, 115), (178, 61, 114), (179, 61, 114), (181, 62, 113),
    (182, 63, 112), (184, 64, 111), (185, 64, 110), (187, 65, 110), (188, 66, 109),
    (189, 67, 108), (191, 68, 107), (192, 68, 106), (194, 69, 105), (195, 70, 104),
    (196, 71, 103), (198, 72, 102), (199, 72, 101), (200, 73, 100), (202, 74, 99),
    (203, 75, 98), (204, 76, 97), (205, 77, 96), (207, 78, 95), (208, 79, 94),
    (209, 80, 93), (210, 81, 92), (211, 82, 91), (212, 83, 90), (214, 84, 89),
    (215, 85, 88), (216, 86, 87), (217, 87, 85), (218, 88, 84), (219, 89, 83),
    (220, 90, 82), (221, 91, 81), (222, 93, 80), (223, 94, 79), (224, 95, 78),
    (225, 96, 76), (226, 97, 75), (227, 99, 74), (228, 100, 73), (228, 101, 72),
    (229, 102, 71), (230, 104, 70), (231, 105, 68), (231, 106, 67), (232, 108, 66),
    (233, 109, 65), (233, 111, 64), (234, 112, 63), (235, 114, 62), (235, 115, 60),
    (236, 117, 59), (236, 118, 58), (237, 120, 57), (237, 121, 56), (238, 123, 55),
    (238, 125, 54), (239, 126, 53), (239, 128, 52), (240, 130, 51), (240, 131, 50),
    (240, 133, 49), (241, 135, 48), (241, 137, 47), (241, 138, 46), (242, 140, 45),
    (242, 142, 45), (242, 144, 44), (243, 146, 43), (243, 147, 43), (243, 149, 42),
    (243, 151, 42), (244, 153, 41), (244, 155, 41), (244, 157, 40), (244, 159, 40),
    (244, 161, 40), (245, 163, 40), (245, 165, 40), (245, 167, 40), (245, 168, 40),
    (245, 170, 40), (245, 172, 41), (246, 174, 41), (246, 176, 42), (246, 178, 42),
    (246, 180, 43), (246, 182, 44), (246, 184, 45), (246, 186, 46), (246, 188, 47),
    (246, 190, 48), (246, 192, 50), (246, 194, 51), (246, 196, 53), (246, 198, 54),
    (246, 200, 56), (247, 202, 58), (247, 204, 60), (247, 205, 62), (247, 207, 64),
    (247, 209, 66), (247, 211, 68), (247, 213, 70), (247, 215, 73), (247, 217, 75),
    (248, 219, 77), (248, 221, 80), (248, 223, 82), (248, 225, 85), (248, 227, 88),
    (249, 229, 90), (249, 230, 93), (249, 232, 96), (249, 234, 99), (250, 236, 102),
    (250, 238, 105), (250, 240, 108), (251, 242, 111), (251, 244, 115), (252, 246, 118),
    (252, 247, 121), (253, 249, 125), (253, 251, 128),
]

def _build_colormap_lut(colormap_data: list[tuple[int, int, int]]) -> wp.array:
    """Build a warp array LUT from colormap data."""
    packed = np.array([
        (0xFF << 24) | (b << 16) | (g << 8) | r
        for r, g, b in colormap_data
    ], dtype=np.uint32)
    return wp.array(packed, dtype=wp.uint32)


# Pre-built colormap LUTs (created lazily)
_viridis_lut: wp.array | None = None
_magma_lut: wp.array | None = None


def get_colormap_lut(colormap: str) -> wp.array:
    """Get the warp array LUT for a colormap."""
    global _viridis_lut, _magma_lut
    
    if colormap == DepthColormap.VIRIDIS:
        if _viridis_lut is None:
            _viridis_lut = _build_colormap_lut(_VIRIDIS_DATA)
        return _viridis_lut
    elif colormap == DepthColormap.MAGMA:
        if _magma_lut is None:
            _magma_lut = _build_colormap_lut(_MAGMA_DATA)
        return _magma_lut
    else:
        raise ValueError(f"Unknown colormap: {colormap}")


# =============================================================================
# Warp Kernels for Shape Index → Color Mapping
# =============================================================================

@wp.kernel
def shape_index_to_color_lut(
    shape_indices: wp.array(dtype=wp.uint32, ndim=3),
    color_lut: wp.array(dtype=wp.uint32),
    out_rgba: wp.array(dtype=wp.uint32, ndim=3),
):
    """Map shape indices to colors using a lookup table."""
    world_id, camera_id, pixel_id = wp.tid()
    shape_index = shape_indices[world_id, camera_id, pixel_id]
    if shape_index < wp.uint32(color_lut.shape[0]):
        out_rgba[world_id, camera_id, pixel_id] = color_lut[wp.int32(shape_index)]
    else:
        # Background or invalid - transparent black
        out_rgba[world_id, camera_id, pixel_id] = wp.uint32(0xFF404040)


@wp.kernel
def depth_to_grayscale(
    depth_image: wp.array(dtype=wp.float32, ndim=3),
    min_depth: wp.float32,
    max_depth: wp.float32,
    out_rgba: wp.array(dtype=wp.uint32, ndim=3),
):
    """Convert depth values to grayscale visualization (closer = brighter)."""
    world_id, camera_id, pixel_id = wp.tid()
    depth = depth_image[world_id, camera_id, pixel_id]
    
    if depth <= 0.0:
        # No hit - dark gray background
        out_rgba[world_id, camera_id, pixel_id] = wp.uint32(0xFF404040)
        return
    
    # Normalize and invert (closer = brighter)
    denom = wp.max(max_depth - min_depth, 0.001)
    normalized = (depth - min_depth) / denom
    value = wp.uint32((1.0 - normalized) * 255.0)
    value = wp.min(value, wp.uint32(255))
    
    # Pack as grayscale RGBA
    out_rgba[world_id, camera_id, pixel_id] = (
        wp.uint32(0xFF000000) | (value << wp.uint32(16)) | (value << wp.uint32(8)) | value
    )


@wp.kernel
def normal_to_rgb(
    normal_image: wp.array(dtype=wp.vec3f, ndim=3),
    out_rgba: wp.array(dtype=wp.uint32, ndim=3),
):
    """Convert world-space normals to RGB visualization."""
    world_id, camera_id, pixel_id = wp.tid()
    normal = normal_image[world_id, camera_id, pixel_id]
    
    # Check for zero normal (no hit)
    length = wp.length(normal)
    if length < 0.001:
        out_rgba[world_id, camera_id, pixel_id] = wp.uint32(0xFF404040)
        return
    
    # Map from [-1,1] to [0,1] then to [0,255]
    r = wp.uint32((normal[0] * 0.5 + 0.5) * 255.0)
    g = wp.uint32((normal[1] * 0.5 + 0.5) * 255.0)
    b = wp.uint32((normal[2] * 0.5 + 0.5) * 255.0)
    
    r = wp.min(wp.max(r, wp.uint32(0)), wp.uint32(255))
    g = wp.min(wp.max(g, wp.uint32(0)), wp.uint32(255))
    b = wp.min(wp.max(b, wp.uint32(0)), wp.uint32(255))
    
    out_rgba[world_id, camera_id, pixel_id] = (
        wp.uint32(0xFF000000) | (b << wp.uint32(16)) | (g << wp.uint32(8)) | r
    )


@wp.kernel
def find_depth_range(
    depth_image: wp.array(dtype=wp.float32, ndim=3),
    depth_range: wp.array(dtype=wp.float32),
):
    """Find min/max depth values for normalization."""
    world_id, camera_id, pixel_id = wp.tid()
    depth = depth_image[world_id, camera_id, pixel_id]
    if depth > 0.0:
        wp.atomic_min(depth_range, 0, depth)
        wp.atomic_max(depth_range, 1, depth)


@wp.kernel
def depth_to_colormap(
    depth_image: wp.array(dtype=wp.float32, ndim=3),
    min_depth: wp.float32,
    max_depth: wp.float32,
    colormap_lut: wp.array(dtype=wp.uint32),
    out_rgba: wp.array(dtype=wp.uint32, ndim=3),
):
    """
    Convert depth values to colormap visualization with log transform.
    
    Uses log_250 transform: log(1 + depth * 249) / log(250)
    This spreads near values more evenly across the colormap.
    
    Closer = warm (high LUT index), Far = cool (low LUT index).
    Background uses the far end of the colormap.
    """
    world_id, camera_id, pixel_id = wp.tid()
    depth = depth_image[world_id, camera_id, pixel_id]
    
    lut_size = colormap_lut.shape[0]
    
    if depth <= 0.0:
        # No hit - use far end of colormap (index 0 after inversion)
        out_rgba[world_id, camera_id, pixel_id] = colormap_lut[0]
        return
    
    # Normalize depth to 0-1 (0=near, 1=far)
    denom = wp.max(max_depth - min_depth, 0.001)
    normalized = (depth - min_depth) / denom
    
    # Apply log_250 transform: spreads near values more evenly
    # log(1 + x * 249) / log(250) maps [0,1] -> [0,1] with log stretching
    log_base = 250.0
    transformed = wp.log(1.0 + normalized * (log_base - 1.0)) / wp.log(log_base)
    
    # Invert so closer = higher value (brighter/warmer in colormap)
    inverted = 1.0 - transformed
    
    # Map to colormap index
    index = wp.int32(inverted * wp.float32(lut_size - 1))
    index = wp.clamp(index, 0, lut_size - 1)
    
    out_rgba[world_id, camera_id, pixel_id] = colormap_lut[index]


@wp.func
def unpack_lut_to_rgb(packed: wp.uint32) -> wp.vec3f:
    """Unpack uint32 RGBA to float RGB in [0, 1]."""
    r = wp.float32((packed >> wp.uint32(0)) & wp.uint32(255)) / 255.0
    g = wp.float32((packed >> wp.uint32(8)) & wp.uint32(255)) / 255.0
    b = wp.float32((packed >> wp.uint32(16)) & wp.uint32(255)) / 255.0
    return wp.vec3f(r, g, b)


@wp.func
def sample_colormap_interpolated(colormap_lut: wp.array(dtype=wp.uint32), t: wp.float32) -> wp.vec3f:
    """
    Sample colormap with linear interpolation for smooth gradients.
    
    Args:
        colormap_lut: LUT with packed uint32 RGB values (256 entries)
        t: Normalized position in [0, 1]
        
    Returns:
        Interpolated RGB color in [0, 1]
    """
    lut_size = colormap_lut.shape[0]
    
    # Scale t to LUT range
    pos = t * wp.float32(lut_size - 1)
    
    # Get integer indices and fractional part
    idx0 = wp.int32(wp.floor(pos))
    idx1 = idx0 + 1
    frac = pos - wp.float32(idx0)
    
    # Clamp indices
    idx0 = wp.clamp(idx0, 0, lut_size - 1)
    idx1 = wp.clamp(idx1, 0, lut_size - 1)
    
    # Unpack and interpolate
    c0 = unpack_lut_to_rgb(colormap_lut[idx0])
    c1 = unpack_lut_to_rgb(colormap_lut[idx1])
    
    return c0 * (1.0 - frac) + c1 * frac


@wp.kernel
def depth_to_colormap_float(
    depth_image: wp.array(dtype=wp.float32, ndim=3),
    min_depth: wp.float32,
    max_depth: wp.float32,
    colormap_lut: wp.array(dtype=wp.uint32),
    out_rgb: wp.array(dtype=wp.vec3f, ndim=3),
):
    """
    Convert depth values to colormap visualization with interpolation.
    
    Uses log_250 transform and linear interpolation between LUT entries
    for smooth gradients. Outputs float RGB in [0, 1].
    
    Closer = warm (high LUT index), Far = cool (low LUT index).
    Background uses the far end of the colormap.
    """
    world_id, camera_id, pixel_id = wp.tid()
    depth = depth_image[world_id, camera_id, pixel_id]
    
    if depth <= 0.0:
        # No hit - use far end of colormap (index 0 after inversion)
        out_rgb[world_id, camera_id, pixel_id] = unpack_lut_to_rgb(colormap_lut[0])
        return
    
    # Normalize depth to 0-1 (0=near, 1=far)
    denom = wp.max(max_depth - min_depth, 0.001)
    normalized = (depth - min_depth) / denom
    
    # Apply log_250 transform: spreads near values more evenly
    log_base = 250.0
    transformed = wp.log(1.0 + normalized * (log_base - 1.0)) / wp.log(log_base)
    
    # Invert so closer = higher value (brighter/warmer in colormap)
    t = 1.0 - transformed
    
    # Sample with interpolation
    out_rgb[world_id, camera_id, pixel_id] = sample_colormap_interpolated(colormap_lut, t)


# =============================================================================
# Type Aliases
# =============================================================================

RGB = tuple[float, float, float]


# =============================================================================
# Color Extractor Protocol
# =============================================================================

@dataclass(frozen=True)
class ExtractionContext:
    """Immutable context passed to color extractors."""
    usd_file_dir: str
    time_code: Usd.TimeCode
    prim_index: int  # For deterministic random colors


class ColorExtractor(Protocol):
    """Protocol for functions that extract a color from a USD prim."""
    def __call__(self, prim: Usd.Prim, ctx: ExtractionContext) -> RGB: ...


# =============================================================================
# Diegetic: The Core Scene Element
# =============================================================================

@dataclass(frozen=True)
class Diegetic:
    """
    An immutable scene element with geometry and multiple color channels.
    
    "Diegetic" (from film theory): existing within the world of the narrative.
    Each Diegetic represents a visible piece of the scene with pre-computed colors
    for different rendering purposes.

    In film theory, diegesis refers to the world of the story. A "diegetic" object 
    is anything that exists within the characters' reality (a chair, a gun, an actor, a car).

    It specifically excludes "non-diegetic" elements like boom mics, C-stands, film lights, 
    and the camera itself, because the characters do not "see" those things.
    
    Color channels are stored as raw data for use by different render passes:
    - diffuse_albedo: Used by the lit color render pass
    - object_id: Used to create object ID lookup table for post-processing
    - semantic: Used to create semantic lookup table for post-processing
    """
    name: str
    path: str
    vertices: np.ndarray  # World-space positions (N, 3), float32
    faces: np.ndarray     # Triangle indices (M, 3), int32
    diffuse_albedo: RGB   # For photorealistic rendering (used in lit pass)
    object_id: RGB        # For instance segmentation (post-process LUT)
    semantic: RGB         # For semantic segmentation (post-process LUT)


@dataclass(frozen=True)
class ParseOptions:
    """Immutable options for parsing USD stages."""
    time_code: Usd.TimeCode
    path_filter: Callable[[str], bool] | None = None
    skip_invisible: bool = True
    skip_proxy: bool = True
    require_render_purpose: bool = True  # Only include prims with purpose="render"


# =============================================================================
# Color Extractor Factory Functions
# =============================================================================

def constant_color(rgb: RGB) -> ColorExtractor:
    """
    Create an extractor that always returns the same color.
    
    Usage:
        extractor = constant_color((0.5, 0.5, 0.5))
    """
    def extract(prim: Usd.Prim, ctx: ExtractionContext) -> RGB:
        return rgb
    return extract


def random_color(seed: int = 42, brightness_range: tuple[float, float] = (0.3, 0.9)) -> ColorExtractor:
    """
    Create an extractor that returns a deterministic random color per prim.
    
    Uses prim path hash + seed for reproducibility.
    Colors are in HSV with full saturation, converted to RGB.
    
    Usage:
        extractor = random_color(seed=123)
    """
    lo, hi = brightness_range
    
    def extract(prim: Usd.Prim, ctx: ExtractionContext) -> RGB:
        # Hash prim path + seed for deterministic but varied colors
        path_hash = hashlib.md5(f"{prim.GetPath()}{seed}".encode()).digest()
        hue = int.from_bytes(path_hash[:2], 'little') / 65535.0
        
        # HSV to RGB with S=1, V in brightness range
        v = lo + (hi - lo) * (int.from_bytes(path_hash[2:4], 'little') / 65535.0)
        
        # Simple HSV->RGB
        h = hue * 6.0
        c = v  # S=1, so c=v*s=v
        x = c * (1 - abs(h % 2 - 1))
        
        if h < 1: r, g, b = c, x, 0.0
        elif h < 2: r, g, b = x, c, 0.0
        elif h < 3: r, g, b = 0.0, c, x
        elif h < 4: r, g, b = 0.0, x, c
        elif h < 5: r, g, b = x, 0.0, c
        else: r, g, b = c, 0.0, x
        
        return (r, g, b)
    return extract


def primvar_color(primvar_name: str, fallback: RGB) -> ColorExtractor:
    """
    Create an extractor that reads a named primvar (color3f, constant interpolation).
    
    Usage:
        extractor = primvar_color("objectid_color", fallback=(1.0, 0.0, 1.0))
    """
    def extract(prim: Usd.Prim, ctx: ExtractionContext) -> RGB:
        primvars_api = UsdGeom.PrimvarsAPI(prim)
        primvar = primvars_api.GetPrimvar(primvar_name)
        
        if not primvar or not primvar.HasValue():
            return fallback
        
        if primvar.GetInterpolation() != UsdGeom.Tokens.constant:
            return fallback
        
        value = primvar.Get(ctx.time_code)
        if value is None:
            return fallback
        
        try:
            if hasattr(value, '__getitem__') and len(value) >= 3:
                return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, IndexError):
            pass
        
        return fallback
    return extract


def display_color_extractor(fallback: RGB) -> ColorExtractor:
    """
    Create an extractor that reads the displayColor primvar.
    
    Usage:
        extractor = display_color_extractor(fallback=(0.7, 0.7, 0.7))
    """
    def extract(prim: Usd.Prim, ctx: ExtractionContext) -> RGB:
        if not prim.IsA(UsdGeom.Mesh):
            return fallback
        
        mesh = UsdGeom.Mesh(prim)
        display_color_raw = mesh.GetDisplayColorPrimvar().Get(ctx.time_code)
        
        if display_color_raw is None or len(display_color_raw) == 0:
            return fallback
        
        first = display_color_raw[0]
        if hasattr(first, '__getitem__'):
            return (float(first[0]), float(first[1]), float(first[2]))
        
        return fallback
    return extract


def material_diffuse_extractor(fallback: RGB, load_textures: bool = False) -> ColorExtractor:
    """
    Create an extractor that reads UsdPreviewSurface diffuseColor.
    
    If load_textures is True and diffuseColor is connected to a texture,
    attempts to sample the texture's average color.
    
    Usage:
        extractor = material_diffuse_extractor(fallback=(0.5, 0.5, 0.5))
    """
    def extract(prim: Usd.Prim, ctx: ExtractionContext) -> RGB:
        materialBinding = UsdShade.MaterialBindingAPI(prim)
        boundMaterial, _ = materialBinding.ComputeBoundMaterial()
        
        if not boundMaterial:
            return fallback
        
        surfaceOutput = boundMaterial.GetSurfaceOutput()
        if not surfaceOutput:
            return fallback
        
        connectedSource = surfaceOutput.GetConnectedSource()
        if not connectedSource:
            return fallback
        
        shaderPrim = connectedSource[0].GetPrim()
        shader = UsdShade.Shader(shaderPrim)
        
        diffuseInput = shader.GetInput("diffuseColor")
        if not diffuseInput:
            return fallback
        
        connSrc = diffuseInput.GetConnectedSource()
        if connSrc:
            # Connected to texture
            connectedShader = UsdShade.Shader(connSrc[0].GetPrim())
            
            if load_textures and _PIL_AVAILABLE:
                fileInput = connectedShader.GetInput("file")
                if fileInput:
                    fileVal = fileInput.Get(ctx.time_code)
                    if fileVal:
                        avg_color = get_texture_average_color(
                            fileVal.path, ctx.usd_file_dir, verbose=False
                        )
                        if avg_color is not None:
                            return avg_color
            
            # Try fallback from texture shader
            fallbackInput = connectedShader.GetInput("fallback")
            if fallbackInput:
                fallbackVal = fallbackInput.Get(ctx.time_code)
                if fallbackVal is not None:
                    return (float(fallbackVal[0]), float(fallbackVal[1]), float(fallbackVal[2]))
        else:
            # Direct value
            val = diffuseInput.Get(ctx.time_code)
            if val is not None and hasattr(val, '__iter__'):
                return (float(val[0]), float(val[1]), float(val[2]))
        
        return fallback
    return extract


def first_of(*extractors: ColorExtractor, fallback: RGB = (0.5, 0.5, 0.5)) -> ColorExtractor:
    """
    Create an extractor that tries each extractor in order, returning the first non-fallback result.
    
    This is a combinator for building fallback chains.
    
    Usage:
        extractor = first_of(
            material_diffuse_extractor((0.5, 0.5, 0.5)),
            display_color_extractor((0.5, 0.5, 0.5)),
            fallback=(0.7, 0.7, 0.7)
        )
    """
    def extract(prim: Usd.Prim, ctx: ExtractionContext) -> RGB:
        for ext in extractors:
            result = ext(prim, ctx)
            if result != fallback:
                return result
        return fallback
    return extract


@dataclass
class CameraData:
    """Container for extracted USD camera data."""
    position: np.ndarray      # World-space position (3,)
    right: np.ndarray         # Camera +X axis in world space (normalized)
    up: np.ndarray            # Camera +Y axis in world space (normalized)
    forward: np.ndarray       # Camera view direction (-Z in local) in world space (normalized)
    fov_vertical: float       # Vertical field of view in radians
    fov_horizontal: float     # Horizontal field of view in radians


@dataclass
class LightData:
    """
    Container for extracted USD light data.
    
    Supports both directional (distant) lights and positional lights.
    The renderer uses:
    - light_type: 0 = spotlight, 1 = directional
    - position: World-space position (used for spotlights)
    - direction: Light direction in world space (used for both types)
    """
    path: str
    light_type: int           # 0 = spotlight, 1 = directional
    position: np.ndarray      # World-space position (3,)
    direction: np.ndarray     # Light direction in world space (3,), normalized
    color: tuple[float, float, float]  # RGB color
    intensity: float
    cast_shadows: bool


@dataclass
class AmbientLightData:
    """
    Container for ambient/dome light data extracted from USD.
    
    Used to configure the renderer's ambient lighting based on DomeLights in the scene.
    Since we don't support HDR environment maps yet, we use the dome light color
    as a constant ambient color.
    """
    path: str
    color: tuple[float, float, float]  # RGB color (sky/dome tint)
    intensity: float                    # USD intensity value
    
    def get_normalized_color(self, target_intensity: float = 0.5) -> tuple[float, float, float]:
        """
        Get color normalized to a target intensity for ambient use.
        
        USD dome lights can have very high intensity values (100-1000+).
        This normalizes to a reasonable ambient contribution level.
        """
        # Normalize the intensity to our target range
        # Typical USD dome intensities are 100-500, we want 0.3-0.8 ambient
        scale = min(1.0, self.intensity / 500.0) * target_intensity
        return (
            self.color[0] * scale,
            self.color[1] * scale,
            self.color[2] * scale,
        )


# =============================================================================
# USD Light Extraction
# =============================================================================

def _get_light_direction_from_xform(
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
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    
    return direction.astype(np.float32)


def _get_light_position_from_xform(
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


def _is_light_visible(prim: Usd.Prim, time_code: Usd.TimeCode) -> bool:
    """
    Check if a light prim is visible for rendering.
    
    Mirrors the visibility check we do for geometry.
    """
    # Lights can be UsdGeom.Imageable, check visibility
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        visibility = imageable.ComputeVisibility(time_code)
        if visibility == UsdGeom.Tokens.invisible:
            return False
    return True


def _extract_light_data(
    prim: Usd.Prim,
    xcache: UsdGeom.XformCache,
    time_code: Usd.TimeCode,
    check_visibility: bool = True,
) -> LightData | None:
    """
    Extract light data from a USD light prim.
    
    Converts USD light types to our simplified light model:
    - DistantLight -> directional (type 1)
    - SphereLight, RectLight, etc. -> spotlight (type 0)
    - DomeLight -> skipped (ambient, handled separately)
    
    Returns None if the prim is not a supported light type or is invisible.
    """
    # Skip dome lights (ambient) - we handle those separately
    if prim.IsA(UsdLux.DomeLight):
        return None
    
    # Check visibility
    if check_visibility and not _is_light_visible(prim, time_code):
        return None
    
    # Determine renderer light type
    if prim.IsA(UsdLux.DistantLight):
        light_type = 1  # directional
    elif prim.IsA(UsdLux.SphereLight) or prim.IsA(UsdLux.RectLight) or prim.IsA(UsdLux.DiskLight):
        light_type = 0  # spotlight (positional)
    else:
        return None
    
    # Get the light API for common properties
    light = UsdLux.LightAPI(prim)
    
    # Extract color
    color = (1.0, 1.0, 1.0)
    color_attr = light.GetColorAttr()
    if color_attr and color_attr.HasValue():
        c = color_attr.Get(time_code)
        if c:
            color = (float(c[0]), float(c[1]), float(c[2]))
    
    # Extract intensity
    intensity = 1.0
    intensity_attr = light.GetIntensityAttr()
    if intensity_attr and intensity_attr.HasValue():
        intensity = float(intensity_attr.Get(time_code))
    
    # Extract shadow settings
    cast_shadows = True
    shadow_api = UsdLux.ShadowAPI(prim)
    if shadow_api:
        enable_attr = shadow_api.GetShadowEnableAttr()
        if enable_attr and enable_attr.HasValue():
            cast_shadows = bool(enable_attr.Get(time_code))
    
    # Get position and direction
    position = _get_light_position_from_xform(prim, xcache)
    direction = _get_light_direction_from_xform(prim, xcache)
    
    return LightData(
        path=str(prim.GetPath()),
        light_type=light_type,
        position=position,
        direction=direction,
        color=color,
        intensity=intensity,
        cast_shadows=cast_shadows,
    )


def find_lights(
    stage: Usd.Stage,
    time_code: Usd.TimeCode,
    path_filter: Callable[[str], bool] | None = None,
    verbose: bool = False,
) -> list[LightData]:
    """
    Find and extract all lights from a USD stage.
    
    Uses UsdLux.ListAPI for comprehensive light discovery.
    Filters out DomeLights (ambient) as those are handled separately.
    
    Args:
        stage: Opened USD stage
        time_code: Time code for sampling animated lights
        path_filter: Optional filter function (path -> bool) to exclude lights
        verbose: If True, print discovered lights
        
    Returns:
        List of LightData for directional and positional lights
    """
    xcache = UsdGeom.XformCache(time_code)
    lights: list[LightData] = []
    
    # Try LightListAPI for comprehensive discovery
    try:
        light_list_api = UsdLux.ListAPI(stage.GetPseudoRoot())
        light_paths = light_list_api.ComputeLightList(UsdLux.ListAPI.ComputeModeIgnoreCache)
        
        for path in light_paths:
            # Apply path filter
            path_str = str(path)
            if path_filter and not path_filter(path_str):
                continue
            
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                light_data = _extract_light_data(prim, xcache, time_code, check_visibility=True)
                if light_data:
                    lights.append(light_data)
                    if verbose:
                        print(f"  Found light: {light_data.path} "
                              f"(type={'directional' if light_data.light_type == 1 else 'positional'}, "
                              f"dir=[{light_data.direction[0]:.3f}, {light_data.direction[1]:.3f}, {light_data.direction[2]:.3f}])")
                elif verbose and not prim.IsA(UsdLux.DomeLight):
                    # Light was skipped (invisible or unsupported type)
                    if not _is_light_visible(prim, time_code):
                        print(f"  Skipping invisible light: {path_str}")
        
    except AttributeError:
        # Fallback: Traverse stage looking for light prims
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.BoundableLightBase) or prim.IsA(UsdLux.NonboundableLightBase):
                path_str = str(prim.GetPath())
                if path_filter and not path_filter(path_str):
                    continue
                
                light_data = _extract_light_data(prim, xcache, time_code)
                if light_data:
                    lights.append(light_data)
    
    return lights


def get_default_light() -> LightData:
    """
    Get default directional light for scenes without USD lights.
    
    Returns a sun-like directional light pointing roughly downward.
    """
    return LightData(
        path="/DefaultDirectionalLight",
        light_type=1,  # directional
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        direction=np.array([-0.577, 0.577, -0.577], dtype=np.float32),  # 45° from above
        color=(1.0, 1.0, 1.0),
        intensity=1.0,
        cast_shadows=True,
    )


def create_fill_light_from_ambient(
    ambient: AmbientLightData | None,
    key_light_direction: np.ndarray | None = None,
) -> LightData:
    """
    Create a fill light based on the ambient/dome light from the USD scene.
    
    Since the renderer's ambient lighting is hardcoded, we simulate softer
    ambient illumination by adding a fill light - a directional light from
    roughly opposite the key light, with the dome light's color tint.
    
    This is similar to the classic key+fill lighting setup in cinematography.
    
    Args:
        ambient: AmbientLightData from find_ambient_light(), or None for defaults
        key_light_direction: Optional direction of the key light to compute fill direction
        
    Returns:
        LightData for a fill light (no shadows, tinted by dome color)
    """
    # Default fill light direction: from below/front (opposite of typical key)
    # If we have a key light, make fill come from roughly opposite direction
    if key_light_direction is not None:
        # Flip the key direction and shift toward camera-front
        fill_dir = -key_light_direction
        # Lift it up slightly so it's not purely from behind
        fill_dir[2] = abs(fill_dir[2]) * 0.3  # Reduce vertical component
        norm = np.linalg.norm(fill_dir)
        if norm > 0:
            fill_dir = fill_dir / norm
    else:
        # Default: from front-below at 45°
        fill_dir = np.array([0.0, -0.707, 0.707], dtype=np.float32)
    
    # Get color and intensity from ambient light, or use warm fill defaults
    if ambient:
        # Use dome color, but reduce intensity significantly (fill is much dimmer than key)
        # USD intensities are often 100-500, we want a subtle fill contribution
        fill_intensity = min(1.0, ambient.intensity / 1000.0)  # ~0.2-0.5 for typical USD values
        color = ambient.color
    else:
        # Warm fill light default (slight warmth to counteract cool ambient)
        color = (1.0, 0.95, 0.9)
        fill_intensity = 0.3
    
    return LightData(
        path="/FillLight",
        light_type=1,  # directional
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        direction=fill_dir.astype(np.float32),
        color=color,
        intensity=fill_intensity,
        cast_shadows=False,  # Fill lights don't cast shadows
    )


def find_ambient_light(
    stage: Usd.Stage,
    time_code: Usd.TimeCode,
    path_filter: Callable[[str], bool] | None = None,
    verbose: bool = False,
) -> AmbientLightData | None:
    """
    Find the primary DomeLight in the scene to use for ambient lighting.
    
    DomeLights in USD represent environment/sky lighting. Since we don't
    support HDR environment maps, we extract the dome color and intensity
    to use as a constant ambient color.
    
    Args:
        stage: Opened USD stage
        time_code: Time code for sampling
        path_filter: Optional filter function to select specific dome light
        verbose: If True, print found dome lights
        
    Returns:
        AmbientLightData for the first matching DomeLight, or None if not found
    """
    # Try LightListAPI first
    try:
        light_list_api = UsdLux.ListAPI(stage.GetPseudoRoot())
        light_paths = light_list_api.ComputeLightList(UsdLux.ListAPI.ComputeModeIgnoreCache)
        
        for path in light_paths:
            path_str = str(path)
            if path_filter and not path_filter(path_str):
                continue
            
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid() and prim.IsA(UsdLux.DomeLight):
                # Check visibility
                if not _is_light_visible(prim, time_code):
                    if verbose:
                        print(f"  Skipping invisible DomeLight: {path_str}")
                    continue
                
                dome = UsdLux.DomeLight(prim)
                light_api = UsdLux.LightAPI(prim)
                
                # Extract color
                color = (1.0, 1.0, 1.0)
                color_attr = light_api.GetColorAttr()
                if color_attr and color_attr.HasValue():
                    c = color_attr.Get(time_code)
                    if c:
                        color = (float(c[0]), float(c[1]), float(c[2]))
                
                # Extract intensity
                intensity = 1.0
                intensity_attr = light_api.GetIntensityAttr()
                if intensity_attr and intensity_attr.HasValue():
                    intensity = float(intensity_attr.Get(time_code))
                
                if verbose:
                    print(f"  Found DomeLight: {path_str}")
                    print(f"    Color: RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
                    print(f"    Intensity: {intensity}")
                
                return AmbientLightData(
                    path=path_str,
                    color=color,
                    intensity=intensity,
                )
    
    except AttributeError:
        # Fallback traversal
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.DomeLight):
                path_str = str(prim.GetPath())
                if path_filter and not path_filter(path_str):
                    continue
                
                light_api = UsdLux.LightAPI(prim)
                
                color = (1.0, 1.0, 1.0)
                color_attr = light_api.GetColorAttr()
                if color_attr and color_attr.HasValue():
                    c = color_attr.Get(time_code)
                    if c:
                        color = (float(c[0]), float(c[1]), float(c[2]))
                
                intensity = 1.0
                intensity_attr = light_api.GetIntensityAttr()
                if intensity_attr and intensity_attr.HasValue():
                    intensity = float(intensity_attr.Get(time_code))
                
                if verbose:
                    print(f"  Found DomeLight: {path_str}")
                    print(f"    Color: RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
                    print(f"    Intensity: {intensity}")
                
                return AmbientLightData(
                    path=path_str,
                    color=color,
                    intensity=intensity,
                )
    
    return None


# =============================================================================
# Render Configuration
# =============================================================================

@dataclass(frozen=True)
class RenderConfig:
    """Immutable configuration for rendering output."""
    width: int
    height: int
    output_dir: Path
    filename_pattern: str = "render.{frame}.png"
    
    def get_output_path(self, frame: int) -> Path:
        """Get the full output path for a given frame number."""
        return self.output_dir / self.filename_pattern.format(frame=frame)


# =============================================================================
# Geometry Utilities
# =============================================================================

def triangulate(indices: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Convert polygon mesh to triangles via fan triangulation.
    
    Args:
        indices: Face vertex indices array
        counts: Face vertex counts array (e.g., [4, 4, 3] for 2 quads and 1 triangle)
        
    Returns:
        Triangle indices as (N, 3) numpy array
    """
    tris = []
    idx = 0
    for count in counts:
        if count >= 3:
            for i in range(1, count - 1):
                tris.append([indices[idx], indices[idx + i], indices[idx + i + 1]])
        idx += count
    return np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32)


def transform_points_to_world(
    points: np.ndarray,
    world_matrix: Gf.Matrix4d
) -> np.ndarray:
    """
    Transform points from local to world space using a 4x4 matrix.
    
    Args:
        points: Local-space points (N, 3)
        world_matrix: USD world transform matrix
        
    Returns:
        World-space points (N, 3)
    """
    M = np.array(world_matrix, dtype=np.float64)
    pts = np.array(points, dtype=np.float64)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    pts_world_h = pts_h @ M  # Row-vectors * matrix
    return pts_world_h[:, :3]


# =============================================================================
# Geometry Extraction
# =============================================================================

@dataclass(frozen=True)
class ExtractedGeometry:
    """Intermediate result from geometry extraction."""
    name: str
    path: str
    vertices: np.ndarray  # World-space (N, 3) float32
    faces: np.ndarray     # Triangle indices (M, 3) int32


def extract_geometry(
    prim: Usd.Prim,
    xcache: UsdGeom.XformCache,
    time_code: Usd.TimeCode
) -> ExtractedGeometry | None:
    """
    Extract world-space triangulated geometry from a mesh prim.
    
    Handles instanced geometry by getting geometry from prototype
    and transform from the proxy prim.
    
    Returns None if geometry cannot be extracted.
    """
    # Get the actual geometry prim (prototype for instances)
    if prim.IsInstanceProxy():
        geom_prim = prim.GetPrimInPrototype()
    else:
        geom_prim = prim
    
    mesh = UsdGeom.Mesh(geom_prim)
    
    points = mesh.GetPointsAttr().Get(time_code)
    indices = mesh.GetFaceVertexIndicesAttr().Get(time_code)
    counts = mesh.GetFaceVertexCountsAttr().Get(time_code)
    
    if not points or not indices or not counts:
        return None
    
    # Transform using the proxy prim's world transform
    world_mat = xcache.GetLocalToWorldTransform(prim)
    vertices = transform_points_to_world(points, world_mat).astype(np.float32)
    
    # Triangulate
    faces = triangulate(np.array(indices), np.array(counts))
    
    path_str = str(prim.GetPath())
    name = path_str.split("/")[-1]
    
    return ExtractedGeometry(
        name=name,
        path=path_str,
        vertices=vertices,
        faces=faces,
    )


def is_visible_mesh(prim: Usd.Prim, time_code: Usd.TimeCode) -> bool:
    """
    Check if a prim is a visible mesh (not invisible for rendering).
    
    For instance proxies, we check visibility on the proxy itself (which is
    in the scene hierarchy and inherits visibility from ancestors), NOT on
    the prototype prim. The prototype doesn't have scene hierarchy visibility.
    """
    if not prim.IsA(UsdGeom.Mesh):
        return False
    
    # Check visibility on the prim in the scene hierarchy (proxy, not prototype)
    # Instance proxies inherit visibility from their scene-graph ancestors
    mesh = UsdGeom.Mesh(prim)
    visibility = mesh.ComputeEffectiveVisibility(UsdGeom.Tokens.render, time_code)
    return visibility != UsdGeom.Tokens.invisible


def has_render_purpose(prim: Usd.Prim) -> bool:
    """
    Check if a prim has 'render' purpose.
    
    USD purpose values:
    - 'default': No special purpose, included in all rendering paths
    - 'render': Final quality data for high-quality/offline rendering
    - 'proxy': Lightweight representation for interactive viewports
    - 'guide': Visual aids like rig controllers, not for rendering
    
    This function returns True only for 'render' purpose, which represents
    the final quality geometry intended for offline rendering.
    
    For instance proxies, we check purpose on the proxy itself (which is
    in the scene hierarchy and inherits purpose from ancestors), NOT on
    the prototype prim. Purpose is inherited from scene-graph ancestors.
    
    Args:
        prim: The USD prim to check
        
    Returns:
        True if the prim's computed purpose is 'render'
    """
    # Check purpose on the prim in the scene hierarchy (proxy, not prototype)
    # Instance proxies inherit purpose from their scene-graph ancestors
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return False
    
    # ComputePurpose() returns a TfToken with the computed purpose,
    # considering inheritance from ancestors
    purpose = imageable.ComputePurpose()
    
    # We want only 'render' purpose
    return purpose == UsdGeom.Tokens.render


# =============================================================================
# Parse Diegetics: The Fold Function
# =============================================================================

def parse_diegetics(
    stage: Usd.Stage,
    usd_file_path: str,
    options: ParseOptions,
    diffuse_extractor: ColorExtractor,
    object_id_extractor: ColorExtractor,
    semantic_extractor: ColorExtractor,
    verbose: bool = False,
    show_progress: bool = True,
    progress_interval: int = 1000,
) -> list[Diegetic]:
    """
    Parse a USD stage into a list of Diegetic scene elements.
    
    This is the core "fold" function that maps:
    - A geometry extractor (hardcoded for triangle meshes)
    - Three color extractors (one per color channel)
    
    Over the USD stage to produce immutable Diegetics.
    
    Filtering (controlled by ParseOptions):
    - skip_invisible: Exclude prims with visibility="invisible"
    - require_render_purpose: Only include prims with purpose="render"
      (excludes "default", "proxy", and "guide" purpose prims)
    - skip_proxy: Exclude prims with "/proxy/" in their path
    - path_filter: Custom filter function for prim paths
    
    Args:
        stage: Opened USD stage
        usd_file_path: Path to USD file (for texture resolution)
        options: ParseOptions controlling traversal behavior
        diffuse_extractor: Extracts diffuse_albedo color
        object_id_extractor: Extracts object_id color
        semantic_extractor: Extracts semantic color
        verbose: Print detailed progress
        show_progress: Print periodic progress updates
        progress_interval: Update interval for progress messages
        
    Returns:
        List of Diegetic objects with geometry and all color channels populated
    """
    usd_file_dir = os.path.dirname(usd_file_path)
    time_code = options.time_code
    xcache = UsdGeom.XformCache(time_code)
    
    diegetics: list[Diegetic] = []
    prims_processed = 0
    prims_skipped = 0
    
    if show_progress:
        print("Parsing USD stage into Diegetics...", flush=True)
    
    for prim_index, prim in enumerate(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())):
        prims_processed += 1
        
        if show_progress and prims_processed % progress_interval == 0:
            print(f"  Processed {prims_processed} prims, found {len(diegetics)} diegetics...", flush=True)
        
        # Filter: must be a mesh
        if not prim.IsA(UsdGeom.Mesh):
            continue
        
        path_str = str(prim.GetPath())
        
        # Filter: user-provided path filter
        if options.path_filter is not None and not options.path_filter(path_str):
            continue
        
        # Filter: visibility
        if options.skip_invisible and not is_visible_mesh(prim, time_code):
            if verbose:
                print(f"  SKIP (invisible): {path_str}")
            prims_skipped += 1
            continue
        
        # Filter: purpose (only include 'render' purpose geometry)
        if options.require_render_purpose and not has_render_purpose(prim):
            if verbose:
                print(f"  SKIP (not render purpose): {path_str}")
            prims_skipped += 1
            continue
        
        # Filter: proxy geometry (path-based heuristic, separate from purpose)
        if options.skip_proxy and is_proxy_path(path_str):
            if verbose:
                print(f"  SKIP (proxy path): {path_str}")
            prims_skipped += 1
            continue
        
        # Extract geometry
        geom = extract_geometry(prim, xcache, time_code)
        if geom is None:
            if verbose:
                print(f"  SKIP (no geometry): {path_str}")
            prims_skipped += 1
            continue
        
        # Create extraction context
        ctx = ExtractionContext(
            usd_file_dir=usd_file_dir,
            time_code=time_code,
            prim_index=prim_index,
        )
        
        # Extract all three colors
        diffuse = diffuse_extractor(prim, ctx)
        object_id = object_id_extractor(prim, ctx)
        semantic = semantic_extractor(prim, ctx)
        
        # Create immutable Diegetic
        diegetic = Diegetic(
            name=geom.name,
            path=geom.path,
            vertices=geom.vertices,
            faces=geom.faces,
            diffuse_albedo=diffuse,
            object_id=object_id,
            semantic=semantic,
        )
        
        diegetics.append(diegetic)
        
        if verbose:
            print(f"  PARSED: {path_str} ({len(geom.vertices)} verts, {len(geom.faces)} tris)")
    
    if show_progress:
        print(f"  Done: {prims_processed} prims -> {len(diegetics)} diegetics, {prims_skipped} skipped", flush=True)
    
    return diegetics


def make_diegetic_names_unique(diegetics: list[Diegetic]) -> list[Diegetic]:
    """
    Return new list of Diegetics with unique names.
    
    Unlike make_mesh_names_unique, this is a pure function that returns
    new Diegetic objects (since Diegetic is frozen).
    """
    name_counts: dict[str, int] = {}
    result: list[Diegetic] = []
    
    for d in diegetics:
        name = d.name
        if name in name_counts:
            name_counts[name] += 1
            new_name = f"{name}_{name_counts[name]}"
        else:
            name_counts[name] = 1
            new_name = name
        
        # Create new Diegetic with updated name (if changed)
        if new_name != name:
            result.append(Diegetic(
                name=new_name,
                path=d.path,
                vertices=d.vertices,
                faces=d.faces,
                diffuse_albedo=d.diffuse_albedo,
                object_id=d.object_id,
                semantic=d.semantic,
            ))
        else:
            result.append(d)
    
    return result


# =============================================================================
# Camera Utilities
# =============================================================================

def get_camera_from_stage(
    stage: Usd.Stage,
    camera_path: str,
    time_code: Usd.TimeCode,
    verbose: bool = False
) -> CameraData:
    """
    Extract camera world-space basis and FOV from a USD camera prim.
    
    USD/RenderMan camera coordinate system:
    - X: right
    - Y: up
    - Z: toward the sensor ("into the lens")
    
    The view direction is -Z (same as OpenGL convention).
    
    Args:
        stage: USD stage
        camera_path: Path to camera prim (e.g., "/World/Camera")
        time_code: Time code for sampling animated cameras
        verbose: If True, print camera info
        
    Returns:
        CameraData with position and orientation in world space
        
    Raises:
        RuntimeError: If camera prim not found or invalid
    """
    import math
    
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Camera not found at {camera_path}")

    usd_camera = UsdGeom.Camera(camera_prim)
    
    # Get intrinsics from Gf.Camera
    gf_camera = usd_camera.GetCamera(time_code)
    fov_v = gf_camera.GetFieldOfView(Gf.Camera.FOVVertical)
    fov_h = gf_camera.GetFieldOfView(Gf.Camera.FOVHorizontal)

    # Get extrinsics using XformCache (same approach as mesh loading)
    xcache = UsdGeom.XformCache(time_code)
    world_mat = xcache.GetLocalToWorldTransform(camera_prim)

    # Transform basis points to world space
    pts_camera = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    pts_world = transform_points_to_world(pts_camera, world_mat)

    cam_origin, x_pt, y_pt, z_pt = pts_world

    # Extract and normalize basis vectors
    right = x_pt - cam_origin
    up = y_pt - cam_origin
    z_axis = z_pt - cam_origin  # +Z points toward sensor
    
    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)
    forward = -z_axis / np.linalg.norm(z_axis)  # View direction is -Z

    if verbose:
        print(f"Camera at frame {time_code}: FOV={fov_v:.2f}°")
        print(f"  Position: [{cam_origin[0]:.3f}, {cam_origin[1]:.3f}, {cam_origin[2]:.3f}]")
        print(f"  Forward:  [{forward[0]:.3f}, {forward[1]:.3f}, {forward[2]:.3f}]")

    return CameraData(
        position=cam_origin.astype(np.float32),
        right=right.astype(np.float32),
        up=up.astype(np.float32),
        forward=forward.astype(np.float32),
        fov_vertical=math.radians(fov_v),
        fov_horizontal=math.radians(fov_h),
    )


def camera_rotation_matrix(camera: CameraData) -> np.ndarray:
    """
    Build a 3x3 rotation matrix from camera basis vectors.
    
    Columns are [right, up, forward].
    
    Args:
        camera: CameraData with basis vectors
        
    Returns:
        3x3 orthonormal rotation matrix
    """
    return np.column_stack([camera.right, camera.up, camera.forward]).astype(np.float32)


# =============================================================================
# Warp Kernel
# =============================================================================

@wp.kernel
def generate_world_rays_kernel(
    width: int,
    height: int,
    fov_y: float,
    cam_pos: wp.vec3f,
    cam_right: wp.vec3f,
    cam_up: wp.vec3f,
    cam_forward: wp.vec3f,
    out_rays: wp.array(dtype=wp.vec3, ndim=4),
):
    """Generate rays directly in world space using camera basis vectors."""
    x, y = wp.tid()
    
    if x >= width or y >= height:
        return

    aspect_ratio = float(width) / float(height)
    u = (float(x) + 0.5) / float(width) - 0.5
    v = (float(y) + 0.5) / float(height) - 0.5
    h = wp.tan(fov_y / 2.0)
    
    # Local ray direction (camera looks down -Z)
    x_local = u * 2.0 * h * aspect_ratio
    y_local = -v * 2.0 * h
    
    # Transform to world space
    ray_dir_world = x_local * cam_right + y_local * cam_up + (-1.0) * (-cam_forward)
    
    out_rays[0, y, x, 0] = cam_pos
    out_rays[0, y, x, 1] = wp.normalize(ray_dir_world)


def transforms_and_rays_from_camera_data(camera: CameraData, width: int, height: int) -> tuple[wp.array(dtype=wp.transformf), wp.array(dtype=wp.vec3f)]:
    cam_pos = wp.vec3f(*camera.position)
    cam_right = wp.vec3f(*camera.right)
    cam_up = wp.vec3f(*camera.up)
    cam_forward = wp.vec3f(*camera.forward)
    
    camera_rays = wp.zeros((1, height, width, 2), dtype=wp.vec3f)
    
    wp.launch(
        kernel=generate_world_rays_kernel,
        dim=(width, height),
        inputs=[width, height, camera.fov_vertical, cam_pos, cam_right, cam_up, cam_forward, camera_rays]
    )

    identity_transform = wp.transformf(wp.vec3f(0, 0, 0), wp.quatf(0, 0, 0, 1))
    camera_transforms = wp.array([[identity_transform]], dtype=wp.transformf)

    return camera_transforms, camera_rays


# =============================================================================
# Rendering (Diegetic -> Pixels)
# =============================================================================

def rgb_to_packed_uint32(r: float, g: float, b: float, a: float = 1.0) -> int:
    """Convert RGB floats (0-1) to packed RGBA uint32."""
    return (
        (int(a * 255) << 24) |
        (int(b * 255) << 16) |
        (int(g * 255) << 8) |
        int(r * 255)
    )


def setup_render_context(
    diegetics: list[Diegetic],
    config: RenderConfig,
    lights: list[LightData] | None = None,
) -> tuple[RenderContext, ColorLUTs, list]:
    """
    Create RenderContext and ColorLUTs from Diegetics for multi-AOV rendering.
    
    This sets up the scene for a single render pass that outputs multiple AOVs:
    - Lit color render (uses diffuse_albedo)
    - Depth
    - Surface normals
    - Shape indices (for post-processing to object_id and semantic passes)
    
    The ColorLUTs can be used to convert shape_index output to object_id or
    semantic color passes in post-processing.
    
    Args:
        diegetics: List of Diegetic scene elements
        config: Render configuration (resolution, output path)
        lights: Optional list of LightData from USD scene. If None, uses default lighting.
        
    Returns:
        Tuple of (RenderContext, ColorLUTs, list of warp.Mesh objects)
    """
    num_shapes = len(diegetics)
    print(f"Setting up render context with {num_shapes} diegetics for multi-AOV rendering...")

    warp_meshes = []
    mesh_bounds = np.zeros((num_shapes, 2, 3), dtype=np.float32)
    scene_min = np.full(3, np.inf)
    scene_max = np.full(3, -np.inf)

    for i, d in enumerate(diegetics):
        mesh = wp.Mesh(
            points=wp.array(d.vertices, dtype=wp.vec3f),
            indices=wp.array(d.faces.flatten(), dtype=wp.int32),
        )
        warp_meshes.append(mesh)
        mesh_bounds[i, 0] = d.vertices.min(axis=0)
        mesh_bounds[i, 1] = d.vertices.max(axis=0)
        scene_min = np.minimum(scene_min, mesh_bounds[i, 0])
        scene_max = np.maximum(scene_max, mesh_bounds[i, 1])
    
    print(f"Scene bounds: [{scene_min[0]:.1f}, {scene_min[1]:.1f}, {scene_min[2]:.1f}] to "
          f"[{scene_max[0]:.1f}, {scene_max[1]:.1f}, {scene_max[2]:.1f}]")

    ctx = RenderContext(
        width=config.width, height=config.height,
        enable_textures=False, enable_shadows=True,
        enable_ambient_lighting=True, enable_particles=False,
        num_worlds=1, num_cameras=1,
    )

    ctx.num_shapes = num_shapes
    ctx.mesh_ids = wp.array([m.id for m in warp_meshes], dtype=wp.uint64)
    ctx.mesh_bounds = wp.array(mesh_bounds, dtype=wp.vec3f)
    ctx.shape_types = wp.array([RenderShapeType.MESH] * num_shapes, dtype=wp.int32)
    ctx.shape_enabled = wp.array(list(range(num_shapes)), dtype=wp.uint32)
    ctx.shape_world_index = wp.array([0] * num_shapes, dtype=wp.int32)
    ctx.shape_mesh_indices = wp.array(list(range(num_shapes)), dtype=wp.int32)
    ctx.shape_materials = wp.array([-1] * num_shapes, dtype=wp.int32)

    transforms = [wp.transformf(wp.vec3f(0, 0, 0), wp.quatf(0, 0, 0, 1))] * num_shapes
    ctx.shape_transforms = wp.array(transforms, dtype=wp.transformf)
    ctx.shape_sizes = wp.array([[1, 1, 1]] * num_shapes, dtype=wp.vec3f)

    # Use diffuse_albedo for the lit color render pass
    diffuse_colors = np.array([
        (*d.diffuse_albedo, 1.0) for d in diegetics
    ], dtype=np.float32)
    ctx.shape_colors = wp.array(diffuse_colors, dtype=wp.vec4f)

    # Set up lights - use provided lights or fall back to default
    if lights and len(lights) > 0:
        num_lights = len(lights)
        print(f"Using {num_lights} lights from USD scene")
        for l in lights:
            light_type_str = "directional" if l.light_type == 1 else "positional"
            print(f"  - {l.path}: {light_type_str}, dir=[{l.direction[0]:.3f}, {l.direction[1]:.3f}, {l.direction[2]:.3f}]")
        
        ctx.lights_active = wp.array([True] * num_lights, dtype=wp.bool)
        ctx.lights_type = wp.array([l.light_type for l in lights], dtype=wp.int32)
        ctx.lights_cast_shadow = wp.array([l.cast_shadows for l in lights], dtype=wp.bool)
        ctx.lights_position = wp.array([l.position.tolist() for l in lights], dtype=wp.vec3f)
        ctx.lights_orientation = wp.array([l.direction.tolist() for l in lights], dtype=wp.vec3f)
    else:
        # Default lighting: single directional light from above
        print("Using default lighting (no USD lights provided)")
        ctx.lights_active = wp.array([True], dtype=wp.bool)
        ctx.lights_type = wp.array([1], dtype=wp.int32)  # directional
        ctx.lights_cast_shadow = wp.array([True], dtype=wp.bool)
        ctx.lights_position = wp.array([[0, 0, 0]], dtype=wp.vec3f)
        ctx.lights_orientation = wp.array([[-0.577, 0.577, -0.577]], dtype=wp.vec3f)

    # Build ColorLUTs for post-processing shape_index to object_id and semantic
    object_id_lut = np.array([
        rgb_to_packed_uint32(*d.object_id) for d in diegetics
    ], dtype=np.uint32)
    semantic_lut = np.array([
        rgb_to_packed_uint32(*d.semantic) for d in diegetics
    ], dtype=np.uint32)
    
    color_luts = ColorLUTs(
        object_id=wp.array(object_id_lut, dtype=wp.uint32),
        semantic=wp.array(semantic_lut, dtype=wp.uint32),
    )

    return ctx, color_luts, warp_meshes


def update_lights_for_headlight(ctx: RenderContext, camera_forward: np.ndarray) -> None:
    """
    Update render context lights to use headlight pointing along camera forward.
    
    A "headlight" is a light attached to the camera that always points in the
    direction the camera is looking. This provides uniform illumination of
    visible surfaces without the complexity of scene lighting.
    
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


def render_all_aovs(
    ctx: RenderContext,
    camera: CameraData,
    config: RenderConfig,
) -> RenderOutputs:
    """
    Render the scene from the given camera viewpoint, outputting all AOVs.
    
    This performs a single render pass that outputs:
    - color_image: Lit diffuse render (packed RGBA uint32)
    - depth_image: Ray hit distance (float32)
    - shape_index_image: Shape index at each pixel (uint32)
    - normal_image: Surface normal at hit point (vec3f)
    
    The shape_index_image can be post-processed with ColorLUTs to create
    additional passes like object_id and semantic.
    
    Args:
        ctx: RenderContext configured with scene geometry
        camera: Camera position and orientation
        config: Render configuration (resolution)
        
    Returns:
        RenderOutputs containing all raw GPU output arrays
    """
    camera_transforms, camera_rays = transforms_and_rays_from_camera_data(
        camera, config.width, config.height
    )

    # Create all output arrays
    color_image = ctx.create_color_image_output()
    depth_image = ctx.create_depth_image_output()
    shape_index_image = ctx.create_shape_index_image_output()
    normal_image = ctx.create_normal_image_output()

    # Single render pass outputs all AOVs
    ctx.render(
        camera_transforms=camera_transforms,
        camera_rays=camera_rays,
        color_image=color_image,
        depth_image=depth_image,
        shape_index_image=shape_index_image,
        normal_image=normal_image,
        refit_bvh=True,
        clear_data=ClearData(clear_color=0xFF404040),
    )

    return RenderOutputs(
        color_image=color_image,
        depth_image=depth_image,
        shape_index_image=shape_index_image,
        normal_image=normal_image,
    )


def packed_uint32_to_rgb(packed: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert packed RGBA uint32 array to RGB uint8 array."""
    pixels = packed.reshape(height, width)
    r = (pixels >> 0) & 0xFF
    g = (pixels >> 8) & 0xFF
    b = (pixels >> 16) & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def convert_aovs_to_pixels(
    outputs: RenderOutputs,
    color_luts: ColorLUTs,
    config: RenderConfig,
    depth_colormap: str = DepthColormap.MAGMA,
) -> PixelOutputs:
    """
    Convert raw GPU render outputs to numpy RGB arrays for saving.
    
    Applies ColorLUTs to shape_index_image to create object_id and semantic passes.
    
    Args:
        outputs: Raw RenderOutputs from render_all_aovs
        color_luts: ColorLUTs for object_id and semantic mapping
        config: Render configuration (for dimensions)
        depth_colormap: Colormap for depth_heat pass (DepthColormap.VIRIDIS or .MAGMA)
        
    Returns:
        PixelOutputs with all passes as (height, width, 3) uint8 arrays
    """
    height, width = config.height, config.width
    
    # Color pass - direct conversion
    color_rgb = packed_uint32_to_rgb(
        outputs.color_image.numpy()[0, 0], height, width
    )
    
    # Compute depth range once (used by both depth passes)
    depth_range = wp.array([1e10, 0.0], dtype=wp.float32)
    wp.launch(
        find_depth_range,
        outputs.depth_image.shape,
        [outputs.depth_image, depth_range],
    )
    depth_range_np = depth_range.numpy()
    
    # Depth pass - grayscale
    depth_rgba = wp.zeros_like(outputs.color_image)
    wp.launch(
        depth_to_grayscale,
        outputs.depth_image.shape,
        [outputs.depth_image, depth_range_np[0], depth_range_np[1], depth_rgba],
    )
    depth_rgb = packed_uint32_to_rgb(depth_rgba.numpy()[0, 0], height, width)
    
    # Depth heat pass - colormap visualization
    colormap_lut = get_colormap_lut(depth_colormap)
    depth_heat_rgba = wp.zeros_like(outputs.color_image)
    wp.launch(
        depth_to_colormap,
        outputs.depth_image.shape,
        [outputs.depth_image, depth_range_np[0], depth_range_np[1], colormap_lut, depth_heat_rgba],
    )
    depth_heat_rgb = packed_uint32_to_rgb(depth_heat_rgba.numpy()[0, 0], height, width)
    
    # Normal pass - convert normals to RGB
    normal_rgba = wp.zeros_like(outputs.color_image)
    wp.launch(
        normal_to_rgb,
        outputs.normal_image.shape,
        [outputs.normal_image, normal_rgba],
    )
    normal_rgb = packed_uint32_to_rgb(normal_rgba.numpy()[0, 0], height, width)
    
    # Object ID pass - use LUT on shape indices
    object_id_rgba = wp.zeros_like(outputs.color_image)
    wp.launch(
        shape_index_to_color_lut,
        outputs.shape_index_image.shape,
        [outputs.shape_index_image, color_luts.object_id, object_id_rgba],
    )
    object_id_rgb = packed_uint32_to_rgb(object_id_rgba.numpy()[0, 0], height, width)
    
    # Semantic pass - use LUT on shape indices
    semantic_rgba = wp.zeros_like(outputs.color_image)
    wp.launch(
        shape_index_to_color_lut,
        outputs.shape_index_image.shape,
        [outputs.shape_index_image, color_luts.semantic, semantic_rgba],
    )
    semantic_rgb = packed_uint32_to_rgb(semantic_rgba.numpy()[0, 0], height, width)
    
    return PixelOutputs(
        color=color_rgb,
        depth=depth_rgb,
        depth_heat=depth_heat_rgb,
        normal=normal_rgb,
        object_id=object_id_rgb,
        semantic=semantic_rgb,
    )


def save_pixels_to_png(pixels: np.ndarray, output_path: Path) -> None:
    """Save RGB pixel array to PNG file."""
    if not _PIL_AVAILABLE:
        raise ImportError("PIL/Pillow is required for saving PNG files.")
    img = Image.fromarray(pixels, mode="RGB")
    img.save(output_path)
    print(f"Saved: {output_path}")


def save_all_aovs(
    pixel_outputs: PixelOutputs,
    output_dir: Path,
    frame_num: int,
    base_name: str = "render",
    ext: str = "png",
) -> None:
    """
    Save all AOV passes to disk as image files.
    
    Creates files named {base}_{AOV}.{frame:04d}.{ext}:
    - {base}_color.0001.png
    - {base}_depth.0001.png
    - {base}_depth_heat.0001.png
    - {base}_normal.0001.png
    - {base}_object_id.0001.png
    - {base}_semantic.0001.png
    
    Args:
        pixel_outputs: PixelOutputs from convert_aovs_to_pixels
        output_dir: Directory to save files
        frame_num: Frame number for filename (formatted as 4-digit zero-padded)
        base_name: Base filename (before the AOV suffix)
        ext: File extension (default: "png")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def aov_path(aov: str) -> Path:
        return output_dir / f"{base_name}_{aov}.{frame_num:04d}.{ext}"
    
    save_pixels_to_png(pixel_outputs.color, aov_path("color"))
    save_pixels_to_png(pixel_outputs.depth, aov_path("depth"))
    save_pixels_to_png(pixel_outputs.depth_heat, aov_path("depth_heat"))
    save_pixels_to_png(pixel_outputs.normal, aov_path("normal"))
    save_pixels_to_png(pixel_outputs.object_id, aov_path("object_id"))
    save_pixels_to_png(pixel_outputs.semantic, aov_path("semantic"))


# =============================================================================
# OpenEXR Output Support
# =============================================================================

def packed_uint32_to_float_rgb(packed: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert packed RGBA uint32 array to float32 RGB array in [0, 1]."""
    pixels = packed.reshape(height, width)
    r = ((pixels >> 0) & 0xFF).astype(np.float32) / 255.0
    g = ((pixels >> 8) & 0xFF).astype(np.float32) / 255.0
    b = ((pixels >> 16) & 0xFF).astype(np.float32) / 255.0
    return np.stack([r, g, b], axis=-1)


def convert_aovs_to_exr_data(
    outputs: RenderOutputs,
    color_luts: ColorLUTs,
    config: RenderConfig,
    depth_colormap: str = DepthColormap.MAGMA,
) -> ExrOutputs:
    """
    Convert raw GPU render outputs to float32 arrays for OpenEXR saving.
    
    Values are normalized/mapped to [0, 1]:
    - Depth is normalized to [0, 1] (0=near, 1=far, background=1)
    - Depth heat uses interpolated colormap sampling for smooth gradients
    - Normals are mapped from [-1, 1] to [0, 1]
    
    Args:
        outputs: Raw RenderOutputs from render_all_aovs
        color_luts: ColorLUTs for object_id and semantic mapping
        config: Render configuration (for dimensions)
        depth_colormap: Colormap for depth_heat (DepthColormap.VIRIDIS or .MAGMA)
        
    Returns:
        ExrOutputs with all passes as float32 arrays in [0, 1]
    """
    height, width = config.height, config.width
    
    # Color pass - convert packed uint32 to float RGB
    color_rgb = packed_uint32_to_float_rgb(
        outputs.color_image.numpy()[0, 0], height, width
    )
    
    # Compute depth range once (used by both depth passes)
    depth_range = wp.array([1e10, 0.0], dtype=wp.float32)
    wp.launch(
        find_depth_range,
        outputs.depth_image.shape,
        [outputs.depth_image, depth_range],
    )
    depth_range_np = depth_range.numpy()
    min_depth, max_depth = depth_range_np[0], depth_range_np[1]
    
    # Depth pass - normalize to 0-1
    depth_raw = outputs.depth_image.numpy()[0, 0].reshape(height, width)
    valid_mask = depth_raw > 0
    if valid_mask.any():
        denom = max(max_depth - min_depth, 0.001)
        # Normalize: 0=near, 1=far
        depth = np.where(valid_mask, (depth_raw - min_depth) / denom, 1.0)
    else:
        depth = np.ones_like(depth_raw)
    
    # Depth heat pass - interpolated colormap for smooth gradients
    colormap_lut = get_colormap_lut(depth_colormap)
    depth_heat_rgb = wp.zeros((1, 1, width * height), dtype=wp.vec3f)
    wp.launch(
        depth_to_colormap_float,
        outputs.depth_image.shape,
        [outputs.depth_image, min_depth, max_depth, colormap_lut, depth_heat_rgb],
    )
    depth_heat = depth_heat_rgb.numpy()[0, 0].reshape(height, width, 3)
    
    # Normal pass - map from [-1, 1] to [0, 1]
    normal_raw = outputs.normal_image.numpy()[0, 0].reshape(height, width, 3)
    normal = normal_raw * 0.5 + 0.5
    
    # Object ID pass - use LUT on shape indices
    object_id_rgba = wp.zeros_like(outputs.color_image)
    wp.launch(
        shape_index_to_color_lut,
        outputs.shape_index_image.shape,
        [outputs.shape_index_image, color_luts.object_id, object_id_rgba],
    )
    object_id_rgb = packed_uint32_to_float_rgb(
        object_id_rgba.numpy()[0, 0], height, width
    )
    
    # Semantic pass - use LUT on shape indices
    semantic_rgba = wp.zeros_like(outputs.color_image)
    wp.launch(
        shape_index_to_color_lut,
        outputs.shape_index_image.shape,
        [outputs.shape_index_image, color_luts.semantic, semantic_rgba],
    )
    semantic_rgb = packed_uint32_to_float_rgb(
        semantic_rgba.numpy()[0, 0], height, width
    )
    
    # Shape index pass - raw integer indices as float32 for EXR compatibility
    # Background pixels have shape_index = 0xFFFFFFFF, we map to -1.0 for clarity
    shape_index_raw = outputs.shape_index_image.numpy()[0, 0].reshape(height, width)
    shape_index = np.where(
        shape_index_raw == 0xFFFFFFFF,
        -1.0,
        shape_index_raw.astype(np.float32)
    )
    
    return ExrOutputs(
        color=color_rgb.astype(np.float32),
        depth=depth.astype(np.float32),
        depth_heat=depth_heat.astype(np.float32),
        normal=normal.astype(np.float32),
        object_id=object_id_rgb.astype(np.float32),
        semantic=semantic_rgb.astype(np.float32),
        shape_index=shape_index.astype(np.float32),
    )


def save_exr_rgb(pixels: np.ndarray, output_path: Path) -> None:
    """
    Save RGB float array to OpenEXR file with half-float precision.
    
    Uses OpenEXR 3.x API with default ZIP compression.
    
    Args:
        pixels: (height, width, 3) float32 array
        output_path: Path to save the EXR file
    """
    if not _OPENEXR_AVAILABLE:
        raise ImportError("OpenEXR package required for EXR output. Install with: uv pip install openexr")
    
    # Convert to half-float (float16) - OpenEXR 3.x detects dtype automatically
    half_pixels = pixels.astype(np.float16)
    
    # Separate into channels
    channels = {
        "R": half_pixels[:, :, 0].copy(),
        "G": half_pixels[:, :, 1].copy(),
        "B": half_pixels[:, :, 2].copy(),
    }
    
    # Create file with default compression (ZIP) and write
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    exr_file = OpenEXR.File(header, channels)
    exr_file.write(str(output_path))
    
    print(f"Saved: {output_path}")


def save_exr_depth(depth: np.ndarray, output_path: Path) -> None:
    """
    Save single-channel depth to OpenEXR file with half-float precision.
    
    Uses OpenEXR 3.x API with default ZIP compression.
    
    Args:
        depth: (height, width) float32 array
        output_path: Path to save the EXR file
    """
    if not _OPENEXR_AVAILABLE:
        raise ImportError("OpenEXR package required for EXR output. Install with: uv pip install openexr")
    
    # Convert to half-float
    half_depth = depth.astype(np.float16)
    
    # Single Y channel for depth/luminance
    channels = {"Y": half_depth.copy()}
    
    # Create file with default compression and write
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    exr_file = OpenEXR.File(header, channels)
    exr_file.write(str(output_path))
    
    print(f"Saved: {output_path}")


def save_exr_shape_index(shape_index: np.ndarray, output_path: Path) -> None:
    """
    Save single-channel shape index to OpenEXR file with full float32 precision.
    
    Uses float32 instead of float16 to preserve integer precision for shape indices.
    Background pixels are stored as -1.0.
    
    Args:
        shape_index: (height, width) float32 array containing shape indices
        output_path: Path to save the EXR file
    """
    if not _OPENEXR_AVAILABLE:
        raise ImportError("OpenEXR package required for EXR output. Install with: uv pip install openexr")
    
    # Keep as float32 to preserve integer precision (float32 can exactly represent
    # integers up to 2^24 = 16 million, more than enough for shape indices)
    float_shape_index = shape_index.astype(np.float32)
    
    # Single Y channel for shape index
    channels = {"Y": float_shape_index.copy()}
    
    # Create file with default compression and write
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    exr_file = OpenEXR.File(header, channels)
    exr_file.write(str(output_path))
    
    print(f"Saved: {output_path}")


def save_shape_index_mapping(
    diegetics: list,
    output_path: Path,
) -> None:
    """
    Save a JSON mapping from shape index to prim path.
    
    This mapping allows looking up which prim corresponds to a given shape index
    in the shape_index EXR output. Useful for debugging and identifying geometry.
    
    Background pixels have shape_index = -1 (not in mapping).
    
    Args:
        diegetics: List of Diegetic objects (order defines shape indices)
        output_path: Path to save the JSON file
    """
    import json
    
    mapping = {
        str(i): {
            "path": d.path,
            "name": d.name,
        }
        for i, d in enumerate(diegetics)
    }
    
    # Add metadata
    output = {
        "description": "Shape index to prim path mapping for shape_index AOV",
        "note": "Background pixels have shape_index = -1 (not in mapping)",
        "num_shapes": len(diegetics),
        "shapes": mapping,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved shape index mapping: {output_path}")


def save_all_aovs_exr(
    exr_outputs: ExrOutputs,
    output_dir: Path,
    frame_num: int,
    base_name: str = "render",
) -> None:
    """
    Save all AOV passes to disk as OpenEXR files with half-float precision.
    
    Creates files named {base}_{AOV}.{frame:04d}.exr:
    - {base}_color.0001.exr: RGB color (half-float)
    - {base}_depth.0001.exr: Single-channel depth (half-float)
    - {base}_depth_heat.0001.exr: Depth heat map with interpolated colormap (half-float RGB)
    - {base}_normal.0001.exr: RGB normals (half-float, values in [0, 1])
    - {base}_object_id.0001.exr: RGB object ID colors (half-float)
    - {base}_semantic.0001.exr: RGB semantic colors (half-float)
    
    Args:
        exr_outputs: ExrOutputs from convert_aovs_to_exr_data
        output_dir: Directory to save files
        frame_num: Frame number for filename (formatted as 4-digit zero-padded)
        base_name: Base filename (before the AOV suffix)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def aov_path(aov: str) -> Path:
        return output_dir / f"{base_name}_{aov}.{frame_num:04d}.exr"
    
    save_exr_rgb(exr_outputs.color, aov_path("color"))
    save_exr_depth(exr_outputs.depth, aov_path("depth"))
    save_exr_rgb(exr_outputs.depth_heat, aov_path("depth_heat"))
    save_exr_rgb(exr_outputs.normal, aov_path("normal"))
    save_exr_rgb(exr_outputs.object_id, aov_path("object_id"))
    save_exr_rgb(exr_outputs.semantic, aov_path("semantic"))


def render_and_save_all_aovs(
    ctx: RenderContext,
    color_luts: ColorLUTs,
    camera: CameraData,
    config: RenderConfig,
    frame_num: int,
    base_name: str = "render",
    output_format: str = "png",
    depth_colormap: str = DepthColormap.MAGMA,
) -> PixelOutputs | ExrOutputs:
    """
    Render all AOVs and save them to disk.
    
    Convenience function that combines render_all_aovs, convert_aovs_to_pixels/exr,
    and save_all_aovs/exr.
    
    Args:
        ctx: RenderContext configured with scene geometry
        color_luts: ColorLUTs for post-processing
        camera: Camera position and orientation
        config: Render configuration (resolution, output dir)
        frame_num: Frame number for filename (formatted as 4-digit zero-padded)
        base_name: Base filename (before the AOV suffix)
        output_format: Output format - "png" for uint8 PNG or "exr" for half-float OpenEXR
        depth_colormap: Colormap for depth_heat (DepthColormap.VIRIDIS or .MAGMA)
        
    Returns:
        PixelOutputs (for PNG) or ExrOutputs (for EXR) with all passes as numpy arrays
    """
    print(f"Rendering frame {frame_num} (all AOVs, format={output_format})...")
    
    # Single render pass
    outputs = render_all_aovs(ctx, camera, config)
    
    if output_format.lower() == "exr":
        # Convert to float data (with interpolated depth_heat) and save as EXR
        exr_outputs = convert_aovs_to_exr_data(outputs, color_luts, config, depth_colormap)
        save_all_aovs_exr(exr_outputs, config.output_dir, frame_num, base_name)
        return exr_outputs
    else:
        # Convert to pixels and save as PNG
        pixel_outputs = convert_aovs_to_pixels(outputs, color_luts, config, depth_colormap)
        save_all_aovs(pixel_outputs, config.output_dir, frame_num, base_name, output_format)
        return pixel_outputs


# =============================================================================
# Texture Utilities
# =============================================================================

class TextureColorCache:
    """Cache for texture average colors to avoid reloading."""
    
    def __init__(self):
        self._cache: dict[str, tuple[float, float, float] | None] = {}
    
    def get(self, path: str) -> tuple[float, float, float] | None:
        """Get cached color for path, or None if not cached."""
        return self._cache.get(path)
    
    def set(self, path: str, color: tuple[float, float, float] | None):
        """Cache a color for a path."""
        self._cache[path] = color
    
    def has(self, path: str) -> bool:
        """Check if path is in cache."""
        return path in self._cache
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()


# Global cache instance
_texture_color_cache = TextureColorCache()


def resolve_udim_textures(texture_path: str, usd_file_dir: str) -> list[str]:
    """
    Resolve a texture path that may contain <UDIM> placeholder.
    
    Args:
        texture_path: Path that may contain <UDIM> (e.g., "textures/foo.<UDIM>.png")
        usd_file_dir: Directory of the USD file for resolving relative paths
        
    Returns:
        List of resolved file paths (sorted by UDIM number)
    """
    # Resolve relative to USD file directory
    if os.path.isabs(texture_path):
        base_path = texture_path
    else:
        base_path = os.path.join(usd_file_dir, texture_path)
    
    base_path = os.path.normpath(base_path)
    
    # Check for UDIM pattern
    if "<UDIM>" in base_path:
        # Replace <UDIM> with a glob pattern to find all tiles
        glob_pattern = base_path.replace("<UDIM>", "[0-9][0-9][0-9][0-9]")
        matching_files = glob.glob(glob_pattern)
        
        if matching_files:
            # Sort by UDIM number (extract the 4-digit number and sort)
            def extract_udim(path):
                match = re.search(r'\.(\d{4})\.', os.path.basename(path))
                return int(match.group(1)) if match else 9999
            
            matching_files.sort(key=extract_udim)
            return matching_files
        else:
            return []
    else:
        # No UDIM, just return the single path if it exists
        if os.path.exists(base_path):
            return [base_path]
        else:
            return []


def get_texture_average_color(
    texture_path: str,
    usd_file_dir: str,
    cache: TextureColorCache | None = None,
    verbose: bool = False
) -> tuple[float, float, float] | None:
    """
    Load a texture and compute its average RGB color.
    
    Handles <UDIM> patterns by using the lowest UDIM tile (1001).
    
    Args:
        texture_path: Path to the texture (may be relative, may contain <UDIM>)
        usd_file_dir: Directory of the USD file for resolving relative paths
        cache: Optional TextureColorCache for caching results
        verbose: If True, print debug information
        
    Returns:
        tuple (r, g, b) normalized to 0-1, or None if failed
    """
    if not _PIL_AVAILABLE:
        if verbose:
            print("        PIL not available for texture loading")
        return None
    
    if cache is None:
        cache = _texture_color_cache
    
    # Check cache first
    if cache.has(texture_path):
        return cache.get(texture_path)
    
    # Resolve UDIM patterns and get list of texture files
    texture_files = resolve_udim_textures(texture_path, usd_file_dir)
    
    if not texture_files:
        if verbose:
            print(f"        Texture not found: {texture_path}")
            print(f"          (resolved from USD dir: {usd_file_dir})")
        cache.set(texture_path, None)
        return None
    
    # Use the first (lowest UDIM) texture
    resolved_path = texture_files[0]
    
    try:
        if verbose:
            if len(texture_files) > 1:
                print(f"        Found {len(texture_files)} UDIM tiles")
            print(f"        Loading texture: {resolved_path}")
        
        img = Image.open(resolved_path)
        
        # Convert to RGB if necessary (handle RGBA, palette, etc.)
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        elif img.mode == 'RGBA':
            # For RGBA, composite over white background to handle transparency
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        # Resize to speed up averaging (we don't need full resolution)
        img.thumbnail((64, 64), Image.Resampling.LANCZOS)
        
        # Get average color
        pixels = np.array(img, dtype=np.float32) / 255.0
        avg_color = pixels.mean(axis=(0, 1))
        
        result = tuple(avg_color[:3])
        if verbose:
            print(f"        Average color: RGB({result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f})")
        
        cache.set(texture_path, result)
        return result
        
    except Exception as e:
        if verbose:
            print(f"        Failed to load texture: {e}")
        cache.set(texture_path, None)
        return None


# =============================================================================
# USD Stage Loading
# =============================================================================

def open_usd_stage(usd_path: str, show_progress: bool = True) -> Usd.Stage:
    """
    Open a USD stage with progress messaging.
    
    Since Usd.Stage.Open() is a blocking call without progress callbacks,
    this prints before/after messages so the user knows parsing is in progress.
    
    Args:
        usd_path: Path to the USD file
        show_progress: If True, print progress messages
        
    Returns:
        Opened USD stage
        
    Raises:
        RuntimeError: If the stage fails to open
    """
    import time
    
    if show_progress:
        print(f"Opening USD stage: {usd_path}", flush=True)
        print("  (This may take a while for large files...)", flush=True)
    
    start_time = time.time()
    stage = Usd.Stage.Open(usd_path)
    elapsed = time.time() - start_time
    
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_path}")
    
    if show_progress:
        print(f"  Stage opened in {elapsed:.1f}s", flush=True)
    
    return stage


def is_proxy_path(path: str) -> bool:
    """Check if a prim path appears to be proxy geometry."""
    return "/proxy/" in path.lower()


# =============================================================================
# Polyscope Visualization (Alternative Consumer)
# =============================================================================

def setup_polyscope(up_direction: str = "z_up") -> None:
    """
    Initialize polyscope with standard settings.
    
    Args:
        up_direction: Up direction ("z_up", "y_up", etc.)
    """
    import polyscope as ps
    ps.init()
    ps.set_up_dir(up_direction)


def register_diegetics_with_polyscope(
    diegetics: list[Diegetic],
    use_diffuse: bool = True,
    verbose: bool = False,
) -> None:
    """
    Register Diegetics with polyscope for interactive visualization.
    
    Args:
        diegetics: List of Diegetic scene elements
        use_diffuse: If True, use diffuse_albedo colors; otherwise use object_id
        verbose: If True, print registration info
    """
    import polyscope as ps
    
    for d in diegetics:
        pm = ps.register_surface_mesh(d.name, d.vertices, d.faces)
        color = d.diffuse_albedo if use_diffuse else d.object_id
        pm.set_color(color)
        
        if verbose:
            print(f"  {d.name}: {color}")


def show_polyscope() -> None:
    """Show the polyscope viewer (blocking)."""
    import polyscope as ps
    ps.show()

