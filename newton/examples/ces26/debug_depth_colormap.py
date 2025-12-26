# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Test utility for iterating on depth colormap transforms.

Reads an existing grayscale depth PNG, converts to normalized floats,
applies various transforms (gamma, log), and outputs colorized versions.

This allows fast iteration without re-rendering the scene.

Usage: uv run python newton/examples/ces26/debug_depth_colormap.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

# =============================================================================
# Colormaps (copied from ces26_utils for standalone use)
# =============================================================================

# Viridis colormap - 256 entries, RGB 0-255
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
    (254, 225, 43), (254, 224, 45), (254, 224, 47),
]

# Magma colormap - 256 entries, RGB 0-255
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
    (253, 252, 131), (254, 254, 134), (254, 255, 137),
]

# Build numpy LUTs
VIRIDIS_LUT = np.array(_VIRIDIS_DATA, dtype=np.uint8)  # (256, 3)
MAGMA_LUT = np.array(_MAGMA_DATA, dtype=np.uint8)  # (256, 3)


# =============================================================================
# Transform Functions
# =============================================================================

def transform_linear(depth: np.ndarray) -> np.ndarray:
    """No transform - linear mapping."""
    return depth


def transform_gamma(depth: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    Gamma correction: output = input^(1/gamma)
    
    gamma > 1: brightens dark areas, spreads near values
    gamma < 1: darkens, spreads far values
    """
    return np.power(depth, 1.0 / gamma)


def transform_log(depth: np.ndarray, base: float = 10.0) -> np.ndarray:
    """
    Logarithmic transform: spreads near values more than far.
    
    output = log(1 + depth * (base - 1)) / log(base)
    
    This maps [0, 1] -> [0, 1] with logarithmic stretching.
    Higher base = more stretching of near values.
    """
    return np.log1p(depth * (base - 1)) / np.log(base)


def transform_sqrt(depth: np.ndarray) -> np.ndarray:
    """Square root transform - like gamma=2."""
    return np.sqrt(depth)


def transform_power(depth: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Power transform: output = depth^power."""
    return np.power(depth, power)


# =============================================================================
# Colormap Application
# =============================================================================

def apply_colormap(
    normalized: np.ndarray,
    lut: np.ndarray,
    background_mask: np.ndarray | None = None,
    use_far_for_background: bool = True,
) -> np.ndarray:
    """
    Apply a colormap LUT to normalized depth values.
    
    Args:
        normalized: Depth values in [0, 1], where 0=near, 1=far
        lut: Colormap LUT array (N, 3) uint8
        background_mask: Boolean mask for background pixels (no depth)
        use_far_for_background: If True, background uses far color
        
    Returns:
        RGB image (H, W, 3) uint8
    """
    lut_size = len(lut)
    
    # Invert so near=high value (bright/warm), far=low value (dark/cool)
    inverted = 1.0 - normalized
    
    # Map to LUT indices (scale to actual LUT size)
    indices = (inverted * (lut_size - 1)).astype(np.int32)
    indices = np.clip(indices, 0, lut_size - 1)
    
    # Apply LUT
    rgb = lut[indices]
    
    # Handle background
    if background_mask is not None:
        if use_far_for_background:
            # Background gets the far color (index 0 after inversion, which is lut[0])
            rgb[background_mask] = lut[0]
        else:
            # Gray background
            rgb[background_mask] = [64, 64, 64]
    
    return rgb


# =============================================================================
# Main Test Function
# =============================================================================

def load_depth_from_grayscale(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load grayscale depth image and convert to normalized floats.
    
    The grayscale depth image has:
    - Brighter = closer (inverted already)
    - Background = dark gray (around 64)
    
    Returns:
        normalized: Depth values [0, 1] where 0=near, 1=far
        background_mask: Boolean mask for background pixels
    """
    img = Image.open(path).convert("L")
    pixels = np.array(img, dtype=np.float32)
    
    # Detect background (gray around 64)
    background_mask = pixels < 65
    
    # The grayscale is already inverted (brighter = near)
    # Convert back: near=0, far=1
    # Normalize to [0, 1] excluding background
    valid_pixels = pixels[~background_mask]
    if len(valid_pixels) == 0:
        return np.zeros_like(pixels), background_mask
    
    min_val = valid_pixels.min()
    max_val = valid_pixels.max()
    
    # Brighter = near, so we invert
    normalized = (max_val - pixels) / (max_val - min_val + 1e-6)
    normalized = np.clip(normalized, 0, 1)
    
    return normalized, background_mask


def test_transforms(
    input_path: Path,
    output_dir: Path,
    colormap: str = "magma",
) -> None:
    """
    Test various depth transforms and save results.
    
    Args:
        input_path: Path to grayscale depth PNG
        output_dir: Directory to save output images
        colormap: "magma" or "viridis"
    """
    print(f"Loading depth from: {input_path}")
    normalized, background_mask = load_depth_from_grayscale(input_path)
    
    lut = MAGMA_LUT if colormap == "magma" else VIRIDIS_LUT
    base_name = input_path.stem.replace("_depth", "")
    
    # Define transforms to test
    transforms = [
        ("linear", lambda d: transform_linear(d)),
        ("gamma_1.5", lambda d: transform_gamma(d, 1.5)),
        ("gamma_2.0", lambda d: transform_gamma(d, 2.0)),
        ("gamma_2.2", lambda d: transform_gamma(d, 2.2)),
        ("gamma_3.0", lambda d: transform_gamma(d, 3.0)),
        ("sqrt", lambda d: transform_sqrt(d)),
        ("log_10", lambda d: transform_log(d, 10.0)),
        ("log_100", lambda d: transform_log(d, 100.0)),
        ("log_1000", lambda d: transform_log(d, 1000.0)),
        ("power_0.3", lambda d: transform_power(d, 0.3)),
        ("power_0.5", lambda d: transform_power(d, 0.5)),
        ("power_0.7", lambda d: transform_power(d, 0.7)),
    ]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, transform_fn in transforms:
        transformed = transform_fn(normalized)
        rgb = apply_colormap(transformed, lut, background_mask, use_far_for_background=True)
        
        output_path = output_dir / f"{base_name}_heat_{colormap}_{name}.png"
        Image.fromarray(rgb).save(output_path)
        print(f"  Saved: {output_path.name}")


# =============================================================================
# Configuration
# =============================================================================

# Input depth image (grayscale)
INPUT_DEPTH = Path(__file__).parent / "debug_all_visible_depth.2920.png"

# Output directory for test images
OUTPUT_DIR = Path(__file__).parent / "depth_colormap_tests"

# Colormap to use
#COLORMAP = "magma"  # or "viridis"
COLORMAP = "viridis"


if __name__ == "__main__":
    test_transforms(INPUT_DEPTH, OUTPUT_DIR, COLORMAP)
    print(f"\nDone! Check {OUTPUT_DIR} for results.")
    print("\nLook for transforms that spread the colors more evenly:")
    print("  - gamma_2.2 or gamma_3.0: Good for spreading near values")
    print("  - log_100 or log_1000: Aggressive log stretch")
    print("  - power_0.3: Strong power curve")

