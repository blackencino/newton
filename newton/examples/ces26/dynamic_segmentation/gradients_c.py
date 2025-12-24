"""
Semantic color gradients for robotics telemetry display.

Provides three gradient functions that map a 0-1 float to RGB colors (0-1 float32):
  - gradient_cool(t): Blues/purples - for ground/background elements
  - gradient_hot(t):  Reds/pinks - for danger/avoid zones
  - gradient_happy(t): Greens/yellows - for interactive/safe elements

This implementation finds a perceptually smooth "single swoop" path through OKLab
color space by using PCA to identify the principal direction of variation, then
projecting colors onto that axis for ordering. This creates monotonic gradients
that feel like moving smoothly in one direction through color space.
"""

from __future__ import annotations

import colorsys
from typing import Callable, Dict, List

import numpy as np


# =========================
# Palette (18 colors total)
# =========================

# Top & bottom rows below were sampled from the provided palette image (median color
# in the center of each cell). You can edit them freely if your palette changes.
TOP_ROW_HEX = [
    "#09065D",  # deep navy
    "#5125AE",  # indigo
    "#A92672",  # dark magenta
    "#B64138",  # brick red
    "#92B211",  # olive
    "#40A906",  # green
]

# Middle row (these are printed in the palette image)
MID_ROW_HEX = [
    "#3B6EBA",  # NOTE: left label is cropped; update if needed
    "#7C82FA",
    "#E47AF5",
    "#FF97BE",
    "#FDB15C",
    "#FFDA3B",
]

BOTTOM_ROW_HEX = [
    "#0E701B",  # dark green
    "#41DC90",  # mint
    "#417FD4",  # blue
    "#7D5EEA",  # violet
    "#D924AF",  # hot magenta
    "#D61161",  # crimson
]

PALETTE_HEX: List[str] = TOP_ROW_HEX + MID_ROW_HEX + BOTTOM_ROW_HEX


# ==================================
# Color helpers (hex <-> float RGB)
# ==================================

def hex_to_rgb01(hx: str) -> np.ndarray:
    hx = hx.strip().lstrip("#")
    return np.array([int(hx[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64)


# ==========================================
# OKLab (perceptual) conversion (pure numpy)
# ==========================================
# Reference: Björn Ottosson, "OKLab: a perceptual color space for image processing"
# https://bottosson.github.io/posts/oklab/

_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)

_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)

_M2_INV = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ],
    dtype=np.float64,
)

_M1_INV = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ],
    dtype=np.float64,
)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb_lin: np.ndarray) -> np.ndarray:
    """
    Safe piecewise conversion (avoids NaNs for negative linear values caused by
    out-of-gamut interpolation before final clipping).
    """
    rgb_lin = np.asarray(rgb_lin, dtype=np.float64)
    out = np.empty_like(rgb_lin)
    mask = rgb_lin <= 0.0031308
    out[mask] = 12.92 * rgb_lin[mask]
    out[~mask] = 1.055 * np.power(np.clip(rgb_lin[~mask], 0.0, None), 1 / 2.4) - 0.055
    return out


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    lin = srgb_to_linear(rgb)
    lms = lin @ _M1.T
    lms_cbrt = np.cbrt(np.clip(lms, 0.0, None))
    return lms_cbrt @ _M2.T


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    lms_cbrt = lab @ _M2_INV.T
    lms = np.power(lms_cbrt, 3)
    lin = lms @ _M1_INV.T
    return linear_to_srgb(lin)


# ===========================
# Smooth interpolation (C1)
# ===========================

def catmull_rom(points: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Uniform Catmull-Rom spline through control points.

    points: (n, d)
    t: (...,) in [0,1]
    returns: (..., d)
    """
    points = np.asarray(points, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    n = points.shape[0]
    if n == 1:
        return np.broadcast_to(points[0], t.shape + (points.shape[1],))

    u = t * (n - 1)
    i = np.floor(u).astype(int)
    i = np.clip(i, 0, n - 2)
    u = u - i  # local segment parameter

    i0 = np.clip(i - 1, 0, n - 1)
    i1 = i
    i2 = i + 1
    i3 = np.clip(i + 2, 0, n - 1)

    P0, P1, P2, P3 = points[i0], points[i1], points[i2], points[i3]

    u2 = u * u
    u3 = u2 * u
    u = u[..., None]
    u2 = u2[..., None]
    u3 = u3[..., None]

    return 0.5 * (
        (2 * P1)
        + (-P0 + P2) * u
        + (2 * P0 - 5 * P1 + 4 * P2 - P3) * u2
        + (-P0 + 3 * P1 - 3 * P2 + P3) * u3
    )


# ==========================================
# Semantic grouping + smooth path ordering
# ==========================================


def classify_semantic_group(hx: str) -> str:
    """
    Group by hue:
      - HOT   : magenta/pink/red
      - COOL  : blue/indigo/violet
      - HAPPY : yellow/green/orange

    Hue thresholds chosen to match your palette intent:
      HOT   if hue >= 285° or hue <= 15°
      COOL  if 180° <= hue < 285°
      HAPPY otherwise
    """
    r, g, b = hex_to_rgb01(hx)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue_deg = h * 360.0

    if hue_deg >= 285.0 or hue_deg <= 15.0:
        return "hot"
    if hue_deg >= 180.0:
        return "cool"
    return "happy"


def order_smooth_path(hex_list: List[str], group_name: str = "") -> List[str]:
    """
    Order colors along a perceptually smooth "single swoop" path through OKLab space.
    
    Uses a greedy nearest-neighbor path-finding algorithm with group-specific biases:
    - Cool: Penalizes pastel colors (high lightness, low chroma), favors darker saturated purples/blues
    - Hot/Happy: Favors higher chroma (more saturated/vibrant colors)
    - All: Maintains smooth perceptual transitions
    
    The path feels like a single continuous sweep through color space, avoiding
    the "roller coaster" effect of pure lightness sorting.
    """
    if len(hex_list) <= 1:
        return hex_list
    
    # Convert to OKLab
    labs = np.stack([rgb_to_oklab(hex_to_rgb01(h)) for h in hex_list], axis=0)
    lightness = labs[:, 0]
    # Chroma in OKLab: sqrt(a^2 + b^2)
    chroma = np.sqrt(labs[:, 1]**2 + labs[:, 2]**2)
    
    # Start with the darkest color
    ordered_idx = [np.argmin(lightness)]
    remaining = set(range(len(hex_list))) - {ordered_idx[0]}
    
    # Greedily build path with group-specific biases
    while remaining:
        current_lab = labs[ordered_idx[-1]]
        current_L = lightness[ordered_idx[-1]]
        current_chroma = chroma[ordered_idx[-1]]
        
        best_idx = None
        best_score = float('inf')
        
        for candidate_idx in remaining:
            candidate_lab = labs[candidate_idx]
            candidate_L = lightness[candidate_idx]
            candidate_chroma = chroma[candidate_idx]
            
            # Perceptual distance in OKLab (Euclidean)
            perceptual_dist = np.linalg.norm(current_lab - candidate_lab)
            
            # Base lightness penalty (allow small backward steps for smoothness)
            lightness_penalty = max(0, (current_L - candidate_L - 0.05) * 3.0)
            
            # Group-specific adjustments
            if group_name == "cool":
                # Penalize pastel colors (high lightness + low chroma)
                # Want more time in darker, saturated purples/blues
                is_pastel = (candidate_L > 0.7) and (candidate_chroma < 0.15)
                pastel_penalty = 2.0 if is_pastel else 0.0
                # Favor darker colors in the middle range
                if 0.3 < candidate_L < 0.7:
                    darkness_bonus = -0.3 * (0.7 - candidate_L)  # Prefer darker in middle
                else:
                    darkness_bonus = 0.0
                score = perceptual_dist + lightness_penalty + pastel_penalty - darkness_bonus
            elif group_name in ["hot", "happy"]:
                # Favor higher chroma (more saturated/vibrant)
                chroma_bonus = -0.5 * candidate_chroma  # Prefer higher chroma
                score = perceptual_dist + lightness_penalty + chroma_bonus
            else:
                # Default: just minimize distance with lightness bias
                score = perceptual_dist + lightness_penalty
            
            if score < best_score:
                best_score = score
                best_idx = candidate_idx
        
        ordered_idx.append(best_idx)
        remaining.remove(best_idx)
    
    return [hex_list[i] for i in ordered_idx]


def make_oklab_gradient(hex_colors: List[str], group_name: str = "") -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a gradient function f(t)->RGB in [0,1] using OKLab + Catmull-Rom.
    
    For hot/happy groups, boosts saturation (chroma) to increase "sizzle".
    For cool group, ensures perceptually linear lightness progression.
    """
    rgbs = np.stack([hex_to_rgb01(h) for h in hex_colors], axis=0)
    labs = rgb_to_oklab(rgbs)

    def f(t: np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
        scalar_input = (t.ndim == 0)
        if scalar_input:
            t = np.atleast_1d(t)
        
        lab = catmull_rom(labs, t)
        
        # Group-specific adjustments
        if group_name == "cool":
            # Ensure perceptually linear lightness progression
            # Re-map t to spend more time in darker colors (less time in pastels)
            # Use a power curve that favors darker end
            t_remapped = t ** 0.75  # More aggressive easing toward darker end
            lab = catmull_rom(labs, t_remapped)
        
        elif group_name in ["hot", "happy"]:
            # Boost saturation (chroma) for more "sizzle"
            # Extract chroma and boost it
            chroma = np.sqrt(lab[:, 1]**2 + lab[:, 2]**2)
            # Boost chroma by 15-25% depending on current chroma
            boost_factor = 1.0 + 0.15 + 0.10 * (1.0 - np.clip(chroma / 0.3, 0.0, 1.0))
            chroma_boosted = chroma * boost_factor
            
            # Reconstruct a and b with boosted chroma
            mask = chroma > 1e-6
            lab_a = np.where(mask, lab[:, 1] * (chroma_boosted / chroma), lab[:, 1])
            lab_b = np.where(mask, lab[:, 2] * (chroma_boosted / chroma), lab[:, 2])
            
            # For happy, specifically boost yellow (high L, positive b) to be sunnier
            if group_name == "happy":
                # Detect yellow-ish colors (high L, positive b)
                is_yellowish = (lab[:, 0] > 0.75) & (lab[:, 2] > 0.05)
                # Boost chroma more for yellows, shift b slightly more positive (less mustard)
                yellow_chroma_boost = 1.0 + 0.20  # Extra boost for yellow
                lab_a = np.where(is_yellowish, lab_a * yellow_chroma_boost, lab_a)
                lab_b = np.where(is_yellowish, lab_b * yellow_chroma_boost + 0.02, lab_b)
            
            lab = np.column_stack([lab[:, 0], lab_a, lab_b])
        
        rgb = oklab_to_rgb(lab)
        rgb = np.clip(rgb, 0.0, 1.0)
        
        if scalar_input:
            return rgb[0].astype(np.float32)
        return rgb.astype(np.float32)

    return f


def _build_group_gradients(palette_hex: List[str]) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Build gradient functions for each semantic group."""
    grouped: Dict[str, List[str]] = {"cool": [], "hot": [], "happy": []}
    for hx in palette_hex:
        grouped[classify_semantic_group(hx)].append(hx)

    results: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for name, hx_list in grouped.items():
        ordered = order_smooth_path(hx_list, group_name=name)
        results[name] = make_oklab_gradient(ordered, group_name=name)
    return results


# Pre-compute the gradient functions
_GRADIENTS = _build_group_gradients(PALETTE_HEX)


# =============================================================================
# Public API: Three gradient functions
# =============================================================================

def gradient_cool(t: float | np.ndarray) -> np.ndarray:
    """
    Cool/blue gradient for ground and background elements.
    
    Args:
        t: Value(s) in range 0-1. 0=dark navy, 1=light lavender
        
    Returns:
        RGB color(s) as float32 in range 0-1
    """
    t = np.asarray(t, dtype=np.float64)
    scalar_input = (t.ndim == 0)
    result = _GRADIENTS["cool"](t)
    if scalar_input:
        return result.astype(np.float32)
    return result.astype(np.float32)


def gradient_hot(t: float | np.ndarray) -> np.ndarray:
    """
    Hot/red gradient for danger and avoidance zones.
    
    Args:
        t: Value(s) in range 0-1. 0=dark plum, 1=light pink
        
    Returns:
        RGB color(s) as float32 in range 0-1
    """
    t = np.asarray(t, dtype=np.float64)
    scalar_input = (t.ndim == 0)
    result = _GRADIENTS["hot"](t)
    if scalar_input:
        return result.astype(np.float32)
    return result.astype(np.float32)


def gradient_happy(t: float | np.ndarray) -> np.ndarray:
    """
    Happy/green-yellow gradient for interactive and safe elements.
    
    Args:
        t: Value(s) in range 0-1. 0=dark forest, 1=bright yellow
        
    Returns:
        RGB color(s) as float32 in range 0-1
    """
    t = np.asarray(t, dtype=np.float64)
    scalar_input = (t.ndim == 0)
    result = _GRADIENTS["happy"](t)
    if scalar_input:
        return result.astype(np.float32)
    return result.astype(np.float32)


# =============================================================================
# Visualization (only runs when executed directly)
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    width = 1000
    height_per_row = 100
    
    t_vals = np.linspace(0, 1, width)
    
    # Generate gradient bars
    bar_cool = gradient_cool(t_vals)
    bar_hot = gradient_hot(t_vals)
    bar_happy = gradient_happy(t_vals)
    
    # Build image
    img = np.zeros((height_per_row * 3, width, 3), dtype=np.float32)
    img[0:height_per_row] = np.tile(bar_cool, (height_per_row, 1, 1))
    img[height_per_row:2*height_per_row] = np.tile(bar_hot, (height_per_row, 1, 1))
    img[2*height_per_row:] = np.tile(bar_happy, (height_per_row, 1, 1))
    
    # Add separators
    sep = np.ones((5, width, 3), dtype=np.float32)
    final_view = np.vstack([
        img[0:height_per_row], sep,
        img[height_per_row:2*height_per_row], sep,
        img[2*height_per_row:]
    ])
    
    plt.figure(figsize=(12, 5))
    plt.imshow(final_view)
    plt.axis('off')
    plt.title("Cool (Background) | Hot (Danger) | Happy (Interactive)")
    plt.tight_layout()
    plt.show()

