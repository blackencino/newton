#!/usr/bin/env python3
"""
palette_gradients.py

Given an art-directed false-color palette for robotics segmentation/telemetry,
this script:

1) Groups *all* palette colors into 3 semantic groups:
   - COOL  : ground/background/non-interacting
   - HOT   : potentially hazardous / avoid
   - HAPPY : likely to enter interaction space but safe contact/collision

2) Orders each group "less -> more" using perceptual lightness (OKLab L),
   which typically corresponds to dark -> light.

3) Builds a *continuous, smooth* gradient for each group using a Catmull-Rom
   spline in the OKLab perceptual colorspace. Each gradient is a function
   f(t) where t in [0,1] and f(t) returns an sRGB triple in [0,1].

4) Renders a 3-row showcase image of the three gradients.

Dependencies:
  - numpy
  - Pillow (PIL) for image output
  - colorsys (standard library)

Note:
  The middle row colors are taken from the hex labels visible in your palette image.
  The leftmost labeled hex is partially cropped in the provided JPEG; it appears
  to be "3B6EBA". If your source palette says otherwise, update MID_ROW_HEX[0].
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from PIL import Image


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


def make_oklab_gradient(hex_colors: List[str]) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a gradient function f(t)->RGB in [0,1] using OKLab + Catmull-Rom.
    """
    rgbs = np.stack([hex_to_rgb01(h) for h in hex_colors], axis=0)
    labs = rgb_to_oklab(rgbs)

    def f(t: np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
        lab = catmull_rom(labs, t)
        rgb = oklab_to_rgb(lab)
        return np.clip(rgb, 0.0, 1.0)

    return f


# ==========================================
# Semantic grouping + "less->more" ordering
# ==========================================

@dataclass(frozen=True)
class GroupResult:
    ordered_hex: List[str]
    gradient: Callable[[np.ndarray], np.ndarray]


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


def order_dark_to_light(hex_list: List[str]) -> List[str]:
    """
    Order by OKLab L component (perceptual lightness).
    """
    labs = np.stack([rgb_to_oklab(hex_to_rgb01(h)) for h in hex_list], axis=0)
    L = labs[:, 0]
    idx = np.argsort(L)
    return [hex_list[i] for i in idx]


def build_group_gradients(palette_hex: List[str]) -> Dict[str, GroupResult]:
    grouped: Dict[str, List[str]] = {"cool": [], "hot": [], "happy": []}
    for hx in palette_hex:
        grouped[classify_semantic_group(hx)].append(hx)

    results: Dict[str, GroupResult] = {}
    for name, hx_list in grouped.items():
        ordered = order_dark_to_light(hx_list)
        results[name] = GroupResult(
            ordered_hex=ordered,
            gradient=make_oklab_gradient(ordered),
        )
    return results


# =========================
# Gradient showcase render
# =========================

def render_gradient_rows(
    gradients: List[Callable[[np.ndarray], np.ndarray]],
    width: int = 1200,
    row_height: int = 120,
) -> Image.Image:
    t = np.linspace(0.0, 1.0, width, dtype=np.float64)

    rows = []
    for grad in gradients:
        rgb = grad(t)  # (W,3) in [0,1]
        strip = (rgb * 255.0).astype(np.uint8)[None, :, :]   # (1,W,3)
        strip = np.repeat(strip, row_height, axis=0)         # (H,W,3)
        rows.append(strip)

    img_arr = np.vstack(rows)  # (3*H, W, 3)
    return Image.fromarray(img_arr)


def main() -> None:
    groups = build_group_gradients(PALETTE_HEX)

    # Print grouped colors in the final dark->light order
    print("\nGrouped palette (dark -> light):\n")
    for name in ["cool", "hot", "happy"]:
        print(f"{name.upper()}:")
        for hx in groups[name].ordered_hex:
            print(f"  {hx}")
        print()

    # Render showcase image (3 rows: cool / hot / happy)
    img = render_gradient_rows(
        gradients=[groups["cool"].gradient, groups["hot"].gradient, groups["happy"].gradient],
        width=1200,
        row_height=120,
    )
    out_path = str(Path(__file__).with_name("three_gradients_oklab.png"))
    img.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
