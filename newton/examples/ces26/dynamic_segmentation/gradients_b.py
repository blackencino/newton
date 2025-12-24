"""
Semantic color gradients for robotics telemetry display.

Provides three gradient functions that map a 0-1 float to RGB colors (0-1 float32):
  - gradient_cool(t): Blues/purples - for ground/background elements
  - gradient_hot(t):  Reds/pinks - for danger/avoid zones
  - gradient_happy(t): Greens/yellows - for interactive/safe elements
"""

import numpy as np

# =============================================================================
# Color Palette Definition (18 colors from 6x3 grid)
# =============================================================================

_PALETTE_HEX = {
    # Row 1 (Darker/Earthy)
    'R1C1': '#050533',  # Dark Navy
    'R1C2': '#280099',  # Deep Indigo
    'R1C3': '#660033',  # Dark Plum
    'R1C4': '#901010',  # Rust
    'R1C5': '#556B2F',  # Dark Olive
    'R1C6': '#32CD32',  # Lime Green
    # Row 2 (Middle)
    'R2C1': '#3B6EBA',  # Medium Blue
    'R2C2': '#7C82FA',  # Periwinkle
    'R2C3': '#E47AF5',  # Orchid
    'R2C4': '#FF97BE',  # Pink
    'R2C5': '#FDB15C',  # Orange
    'R2C6': '#FFDA3B',  # Yellow
    # Row 3 (Vibrant/Accents)
    'R3C1': '#004400',  # Forest Green
    'R3C2': '#40E0D0',  # Turquoise
    'R3C3': '#5D9CEC',  # Sky Blue
    'R3C4': '#AC92EB',  # Lavender
    'R3C5': '#C71585',  # Medium Violet Red
    'R3C6': '#DC143C',  # Crimson
}

# Semantic groupings
_GROUP_KEYS = {
    'cool':  ['R1C1', 'R1C2', 'R2C1', 'R3C3', 'R2C2', 'R3C4'],  # Blues/purples
    'hot':   ['R1C3', 'R1C4', 'R3C6', 'R3C5', 'R2C3', 'R2C4'],  # Reds/pinks
    'happy': ['R3C1', 'R1C5', 'R3C2', 'R1C6', 'R2C5', 'R2C6'],  # Greens/yellows
}


def _hex_to_rgb(h: str) -> np.ndarray:
    """Convert hex color string to RGB array (0-1 float32)."""
    h = h.lstrip('#')
    return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def _luminance(rgb: np.ndarray) -> float:
    """Compute perceptual luminance for sorting."""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _build_sorted_gradient(keys: list) -> np.ndarray:
    """Build a gradient array sorted dark-to-light."""
    rgbs = [_hex_to_rgb(_PALETTE_HEX[k]) for k in keys]
    rgbs.sort(key=_luminance)
    return np.array(rgbs, dtype=np.float32)


# Pre-compute the sorted gradient arrays
_GRADIENT_COOL = _build_sorted_gradient(_GROUP_KEYS['cool'])
_GRADIENT_HOT = _build_sorted_gradient(_GROUP_KEYS['hot'])
_GRADIENT_HAPPY = _build_sorted_gradient(_GROUP_KEYS['happy'])


def _interpolate_gradient(colors: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """
    Smoothly interpolate through a color array.
    
    Args:
        colors: Array of RGB colors, shape (N, 3), values in 0-1
        t: Input value(s) in range 0-1
        
    Returns:
        RGB color(s) as float32, shape (3,) for scalar t, or (len(t), 3) for array t
    """
    t = np.asarray(t, dtype=np.float32)
    t = np.clip(t, 0.0, 1.0)
    
    n = len(colors)
    pos = t * (n - 1)
    
    scalar_input = (t.ndim == 0)
    if scalar_input:
        pos = np.atleast_1d(pos)
    
    idx = np.floor(pos).astype(int)
    idx = np.clip(idx, 0, n - 2)
    frac = (pos - idx).astype(np.float32)
    
    c1 = colors[idx]
    c2 = colors[idx + 1]
    
    result = c1 + frac[..., np.newaxis] * (c2 - c1)
    
    if scalar_input:
        return result[0].astype(np.float32)
    return result.astype(np.float32)


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
    # CJH HACK
    t = (t ** 2.2) * 0.6
    return _interpolate_gradient(_GRADIENT_COOL, t)


def gradient_hot(t: float | np.ndarray) -> np.ndarray:
    """
    Hot/red gradient for danger and avoidance zones.
    
    Args:
        t: Value(s) in range 0-1. 0=dark plum, 1=light pink
        
    Returns:
        RGB color(s) as float32 in range 0-1
    """
    return _interpolate_gradient(_GRADIENT_HOT, t)


def gradient_happy(t: float | np.ndarray) -> np.ndarray:
    """
    Happy/green-yellow gradient for interactive and safe elements.
    
    Args:
        t: Value(s) in range 0-1. 0=dark forest, 1=bright yellow
        
    Returns:
        RGB color(s) as float32 in range 0-1
    """
    return _interpolate_gradient(_GRADIENT_HAPPY, t)


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
