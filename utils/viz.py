"""Plotting helpers shared by notebooks, CLI reports and the GUI.

The palette is colour-blind safe and stays legible on both light and dark
backgrounds, which matters because the same figures appear in the README,
the technical report and the slide deck.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIG_DIR = Path("results/figures")

# Okabe-Ito derived palette (colour-blind safe)
PALETTE = {
    "primary": "#0072B2",
    "secondary": "#D55E00",
    "accent": "#009E73",
    "warn": "#E69F00",
    "purple": "#CC79A7",
    "dark": "#333333",
    "grey": "#8C8C8C",
}

# Distinct colours for instance overlays
INSTANCE_COLORS = np.array([
    [0.00, 0.45, 0.70], [0.84, 0.37, 0.00], [0.00, 0.62, 0.45],
    [0.90, 0.62, 0.00], [0.80, 0.47, 0.65], [0.34, 0.71, 0.91],
    [0.94, 0.89, 0.26], [0.60, 0.60, 0.60], [0.10, 0.30, 0.60],
    [0.55, 0.20, 0.30],
])


def palette(key: str) -> str:
    return PALETTE.get(key, PALETTE["primary"])


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 0.45, 0.70), alpha: float = 0.45) -> np.ndarray:
    """Blend a binary mask over a grayscale image, returning a float RGB array."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        lo, hi = float(img.min()), float(img.max())
        img = (img - lo) / (hi - lo + 1e-8)
        rgb = np.repeat(img[..., None], 3, axis=2)
    else:
        rgb = img[..., :3].astype(np.float32)
        if rgb.max() > 1:
            rgb /= 255.0
    m = np.asarray(mask) > 0.5
    out = rgb.copy()
    out[m] = (1 - alpha) * rgb[m] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 1)


def style_axes(ax, title: str | None = None, grid: bool = True) -> None:
    """Consistent, desk-friendly axis styling."""
    if title:
        ax.set_title(title, fontsize=11, fontweight="semibold", color=PALETTE["dark"])
    if grid:
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#CCCCCC")
    ax.tick_params(colors="#555555", labelsize=8.5)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
