"""Prompt generation strategies.

MitoSAM is promptable (SAM), so the same network can be driven in three ways:

* ``random``  — random foreground/background points sampled from the GT mask
                (used during fine-tuning, exactly as in the thesis).
* ``grid``    — a dense point grid giving full-coverage "automatic" masks.
* ``box``     — bounding-box prompts from a coarse detector / manual ROI.

Keeping prompts in one module makes the training objective and the inference
path share the same code — a common source of train/test mismatch.
"""

from __future__ import annotations

import numpy as np


def random_points_from_mask(
    mask: np.ndarray, num_points: int = 10, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``num_points`` points, half foreground and half background.

    Returns ``(coords[N,2] as (x, y), labels[N])`` with 1 = object, 0 = background.
    Falls back to the image centre if the mask is empty.
    """
    rng = rng or np.random.default_rng()
    mask_bool = np.asarray(mask) > 0.5
    h, w = mask_bool.shape
    fg = np.argwhere(mask_bool)          # (y, x)
    bg = np.argwhere(~mask_bool)

    coords: list[list[int]] = []
    labels: list[int] = []
    n_fg = num_points // 2

    if len(fg):
        picks = rng.choice(len(fg), size=min(n_fg, len(fg)), replace=len(fg) < n_fg)
        for y, x in fg[picks]:
            coords.append([int(x), int(y)])
            labels.append(1)
    if len(bg):
        need = num_points - len(coords)
        picks = rng.choice(len(bg), size=min(need, len(bg)), replace=len(bg) < need)
        for y, x in bg[picks]:
            coords.append([int(x), int(y)])
            labels.append(0)
    if not coords:
        coords = [[w // 2, h // 2]]
        labels = [1]
    return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def grid_points(height: int, width: int, n_per_side: int = 32, margin: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Regular grid of positive prompts within an optional border margin."""
    xs = np.linspace(width * margin, width * (1 - margin), n_per_side)
    ys = np.linspace(height * margin, height * (1 - margin), n_per_side)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return grid.astype(np.float32), np.ones(len(grid), dtype=np.int64)


def box_to_points(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``[x0, y0, x1, y1]`` boxes to centre-point prompts."""
    box = np.asarray(box, dtype=np.float32).reshape(-1, 4)
    centres = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], axis=1)
    return centres, np.ones(len(centres), dtype=np.int64)


def scale_points(points: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Scale (x, y) coordinates — used when the image is resized before prompting."""
    out = np.asarray(points, dtype=np.float32).copy()
    out[:, 0] *= scale_x
    out[:, 1] *= scale_y
    return out
