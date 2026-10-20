"""Image / mask IO.

Fluorescence stacks come in wildly different containers, so everything is read
through PIL (16-bit PNG / TIFF are native there) and normalised defensively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # microscopy tiles can be huge

IMG_EXTS: tuple[str, ...] = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")


def list_images(directory: str | Path, exts: Sequence[str] = IMG_EXTS) -> list[Path]:
    """Return a sorted list of image files inside ``directory`` (non-recursive)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    files = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in exts]
    return files


def _to_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        # RGBA -> drop alpha, RGB -> luminance
        if array.shape[2] == 4:
            array = array[:, :, :3]
        return (0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2])
    raise ValueError(f"Unsupported image shape {array.shape}")


def load_image_gray(path: str | Path, as_float: bool = True) -> np.ndarray:
    """Load an image as a 2-D float32 array scaled to [0, 1].

    Percentile clipping (0.2 / 99.8) prevents a handful of saturated pixels from
    crushing the dynamic range, which matters for 12/16-bit fluorescence data.
    """
    arr = np.asarray(Image.open(path)).astype(np.float32)
    arr = _to_2d(arr)
    lo, hi = np.percentile(arr, [0.2, 99.8])
    if hi <= lo:
        hi = arr.max() if arr.max() > lo else lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr if as_float else (arr * 255).astype(np.uint8)


def load_mask(path: str | Path, threshold: float = 0.5) -> np.ndarray:
    """Load a binary mask as a bool array.

    Accepts 0/1 arrays, 0/255 arrays (threshold applied on the normalised scale)
    and multi-label instance masks (any non-zero becomes foreground).
    """
    arr = _to_2d(np.asarray(Image.open(path)).astype(np.float32))
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val <= 1.0:
        return arr >= threshold
    if max_val <= 255.0:
        return arr >= (threshold * 255.0)
    return arr >= (threshold * max_val)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale any numeric array into 0-255 uint8 for display / saving."""
    arr = np.asarray(image, dtype=np.float32)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi - lo < 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    """Save a boolean / {0,1} / label mask as an 8-bit PNG (0/255)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(mask)
    if arr.dtype == bool:
        arr = (arr.astype(np.uint8) * 255)
    elif arr.max() <= 1:
        arr = (arr.astype(np.uint8) * 255)
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return path


def find_mask_for_image(image_path: str | Path, mask_dir: str | Path) -> Path | None:
    """Resolve the mask that belongs to ``image_path``.

    Supports the three conventions seen in the lab dataset:
    ``<stem>.png``, ``<stem>_mask.png`` and ``<stem>-label.png``.
    """
    image_path, mask_dir = Path(image_path), Path(mask_dir)
    stem = image_path.stem
    candidates: Iterable[str] = (
        f"{stem}.png", f"{stem}.tif", f"{stem}.tiff",
        f"{stem}_mask.png", f"{stem}_mask.tif",
        f"{stem}-label.png", f"{stem}_label.png",
    )
    for name in candidates:
        candidate = mask_dir / name
        if candidate.exists():
            return candidate
    return None
