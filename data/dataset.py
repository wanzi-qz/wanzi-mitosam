"""Datasets and patching utilities for mitochondrial segmentation.

Two dataset classes are provided:

* :class:`MitoPatchDataset` — random patches for fine-tuning (thesis setting).
* :class:`MitoImageDataset` — whole images for validation / inference.

Both accept 16-bit PNG/TIFF, 8-bit PNG and RGB inputs and return float32
tensors in [0, 1], so no upstream conversion step is required.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import numpy as np

from ..utils.io import find_mask_for_image, list_images, load_image_gray, load_mask
from ..utils.logging import get_logger

log = get_logger("data")

try:  # torch is optional for feature-only usage
    import torch
    from torch.utils.data import Dataset

    _TORCH = True
except ImportError:  # pragma: no cover
    Dataset = object  # type: ignore
    _TORCH = False


def pair_images_and_masks(image_dir: str | Path, mask_dir: str | Path) -> list[tuple[Path, Path]]:
    """Return ``[(image, mask), ...]`` for every image that has a matching mask."""
    pairs: list[tuple[Path, Path]] = []
    for img in list_images(image_dir):
        mask = find_mask_for_image(img, mask_dir)
        if mask is not None:
            pairs.append((img, mask))
    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs found.\n  images: {image_dir}\n  masks:  {mask_dir}"
        )
    log.info("paired %d images with masks", len(pairs))
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]], val_split: float = 0.2, seed: int = 42
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Deterministic image-level split (never splits patches of the same FOV)."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_split))) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def random_patch(
    image: np.ndarray, mask: np.ndarray, size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a random patch, padding with zeros if the image is smaller than ``size``."""
    h, w = image.shape[:2]
    if h < size or w < size:
        pad_h, pad_w = max(0, size - h), max(0, size - w)
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
        h, w = image.shape[:2]
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    return image[y:y + size, x:x + size], mask[y:y + size, x:x + size]


def resize_stack(arr: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
    """Resize a 2-D array to ``size x size`` using cv2 when available."""
    try:
        import cv2

        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(arr.astype(np.float32), (size, size), interpolation=interp)
    except ImportError:
        # nearest-neighbour fallback (no dependency)
        idx_y = (np.linspace(0, arr.shape[0] - 1, size)).astype(int)
        idx_x = (np.linspace(0, arr.shape[1] - 1, size)).astype(int)
        return arr[np.ix_(idx_y, idx_x)].astype(np.float32)


def augment_pair(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Light, geometry-only augmentation (flips / 90° rotations / gamma).

    Mitochondria have no canonical orientation, so only label-preserving
    transforms are used — no colour jitter that would corrupt intensity features.
    """
    k = int(rng.integers(0, 4))
    if k:
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)
    if rng.random() < 0.5:
        image, mask = np.fliplr(image), np.fliplr(mask)
    if rng.random() < 0.5:
        image, mask = np.flipud(image), np.flipud(mask)
    if rng.random() < 0.3:  # intensity gamma jitter (0.8 - 1.25)
        gamma = float(rng.uniform(0.8, 1.25))
        image = np.clip(image, 0, 1) ** gamma
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class MitoPatchDataset(Dataset):  # type: ignore[misc]
    """Yields ``(image[3,H,W], mask[1,h,w])`` patch pairs for fine-tuning."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        patch_size: int = 512,
        samples_per_image: int = 16,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        if not _TORCH:
            raise ImportError("PyTorch is required for MitoPatchDataset")
        self.pairs = pairs
        self.patch_size = patch_size
        self.samples = max(1, samples_per_image)
        self.augment = augment
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.pairs) * self.samples

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if idx not in self._cache:
            img_path, mask_path = self.pairs[idx]
            self._cache[idx] = (load_image_gray(img_path), load_mask(mask_path))
        return self._cache[idx]

    def __getitem__(self, index: int):
        img_idx = index % len(self.pairs)
        image, mask = self._load(img_idx)
        image, mask = random_patch(image, mask, self.patch_size, self._rng)
        if self.augment:
            image, mask = augment_pair(image, mask, self._rng)

        image_t = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0).repeat(3, 1, 1)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        return image_t, mask_t


class MitoImageDataset(Dataset):  # type: ignore[misc]
    """Yields full images resized to ``input_size`` (for validation/inference)."""

    def __init__(self, pairs: list[tuple[Path, Path]], input_size: int = 1024) -> None:
        if not _TORCH:
            raise ImportError("PyTorch is required for MitoImageDataset")
        self.pairs = pairs
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        img_path, mask_path = self.pairs[index]
        image = resize_stack(load_image_gray(img_path), self.input_size)
        mask = resize_stack(load_mask(mask_path).astype(np.float32), self.input_size, is_mask=True) > 0.5
        image_t = torch.from_numpy(image).float().unsqueeze(0).repeat(3, 1, 1)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return image_t, mask_t, img_path.name
