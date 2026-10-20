"""Shared utilities: logging, device selection, image IO, metrics, plotting style."""

from .logging import get_logger, setup_logging
from .device import resolve_device
from .io import (
    list_images,
    load_image_gray,
    load_mask,
    normalize_to_uint8,
    save_mask,
    find_mask_for_image,
)
from .metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score,
    boundary_f1,
    instance_ap,
    evaluate_masks,
)
from .viz import overlay_mask, palette, style_axes, FIG_DIR

__all__ = [
    "get_logger",
    "setup_logging",
    "resolve_device",
    "list_images",
    "load_image_gray",
    "load_mask",
    "normalize_to_uint8",
    "save_mask",
    "find_mask_for_image",
    "dice_score",
    "iou_score",
    "precision_score",
    "recall_score",
    "boundary_f1",
    "instance_ap",
    "evaluate_masks",
    "overlay_mask",
    "palette",
    "style_axes",
    "FIG_DIR",
]
