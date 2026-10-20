"""Segmentation metrics.

Pixel metrics (Dice / IoU / precision / recall) are what the thesis reported;
the boundary-F1 and instance-AP scores are added because mitochondrial networks
are thin and highly branched, so a 1-pixel shift can look like a total miss to
pixel-wise metrics while being visually correct.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0.5


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred, gt = _as_bool(pred), _as_bool(gt)
    inter = np.logical_and(pred, gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred, gt = _as_bool(pred), _as_bool(gt)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


def precision_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred, gt = _as_bool(pred), _as_bool(gt)
    tp = np.logical_and(pred, gt).sum()
    return float((tp + eps) / (pred.sum() + eps))


def recall_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred, gt = _as_bool(pred), _as_bool(gt)
    tp = np.logical_and(pred, gt).sum()
    return float((tp + eps) / (gt.sum() + eps))


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = _as_bool(mask)
    eroded = ndi.binary_erosion(mask, iterations=1, border_value=0)
    return mask & ~eroded


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    """Boundary F1 with a pixel tolerance (the standard 2-px setting for EM)."""
    pred_b, gt_b = _boundary(pred), _boundary(gt)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0
    dt_gt = ndi.distance_transform_edt(~gt_b)
    dt_pred = ndi.distance_transform_edt(~pred_b)
    match_p = (dt_gt[pred_b] <= tolerance).mean()
    match_r = (dt_pred[gt_b] <= tolerance).mean()
    if match_p + match_r == 0:
        return 0.0
    return float(2 * match_p * match_r / (match_p + match_r))


def instance_ap(pred: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> float:
    """Single-threshold instance AP (greedy matching over connected components)."""
    pred_lab, n_pred = ndi.label(_as_bool(pred))
    gt_lab, n_gt = ndi.label(_as_bool(gt))
    if n_pred == 0 or n_gt == 0:
        return 0.0

    pred_objs = ndi.find_objects(pred_lab)
    gt_objs = ndi.find_objects(gt_lab)
    ious = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, sl_p in enumerate(pred_objs, start=1):
        if sl_p is None:
            continue
        for j, sl_g in enumerate(gt_objs, start=1):
            if sl_g is None:
                continue
            union_box = (
                slice(min(sl_p[0].start, sl_g[0].start), max(sl_p[0].stop, sl_g[0].stop)),
                slice(min(sl_p[1].start, sl_g[1].start), max(sl_p[1].stop, sl_g[1].stop)),
            )
            p = pred_lab[union_box] == i
            g = gt_lab[union_box] == j
            inter = np.logical_and(p, g).sum()
            if inter == 0:
                continue
            ious[i - 1, j - 1] = inter / np.logical_or(p, g).sum()

    matched, used = 0, set()
    for flat_idx in np.argsort(ious, axis=None)[::-1]:
        i, j = divmod(int(flat_idx), n_gt)
        if ious[i, j] < iou_thr:
            break
        if j in used:
            continue
        used.add(j)
        matched += 1
    precision = matched / n_pred
    recall = matched / n_gt
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def evaluate_masks(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """All metrics for one image pair."""
    return {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "precision": precision_score(pred, gt),
        "recall": recall_score(pred, gt),
        "boundary_f1": boundary_f1(pred, gt),
        "instance_f1@0.5": instance_ap(pred, gt),
    }
