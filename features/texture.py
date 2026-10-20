"""Texture features: first-order statistics, GLCM and LBP.

Texture is where mitochondrial *inner* state shows up: cristae density and
ordering change the grey-level statistics long before the organelle changes
shape. All features are computed **inside the mask**, and GLCM is computed on a
masked ROI with the background filled in, so neighbouring bright objects do not
leak into the statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def _quantize(roi: np.ndarray, levels: int) -> np.ndarray:
    lo, hi = float(roi.min()), float(roi.max())
    if hi - lo < 1e-8:
        return np.zeros(roi.shape, dtype=np.uint8)
    norm = (roi - lo) / (hi - lo) * (levels - 1)
    return norm.astype(np.uint8)


def first_order(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = image[mask]
    if values.size == 0:
        return {}
    feats = {
        "intensity_mean": float(np.mean(values)),
        "intensity_std": float(np.std(values)),
        "intensity_entropy": float(shannon_entropy(values)),
        "intensity_cv": float(np.std(values) / (np.mean(values) + 1e-8)),
        "intensity_range": float(np.ptp(values)),
    }
    if values.size >= 3:
        feats["intensity_skewness"] = float(stats.skew(values))
        feats["intensity_kurtosis"] = float(stats.kurtosis(values))
    else:
        feats["intensity_skewness"] = 0.0
        feats["intensity_kurtosis"] = 0.0
    for p in (10, 25, 50, 75, 90):
        feats[f"intensity_p{p}"] = float(np.percentile(values, p))
    return feats


def glcm_features(image: np.ndarray, mask: np.ndarray, levels: int = 64,
                  distances: tuple[int, ...] = (1, 2),
                  angles: tuple[float, ...] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)) -> dict[str, float]:
    """GLCM on the masked bounding box (background zero-filled)."""
    if not mask.any():
        return {}
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    box_img = image[r0:r1 + 1, c0:c1 + 1]
    box_mask = mask[r0:r1 + 1, c0:c1 + 1]
    if box_img.size < 4:
        return {}

    masked = np.where(box_mask, box_img, 0.0)
    quant = _quantize(masked, levels)
    glcm = graycomatrix(quant, distances=list(distances), angles=list(angles),
                        levels=levels, symmetric=True, normed=True)

    feats: dict[str, float] = {}
    for prop in GLCM_PROPS:
        values = graycoprops(glcm, prop)
        for i, d in enumerate(distances):
            for j, a in enumerate(angles):
                feats[f"GLCM_{prop}_d{d}_a{int(np.degrees(a))}"] = float(values[i, j])
    # rotation-invariant summary (mean over angles) — better for small cohorts
    for prop in GLCM_PROPS:
        values = graycoprops(glcm, prop).mean(axis=1)
        for i, d in enumerate(distances):
            feats[f"GLCM_{prop}_d{d}_mean"] = float(values[i])
    return feats


def lbp_features(image: np.ndarray, mask: np.ndarray, points: int = 24,
                 radius: int = 3) -> dict[str, float]:
    if not mask.any():
        return {}
    if image.max() <= 1.0:
        work = (image * 255).astype(np.uint8)
    else:
        work = np.clip(image, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(work, points, radius, method="uniform")
    values = lbp[mask]
    if values.size == 0:
        return {}
    n_bins = points + 2
    hist, _ = np.histogram(values, bins=n_bins, range=(0, n_bins))
    prob = hist.astype(float) / (hist.sum() + 1e-8)
    return {
        "LBP_mean": float(np.mean(values)),
        "LBP_std": float(np.std(values)),
        "LBP_entropy": float(-np.sum(prob * np.log2(prob + 1e-8))),
        "LBP_energy": float(np.sum(prob ** 2)),
        "LBP_skewness": float(stats.skew(prob)),
        "LBP_kurtosis": float(stats.kurtosis(prob)),
        "LBP_uniformity": float(np.max(prob)),
    }


def extract_texture(image: np.ndarray, mask: np.ndarray, levels: int = 64,
                    lbp_points: int = 24, lbp_radius: int = 3,
                    image_id: str = "", per_instance: bool = False) -> dict | pd.DataFrame:
    """Whole-ROI texture, or per-organelle texture when ``per_instance``."""
    image = np.asarray(image, dtype=np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    mask = np.asarray(mask) > 0.5
    if not mask.any():
        return pd.DataFrame() if per_instance else {}

    if not per_instance:
        feats = {"image_id": image_id}
        feats.update(first_order(image, mask))
        feats.update(glcm_features(image, mask, levels=levels))
        feats.update(lbp_features(image, mask, lbp_points, lbp_radius))
        return feats

    from skimage import measure

    labeled = measure.label(mask, connectivity=2)
    rows = []
    for region in measure.regionprops(labeled):
        sub = labeled == region.label
        if region.area < 30:
            continue
        feats = {"image_id": image_id, "instance_id": int(region.label)}
        feats.update(first_order(image, sub))
        feats.update(glcm_features(image, sub, levels=levels, distances=(1,)))
        feats.update(lbp_features(image, sub, lbp_points, lbp_radius))
        rows.append(feats)
    return pd.DataFrame(rows)
