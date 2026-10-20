"""Morphology features from a binary mask.

Per connected mitochondrion: size, shape descriptors and network topology.
These are the quantities a biologist actually reports (area, aspect ratio,
circularity, branch counts) and they are cheap enough to run on thousands of
organelles.

Note on axis convention: ``skimage`` reports ``(row, col)`` = ``(y, x)``; all
coordinates exposed by MitoSAM are converted to ``(x, y)`` at the boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure
from skimage.morphology import skeletonize

# Feature groups used for ablation / SHAP grouping downstream
MORPHOLOGY_GROUPS = {
    "size": ["area", "perimeter", "equivalent_diameter", "major_axis_length",
             "minor_axis_length", "convex_area", "bbox_area", "area_fraction"],
    "shape": ["aspect_ratio", "circularity", "solidity", "eccentricity",
              "form_factor", "elongation", "compactness", "roundness"],
    "topology": ["skeleton_length", "n_branches", "n_branch_points", "n_end_points",
                 "mean_branch_length", "network_complexity", "n_components"],
}


def _safe_ratio(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b not in (0, 0.0) and np.isfinite(b) else default


def _prop(region, new_name: str, old_name: str) -> float:
    """Read a regionprops attribute across scikit-image versions."""
    value = getattr(region, new_name, None)
    if value is None:
        value = getattr(region, old_name)
    return float(value)


def extract_morphology(
    mask: np.ndarray,
    pixel_size_um: float = 0.1,
    min_area: int = 50,
    max_area: int = 12_000,
    image_id: str = "",
) -> pd.DataFrame:
    """Extract one row per mitochondrion.

    Args:
        mask: binary mask (bool or 0/1), HxW.
        pixel_size_um: physical pixel pitch; areas are reported in µm².
        min_area / max_area: size gates in **pixels** to drop debris and clumps.
    """
    mask = np.asarray(mask) > 0.5
    labeled = measure.label(mask, connectivity=2)
    regions = [r for r in measure.regionprops(labeled) if min_area <= r.area <= max_area]
    total_px = float(mask.sum())

    skel = skeletonize(mask) if mask.any() else np.zeros_like(mask)
    skel_len = float(skel.sum())
    n_end_points, n_branch_points = _skeleton_topology(skel)

    rows: list[dict] = []
    for idx, region in enumerate(regions, start=1):
        perim = float(region.perimeter) or 1.0
        area = float(region.area)
        major = float(_prop(region, "axis_major_length", "major_axis_length"))
        minor = float(_prop(region, "axis_minor_length", "minor_axis_length"))
        y, x = region.centroid

        rows.append({
            "image_id": image_id,
            "instance_id": idx,
            "area": area * pixel_size_um ** 2,
            "perimeter": perim * pixel_size_um,
            "equivalent_diameter": float(_prop(region, "equivalent_diameter_area",
                                               "equivalent_diameter")) * pixel_size_um,
            "major_axis_length": major * pixel_size_um,
            "minor_axis_length": minor * pixel_size_um,
            "convex_area": float(_prop(region, "area_convex", "convex_area")) * pixel_size_um ** 2,
            "bbox_area": float(_prop(region, "area_bbox", "bbox_area")) * pixel_size_um ** 2,
            "area_fraction": _safe_ratio(area, total_px),
            "aspect_ratio": _safe_ratio(major, minor, 1.0),
            "circularity": float(4 * np.pi * area / perim ** 2),
            "roundness": float(4 * area / (np.pi * major ** 2)) if major > 0 else 0.0,
            "solidity": float(region.solidity),
            "eccentricity": float(region.eccentricity),
            "form_factor": float(region.feret_diameter_max / perim) if perim > 0 else 0.0,
            "elongation": _safe_ratio(major, minor, 1.0) - 1.0,
            "compactness": _safe_ratio(perim ** 2, 4 * np.pi * area),
            "centroid_x": float(x) * pixel_size_um,
            "centroid_y": float(y) * pixel_size_um,
            "skeleton_length": skel_len * pixel_size_um,
            "n_branches": int(n_end_points + n_branch_points),
            "n_branch_points": int(n_branch_points),
            "n_end_points": int(n_end_points),
            "mean_branch_length": _safe_ratio(skel_len, max(n_branch_points + n_end_points, 1)),
            "network_complexity": _safe_ratio(skel_len, perim),
            "n_components": int(labeled.max()),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame({c: [] for c in [
            "image_id", "instance_id", "area", "perimeter", "equivalent_diameter",
            "major_axis_length", "minor_axis_length", "convex_area", "bbox_area",
            "area_fraction", "aspect_ratio", "circularity", "roundness", "solidity",
            "eccentricity", "form_factor", "elongation", "compactness", "centroid_x",
            "centroid_y", "skeleton_length", "n_branches", "n_branch_points",
            "n_end_points", "mean_branch_length", "network_complexity", "n_components"]})
    return df


def _skeleton_topology(skel: np.ndarray) -> tuple[int, int]:
    """Count skeleton end-points and branch-points via 8-neighbour analysis."""
    if not skel.any():
        return 0, 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbours = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant")
    end_points = int(np.logical_and(skel, neighbours == 1).sum())
    branch_points = int(np.logical_and(skel, neighbours >= 3).sum())
    return end_points, branch_points


def aggregate_morphology(df: pd.DataFrame) -> dict[str, float]:
    """Field-of-view level summary (mean / std / count of the core descriptors)."""
    if df.empty:
        return {}
    out: dict[str, float] = {"n_mitochondria": float(len(df))}
    for col in ["area", "perimeter", "aspect_ratio", "circularity", "solidity",
                "eccentricity", "skeleton_length", "network_complexity", "area_fraction"]:
        if col in df:
            out[f"{col}_mean"] = float(df[col].mean())
            out[f"{col}_std"] = float(df[col].std(ddof=0))
            out[f"{col}_cv"] = float(df[col].std(ddof=0) / df[col].mean()) if df[col].mean() else 0.0
    out["n_components"] = float(df["n_components"].iloc[0])
    return out
