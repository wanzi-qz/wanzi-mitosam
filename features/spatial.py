"""Spatial distribution features.

Morphology and texture describe *individual* organelles; spatial features
describe how the population is arranged in the cell, which is what reveals
fission/fusion imbalance (clustered small fragments vs. a spread-out network).

Implemented:
* density statistics (occupancy, mitochondria per unit cell area),
* nearest-neighbour distances (mean / CV / skew),
* Voronoi tessellation of centroids (area statistics, shape regularity),
* Ripley's K/L function with complete spatial randomness (CSR) reference,
* skeleton network descriptors (already in morphology, aggregated here).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree, Voronoi, QhullError

from .morphology import _skeleton_topology


def ripley_l(centroids: np.ndarray, area: float, max_radius: float, steps: int = 16) -> dict[str, float]:
    """Ripley's L(r) = sqrt(K(r)/pi) - r for a point set in a window of ``area``.

    L(r) > 0 indicates clustering, L(r) < 0 regularity/dispersion.
    """
    n = len(centroids)
    if n < 3 or area <= 0:
        return {"ripley_l_max": 0.0, "ripley_l_auc": 0.0, "ripley_cluster_index": 0.0,
                "spatial_pattern": "insufficient_points"}
    tree = cKDTree(centroids)
    radii = np.linspace(max_radius / steps, max_radius, steps)
    counts = np.array([len(tree.query_ball_point(c, r)) - 1 for c in centroids for r in radii])
    counts = counts.reshape(n, steps)
    k_obs = counts.sum(axis=0) / n * (area / n)
    l_obs = np.sqrt(np.maximum(k_obs, 0) / np.pi) - radii
    return {
        "ripley_l_max": float(l_obs.max()),
        "ripley_l_auc": float(np.trapezoid(l_obs, radii) / (radii[-1] - radii[0])),
        "ripley_cluster_index": float(l_obs.argmax() / steps),
        "spatial_pattern": "clustered" if l_obs.max() > 0.5 else "dispersed",
    }


def voronoi_features(centroids: np.ndarray, shape: tuple[int, int]) -> dict[str, float]:
    n = len(centroids)
    if n < 6:
        return {"voronoi_area_mean": 0.0, "voronoi_area_cv": 0.0, "voronoi_len_cv": 0.0}
    try:
        vor = Voronoi(centroids)
        regions = [r for r in vor.regions if r and -1 not in r]
        areas = []
        for region in regions:
            poly = vor.vertices[region]
            x, y = poly[:, 0], poly[:, 1]
            areas.append(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        areas_arr = np.asarray(areas, dtype=float)
        lengths = []
        for ridge_vertices in vor.ridge_vertices:
            if -1 in ridge_vertices:
                continue
            p1, p2 = ridge_vertices
            if p1 >= len(vor.vertices) or p2 >= len(vor.vertices):
                continue
            lengths.append(float(np.linalg.norm(vor.vertices[p1] - vor.vertices[p2])))
    except (QhullError, IndexError):
        # degenerate small/collinear point sets
        return {"voronoi_area_mean": 0.0, "voronoi_area_cv": 0.0, "voronoi_len_cv": 0.0}
    lengths_arr = np.asarray(lengths, dtype=float) if lengths else np.array([0.0])
    return {
        "voronoi_area_mean": float(areas_arr.mean()) if areas_arr.size else 0.0,
        "voronoi_area_cv": float(areas_arr.std() / areas_arr.mean()) if areas_arr.size and areas_arr.mean() else 0.0,
        "voronoi_len_cv": float(lengths_arr.std() / (lengths_arr.mean() + 1e-8)),
        "n_voronoi_cells": float(len(regions)),
    }


def nearest_neighbour(centroids: np.ndarray) -> dict[str, float]:
    n = len(centroids)
    if n < 2:
        return {"nn_mean": 0.0, "nn_cv": 0.0, "nn_skew": 0.0, "nn_min": 0.0}
    tree = cKDTree(centroids)
    dist, _ = tree.query(centroids, k=2)
    d = dist[:, 1]
    mean = float(d.mean())
    return {
        "nn_mean": mean,
        "nn_cv": float(d.std() / (mean + 1e-8)),
        "nn_skew": float(((d - mean) ** 3).mean() / (d.std() ** 3 + 1e-8)),
        "nn_min": float(d.min()),
        "nn_max": float(d.max()),
    }


def extract_spatial(
    mask: np.ndarray,
    pixel_size_um: float = 0.1,
    max_radius_px: float = 80.0,
    steps: int = 16,
    min_area: int = 50,
    image_id: str = "",
) -> dict[str, float]:
    """One row of spatial descriptors for a whole field of view."""
    from skimage import measure

    mask = np.asarray(mask) > 0.5
    if not mask.any():
        return {"image_id": image_id, "n_mitochondria": 0.0}

    labeled = measure.label(mask, connectivity=2)
    regions = [r for r in measure.regionprops(labeled) if r.area >= min_area]
    if not regions:
        return {"image_id": image_id, "n_mitochondria": 0.0}

    centroids = np.array([[r.centroid[1], r.centroid[0]] for r in regions], dtype=float)
    h, w = mask.shape
    intensity = ndi.gaussian_filter(mask.astype(np.float32), sigma=12)

    feats: dict[str, float] = {
        "image_id": image_id,
        "n_mitochondria": float(len(regions)),
        "area_density": float(mask.mean()),
        "number_density": float(len(regions) / (h * w) * 1e4),  # per 10k px
        "inner_density": float(intensity[ndi.binary_erosion(mask, iterations=8).astype(bool)].mean())
        if ndi.binary_erosion(mask, iterations=8).any() else 0.0,
        "outer_density": float(intensity[~mask].mean()) if (~mask).any() else 0.0,
        "perinuclear_ratio": float(mask.mean() / (mask.mean() + intensity[~mask].mean() + 1e-8)),
    }
    feats.update(nearest_neighbour(centroids))
    feats.update(voronoi_features(centroids, (h, w)))
    feats.update(ripley_l(centroids, area=float(mask.size), max_radius=max_radius_px, steps=steps))
    _, branch_points = _skeleton_topology(mask)
    feats["branch_point_density"] = float(branch_points / (h * w) * 1e4)
    return feats
