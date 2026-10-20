"""Feature extraction pipeline: masks + images -> tidy feature tables.

Two granularities are produced:

* **per-instance** (``features_per_mitochondrion.csv``) — one row per organelle,
  the substrate for single-organelle classification / clustering;
* **per-image** (``features_per_image.csv``) — one row per field of view,
  the substrate for the group-level (control vs. treatment) classifier.

Family selection is explicit so ablation studies ("morphology only", "texture +
spatial", …) reuse exactly the same extraction code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MitoSAMConfig, FeatureConfig
from ..utils.io import find_mask_for_image, list_images, load_image_gray, load_mask
from ..utils.logging import get_logger
from .morphology import MORPHOLOGY_GROUPS, aggregate_morphology, extract_morphology
from .spatial import extract_spatial
from .texture import extract_texture, first_order, glcm_features, lbp_features

log = get_logger("features")

FAMILY_PREFIXES = {
    "morphology": ("area", "perimeter", "aspect_ratio", "circularity", "solidity",
                   "eccentricity", "form_factor", "elongation", "compactness",
                   "roundness", "major_axis", "minor_axis", "equivalent_diameter",
                   "convex_area", "bbox_area", "area_fraction", "skeleton",
                   "branch", "network", "n_branches", "n_components", "mean_branch",
                   "n_mitochondria"),
    "texture": ("intensity_", "GLCM_", "LBP_"),
    "spatial": ("nn_", "voronoi_", "ripley_", "density", "perinuclear", "branch_point_density"),
}

META_COLS = ("image_id", "instance_id", "file_name", "label", "group")


def feature_family(column: str) -> str:
    """Map a column name to its feature family (used for ablation subsets)."""
    for family, prefixes in FAMILY_PREFIXES.items():
        if column.startswith(prefixes):
            return family
    if column in MORPHOLOGY_GROUPS["topology"]:
        return "morphology"
    return "other"


def extract_image_features(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: FeatureConfig,
    image_id: str,
) -> tuple[dict, pd.DataFrame]:
    """Per-field-of-view row + per-organelle table for one image pair."""
    families = set(cfg.families)
    row: dict[str, float] = {"image_id": image_id}

    morph_df = extract_morphology(
        mask, cfg.pixel_size_um, cfg.min_area, cfg.max_area, image_id=image_id
    )
    if "morphology" in families:
        row.update(aggregate_morphology(morph_df))
    if "texture" in families:
        img_f = image * 255.0 if image.max() <= 1.0 else image
        m = np.asarray(mask) > 0.5
        if m.any():
            row.update(first_order(img_f, m))
            row.update(glcm_features(img_f, m, levels=cfg.texture_bins, distances=(1, 2)))
            row.update(lbp_features(img_f, m, cfg.lbp_points, cfg.lbp_radius))
    if "spatial" in families:
        row.update(extract_spatial(
            mask, cfg.pixel_size_um, cfg.ripley_max_radius_px, cfg.ripley_steps,
            cfg.min_area, image_id=image_id,
        ))
    row.pop("image_id", None)
    row = {"image_id": image_id, **row}
    return row, morph_df


def build_feature_tables(
    cfg: MitoSAMConfig,
    out_dir: str | Path = "results/tables",
    group_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run extraction over the configured dataset and write both tables.

    Args:
        group_map: optional ``{image_stem: "control" | "treatment"}`` label map.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(cfg.data.image_dir)
    rows: list[dict] = []
    instances: list[pd.DataFrame] = []

    for img_path in images:
        mask_path = find_mask_for_image(img_path, cfg.data.mask_dir)
        if mask_path is None:
            log.warning("no mask for %s -> skipped", img_path.name)
            continue
        image = load_image_gray(img_path)
        mask = load_mask(mask_path)
        if mask.shape != image.shape:
            from ..data.dataset import resize_stack

            mask = resize_stack(mask.astype(np.float32), image.shape[0], is_mask=True) > 0.5
        row, inst = extract_image_features(image, mask, cfg.features, img_path.stem)
        row["file_name"] = img_path.name
        if group_map and img_path.stem in group_map:
            row["group"] = group_map[img_path.stem]
        rows.append(row)
        if not inst.empty:
            inst["file_name"] = img_path.name
            instances.append(inst)

    per_image = pd.DataFrame(rows)
    per_instance = pd.concat(instances, ignore_index=True) if instances else pd.DataFrame()

    per_image.to_csv(out_dir / "features_per_image.csv", index=False)
    if not per_instance.empty:
        per_instance.to_csv(out_dir / "features_per_mitochondrion.csv", index=False)

    log.info("features: %d images x %d columns | %d organelles",
             *per_image.shape, len(per_instance))
    return per_image, per_instance


def select_family_columns(df: pd.DataFrame, families: list[str]) -> list[str]:
    """Numeric columns belonging to the requested families."""
    numeric = df.select_dtypes(include=[np.number]).columns
    keep = []
    for col in numeric:
        if col in META_COLS:
            continue
        if feature_family(col) in families:
            keep.append(col)
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract mitochondrial features")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--groups", default=None, help="CSV with columns: image_id,group")
    parser.add_argument("--out", default="results/tables")
    args = parser.parse_args()

    cfg = MitoSAMConfig.load(args.config)
    if args.image_dir:
        cfg.data.image_dir = args.image_dir
    if args.mask_dir:
        cfg.data.mask_dir = args.mask_dir

    group_map = None
    if args.groups and Path(args.groups).exists():
        gt = pd.read_csv(args.groups)
        group_map = dict(zip(gt["image_id"].astype(str), gt["group"].astype(str)))

    build_feature_tables(cfg, args.out, group_map)


if __name__ == "__main__":
    main()
