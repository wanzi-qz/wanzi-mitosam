"""One-command reproduction pipeline.

    python -m MitoSAM.pipelines.run_all --config configs/default.yaml \
        --image-dir examples/synthetic/images --mask-dir examples/synthetic/masks \
        --groups examples/synthetic/labels.csv --out results

Steps
-----
1. ``features``  — extract morphology / texture / spatial tables
2. ``classify``  — ablation over feature families x models (+ label-permutation control)
3. ``shap``      — global + per-family attribution figures
4. ``figures``   — ablation bars, ROC/PR, distributions, correlation map
5. ``qualitative`` — instance overlays for the visual story
6. ``report``    — rebuild ``results/metrics/summary.json`` consumed by the README

Every step is idempotent and skips gracefully when its inputs are missing, so
the same entry point works for the synthetic demo, for a new lab cohort, and
inside CI.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MitoSAMConfig
from ..utils.logging import get_logger

log = get_logger("pipeline")


def _step_features(cfg: MitoSAMConfig, out_dir: Path, groups: str | None) -> pd.DataFrame:
    from ..features.extract import build_feature_tables

    group_map = None
    if groups and Path(groups).exists():
        gt = pd.read_csv(groups)
        key = "image_id" if "image_id" in gt.columns else gt.columns[0]
        group_map = dict(zip(gt[key].astype(str), gt["group"].astype(str)))
    per_image, _ = build_feature_tables(cfg, out_dir / "tables", group_map)
    return per_image


def _step_classify(cfg: MitoSAMConfig, out_dir: Path, features_csv: Path,
                   permutation_control: bool = True) -> dict:
    from ..evaluation.classification import run_classification
    from ..evaluation.permutation_test import permutation_control as perm_control

    payload = run_classification(cfg, features_csv, out_dir)

    if permutation_control:
        try:
            null = perm_control(cfg, features_csv, n_permutations=50, out_dir=out_dir)
            payload["permutation_control"] = null
            log.info("permutation control: observed AUC=%.3f, shuffled %.3f +- %.3f (p=%.2f)",
                     null["observed_auc"], null["null_mean"], null["null_std"], null["p_value"])
        except Exception as exc:  # pragma: no cover
            log.warning("permutation control skipped: %s", exc)
    return payload


def _step_shap(cfg: MitoSAMConfig, out_dir: Path, features_csv: Path) -> dict | None:
    from ..evaluation.shap_explain import run

    try:
        return run(cfg, str(features_csv), out_dir=str(out_dir))
    except Exception as exc:  # pragma: no cover
        log.warning("SHAP skipped: %s", exc)
        return None


def _step_figures(out_dir: Path, features_csv: Path) -> list[Path]:
    from ..evaluation import figures

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    results_csv = out_dir / "tables" / "classification_results.csv"
    if results_csv.exists():
        paths.append(figures.plot_ablation(results_csv, fig_dir / "ablation_auc.png"))
    seg_csv = out_dir / "tables" / "segmentation_summary.csv"
    if seg_csv.exists():
        paths.append(figures.plot_segmentation_summary(seg_csv, fig_dir / "segmentation_summary.png"))
    paths.append(figures.plot_feature_distributions(features_csv, fig_dir / "feature_distributions.png"))
    paths.append(figures.plot_correlation_heatmap(features_csv, fig_dir / "feature_correlation.png"))
    return paths


def _step_qualitative(cfg: MitoSAMConfig, out_dir: Path, n: int = 3) -> list[Path]:
    from ..evaluation.qualitative import build_qualitative_figures

    try:
        return build_qualitative_figures(cfg.data.image_dir, cfg.data.mask_dir,
                                         n_examples=n, out_dir=out_dir / "figures")
    except Exception as exc:  # pragma: no cover
        log.warning("qualitative figures skipped: %s", exc)
        return []


def _write_summary(out_dir: Path, payload: dict, fig_paths: list[Path]) -> Path:
    """Single machine-readable summary consumed by the README / report."""
    summary = {
        "dataset": {
            "n_images": payload.get("n_samples"),
            "n_features": payload.get("n_features_available"),
            "class_balance": payload.get("class_balance"),
        },
        "classification": {
            "best": payload.get("best", {}),
            "subset_summary": payload.get("subset_summary", []),
            "permutation_control": payload.get("permutation_control"),
        },
        "figures": [str(p.relative_to(out_dir)) for p in fig_paths],
    }
    path = out_dir / "metrics" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run(
    cfg: MitoSAMConfig,
    out_dir: str | Path = "results",
    groups: str | None = None,
    features_csv: str | Path | None = None,
    skip_shap: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    log.info("=== step 1/5 features ===")
    if features_csv and Path(features_csv).exists():
        per_image = pd.read_csv(features_csv)
        log.info("using precomputed features: %s", features_csv)
    else:
        per_image = _step_features(cfg, out_dir, groups)
        features_csv = out_dir / "tables" / "features_per_image.csv"
    features_csv = Path(features_csv)

    log.info("=== step 2/5 classification + ablation ===")
    payload = _step_classify(cfg, out_dir, features_csv)

    if not skip_shap:
        log.info("=== step 3/5 SHAP ===")
        payload["shap"] = _step_shap(cfg, out_dir, features_csv)

    log.info("=== step 4/5 figures ===")
    figs = _step_figures(out_dir, features_csv)

    log.info("=== step 5/5 qualitative overlays ===")
    figs += _step_qualitative(cfg, out_dir)

    summary_path = _write_summary(out_dir, payload, figs)
    log.info("summary -> %s", summary_path)
    return {"payload": payload, "figures": [str(p) for p in figs], "summary": str(summary_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the full MitoSAM analysis")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--groups", default=None, help="CSV with image_id,group")
    parser.add_argument("--features", default=None, help="reuse an existing feature table")
    parser.add_argument("--out", default="results")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    cfg = MitoSAMConfig.load(args.config)
    if args.image_dir:
        cfg.data.image_dir = args.image_dir
    if args.mask_dir:
        cfg.data.mask_dir = args.mask_dir

    run(cfg, args.out, args.groups, args.features, args.skip_shap)


if __name__ == "__main__":
    main()
