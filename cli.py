"""MitoSAM command-line interface.

    mitosam features  --image-dir ... --mask-dir ...
    mitosam classify  --features ... --groups ...
    mitosam shap      --features ...
    mitosam report    --out results
    mitosam demo-data --out examples/synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg(args):
    from .config import MitoSAMConfig

    cfg = MitoSAMConfig.load(getattr(args, "config", None))
    if getattr(args, "image_dir", None):
        cfg.data.image_dir = args.image_dir
    if getattr(args, "mask_dir", None):
        cfg.data.mask_dir = args.mask_dir
    return cfg


def cmd_features(args) -> int:
    from .features.extract import build_feature_tables

    group_map = None
    if args.groups and Path(args.groups).exists():
        import pandas as pd

        gt = pd.read_csv(args.groups)
        key = "image_id" if "image_id" in gt.columns else gt.columns[0]
        group_map = dict(zip(gt[key].astype(str), gt["group"].astype(str)))
    per_image, _ = build_feature_tables(_cfg(args), args.out, group_map)
    print(f"OK: {per_image.shape[0]} images x {per_image.shape[1]} features -> {args.out}")
    return 0


def cmd_classify(args) -> int:
    from .evaluation.classification import run_classification

    payload = run_classification(_cfg(args), args.features, args.out, label_col=args.label)
    best = payload["best"]
    print(f"OK: best={best.get('feature_subset')} test_AUC={best.get('test_auc'):.3f} "
          f"cv_AUC={best.get('cv_auc_mean'):.3f}")
    return 0


def cmd_shap(args) -> int:
    from .evaluation.shap_explain import run

    payload = run(_cfg(args), args.features, args.out)
    print("OK: top features:", ", ".join(list(payload["top_features"])[:5]))
    return 0


def cmd_report(args) -> int:
    """Regenerate every figure from existing result tables."""
    from .evaluation import figures

    out = Path(args.out)
    made = []
    results_csv = out / "tables" / "classification_results.csv"
    if results_csv.exists():
        made.append(figures.plot_ablation(results_csv, out / "figures" / "ablation_auc.png"))
    seg_csv = out / "tables" / "segmentation_summary.csv"
    if seg_csv.exists():
        made.append(figures.plot_segmentation_summary(seg_csv, out / "figures" / "segmentation_summary.png"))
    feats_csv = out / "tables" / "features_per_image.csv"
    if feats_csv.exists():
        made.append(figures.plot_feature_distributions(feats_csv, out / "figures" / "feature_distributions.png"))
        made.append(figures.plot_correlation_heatmap(feats_csv, out / "figures" / "feature_correlation.png"))
    hist = Path("weights/train_history.json")
    if hist.exists():
        made.append(figures.plot_training_curves(hist, out / "figures" / "training_curves.png"))
    print("OK: wrote", len(made), "figures")
    return 0


def cmd_demo_data(args) -> int:
    from .data.synthetic import create_cohort

    path = create_cohort(args.out, args.n, (args.size, args.size), tuple(args.classes), args.seed)
    print(f"OK: synthetic cohort at {path} (demo only — not real data)")
    return 0


def cmd_permutation(args) -> int:
    from .evaluation.permutation_test import permutation_control

    payload = permutation_control(_cfg(args), args.features, args.n, out_dir=args.out)
    print(f"OK: observed={payload['observed_auc']} null={payload['null_mean']}±{payload['null_std']} "
          f"p={payload['p_value']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mitosam", description="MitoSAM toolkit")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("features", help="extract morphology/texture/spatial features")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--groups", default=None, help="CSV: image_id,group")
    p.add_argument("--out", default="results/tables")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("classify", help="ablation + models over a feature table")
    p.add_argument("--features", default="results/tables/features_per_image.csv")
    p.add_argument("--groups", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("permutation", help="label-permutation control")
    p.add_argument("--features", default="results/tables/features_per_image.csv")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_permutation)

    p = sub.add_parser("shap", help="SHAP explanation figures")
    p.add_argument("--features", default="results/tables/features_per_image.csv")
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_shap)

    p = sub.add_parser("report", help="rebuild all figures from result tables")
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("demo-data", help="generate the synthetic demo cohort")
    p.add_argument("--out", default="examples/synthetic")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--classes", nargs="+", default=["control", "treated"])
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_demo_data)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
