"""SHAP explainability for the mitochondrial state classifier.

Two artefacts are produced, and both matter:

* **global** — which features drive the decision across the cohort
  (summary beeswarm + bar), which is what goes in the report;
* **local** — why *this particular* cell was called abnormal (waterfall), which
  is what a biologist asks in front of the GUI.

Tree models use the exact TreeExplainer; everything else falls back to a
permutation-based explainer so the code path never silently produces garbage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MitoSAMConfig
from ..features.extract import feature_family
from ..utils.logging import get_logger
from ..utils.viz import FIG_DIR, palette, style_axes
from .classification import load_feature_table, numeric_feature_columns
from .feature_sets import resolve_subset
from .models import build_model, model_display_name

log = get_logger("eval.shap")

MODEL_TITLES = {
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def compute_shap(
    cfg: MitoSAMConfig,
    features_path: str | Path = "results/tables/features_per_image.csv",
    model_key: str = "random_forest",
    subset: str = "full_features",
    max_samples: int | None = None,
) -> dict:
    """Fit the pipeline on all data and return SHAP values + the fitted pieces."""
    import shap
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df, label_col = load_feature_table(features_path)
    if label_col is None:
        raise ValueError("Feature table has no label column")
    y = df[label_col].astype(int)
    cols = resolve_subset(subset, numeric_feature_columns(df))
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(df[cols].median(numeric_only=True))

    k = min(cfg.evaluation.n_features, len(cols))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", build_model(model_key, cfg.evaluation.random_state)),
    ]).fit(X, y)

    mask = pipe.named_steps["select"].get_support()
    selected = list(np.asarray(cols)[mask])
    X_sel = pd.DataFrame(pipe.named_steps["scaler"].transform(X)[:, mask], columns=selected)
    clf = pipe.named_steps["clf"]

    try:
        explainer = shap.TreeExplainer(clf)
        raw = explainer.shap_values(X_sel)
        if isinstance(raw, list):
            values = raw[1]
        elif getattr(raw, "ndim", 0) == 3:
            values = raw[:, :, 1]
        else:
            values = raw
        expected = explainer.expected_value
        base_value = float(expected[1] if isinstance(expected, (list, np.ndarray)) else expected)
    except Exception:  # non-tree model
        explainer = shap.Explainer(clf.predict_proba, X_sel, algorithm="permutation")
        explanation = explainer(X_sel)
        values = explanation.values[:, :, 1] if explanation.values.ndim == 3 else explanation.values
        base_value = float(np.ravel(explanation.base_values)[0])

    n = max_samples or cfg.evaluation.shap_samples
    if len(X_sel) > n:
        rng = np.random.default_rng(cfg.evaluation.random_state)
        idx = rng.choice(len(X_sel), size=n, replace=False)
        X_sel, values = X_sel.iloc[idx], values[idx]

    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=selected).sort_values(ascending=False)
    family_totals = mean_abs.groupby(mean_abs.index.map(feature_family)).sum()
    family_share = (family_totals / family_totals.sum()).round(4).to_dict()

    return {
        "X": X_sel,
        "values": values,
        "selected": selected,
        "base_value": base_value,
        "mean_abs": mean_abs,
        "family_share": family_share,
        "model_key": model_key,
        "model_name": model_display_name(model_key),
        "pipeline": pipe,
        "y": y.loc[X_sel.index] if len(X_sel) < len(y) else y,
    }


def plot_shap(result: dict, out_dir: str | Path = FIG_DIR) -> list[Path]:
    """Beeswarm + bar summary + family contribution pie, saved as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})

    shap.summary_plot(result["values"], result["X"], show=False, max_display=15)
    fig = plt.gcf()
    fig.set_size_inches(7.6, 5.2)
    path = out_dir / "shap_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    top = result["mean_abs"].head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = [palette({"morphology": "primary", "texture": "secondary",
                       "spatial": "accent"}.get(feature_family(n), "grey")) for n in top.index]
    ax.barh(top.index, top.values, color=colors, height=0.72)
    for y_pos, val in enumerate(top.values):
        ax.text(val, y_pos, f" {val:.3f}", va="center", fontsize=7.5, color="#444444")
    style_axes(ax, f"Top-15 features by mean |SHAP| — {result['model_name']}", grid=False)
    ax.set_xlabel("mean |SHAP value| (impact on predicted probability)")
    fig.tight_layout()
    path = out_dir / "shap_bar.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    shares = result["family_share"]
    if shares:
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        labels = list(shares.keys())
        ax.pie([shares[k] for k in labels], labels=[f"{k}\n{v:.0%}" for k, v in shares.items()],
               colors=[palette({"morphology": "primary", "texture": "secondary",
                                "spatial": "accent"}.get(k, "grey")) for k in labels],
               startangle=90, wedgeprops={"width": 0.42, "edgecolor": "white"})
        ax.set_title("SHAP contribution by feature family", fontsize=10, fontweight="semibold")
        fig.tight_layout()
        path = out_dir / "shap_family_share.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    log.info("SHAP figures -> %s", out_dir)
    return paths


def explain_single(result: dict, index: int = 0, out_dir: str | Path = FIG_DIR) -> Path:
    """Waterfall plot for one sample (used by the GUI 'why this cell?' panel)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    explanation = shap.Explanation(
        values=result["values"][index],
        base_values=result["base_value"],
        data=result["X"].iloc[index].values,
        feature_names=result["X"].columns.tolist(),
    )
    shap.plots.waterfall(explanation, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7.4, 4.8)
    path = out_dir / "shap_waterfall.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def run(cfg: MitoSAMConfig, features_path: str, out_dir: str = "results") -> dict:
    out_dir = Path(out_dir)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    result = compute_shap(cfg, features_path)
    plot_shap(result, out_dir / "figures")
    payload = {
        "model": result["model_name"],
        "base_value": result["base_value"],
        "family_share": result["family_share"],
        "top_features": {k: round(float(v), 4) for k, v in result["mean_abs"].head(20).items()},
    }
    (out_dir / "metrics" / "shap_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="SHAP analysis for MitoSAM classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--features", default="results/tables/features_per_image.csv")
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--subset", default="full_features")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    cfg = MitoSAMConfig.load(args.config)
    run(cfg, args.features, args.out)
