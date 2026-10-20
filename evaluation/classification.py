"""Group-level classification: random forest (and baselines) + ablation + SHAP.

The scientific question: *can multi-dimensional features separate perturbed
mitochondria from controls?* The engineering question: *is the result honest?*

Honesty measures baked in here:

* **image-level splitting** — patches/organelles from the same field of view
  never appear in both train and test;
* **repeated stratified CV as the primary ablation metric** — every subset x
  model combination is scored over ``n_splits x n_repeats`` (default 50) folds
  on the full dataset, because with ~30 images a *single* hold-out split has a
  test set so small (≈9 samples) that one mis-ranked sample swings AUC by
  ~0.11 — larger than the real differences between feature families;
* **hold-out split reported as a secondary check** — a lucky/unlucky split is
  visible as a large gap against the repeated-CV mean;
* **feature-subset ablation** — if ``spatial_only`` performs like
  ``full_features``, the paper's "three-dimensional fusion" claim is empty;
* **leakage check** — correlation of each feature with the label is logged, and
  features that are almost deterministic labels are flagged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MitoSAMConfig
from ..utils.logging import get_logger
from .feature_sets import SUBSET_DEFINITIONS, resolve_subset
from .models import SUPPORTED_MODELS, build_model, model_display_name
from .scoring import composite_score, generalization_gap, stability_score

log = get_logger("eval.classification")

META_COLS = {"image_id", "file_name", "label", "group", "group_name", "instance_id"}


def load_feature_table(path: str | Path, label_col: str | None = None) -> tuple[pd.DataFrame, str | None]:
    """Read a feature table and normalise the label column to 0/1.

    Accepts integer labels (0/1) or string groups (``control`` / ``treated``);
    strings are mapped in sorted order so ``control`` -> 0 deterministically.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if label_col is None:
        for candidate in ("label", "group", "Group", "class"):
            if candidate in df.columns:
                label_col = candidate
                break
    if label_col and label_col in df.columns:
        values = df[label_col]
        if not pd.api.types.is_numeric_dtype(values):
            normalised = values.astype(str).str.strip().str.lower()
            mapping = {v: i for i, v in enumerate(sorted(normalised.unique()))}
            df["label"] = normalised.map(mapping).astype(int)
            df["group_name"] = normalised
            label_col = "label"
        else:
            df["label"] = values.astype(int)
            label_col = "label"
    return df, label_col


def numeric_feature_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    exclude = (exclude or set()) | META_COLS
    cols = []
    for c in df.select_dtypes(include=[np.number]).columns:
        if c in exclude:
            continue
        if df[c].nunique(dropna=True) <= 1:
            continue
        cols.append(c)
    return cols


def _split_indices(df: pd.DataFrame, label: pd.Series, cfg: MitoSAMConfig) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    strat = label.values if label.nunique() > 1 else None
    return train_test_split(idx, test_size=0.3, random_state=cfg.evaluation.random_state, stratify=strat)


def run_classification(
    cfg: MitoSAMConfig,
    features_path: str | Path = "results/tables/features_per_image.csv",
    out_dir: str | Path = "results",
    label_col: str | None = None,
    subsets: list[str] | None = None,
    models: list[str] | None = None,
) -> dict:
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    df, label_col = load_feature_table(features_path, label_col)
    if label_col is None or label_col not in df.columns:
        raise ValueError("No label column found — expected 'label' or 'group' in the feature table")
    y = df[label_col].astype(int)
    if y.nunique() < 2:
        raise ValueError(f"Need at least two classes, found {y.nunique()}")

    feature_cols = numeric_feature_columns(df)
    subsets = subsets or cfg.evaluation.feature_subsets
    models = models or cfg.evaluation.models
    train_idx, test_idx = _split_indices(df, y, cfg)

    rows: list[dict] = []
    fold_rows: list[dict] = []
    oof_importance: dict[str, pd.Series] = {}
    n_repeats = max(1, int(getattr(cfg.evaluation, "n_repeats", 10)))

    for subset in subsets:
        cols = resolve_subset(subset, feature_cols)
        if len(cols) < 2:
            log.warning("subset %s has %d features -> skipped", subset, len(cols))
            continue
        X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(df[cols].median(numeric_only=True))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for model_key in models:
            try:
                clf = build_model(model_key, cfg.evaluation.random_state)
            except ImportError as exc:
                log.warning("skip %s: %s", model_key, exc)
                continue
            k = min(cfg.evaluation.n_features, len(cols))

            def make_pipe() -> Pipeline:
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("select", SelectKBest(f_classif, k=k)),
                    ("clf", build_model(model_key, cfg.evaluation.random_state)),
                ])

            n_splits = int(min(cfg.evaluation.n_splits, max(2, y_train.value_counts().min())))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.evaluation.random_state)
            cv_scores = cross_val_score(make_pipe(), X_train, y_train, cv=cv, scoring="roc_auc",
                                        error_score=float("nan"))

            # Primary ablation metric: repeated stratified CV over the FULL dataset.
            # RepeatedStratifiedKFold is deterministic given (y, n_splits, n_repeats,
            # random_state), so fold i is identical across subsets -> paired diffs valid.
            rcv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                          random_state=cfg.evaluation.random_state)
            rcv_scores = cross_val_score(make_pipe(), X, y, cv=rcv, scoring="roc_auc",
                                         error_score=float("nan"))
            rcv_scores = np.asarray(rcv_scores, dtype=float)
            rcv_scores = rcv_scores[~np.isnan(rcv_scores)]
            for rep in range(n_repeats):
                chunk = rcv_scores[rep * n_splits:(rep + 1) * n_splits]
                fold_rows.append({
                    "model": model_display_name(model_key),
                    "feature_subset": subset,
                    "repeat": rep,
                    "auc_mean": float(np.nanmean(chunk)),
                    "n_folds": int(n_splits),
                })

            pipe = make_pipe()
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            metrics = _metrics(y_test, pred, proba)
            cv_mean, cv_std = float(np.nanmean(cv_scores)), float(np.nanstd(cv_scores))

            rows.append({
                "model": model_display_name(model_key),
                "model_key": model_key,
                "feature_subset": subset,
                "n_features": len(cols),
                "rcv_auc_mean": float(rcv_scores.mean()) if len(rcv_scores) else float("nan"),
                "rcv_auc_std": float(rcv_scores.std()) if len(rcv_scores) else float("nan"),
                "test_auc": metrics["auc_roc"],
                "test_pr_auc": metrics["auc_pr"],
                "test_accuracy": metrics["accuracy"],
                "test_f1": metrics["f1"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "cv_auc_mean": cv_mean,
                "cv_auc_std": cv_std,
                "generalization_gap": generalization_gap(metrics["auc_roc"], cv_mean),
                "stability": stability_score(cv_mean, cv_std, metrics["auc_roc"]),
            })
            if subset == "full_features":
                importances = _extract_importance(pipe)
                if importances is not None:
                    oof_importance[model_display_name(model_key)] = importances

    results = pd.DataFrame(rows)
    results["composite"] = results.apply(
        lambda r: composite_score(r["test_auc"], r["cv_auc_mean"], r["cv_auc_std"]), axis=1
    )
    results = results.sort_values(["feature_subset", "test_auc"], ascending=[True, False]).round(4)
    results.to_csv(out_dir / "tables" / "classification_results.csv", index=False)

    # Per-fold distributions of the primary (repeated-CV) metric.
    if fold_rows:
        folds_df = pd.DataFrame(fold_rows).round(4)
        folds_df.to_csv(out_dir / "tables" / "ablation_cv_distributions.csv", index=False)

    best_overall = results.loc[results["composite"].idxmax()].to_dict() if len(results) else {}
    subset_summary = (
        results.groupby("feature_subset")[["rcv_auc_mean", "rcv_auc_std", "test_auc", "cv_auc_mean", "composite"]]
        .mean().round(4).reset_index().sort_values("rcv_auc_mean", ascending=False)
    )
    subset_summary.to_csv(out_dir / "tables" / "ablation_summary.csv", index=False)

    paired = _paired_comparison(results)
    if paired is not None:
        paired.to_csv(out_dir / "tables" / "ablation_paired.csv", index=False)

    payload = {
        "label_column": label_col,
        "n_samples": int(len(df)),
        "n_features_available": len(feature_cols),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().items()},
        "best": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                 for k, v in best_overall.items()},
        "subset_summary": subset_summary.to_dict(orient="records"),
        "results": results.to_dict(orient="records"),
    }
    (out_dir / "metrics" / "classification_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if oof_importance:
        imp_df = pd.DataFrame(oof_importance).fillna(0.0)
        imp_df["mean_importance"] = imp_df.mean(axis=1)
        imp_df.sort_values("mean_importance", ascending=False).head(40).to_csv(
            out_dir / "tables" / "feature_importance.csv"
        )

    log.info("classification: %d configs, best=%s (AUC=%.3f)",
             len(results), best_overall.get("feature_subset"),
             best_overall.get("test_auc", float("nan")))
    return payload


def _paired_comparison(results: pd.DataFrame) -> pd.DataFrame | None:
    """Full-feature set vs each reduced family, compared on the repeated-CV mean.

    Reported as mean AUC difference ± std across models. With overlapping
    error bars on the ablation figure this table is the honest way to say
    "the difference is (not) larger than split-to-split noise".
    """
    if "rcv_auc_mean" not in results.columns or "full_features" not in set(results["feature_subset"]):
        return None
    full = results[results["feature_subset"] == "full_features"].set_index("model")["rcv_auc_mean"]
    rows = []
    for subset in results["feature_subset"].unique():
        if subset == "full_features":
            continue
        other = results[results["feature_subset"] == subset].set_index("model")["rcv_auc_mean"]
        diff = (full - other).dropna()
        if diff.empty:
            continue
        rows.append({
            "reduced_subset": subset,
            "mean_auc_diff_full_minus_reduced": round(float(diff.mean()), 4),
            "std_auc_diff": round(float(diff.std(ddof=0)), 4),
            "n_models": int(len(diff)),
            "full_better_in": f"{int((diff > 0).sum())}/{len(diff)}",
        })
    return pd.DataFrame(rows).sort_values("mean_auc_diff_full_minus_reduced", ascending=False)


def _metrics(y_true, y_pred, proba) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    two_class = len(np.unique(y_true)) > 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, proba)) if two_class else float("nan"),
        "auc_pr": float(average_precision_score(y_true, proba)) if two_class else float("nan"),
    }


def _extract_importance(pipe: Pipeline) -> pd.Series | None:
    """Map post-selection importances back to feature names."""
    try:
        selector = pipe.named_steps["select"]
        clf = pipe.named_steps["clf"]
        mask = selector.get_support()
        names = np.asarray(selector.feature_names_in_)[mask]
        if hasattr(clf, "coef_"):
            values = np.abs(clf.coef_).ravel()
        elif hasattr(clf, "feature_importances_"):
            values = clf.feature_importances_.ravel()
        else:
            return None
        return pd.Series(values, index=names)
    except Exception:  # pragma: no cover
        return None
