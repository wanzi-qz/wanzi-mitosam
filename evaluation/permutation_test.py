"""Label-permutation control.

The single most useful sanity check for a small-n biological classifier: train
the *same* pipeline on shuffled labels. If the observed AUC is not clearly
outside the shuffled distribution, the model has learned nothing but noise.

This is cheap (10 x k-fold) and it answers the reviewer question "with 60
images and 130 features, isn't this over-fitting?" with a number, not a claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MitoSAMConfig
from ..utils.logging import get_logger
from .classification import load_feature_table, numeric_feature_columns
from .feature_sets import resolve_subset
from .models import build_model

log = get_logger("eval.permutation")


def _fit_auc(X, y, train_idx, test_idx, model_key, random_state, n_features):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
        return float("nan")

    k = min(n_features, X.shape[1])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", build_model(model_key, random_state)),
    ])
    pipe.fit(X[train_idx], y[train_idx])
    return float(roc_auc_score(y[test_idx], pipe.predict_proba(X[test_idx])[:, 1]))


def permutation_control(
    cfg: MitoSAMConfig,
    features_csv: str | Path,
    n_permutations: int = 10,
    model_key: str | None = None,
    subset: str = "full_features",
    out_dir: str | Path = "results",
) -> dict:
    """Compare the real hold-out AUC against a shuffled-label null distribution."""
    from sklearn.model_selection import train_test_split

    df, label_col = load_feature_table(features_csv)
    if label_col is None:
        raise ValueError("no label column in feature table")
    y = df[label_col].astype(int).values
    cols = resolve_subset(subset, numeric_feature_columns(df))
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(df[cols].median(numeric_only=True)).values

    model_key = model_key or ("random_forest" if "random_forest" in cfg.evaluation.models
                              else cfg.evaluation.models[0])
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.3, random_state=cfg.evaluation.random_state, stratify=y
    )
    observed = _fit_auc(X, y, train_idx, test_idx, model_key,
                        cfg.evaluation.random_state, cfg.evaluation.n_features)

    rng = np.random.default_rng(cfg.evaluation.random_state)
    null: list[float] = []
    for _ in range(n_permutations):
        shuffled = y.copy()
        rng.shuffle(shuffled)
        null.append(_fit_auc(X, shuffled, train_idx, test_idx, model_key,
                             cfg.evaluation.random_state, cfg.evaluation.n_features))

    null_arr = np.asarray(null, dtype=float)
    null_arr = null_arr[~np.isnan(null_arr)]
    p_value = float((np.sum(null_arr >= observed) + 1) / (len(null_arr) + 1)) if len(null_arr) else float("nan")

    payload = {
        "model": model_key,
        "feature_subset": subset,
        "observed_auc": round(float(observed), 4),
        "null_mean": round(float(null_arr.mean()), 4) if len(null_arr) else None,
        "null_std": round(float(null_arr.std()), 4) if len(null_arr) else None,
        "null_max": round(float(null_arr.max()), 4) if len(null_arr) else None,
        "n_permutations": int(len(null_arr)),
        "p_value": round(p_value, 4) if p_value == p_value else None,
        "interpretation": (
            "observed AUC is separable from the null -> signal beyond label leakage/noise"
            if p_value == p_value and p_value <= 0.05
            else "observed AUC is inside the null distribution -> treat the result as inconclusive"
        ),
    }
    out_dir = Path(out_dir)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics" / "permutation_control.json").write_text(
        json.dumps({**payload, "null_values": [round(v, 4) for v in null_arr.tolist()]},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        _plot(observed, null_arr, out_dir / "figures" / "permutation_control.png")
    except Exception as exc:  # pragma: no cover
        log.warning("permutation figure skipped: %s", exc)

    return payload


def _plot(observed: float, null: np.ndarray, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..utils.viz import palette, style_axes

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.hist(null, bins=min(12, max(4, len(null))), color="#9DBFD9", edgecolor="white",
            label="shuffled labels (null)")
    ax.axvline(observed, color=palette("secondary"), linewidth=2.0, label=f"observed = {observed:.3f}")
    ax.axvline(0.5, color="#BBBBBB", linestyle=":", linewidth=1)
    style_axes(ax, "Label-permutation control")
    ax.set_xlabel("hold-out ROC-AUC")
    ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
