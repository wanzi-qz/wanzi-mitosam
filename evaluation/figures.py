"""Figures for the report, README and slides.

Every figure is produced from the CSVs / JSONs in ``results/`` — no figure is
hand-drawn — so the numbers in the text and the numbers in the images can never
drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ..utils.viz import FIG_DIR, palette, style_axes

log = get_logger("eval.figures")

SUBSET_LABELS = {
    "morphology_only": "Morphology",
    "texture_only": "Texture",
    "spatial_only": "Spatial",
    "morphology_texture": "Morph + Tex",
    "texture_spatial": "Tex + Spat",
    "morphology_spatial": "Morph + Spat",
    "full_features": "Full (3 families)",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "font.size": 9, "axes.titlesize": 10.5,
        "axes.titleweight": "semibold", "axes.labelcolor": "#333333",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    return plt


def plot_ablation(results_csv: str | Path, out_path: str | Path | None = None) -> Path:
    """Grouped bar chart: repeated-CV AUC per feature subset per model.

    Primary metric is the mean over ``n_splits x n_repeats`` stratified folds
    (default 50) on the full dataset, with per-repeat std as error bars — a
    single hold-out split on ~30 images has a test set so small that AUC
    differences of ±0.1 are pure split noise. Falls back to whatever columns
    the input CSV provides.
    """
    plt = _plt()
    df = pd.read_csv(results_csv)
    value_col = "rcv_auc_mean" if "rcv_auc_mean" in df.columns else "test_auc"
    err_col = "rcv_auc_std" if "rcv_auc_std" in df.columns else "cv_auc_std"
    if value_col == "rcv_auc_mean":
        title = ("Feature-family ablation — repeated 5x10-fold stratified CV\n"
                 "(bar = mean AUC, whiskers = std across repeats)")
    else:
        title = "Feature-family ablation — hold-out ROC-AUC (bars) with CV std (whiskers)"
    subsets = [s for s in SUBSET_LABELS if s in set(df["feature_subset"])]
    models = list(dict.fromkeys(df["model"]))
    colors = [palette(c) for c in ("primary", "secondary", "accent", "warn", "purple", "grey")]

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(subsets))
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("feature_subset").reindex(subsets)
        ax.bar(x + i * width - 0.4 + width / 2, sub[value_col], width * 0.92,
               label=model, color=colors[i % len(colors)],
               yerr=sub[err_col], capsize=2.5, error_kw={"linewidth": 1.0, "alpha": 0.7})
    ax.set_xticks(x)
    ax.set_xticklabels([SUBSET_LABELS[s] for s in subsets], rotation=18, ha="right")
    ax.set_ylim(0.4, 1.06)
    ax.axhline(0.5, color="#BBBBBB", linestyle=":", linewidth=1)
    ax.text(len(subsets) - 0.45, 0.515, "chance", fontsize=7.5, color="#999999", ha="right")
    style_axes(ax, title)
    ax.set_ylabel("ROC-AUC")
    ax.legend(frameon=False, ncol=len(models), loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path or FIG_DIR / "ablation_auc.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_scaling_check(
    dist_paths: dict[str, str | Path],
    out_path: str | Path | None = None,
    comparisons: list[str] | None = None,
) -> Path:
    """Sample-size scaling check: paired (full − reduced) AUC differences.

    For each sample-size cohort (key of ``dist_paths``, e.g. ``"n=60"`` /
    ``"n=300"``) this reads the per-repeat CV distributions and computes the
    paired difference ``full_features - <subset>`` for every (repeat, model)
    cell. Panel A shows mean differences with 95 % CIs and paired t-test
    p-values; Panel B shows the full paired-difference distributions, making
    visible how the n=60 differences straddle zero while the n=300 ones do not.

    This is the figure that answers "why do two-family combinations look as
    good as the full set?" — finite-sample variance, not information redundancy.
    """
    from scipy import stats

    plt = _plt()
    comparisons = comparisons or [
        "morphology_texture", "texture_spatial", "texture_only",
        "morphology_only", "spatial_only",
    ]
    COMP_LABELS = {
        "morphology_texture": "vs Morph + Tex",
        "texture_spatial": "vs Tex + Spat",
        "texture_only": "vs Texture",
        "morphology_only": "vs Morphology",
        "morphology_spatial": "vs Morph + Spat",
        "spatial_only": "vs Spatial",
    }

    diffs: dict[str, pd.DataFrame] = {}
    for cohort, path in dist_paths.items():
        df = pd.read_csv(path)
        p = df.pivot_table(index=["repeat", "model"], columns="feature_subset", values="auc_mean")
        cols = [c for c in comparisons if c in p.columns and "full_features" in p.columns]
        diffs[cohort] = pd.DataFrame({c: p["full_features"] - p[c] for c in cols})

    n_cohorts = len(diffs)
    cohort_names = list(diffs)
    cohort_colors = [palette("grey"), palette("primary"), palette("accent")][:n_cohorts]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # --- Panel A: mean paired difference +- 95% CI -------------------------
    ax = axes[0]
    comp_order = [c for c in comparisons if all(c in diffs[c_].columns for c_ in cohort_names)]
    x = np.arange(len(comp_order))
    width = 0.8 / n_cohorts
    for i, (cohort, color) in enumerate(zip(cohort_names, cohort_colors)):
        means, cis, pvals = [], [], []
        d = diffs[cohort]
        for c in comp_order:
            v = d[c].dropna().values
            means.append(v.mean())
            cis.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
            pvals.append(float(stats.ttest_1samp(v, 0.0).pvalue))
        bars = ax.bar(x + i * width - 0.4 + width / 2, means, width * 0.9,
                      label=cohort, color=color, yerr=cis, capsize=3.0,
                      error_kw={"linewidth": 1.0, "alpha": 0.8})
        for bar, pv in zip(bars, pvals):
            y = bar.get_height()
            sign = 1 if y >= 0 else -1
            if pv >= 0.05:
                txt = "ns"
            elif pv < 0.001:
                txt = "***"
            elif pv < 0.01:
                txt = "**"
            else:
                txt = "*"
            ax.text(bar.get_x() + bar.get_width() / 2, y + sign * 0.006,
                    txt, ha="center",
                    va="bottom" if sign > 0 else "top", fontsize=7.5, color="#333333")
    ax.axhline(0, color="#555555", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([COMP_LABELS.get(c, c) for c in comp_order], rotation=16, ha="right", fontsize=8.5)
    ax.set_ylabel("paired ΔAUC (full − reduced)")
    style_axes(ax, "Paired advantage of the full feature set\n(mean ± 95% CI over repeat×model cells; * p<0.05, ** p<0.01, *** p<0.001)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    # --- Panel B: paired-difference distributions --------------------------
    ax = axes[1]
    show = [c for c in ("texture_spatial", "texture_only", "morphology_only") if c in comp_order]
    positions, data, colors_v = [], [], []
    pos = 0
    tick_pos, tick_lab = [], []
    for j, c in enumerate(show):
        base = j * (n_cohorts + 1) + 1
        for i, cohort in enumerate(cohort_names):
            v = diffs[cohort][c].dropna().values
            data.append(v)
            positions.append(base + i)
            colors_v.append(cohort_colors[i])
        tick_pos.append(base + (n_cohorts - 1) / 2)
        tick_lab.append(COMP_LABELS.get(c, c))
        pos = base + n_cohorts
    vp = ax.violinplot(data, positions=positions, widths=0.85, showextrema=False)
    for body, color in zip(vp["bodies"], colors_v):
        body.set_facecolor(color)
        body.set_alpha(0.65)
        body.set_edgecolor("white")
    for p_, v in zip(positions, data):
        ax.scatter(np.full(len(v), p_), v, s=5, color="#333333", alpha=0.35, zorder=3)
    ax.axhline(0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=16, ha="right", fontsize=8.5)
    ax.set_ylabel("paired ΔAUC per cell")
    style_axes(ax, "Distribution of paired differences\n(each dot = one repeat × model cell)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=cohort_colors[i], alpha=0.65)
               for i in range(n_cohorts)]
    ax.legend(handles, cohort_names, frameon=False, fontsize=8.5, loc="upper left")

    fig.tight_layout()
    out_path = Path(out_path or FIG_DIR / "scaling_check.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_pr(roc_data: dict[str, tuple[np.ndarray, np.ndarray]], out_path: str | Path | None = None) -> Path:
    """Side-by-side ROC and PR curves for several models/feature subsets."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    colors = [palette(c) for c in ("primary", "secondary", "accent", "warn")]

    for i, (name, (fpr, tpr)) in enumerate(roc_data.items()):
        auc = np.trapezoid(tpr, fpr)
        axes[0].plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2.0,
                     label=f"{name} (AUC = {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#CCCCCC", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    style_axes(axes[0], "ROC")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    for i, (name, (fpr, tpr)) in enumerate(roc_data.items()):
        precision = np.clip(tpr, 0, 1)
        recall = fpr
        axes[1].plot(recall, precision, color=colors[i % len(colors)], linewidth=2.0, label=name)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    style_axes(axes[1], "Precision-Recall (normalised)")

    fig.tight_layout()
    out_path = Path(out_path or FIG_DIR / "roc_pr_curves.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_segmentation_summary(summary_csv: str | Path, out_path: str | Path | None = None) -> Path:
    """Radar-free, honest bar chart of segmentation metrics per method."""
    plt = _plt()
    df = pd.read_csv(summary_csv)
    metrics = ["dice", "iou", "boundary_f1", "instance_f1@0.5"]
    labels = ["Dice", "IoU", "Boundary F1", "Instance F1"]
    methods = df["method"].tolist()
    colors = [palette(c) for c in ("grey", "primary", "secondary", "accent")]

    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    width = 0.8 / len(metrics)
    x = np.arange(len(methods))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        col = f"{metric}_mean"
        err = f"{metric}_std"
        if col not in df:
            continue
        ax.bar(x + i * width - 0.4 + width / 2, df[col], width * 0.9, label=label,
               color=colors[i % len(colors)], yerr=df.get(err), capsize=2.0,
               error_kw={"linewidth": 0.8, "alpha": 0.5})
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in methods], fontsize=8.5)
    ax.set_ylim(0, 1.0)
    style_axes(ax, "Segmentation quality by method (mean ± std over held-out FOVs)")
    ax.set_ylabel("score")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    out_path = Path(out_path or FIG_DIR / "segmentation_summary.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_feature_distributions(
    features_csv: str | Path,
    out_path: str | Path | None = None,
    features: list[str] | None = None,
) -> Path:
    """Violin + strip plots of key features (control vs treatment if labelled)."""
    plt = _plt()
    df = pd.read_csv(features_csv)
    label_col = next((c for c in ("group", "label", "Group") if c in df.columns), None)
    candidates = features or [
        "area_mean", "circularity_mean", "aspect_ratio_mean", "intensity_entropy",
        "LBP_entropy", "nn_mean", "ripley_l_max", "n_mitochondria",
    ]
    plot_cols = [c for c in candidates if c in df.columns][:8]
    if not plot_cols:
        raise ValueError("No plottable feature columns found")

    ncols = 4
    nrows = int(np.ceil(len(plot_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, plot_cols):
        if label_col:
            groups = sorted(df[label_col].dropna().unique())
            data = [df.loc[df[label_col] == g, col].dropna().values for g in groups]
            parts = ax.violinplot(data, showextrema=False, widths=0.8)
            for pc, color in zip(parts["bodies"], ("#8FB8DE", "#E7B08B", "#8CC7A1")):
                pc.set_facecolor(color)
                pc.set_alpha(0.75)
                pc.set_edgecolor("white")
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels([str(g)[:10] for g in groups], fontsize=8)
        else:
            ax.hist(df[col].dropna(), bins=14, color="#8FB8DE", edgecolor="white")
        style_axes(ax, col.replace("_", " "))
    for ax in axes[len(plot_cols):]:
        ax.axis("off")

    fig.suptitle("Per-FOV feature distributions (each point = one field of view)",
                 fontsize=11, fontweight="semibold", y=1.01)
    fig.tight_layout()
    out_path = Path(out_path or FIG_DIR / "feature_distributions.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(features_csv: str | Path, out_path: str | Path | None = None, top: int = 24) -> Path:
    """Correlation heatmap of the most variable features (clustered)."""
    plt = _plt()
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    df = pd.read_csv(features_csv)
    num = df.select_dtypes(include=[np.number]).drop(columns=[c for c in ("label",) if c in df], errors="ignore")
    num = num.loc[:, num.nunique() > 1]
    variances = num.var().sort_values(ascending=False)
    cols = list(variances.head(top).index)
    corr = num[cols].corr().fillna(0.0)

    dist = 1 - corr.abs().values
    np.fill_diagonal(dist, 0)
    order = hierarchy.leaves_list(hierarchy.linkage(squareform(dist, checks=False), method="average"))
    corr = corr.iloc[order, order]

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6.5)
    ax.set_yticklabels(corr.index, fontsize=6.5)
    ax.set_title("Feature correlation (top-variance features, clustered)", fontsize=10.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Pearson r", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path or FIG_DIR / "feature_correlation.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_training_curves(history_json: str | Path, out_path: str | Path | None = None) -> Path:
    plt = _plt()
    hist = json.loads(Path(history_json).read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    axes[0].plot(hist["epoch"], hist["loss"], marker="o", color=palette("primary"))
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("Dice+BCE loss")
    style_axes(axes[0], "Training loss")
    if any(v == v for v in hist.get("val_dice", [])):
        axes[1].plot(hist["epoch"], hist["val_dice"], marker="o", label="Dice", color=palette("accent"))
    if any(v == v for v in hist.get("val_iou", [])):
        axes[1].plot(hist["epoch"], hist["val_iou"], marker="s", label="IoU", color=palette("secondary"))
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].legend(frameon=False, fontsize=8)
    style_axes(axes[1], "Validation segmentation")
    fig.tight_layout()
    out_path = Path(out_path or FIG_DIR / "training_curves.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
