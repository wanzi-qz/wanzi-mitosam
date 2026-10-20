"""Qualitative figures: image / GT / prediction side by side, feature overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import load_image_gray, load_mask, normalize_to_uint8
from ..utils.logging import get_logger
from ..utils.viz import FIG_DIR, INSTANCE_COLORS, overlay_mask, palette

log = get_logger("eval.qualitative")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8.5, "figure.dpi": 130})
    return plt


def label_instances(mask: np.ndarray) -> np.ndarray:
    from skimage import measure

    return measure.label(np.asarray(mask) > 0.5, connectivity=2)


def colorize_labels(labels: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    rgb = np.zeros((*labels.shape, 3), dtype=np.float32)
    for lab in range(1, int(labels.max()) + 1):
        rgb[labels == lab] = INSTANCE_COLORS[(lab - 1) % len(INSTANCE_COLORS)]
    return rgb * alpha


def segmentation_panel(
    samples: list[tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None]],
    out_path: str | Path | None = None,
    title: str = "Segmentation: super-resolution input, ground truth, prediction",
) -> Path:
    """Grid of ``[(name, image, gt_mask, pred_mask), ...]`` with perimeter overlays."""
    plt = _plt()
    from skimage.segmentation import find_boundaries

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(8.6, 2.5 * n), squeeze=False)

    for row, (name, image, gt, pred) in enumerate(samples):
        axes[row, 0].imshow(normalize_to_uint8(image), cmap="gray")
        axes[row, 0].set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center", labelpad=22)

        for col, (mask, cmap, title_txt) in enumerate(
            [(gt, None, "ground truth"), (pred, None, "prediction")], start=1
        ):
            ax = axes[row, col]
            if mask is None:
                ax.axis("off")
                continue
            labels = label_instances(mask)
            ax.imshow(colorize_labels(labels), interpolation="nearest")
            boundaries = find_boundaries(labels, mode="outer")
            ax.imshow(np.ma.masked_where(~boundaries, boundaries), cmap="gray_r", alpha=0.5)
            if col == 1:
                ax.set_title(f"{title_txt}  (n={int(labels.max())})", fontsize=8.5)
            else:
                ax.set_title(f"{title_txt}  (n={int(labels.max())})", fontsize=8.5)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    for ax, txt in zip(axes[0], ["input (SR)", "ground truth", "prediction"]):
        ax.set_xlabel(txt, fontsize=8.5, labelpad=6)

    fig.suptitle(title, fontsize=10.5, fontweight="semibold")
    fig.tight_layout()
    out_path = Path(out_path or FIG_DIR / "segmentation_qualitative.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", out_path)
    return out_path


def feature_overlay(image: np.ndarray, mask: np.ndarray, out_path: str | Path | None = None) -> Path:
    """Three-panel anatomy figure: intensity, instance labels, skeleton network."""
    plt = _plt()
    from skimage.morphology import skeletonize

    labels = label_instances(mask)
    skel = skeletonize(labels > 0)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
    axes[0].imshow(normalize_to_uint8(image), cmap="gray")
    axes[0].set_title("Super-resolution input", fontsize=9)
    axes[1].imshow(colorize_labels(labels))
    axes[1].set_title(f"Instance masks (n={int(labels.max())})", fontsize=9)
    rgb = np.repeat(normalize_to_uint8(image)[..., None] / 255.0, 3, axis=2)
    rgb[labels > 0] = 0.82 * rgb[labels > 0] + 0.18 * np.array([0.0, 0.45, 0.70])
    rgb[skel] = np.array([1.0, 0.36, 0.0])
    axes[2].imshow(rgb)
    axes[2].set_title("Skeleton / network topology", fontsize=9)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    for item in axes:
        for spine in item.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    out_path = Path(out_path or FIG_DIR / "feature_overlay.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def spatial_overlay(image: np.ndarray, mask: np.ndarray, out_path: str | Path | None = None) -> Path:
    """Centroid / Voronoi / Ripley-L panel — the visual story for spatial features."""
    plt = _plt()
    from scipy.spatial import Voronoi, voronoi_plot_2d
    from skimage import measure

    labels = measure.label(mask > 0.5, connectivity=2)
    regions = [r for r in measure.regionprops(labels) if r.area >= 50]
    centroids = np.array([[r.centroid[1], r.centroid[0]] for r in regions]) if regions else np.zeros((0, 2))

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    axes[0].imshow(normalize_to_uint8(image), cmap="gray")
    axes[0].scatter(centroids[:, 0], centroids[:, 1], s=6, c=palette("secondary"), alpha=0.9)
    axes[0].set_title(f"Centroids (n={len(centroids)})", fontsize=9)

    axes[1].imshow(colorize_labels(labels), alpha=0.35)
    if len(centroids) > 3:
        try:
            voronoi_plot_2d(Voronoi(centroids), ax=axes[1], show_vertices=False,
                            line_colors=palette("primary"), line_width=0.7, point_size=3)
        except Exception:  # pragma: no cover
            pass
    axes[1].set_xlim(0, mask.shape[1])
    axes[1].set_ylim(mask.shape[0], 0)
    axes[1].set_title("Voronoi tessellation", fontsize=9)

    axes[2].imshow(normalize_to_uint8(image), cmap="gray", alpha=0.45)
    axes[2].scatter(centroids[:, 0], centroids[:, 1], s=4, c=palette("accent"), alpha=0.85)
    axes[2].set_title("Distribution pattern (clustered vs dispersed)", fontsize=9)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out_path = Path(out_path or FIG_DIR / "spatial_overlay.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_qualitative_figures(
    image_dir: str,
    mask_dir: str,
    predictions: dict[str, np.ndarray] | None = None,
    n_examples: int = 3,
    out_dir: str | Path = FIG_DIR,
) -> list[Path]:
    """Convenience wrapper used by the CLI and the notebook."""
    from ..utils.io import find_mask_for_image, list_images

    out_dir = Path(out_dir)
    paths: list[Path] = []
    images = list_images(image_dir)[:n_examples]
    samples = []
    for img_path in images:
        mask_path = find_mask_for_image(img_path, mask_dir)
        if mask_path is None:
            continue
        image = load_image_gray(img_path)
        gt = load_mask(mask_path)
        pred = predictions.get(img_path.stem) if predictions else None
        samples.append((img_path.stem, image, gt, pred))
    if samples:
        paths.append(segmentation_panel(samples, out_dir / "segmentation_qualitative.png"))
        image, gt, _ = samples[0][1], samples[0][2], None
        paths.append(feature_overlay(image, gt, out_dir / "feature_overlay.png"))
        paths.append(spatial_overlay(image, gt, out_dir / "spatial_overlay.png"))
    return paths
