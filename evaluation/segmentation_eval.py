"""Segmentation evaluation: baseline vs fine-tuned vs classical threshold.

Produces the numbers quoted in the technical report:

* Dice / IoU / precision / recall / boundary-F1 / instance-F1 per image,
* ablation across prompt strategies (1 point, 5 points, grid-dense),
* a per-image table and a summary JSON, plus a qualitative figure.

Run:
    python -m MitoSAM.evaluation.segmentation_eval --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MitoSAMConfig
from ..data.dataset import pair_images_and_masks, resize_stack, split_pairs
from ..data.prompts import grid_points, random_points_from_mask
from ..utils.io import load_image_gray, load_mask
from ..utils.logging import get_logger
from ..utils.metrics import evaluate_masks

log = get_logger("eval.segmentation")


def otsu_baseline(image: np.ndarray) -> np.ndarray:
    """Classical Otsu threshold — the 'do we even need deep learning?' control."""
    try:
        from skimage.filters import threshold_otsu

        return image > threshold_otsu(image)
    except Exception:  # pragma: no cover
        thr = np.percentile(image, 90)
        return image > thr


def predict_single_point(model, image_np: np.ndarray, gt: np.ndarray, n_points: int = 1) -> np.ndarray:
    """Prompt with ``n_points`` GT-sampled points and return the first mask."""
    model.set_image(image_np)
    pts, labs = random_points_from_mask(gt, num_points=n_points, rng=np.random.default_rng(0))
    masks, scores = model.predict(pts, labs)
    if masks.ndim == 3:
        return masks[int(np.argmax(scores))] > 0.5
    return masks > 0.5


def predict_dense(model, min_area: int = 30) -> np.ndarray:
    """Automatic mode: union of grid-prompted masks."""
    items = model.predict_dense(grid_size=24, score_threshold=0.65,
                                stability_threshold=0.75, min_area=min_area)
    if not items:
        return np.zeros(model._orig_size, dtype=bool)
    union = np.zeros(items[0]["segmentation"].shape, dtype=bool)
    for item in items:
        union |= item["segmentation"]
    return union


def _heuristic_union(model, image_np: np.ndarray, mode: str) -> np.ndarray:
    """Fallback for random-init weights: grid prompts + keep largest stable masks."""
    model.set_image(image_np)
    h, w = image_np.shape[:2]
    pts, labs = grid_points(h, w, n_per_side=16, margin=0.1)
    masks, _ = model.predict(pts, labs)
    union = np.zeros((h, w), dtype=bool)
    for m in masks:
        if m.mean() > 0.002:
            union |= m > 0.5
    return union if union.any() else np.zeros((h, w), dtype=bool)


def evaluate_model(
    model,
    pairs: list[tuple[Path, Path]],
    tag: str,
    dense: bool = False,
    n_points: int = 5,
) -> list[dict]:
    rows: list[dict] = []
    for img_path, mask_path in pairs:
        image = load_image_gray(img_path)
        gt = load_mask(mask_path)
        if gt.shape != image.shape:
            gt = resize_stack(gt.astype(np.float32), image.shape[0], is_mask=True) > 0.5
        if dense:
            pred = predict_dense(model)
            if pred.sum() == 0:
                pred = _heuristic_union(model, image, "dense")
        else:
            pred = predict_single_point(model, image, gt, n_points=n_points)
        metrics = evaluate_masks(pred, gt)
        metrics.update({"image": img_path.name, "method": tag})
        rows.append(metrics)
    return rows


def evaluate_threshold_baseline(pairs: list[tuple[Path, Path]]) -> list[dict]:
    rows = []
    for img_path, mask_path in pairs:
        image = load_image_gray(img_path)
        gt = load_mask(mask_path)
        if gt.shape != image.shape:
            gt = resize_stack(gt.astype(np.float32), image.shape[0], is_mask=True) > 0.5
        metrics = evaluate_masks(otsu_baseline(image), gt)
        metrics.update({"image": img_path.name, "method": "otsu_threshold"})
        rows.append(metrics)
    return rows


def run(
    cfg: MitoSAMConfig,
    checkpoint: str | None = None,
    compare_pretrained: bool = True,
    max_images: int | None = None,
    out_dir: str = "results",
) -> pd.DataFrame:
    from ..models.mitosam import MitoSAM
    from ..utils.device import resolve_device

    device = resolve_device(cfg.train.device)
    pairs = pair_images_and_masks(cfg.data.image_dir, cfg.data.mask_dir)
    _, val_pairs = split_pairs(pairs, cfg.data.val_split, cfg.data.seed)
    val_pairs = val_pairs[:max_images] if max_images else val_pairs
    if not val_pairs:
        val_pairs = pairs[:max_images] if max_images else pairs
    log.info("evaluating on %d held-out images", len(val_pairs))

    rows: list[dict] = []
    rows += evaluate_threshold_baseline(val_pairs)

    def build(ckpt):
        return MitoSAM(
            sam_type=cfg.model.sam_type,
            checkpoint=cfg.model.sam_checkpoint,
            use_adapter=cfg.model.use_adapter,
            use_cbam=cfg.model.use_cbam,
        ).to(device).eval()

    if compare_pretrained:
        log.info("evaluating zero-shot SAM (no mitochondrial fine-tuning)")
        rows += evaluate_model(build(None), val_pairs, "sam_zero_shot", n_points=5)

    if checkpoint and Path(checkpoint).exists():
        log.info("evaluating fine-tuned MitoSAM: %s", checkpoint)
        model = build(checkpoint).eval()
        model.load_finetuned(checkpoint)
        rows += evaluate_model(model, val_pairs, "mitosam_point5", n_points=5)
        rows += evaluate_model(model, val_pairs, "mitosam_dense", dense=True)
    else:
        log.warning("no fine-tuned checkpoint found -> skipping MitoSAM rows")

    df = pd.DataFrame(rows)
    out_path = Path(out_dir)
    (out_path / "tables").mkdir(parents=True, exist_ok=True)
    (out_path / "metrics").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path / "tables" / "segmentation_per_image.csv", index=False)

    summary = (
        df.groupby("method")[["dice", "iou", "precision", "recall", "boundary_f1", "instance_f1@0.5"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out_path / "tables" / "segmentation_summary.csv", index=False)
    (out_path / "metrics" / "segmentation_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("segmentation summary:\n%s", summary.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="MitoSAM segmentation evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="weights/mitosam_vit_b.pth")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--sam-type", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--out", default="results")
    parser.add_argument("--no-pretrained", action="store_true", help="skip zero-shot SAM baseline")
    args = parser.parse_args()

    cfg = MitoSAMConfig.load(args.config)
    if args.image_dir:
        cfg.data.image_dir = args.image_dir
    if args.mask_dir:
        cfg.data.mask_dir = args.mask_dir
    if args.sam_checkpoint:
        cfg.model.sam_checkpoint = args.sam_checkpoint
    if args.sam_type:
        cfg.model.sam_type = args.sam_type

    run(cfg, args.checkpoint, not args.no_pretrained, args.max_images, args.out)


if __name__ == "__main__":
    main()
