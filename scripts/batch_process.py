"""Command-line tool: segment a folder of images and extract features in one go.

Examples
--------
Segment only::

    python scripts/batch_process.py --input_dir data/images --output_dir out/masks

Segment + quantify (CSV next to the masks)::

    python scripts/batch_process.py --input_dir data/images --output_dir out/masks \\
        --features --features-csv out/features.csv

Classical baseline (no checkpoint needed)::

    python scripts/batch_process.py --input_dir data/images --output_dir out/masks \\
        --method otsu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from MitoSAM.config import MitoSAMConfig  # noqa: E402
from MitoSAM.features.extract import extract_image_features  # noqa: E402
from MitoSAM.utils.io import (  # noqa: E402
    list_images,
    load_image_gray,
    load_mask,
    normalize_to_uint8,
    save_mask,
)
from MitoSAM.utils.logging import get_logger, setup_logging  # noqa: E402
from MitoSAM.utils.metrics import evaluate_masks  # noqa: E402
from MitoSAM.utils.viz import overlay_mask  # noqa: E402

log = get_logger("cli.batch")


def segment_with_mitosam(model, image: np.ndarray, prompt: str = "auto") -> np.ndarray:
    """Prompt strategy: 'auto' = dense grid union, 'single' = centre point."""
    model.set_image(image)
    if prompt == "auto":
        items = model.predict_dense(grid_size=24, score_threshold=0.65,
                                    stability_threshold=0.75, min_area=30)
        union = np.zeros(model._orig_size, dtype=bool)
        for item in items:
            union |= item["segmentation"]
        return union
    h, w = image.shape[:2]
    masks, scores = model.predict(np.array([[w / 2, h / 2]]), np.array([1]))
    return masks[int(np.argmax(scores))] > 0.5


def segment_otsu(image: np.ndarray) -> np.ndarray:
    from skimage.filters import threshold_otsu

    return image > threshold_otsu(image)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="batch_process",
        description="MitoSAM batch segmentation & quantification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True, help="folder with images (png/tif/jpg)")
    parser.add_argument("--output_dir", required=True, help="folder for masks and previews")
    parser.add_argument("--method", choices=["mitosam", "otsu"], default="mitosam",
                        help="mitosam = promptable SAM pipeline; otsu = classical baseline")
    parser.add_argument("--checkpoint", default=None, help="fine-tuned MitoSAM checkpoint (.pth)")
    parser.add_argument("--sam-checkpoint", default=None, help="pretrained SAM weights (optional)")
    parser.add_argument("--sam-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--prompt", choices=["auto", "single"], default="auto")
    parser.add_argument("--features", action="store_true", help="also extract the feature tables")
    parser.add_argument("--features-csv", default=None, help="where to write features_per_image.csv")
    parser.add_argument("--ground-truth-dir", default=None,
                        help="optional folder with GT masks -> writes quality metrics")
    parser.add_argument("--previews", action="store_true", help="save coloured overlay previews")
    parser.add_argument("--pixel-size-um", type=float, default=0.1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args(argv)

    setup_logging()
    cfg = MitoSAMConfig.load(args.config)
    cfg.features.pixel_size_um = args.pixel_size_um
    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if args.method == "mitosam":
        from MitoSAM.models.mitosam import MitoSAM
        from MitoSAM.utils.device import resolve_device

        device = resolve_device(args.device)
        log.info("loading MitoSAM (%s) on %s", args.sam_type, device)
        model = MitoSAM(sam_type=args.sam_type, checkpoint=args.sam_checkpoint)
        if args.checkpoint and Path(args.checkpoint).exists():
            model.load_finetuned(args.checkpoint)
        model = model.to(device).eval()

    images = list_images(in_dir)
    if not images:
        log.error("no images found in %s", in_dir)
        return 1

    rows: list[dict] = []
    t0 = time.time()
    for i, img_path in enumerate(images, 1):
        image = load_image_gray(img_path)
        try:
            if args.method == "otsu" or model is None:
                pred = segment_otsu(image)
            else:
                pred = segment_with_mitosam(model, image, args.prompt)
        except Exception as exc:  # pragma: no cover
            log.error("%s failed: %s", img_path.name, exc)
            continue

        stem = img_path.stem
        save_mask(pred, masks_dir / f"{stem}_mask.png")
        if args.previews:
            rgb = overlay_mask(image, pred)
            preview_dir = out_dir / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.imsave(preview_dir / f"{stem}_overlay.png", (rgb * 255).astype(np.uint8))

        row = {"image": img_path.name, "n_objects": int(pred.sum() > 0),
               "mask_area_fraction": round(float(pred.mean()), 4)}
        gt_path = None
        if args.ground_truth_dir:
            from MitoSAM.utils.io import find_mask_for_image

            gt_path = find_mask_for_image(img_path, args.ground_truth_dir)
        if gt_path is not None:
            gt = load_mask(gt_path)
            if gt.shape != image.shape:
                from MitoSAM.data.dataset import resize_stack

                gt = resize_stack(gt.astype(np.float32), image.shape[0], is_mask=True) > 0.5
            row.update({k: round(v, 4) for k, v in evaluate_masks(pred, gt).items()})
        rows.append(row)
        log.info("[%d/%d] %s -> area=%.3f%s", i, len(images), img_path.name,
                 pred.mean(), " +metrics" if gt_path is not None else "")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "batch_results.csv", index=False)

    if args.features:
        csv_path = Path(args.features_csv) if args.features_csv else out_dir / "features_per_image.csv"
        feature_rows = []
        for i, img_path in enumerate(images, 1):
            mask_path = masks_dir / f"{img_path.stem}_mask.png"
            if not mask_path.exists():
                continue
            image = load_image_gray(img_path)
            row, _ = extract_image_features(image, load_mask(mask_path),
                                            cfg.features, img_path.stem)
            feature_rows.append(row)
        pd.DataFrame(feature_rows).to_csv(csv_path, index=False)
        log.info("features -> %s", csv_path)

    elapsed = time.time() - t0
    log.info("done: %d images in %.1fs (%.2f s/img) | masks in %s",
             len(results), elapsed, elapsed / max(len(results), 1), masks_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
