#!/usr/bin/env python
"""Train MitoSAM on a paired dataset.

    python scripts/train.py --image_dir data/train/images --mask_dir data/train/masks \
        --sam_type vit_b --epochs 3 --out weights

Only the ResNet-CBAM adapter and the mask decoder receive gradients; the ViT
encoder stays frozen (see MitoSAM/models/layers.py for the rationale).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from MitoSAM.config import MitoSAMConfig  # noqa: E402
from MitoSAM.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("cli.train")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune MitoSAM")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--sam_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_checkpoint", default=None, help="pretrained SAM weights")
    parser.add_argument("--resume", default=None, help="MitoSAM checkpoint to continue from")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--patches_per_image", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--out", default="weights")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    setup_logging()
    cfg = MitoSAMConfig.load(args.config)
    cfg.data.image_dir = args.image_dir
    cfg.data.mask_dir = args.mask_dir
    cfg.data.patch_size = args.patch_size
    cfg.data.patches_per_image = args.patches_per_image
    cfg.data.val_split = args.val_split
    cfg.model.sam_type = args.sam_type
    cfg.model.sam_checkpoint = args.sam_checkpoint
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.train.checkpoint_dir = args.out
    cfg.train.device = args.device

    from MitoSAM.models.train import train

    result = train(cfg, resume=args.resume)
    log.info("history -> %s", result["history_path"])
    print(json.dumps(result["history"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
