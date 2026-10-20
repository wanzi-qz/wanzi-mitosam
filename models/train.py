"""Fine-tuning loop for MitoSAM.

Only the adapter and the mask decoder receive gradients; the ViT encoder stays
frozen. Training samples point prompts *from the ground-truth mask*, which
teaches the decoder "given a hint inside this mitochondrion, trace the whole
organelle" — the same regime used at inference time when a user clicks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import MitoSAMConfig
from ..data.dataset import MitoImageDataset, MitoPatchDataset, pair_images_and_masks, split_pairs
from ..data.prompts import random_points_from_mask
from ..utils.logging import get_logger
from ..utils.metrics import dice_score, iou_score
from .layers import DiceBCELoss
from .mitosam import MitoSAM

log = get_logger("models.train")

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def evaluate(model: MitoSAM, loader, device: str, max_batches: int | None = None) -> dict[str, float]:
    """Whole-image validation with GT-derived point prompts."""
    model.eval()
    dices, ious = [], []
    for i, (images, masks, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        for image, mask in zip(images, masks):
            gt = mask[0].numpy() > 0.5
            model.encode(image)
            pts, labs = random_points_from_mask(gt, num_points=6, rng=np.random.default_rng(0))
            pred_masks, _ = model.predict(pts, labs)
            pred = pred_masks[0] > 0.5
            dices.append(dice_score(pred, gt))
            ious.append(iou_score(pred, gt))
    model.train()
    return {
        "val_dice": float(np.mean(dices)) if dices else 0.0,
        "val_iou": float(np.mean(ious)) if ious else 0.0,
        "val_n": len(dices),
    }


def train(cfg: MitoSAMConfig, resume: str | None = None) -> dict:
    """Run fine-tuning and return the training history."""
    if torch is None:
        raise ImportError("PyTorch is required for training")

    device = cfg.train.device
    if device == "auto":
        from ..utils.device import resolve_device

        device = resolve_device()
    log.info("device=%s | backbone=%s", device, cfg.model.sam_type)

    pairs = pair_images_and_masks(cfg.data.image_dir, cfg.data.mask_dir)
    train_pairs, val_pairs = split_pairs(pairs, cfg.data.val_split, cfg.data.seed)

    train_ds = MitoPatchDataset(
        train_pairs,
        patch_size=cfg.data.patch_size,
        samples_per_image=cfg.data.patches_per_image,
        augment=cfg.data.augment,
        seed=cfg.data.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, drop_last=False)
    val_loader = DataLoader(MitoImageDataset(val_pairs, cfg.model.input_size),
                            batch_size=1, shuffle=False) if val_pairs else None

    model = MitoSAM(
        sam_type=cfg.model.sam_type,
        checkpoint=cfg.model.sam_checkpoint,
        use_adapter=cfg.model.use_adapter,
        use_cbam=cfg.model.use_cbam,
        train_decoder=cfg.model.train_decoder,
        train_prompt_encoder=cfg.model.train_prompt_encoder,
        adapter_channels=cfg.model.adapter_channels,
    ).to(device)
    if resume:
        model.load_finetuned(resume)

    log.info("model summary: %s", model.summary())

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.train.epochs))
    use_amp = cfg.train.amp and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.startswith("cuda") else None

    out_dir = Path(cfg.train.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: dict[str, list] = {"epoch": [], "loss": [], "val_dice": [], "val_iou": [], "lr": []}
    started = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        losses: list[float] = []
        rng = np.random.default_rng(cfg.data.seed + epoch)

        for step, (images, masks) in enumerate(train_loader, 1):
            image = images[0].to(device)              # (3,H,W)
            mask = masks[0].to(device)                # (1,H,W)
            gt = mask[0].detach().cpu().numpy() > 0.5

            pts, labs = random_points_from_mask(gt, cfg.train.prompt_points, rng)
            model.encode(image)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                feat = model._features
                pt_t = torch.as_tensor(pts[None], device=device, dtype=torch.float32)
                lab_t = torch.as_tensor(labs[None], device=device)
                sparse, dense = model.sam.prompt_encoder(points=(pt_t, lab_t), boxes=None, masks=None)
                low_res, _ = model.sam.mask_decoder(
                    image_embeddings=feat,
                    image_pe=model.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )
                target = torch.nn.functional.interpolate(
                    mask[None], size=low_res.shape[-2:], mode="nearest"
                )
                loss = criterion(low_res, target)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.train.grad_clip)
                optimizer.step()

            losses.append(float(loss.item()))
            if step % 20 == 0 or step == len(train_loader):
                log.info("epoch %d/%d step %d/%d loss=%.4f (%.0fs)",
                         epoch, cfg.train.epochs, step, len(train_loader),
                         np.mean(losses[-20:]), time.time() - started)

        scheduler.step()
        metrics = evaluate(model, val_loader, device, max_batches=8) if val_loader else {}
        history["epoch"].append(epoch)
        history["loss"].append(float(np.mean(losses)) if losses else 0.0)
        history["val_dice"].append(metrics.get("val_dice", float("nan")))
        history["val_iou"].append(metrics.get("val_iou", float("nan")))
        history["lr"].append(optimizer.param_groups[0]["lr"])

        ckpt_path = out_dir / f"mitosam_{cfg.model.sam_type}_epoch{epoch}.pth"
        model.save(ckpt_path, epoch=epoch, loss=history["loss"][-1], **metrics)
    latest = out_dir / f"mitosam_{cfg.model.sam_type}.pth"
    model.save(latest, epoch=cfg.train.epochs, final=True,
               **{k: history[k][-1] for k in ("loss", "val_dice", "val_iou")})

    history_path = out_dir / "train_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    log.info("training finished in %.1fs | final checkpoint %s", time.time() - started, latest)
    return {"history": history, "checkpoint": str(latest), "history_path": str(history_path)}
