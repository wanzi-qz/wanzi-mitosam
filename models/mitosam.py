"""MitoSAM model definition.

Architecture (same as the thesis, packaged):

    image ─► SAM image encoder (frozen)  ─► (B,256,H/16,W/16)
                                              │
                                    ResNet-CBAM adapter (trainable)
                                              │
    sparse prompt ─► prompt encoder (frozen) ─┴─► mask decoder (trainable) ─► mask logits

The wrapper keeps SAM's native two-step API (``set_image`` / ``predict``) by
pre-computing and caching embeddings, which is what makes interactive and
batched inference cheap: the encoder runs once per image, the decoder runs per
click.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ..utils.io import load_image_gray
from ..utils.logging import get_logger
from .bridge import build_sam
from .layers import EmbeddingAdapter, count_parameters

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False
    nn = object  # type: ignore

log = get_logger("models.mitosam")


class MitoSAM(nn.Module if _TORCH else object):  # type: ignore[misc]
    """SAM + CBAM adapter + fine-tuned decoder, specialised for mitochondria.

    Args:
        sam_type: ``vit_b`` (fast, default) / ``vit_l`` / ``vit_h`` (thesis).
        checkpoint: SAM pretrained weights, or a MitoSAM fine-tuned checkpoint.
        use_adapter: attach the ResNet-CBAM embedding adapter.
        train_decoder: unfreeze SAM's mask decoder.
    """

    def __init__(
        self,
        sam_type: str = "vit_b",
        checkpoint: str | Path | None = None,
        use_adapter: bool = True,
        use_cbam: bool = True,
        train_decoder: bool = True,
        train_prompt_encoder: bool = False,
        adapter_channels: int = 256,
    ) -> None:
        if not _TORCH:
            raise ImportError("PyTorch is required for MitoSAM")
        super().__init__()
        self.sam_type = sam_type
        self.sam = build_sam(sam_type, checkpoint)
        self.adapter = (
            EmbeddingAdapter(adapter_channels, n_blocks=2, use_cbam=use_cbam)
            if use_adapter else nn.Identity()
        )

        # ---- freeze / unfreeze policy
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = train_prompt_encoder
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = train_decoder

        self._features: torch.Tensor | None = None
        self._input_size: tuple[int, int] | None = None
        self._orig_size: tuple[int, int] | None = None

    # ------------------------------------------------------------------ utils
    @property
    def device(self) -> "torch.device":
        return next(self.sam.parameters()).device

    @property
    def image_size(self) -> int:
        return self.sam.image_encoder.img_size

    def trainable_parameters(self) -> Iterable["torch.nn.Parameter"]:
        return (p for p in self.parameters() if p.requires_grad)

    def summary(self) -> dict[str, int]:
        total, trainable = count_parameters(self)
        return {"total_params": total, "trainable_params": trainable,
                "trainable_ratio": round(trainable / max(total, 1), 4)}

    # ------------------------------------------------------------- embedding
    def encode(self, image: np.ndarray | "torch.Tensor") -> "torch.Tensor":
        """Run the frozen encoder + adapter once and cache the embedding.

        Args:
            image: HxW or 3xHxW, values in [0, 1] (uint8/float accepted).
        """
        tensor = self._to_tensor(image).to(self.device)
        self._orig_size = tuple(tensor.shape[-2:])
        with torch.no_grad():
            padded = self.sam.preprocess(tensor[None])  # (1,3,1024,1024)
            self._input_size = tuple(padded.shape[-2:])
            features = self.sam.image_encoder(padded)
        with torch.no_grad():
            self._features = self.adapter(features).detach()
        return self._features

    def _to_tensor(self, image: np.ndarray | "torch.Tensor") -> "torch.Tensor":
        if isinstance(image, torch.Tensor):
            t = image.float()
        else:
            arr = np.asarray(image)
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        if t.ndim == 2:
            t = t[None].repeat(3, 1, 1)
        elif t.ndim == 3 and t.shape[0] not in (1, 3):
            t = t.permute(2, 0, 1)[:3]
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        if t.max() > 1.5:
            t = t / 255.0
        return t

    # ---------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(
        self,
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
        multimask: bool = False,
        mask_input: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict masks for the cached image.

        Point/box coordinates are given in **original image pixels** — the
        rescaling to the 1024-px encoder input happens here, so callers never
        have to reason about SAM's internal resolution.
        """
        if self._features is None:
            raise RuntimeError("Call encode()/set_image() before predict()")

        sx = self._input_size[1] / self._orig_size[1]
        sy = self._input_size[0] / self._orig_size[0]

        pt_t = None
        if points is not None and len(points):
            pts = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
            pts = pts * np.array([sx, sy], dtype=np.float32)
            pt_t = torch.as_tensor(pts, device=self.device)
        lab_t = None
        if labels is not None and len(np.atleast_1d(labels)):
            lab_t = torch.as_tensor(np.asarray(labels).reshape(1, -1), device=self.device)

        box_t = None
        if boxes is not None and len(boxes):
            bx = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            bx = bx * np.array([sx, sy, sx, sy], dtype=np.float32)
            box_t = torch.as_tensor(bx, device=self.device)

        mask_t = None
        if mask_input is not None:
            m = torch.as_tensor(mask_input, device=self.device, dtype=torch.float32)
            if m.ndim == 2:
                m = m[None, None]
            mask_t = F.interpolate(m, size=(256, 256), mode="bilinear", align_corners=False)

        sparse, dense = self.sam.prompt_encoder(
            points=(pt_t, lab_t) if pt_t is not None and lab_t is not None else None,
            boxes=box_t,
            masks=mask_t,
        )
        low_res, iou_pred = self.sam.mask_decoder(
            image_embeddings=self._features,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask,
        )
        masks = self.sam.postprocess_masks(low_res, self._input_size, self._orig_size)
        masks = (masks > 0.0).float().cpu().numpy()[0]
        scores = torch.sigmoid(iou_pred).cpu().numpy()[0]
        return masks, scores

    def set_image(self, image: np.ndarray | str | Path) -> "torch.Tensor":
        """Promptable-API compatible entry point."""
        if isinstance(image, (str, Path)):
            image = load_image_gray(image)
        return self.encode(image)

    # ------------------------------------------------------------- checkpoint
    def save(self, path: str | Path, **meta) -> Path:
        """Save trainable weights only (adapter + decoder) — a few MB, not 2.5 GB."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in self.state_dict().items()
                 if k.startswith("adapter.") or self._is_decoder_key(k)}
        torch.save({"state_dict": state, "sam_type": self.sam_type, "meta": meta}, path)
        log.info("saved %d tensors (%.1f MB) -> %s", len(state),
                 path.stat().st_size / 1e6, path)
        return path

    @staticmethod
    def _is_decoder_key(key: str) -> bool:
        return "mask_decoder" in key or "prompt_encoder" in key

    def load_finetuned(self, path: str | Path) -> "MitoSAM":
        """Load a MitoSAM checkpoint produced by :meth:`save` or ``train.py``."""
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            log.warning("missing keys: %d (e.g. %s)", len(missing), missing[:3])
        if unexpected:
            log.warning("unexpected keys: %d", len(unexpected))
        log.info("loaded MitoSAM weights from %s", path)
        return self

    # ------------------------------------------------------------- inference chunking
    @torch.no_grad()
    def predict_dense(
        self,
        grid_size: int = 32,
        points_per_batch: int = 64,
        score_threshold: float = 0.75,
        stability_threshold: float = 0.80,
        min_area: int = 30,
        nms_iou: float = 0.7,
    ) -> list[dict]:
        """Automatic mask generation over a point grid (SAM-AMG style, simplified).

        Returns a list of ``{"segmentation": bool[H,W], "score": float, "area": int}``
        filtered by predicted IoU, stability under 0.5-pixel logit shifts, and
        area, with greedy IoU-based NMS to remove duplicates.
        """
        if self._features is None:
            raise RuntimeError("Call set_image() first")

        h, w = self._orig_size
        sx, sy = self._input_size[1] / w, self._input_size[0] / h
        xs = np.linspace(0, w - 1, grid_size)
        ys = np.linspace(0, h - 1, grid_size)
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2).astype(np.float32)

        candidates: list[dict] = []
        for start in range(0, len(grid), points_per_batch):
            chunk = grid[start:start + points_per_batch]
            batch = len(chunk)
            pts = torch.as_tensor(chunk * np.array([sx, sy], dtype=np.float32),
                                  device=self.device)[:, None, :]
            lab = torch.ones((batch, 1), device=self.device)
            sparse, dense = self.sam.prompt_encoder(points=(pts, lab), boxes=None, masks=None)
            low_res, iou_pred = self.sam.mask_decoder(
                image_embeddings=self._features.expand(batch, -1, -1, -1),
                image_pe=self.sam.prompt_encoder.get_dense_pe().expand(batch, -1, -1, -1),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
            )
            masks = self.sam.postprocess_masks(low_res, self._input_size, self._orig_size)
            probs = torch.sigmoid(masks)
            scores = torch.sigmoid(iou_pred)

            # stability = IoU between (logit > 0) and (logit > 0 +/- offset)
            for i in range(batch):
                for j in range(masks.shape[1]):
                    score = float(scores[i, j])
                    if score < score_threshold:
                        continue
                    m = probs[i, j]
                    base = m > 0.5
                    stable = ((m > 0.5 - 0.15) & (m > 0.5 + 0.15)).float().mean().item()
                    area = int(base.sum().item())
                    if area < min_area or stable < stability_threshold:
                        continue
                    candidates.append({
                        "segmentation": base.cpu().numpy(),
                        "score": score,
                        "area": area,
                        "stability": float(stable),
                    })

        candidates.sort(key=lambda d: (-d["score"], -d["area"]))
        kept: list[dict] = []
        for cand in candidates:
            if all(_iou(cand["segmentation"], k["segmentation"]) < nms_iou for k in kept):
                kept.append(cand)
        log.info("dense prediction: %d candidates -> %d masks", len(candidates), len(kept))
        return kept


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0
