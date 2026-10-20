"""ONNX export and ONNX-Runtime inference for MitoSAM.

Why bother with ONNX in a thesis project? Because it answers the question an
interviewer will ask: *"could this actually be used?"*

* the exported graph contains **only the decoder path** (prompt encoder +
  adapter + mask decoder, ~20 MB) — the ViT-H encoder is exported separately or
  reused from a precomputed embedding, which keeps deployment artefacts small;
* inference then needs neither PyTorch nor a GPU-specific wheel, runs on CPU at
  low latency per click, and drops straight into C++/Java/C# services.

Usage:
    python -m MitoSAM.models.onnx_export --checkpoint weights/mitosam_vit_b.pth --out weights/mitosam_decoder.onnx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.logging import get_logger

log = get_logger("models.onnx")


def export_decoder_onnx(
    checkpoint: str | Path,
    out_path: str | Path,
    sam_type: str = "vit_b",
    sam_checkpoint: str | Path | None = None,
    opset: int = 17,
) -> Path:
    """Export the prompt-encoder + adapter + decoder sub-graph to ONNX."""
    import torch

    from .mitosam import MitoSAM

    model = MitoSAM(sam_type=sam_type, checkpoint=sam_checkpoint).eval()
    model.load_finetuned(checkpoint)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    emb = torch.randn(1, 256, 64, 64)
    pts = torch.tensor([[[512.0, 512.0]]])
    labs = torch.tensor([[1.0]])

    class DecoderExport(torch.nn.Module):
        def __init__(self, m: MitoSAM):
            super().__init__()
            self.sam = m.sam
            self.adapter = m.adapter
            self.image_pe = m.sam.prompt_encoder.get_dense_pe()

        def forward(self, embedding, point_coords, point_labels):
            feat = self.adapter(embedding)
            sparse, dense = self.sam.prompt_encoder(
                points=(point_coords, point_labels.long()), boxes=None, masks=None
            )
            low_res, iou = self.sam.mask_decoder(
                image_embeddings=feat,
                image_pe=self.image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            return low_res, iou

    torch.onnx.export(
        DecoderExport(model).eval(),
        (emb, pts, labs),
        str(out_path),
        input_names=["image_embedding", "point_coords", "point_labels"],
        output_names=["low_res_masks", "iou_pred"],
        dynamic_axes={
            "image_embedding": {0: "batch"},
            "point_coords": {0: "batch", 1: "num_points"},
            "point_labels": {0: "batch", 1: "num_points"},
            "low_res_masks": {0: "batch"},
            "iou_pred": {0: "batch"},
        },
        opset_version=opset,
    )
    log.info("exported ONNX graph -> %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def export_encoder_onnx(
    checkpoint: str | Path,
    out_path: str | Path,
    sam_type: str = "vit_b",
    size: int = 1024,
) -> Path:
    """Export the full encoder (heavy: 90-400 MB depending on backbone)."""
    import torch

    from .mitosam import MitoSAM

    model = MitoSAM(sam_type=sam_type).eval()
    model.load_finetuned(checkpoint)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class EncoderExport(torch.nn.Module):
        def __init__(self, m: MitoSAM):
            super().__init__()
            self.sam, self.adapter = m.sam, m.adapter

        def forward(self, image):
            return self.adapter(self.sam.image_encoder(self.sam.preprocess(image)))

    torch.onnx.export(
        EncoderExport(model).eval(),
        torch.randn(1, 3, size, size),
        str(out_path),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    log.info("exported encoder ONNX -> %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


class ONNXPredictor:
    """OnnxRuntime-backed predictor — no PyTorch, no CUDA required.

    The encoder session is optional: if you pass precomputed embeddings (e.g.
    produced by a C++ service or batched job), only the small decoder session
    needs to be loaded.
    """

    def __init__(
        self,
        decoder_path: str | Path,
        encoder_path: str | Path | None = None,
        providers: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install onnxruntime to use ONNXPredictor") from exc

        providers = providers or ["CPUExecutionProvider"]
        self.decoder = ort.InferenceSession(str(decoder_path), providers=providers)
        self.encoder = (
            ort.InferenceSession(str(encoder_path), providers=providers) if encoder_path else None
        )
        self._embedding: np.ndarray | None = None
        self._orig_size: tuple[int, int] | None = None
        log.info("ONNX sessions ready (encoder=%s)", bool(self.encoder))

    def set_embedding(self, embedding: np.ndarray) -> None:
        self._embedding = embedding.astype(np.float32)

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Preprocess + run the encoder session (image: float32 CHW or HWC in [0,1])."""
        if self.encoder is None:
            raise RuntimeError("No encoder session loaded")
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[None], 3, axis=0)
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3) and arr.shape[0] not in (1, 3):
            arr = np.transpose(arr, (2, 0, 1))

        self._orig_size = (arr.shape[1], arr.shape[2])
        # pad to a square 1024 canvas, bottom/right (same as SAM.preprocess)
        _, h, w = arr.shape
        scale = 1024 / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        arr = _resize_chw(arr, nh, nw)
        canvas = np.zeros((3, 1024, 1024), dtype=np.float32)
        canvas[:, :nh, :nw] = arr
        self._embedding = self.encoder.run(None, {"image": canvas[None]})[0]
        return self._embedding

    def predict(self, points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
        """Return ``(bool mask [H,W], iou score)`` for the current embedding."""
        if self._embedding is None:
            raise RuntimeError("Call encode() or set_embedding() first")
        coords = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
        if self._orig_size is not None:
            h, w = self._orig_size
            scale = 1024 / max(h, w)
            coords = coords * scale
        labs = np.asarray(labels, dtype=np.int64).reshape(1, -1)
        low_res, iou = self.decoder.run(
            None,
            {"image_embedding": self._embedding, "point_coords": coords, "point_labels": labs},
        )
        mask = _postprocess(low_res, self._orig_size)
        return mask, float(1 / (1 + np.exp(-iou.reshape(-1)[0])))


def _resize_chw(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    try:
        import cv2

        return np.stack([cv2.resize(ch, (w, h), interpolation=cv2.INTER_LINEAR) for ch in arr])
    except ImportError:
        idx_y = np.linspace(0, arr.shape[1] - 1, h).astype(int)
        idx_x = np.linspace(0, arr.shape[2] - 1, w).astype(int)
        return arr[:, idx_y][:, :, idx_x]


def _postprocess(low_res: np.ndarray, orig_size: tuple[int, int] | None) -> np.ndarray:
    """Upsample 256x256 logits to the original image size and threshold."""
    logits = low_res[0, 0]
    if orig_size is None:
        return logits > 0
    h, w = orig_size
    try:
        import cv2

        up = cv2.resize(logits, (w, h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        idx_y = np.linspace(0, logits.shape[0] - 1, h).astype(int)
        idx_x = np.linspace(0, logits.shape[1] - 1, w).astype(int)
        up = logits[np.ix_(idx_y, idx_x)]
    return up > 0
