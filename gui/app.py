"""Gradio web demo — the "show, don't tell" layer of MitoSAM.

Three tabs mirror the pipeline:

1. **Segmentation**  — upload a fluorescence image; Otsu baseline runs instantly,
   MitoSAM loads on demand when a checkpoint is supplied (encoder runs once per
   image, decoder per click — so interactions stay snappy).
2. **Quantification** — image + mask -> per-organelle table, overlay and network
   skeleton figure, all computed in-process.
3. **Cohort analysis** — drop the feature CSV (+ optional labels) -> ablation
   table and the SHAP/family figures regenerated on the spot.

Run: ``python -m MitoSAM.gui.app`` (or ``python gui/app.py``) and open the URL.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

import gradio as gr  # noqa: E402

from MitoSAM.config import MitoSAMConfig  # noqa: E402
from MitoSAM.features.extract import extract_image_features  # noqa: E402
from MitoSAM.utils.io import load_image_gray, load_mask, normalize_to_uint8  # noqa: E402
from MitoSAM.utils.logging import get_logger, setup_logging  # noqa: E402
from MitoSAM.utils.viz import INSTANCE_COLORS, palette  # noqa: E402

log = get_logger("gui")
setup_logging()

_MODEL_CACHE: dict[tuple, object] = {}


# --------------------------------------------------------------------------- helpers
def _fig_to_numpy(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf


def _colorize(labels: np.ndarray) -> np.ndarray:
    from skimage import measure

    labels = measure.label(labels > 0, connectivity=2)
    rgb = np.zeros((*labels.shape, 3), dtype=float)
    for lab in range(1, int(labels.max()) + 1):
        rgb[labels == lab] = INSTANCE_COLORS[(lab - 1) % len(INSTANCE_COLORS)]
    return rgb


def _load_model(checkpoint: str | None, sam_type: str, device: str):
    key = (checkpoint, sam_type, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    from MitoSAM.models.mitosam import MitoSAM
    from MitoSAM.utils.device import resolve_device

    device = resolve_device(device)
    model = MitoSAM(sam_type=sam_type).to(device).eval()
    if checkpoint:
        model.load_finetuned(checkpoint)
    _MODEL_CACHE[key] = model
    log.info("model ready (%s on %s)", sam_type, device)
    return model


# --------------------------------------------------------------------------- tab 1
def run_segmentation(image, use_mitosam: bool, checkpoint: str, sam_type: str, device: str):
    if image is None:
        return None, "请先上传一张图像。"
    arr = np.asarray(image)
    if arr.max() > 1:
        arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    import matplotlib.pyplot as plt
    from skimage.filters import threshold_otsu

    otsu = arr > threshold_otsu(arr)
    pred, note = otsu, "Otsu threshold (classical baseline)"
    if use_mitosam:
        try:
            ckpt = checkpoint.strip() or None
            model = _load_model(ckpt, sam_type, device)
            model.set_image(arr)
            items = model.predict_dense(grid_size=24, score_threshold=0.65,
                                        stability_threshold=0.75, min_area=30)
            union = np.zeros(arr.shape, dtype=bool)
            for item in items:
                union |= item["segmentation"]
            if union.any():
                pred = union
                note = f"MitoSAM dense prompts — {len(items)} masks (checkpoint: {ckpt or 'zero-shot'})"
            else:
                note = "MitoSAM returned no stable masks — showing Otsu fallback"
        except Exception as exc:  # pragma: no cover
            log.exception("mitosam inference failed")
            note = f"MitoSAM failed ({exc}) — showing Otsu fallback"

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    axes[0].imshow(arr, cmap="gray")
    axes[0].set_title("input", fontsize=10)
    axes[1].imshow(_colorize(pred), interpolation="nearest")
    axes[1].set_title(f"{note}\n(objects: {int(pd.Series(pred.ravel()).value_counts().size - 1)})",
                      fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return _fig_to_numpy(fig), note


# --------------------------------------------------------------------------- tab 2
def run_quantify(image, mask):
    if image is None or mask is None:
        return None, None, "请同时上传图像和掩码。"
    arr = np.asarray(image)
    if arr.max() > 1:
        arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m.mean(axis=2)
    m = m > (0.5 * (m.max() if m.max() <= 1 else 255) if m.max() <= 1 else 127)

    cfg = MitoSAMConfig().features
    row, per_instance = extract_image_features(arr, m, cfg, "demo")
    if per_instance.empty:
        return None, None, "掩码中没有检测到目标。"

    import matplotlib.pyplot as plt
    from skimage.morphology import skeletonize

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    axes[0].imshow(arr, cmap="gray")
    axes[0].set_title("input", fontsize=10)
    axes[1].imshow(_colorize(m))
    axes[1].set_title(f"instances (n={len(per_instance)})", fontsize=10)
    skel = skeletonize(m)
    rgb = np.repeat(arr[..., None], 3, axis=2)
    rgb[m] = 0.75 * rgb[m] + 0.25 * np.array([0.0, 0.45, 0.70])
    rgb[skel] = np.array([1.0, 0.36, 0.0])
    axes[2].imshow(np.clip(rgb, 0, 1))
    axes[2].set_title("network skeleton", fontsize=10)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()

    cols = ["instance_id", "area", "perimeter", "aspect_ratio", "circularity",
            "solidity", "eccentricity", "n_branch_points"]
    cols = [c for c in cols if c in per_instance.columns]
    return _fig_to_numpy(fig), per_instance[cols].round(3), (
        f"提取了 {len(per_instance)} 个线粒体实例；视场级汇总特征 "
        f"{len(row)} 列（含纹理与空间分布）。"
    )


# --------------------------------------------------------------------------- tab 3
def run_cohort(features_csv, labels_csv, model_key: str):
    if features_csv is None:
        return None, "请上传 features_per_image.csv（可由 batch_process.py --features 生成）。"
    path = Path(features_csv.name if hasattr(features_csv, "name") else features_csv)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "features.csv"
        tmp_path.write_bytes(features_csv if isinstance(features_csv, bytes)
                             else Path(features_csv).read_bytes())
        if labels_csv is not None:
            labels = Path(tmp) / "labels.csv"
            labels.write_bytes(labels_csv if isinstance(labels_csv, bytes)
                               else Path(labels_csv).read_bytes())
            # merge labels by image_id
            feats = pd.read_csv(tmp_path)
            labs = pd.read_csv(labels)
            key = "image_id" if "image_id" in labs.columns else labs.columns[0]
            mapping = dict(zip(labs[key].astype(str), labs["group"].astype(str)))
            feats["group"] = feats["image_id"].astype(str).map(mapping)
            feats.to_csv(tmp_path, index=False)

        from MitoSAM.evaluation.classification import run_classification

        cfg = MitoSAMConfig()
        cfg.evaluation.models = [model_key]
        try:
            payload = run_classification(cfg, tmp_path, Path(tmp) / "out")
        except Exception as exc:
            return None, f"分析失败：{exc}"
        import shutil

        fig_src = Path(tmp) / "out" / "figures" / "ablation_auc.png"
        fig_bytes = fig_src.read_bytes() if fig_src.exists() else None

    table = pd.DataFrame(payload["results"])[
        ["model", "feature_subset", "test_auc", "cv_auc_mean", "generalization_gap", "composite"]
    ].round(4)
    best = payload.get("best", {})
    note = (
        f"最佳组合：{best.get('feature_subset')}（hold-out AUC = "
        f"{best.get('test_auc', float('nan')):.3f}，CV AUC = {best.get('cv_auc_mean', float('nan')):.3f}）。"
        "每行同时报告泛化差距 —— 差距大说明该子集过拟合。"
    )
    return (fig_bytes, note, table) if fig_bytes else (None, note, table)


# --------------------------------------------------------------------------- app
def build_app() -> "gr.Blocks":
    with gr.Blocks(title="MitoSAM — Mitochondria Segmentation & Quantification") as demo:
        gr.Markdown(
            "# MitoSAM — 线粒体超分辨图像分割与量化分析工具箱\n"
            "基于 SAM 的可提示分割 + 形态/纹理/空间三维特征 + 随机森林消融与 SHAP 解释。\n"
            "所有 tab 均在本地运行；图像不会上传到任何外部服务。"
        )
        with gr.Tab("1 · 分割"):
            with gr.Row():
                with gr.Column():
                    seg_input = gr.Image(label="荧光图像 (PNG/TIF/JPG)")
                    use_mito = gr.Checkbox(label="使用 MitoSAM（需要 checkpoint）", value=False)
                    ckpt = gr.Textbox(label="checkpoint 路径（留空 = 零样本 SAM）", value="")
                    sam_type = gr.Radio(["vit_b", "vit_l", "vit_h"], value="vit_b", label="backbone")
                    device = gr.Radio(["auto", "cpu", "cuda"], value="auto", label="device")
                    seg_btn = gr.Button("运行分割", variant="primary")
                with gr.Column():
                    seg_out = gr.Image(label="分割结果")
                    seg_note = gr.Markdown()
            seg_btn.click(run_segmentation,
                          inputs=[seg_input, use_mito, ckpt, sam_type, device],
                          outputs=[seg_out, seg_note])

        with gr.Tab("2 · 量化"):
            with gr.Row():
                with gr.Column():
                    q_img = gr.Image(label="图像")
                    q_mask = gr.Image(label="二值/实例掩码")
                    q_btn = gr.Button("提取特征", variant="primary")
                with gr.Column():
                    q_fig = gr.Image(label="实例与骨架")
                    q_table = gr.Dataframe(label="逐线粒体特征（前 8 列）")
                    q_note = gr.Markdown()
            q_btn.click(run_quantify, inputs=[q_img, q_mask],
                        outputs=[q_fig, q_table, q_note])

        with gr.Tab("3 · 队列分析"):
            with gr.Row():
                with gr.Column():
                    c_feats = gr.File(label="features_per_image.csv", file_types=[".csv"])
                    c_labels = gr.File(label="labels.csv（可选，列: image_id,group）",
                                       file_types=[".csv"])
                    c_model = gr.Radio(["random_forest", "logreg", "xgboost", "lightgbm"],
                                       value="random_forest", label="分类器")
                    c_btn = gr.Button("运行消融分析", variant="primary")
                with gr.Column():
                    c_fig = gr.Image(label="消融结果")
                    c_note = gr.Markdown()
                    c_table = gr.Dataframe(label="全部结果")
            c_btn.click(run_cohort, inputs=[c_feats, c_labels, c_model],
                        outputs=[c_fig, c_note, c_table])

        gr.Markdown(
            "---\n"
            "**提示**：没有数据？先运行 `python -m MitoSAM.data.synthetic` 生成一个演示队列，"
            "再把 `examples/synthetic/features/features_per_image.csv` 拖进 Tab 3。"
        )
    return demo


def main() -> None:
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
