"""Synthetic cohort generator for the demo / regression tests.

Why synthetic data in a real project? Because the lab's labelled cohort is
small and cannot be published. This generator creates **structurally realistic**
mitochondrial fields with *known* ground truth, so that the whole pipeline —
segmentation, feature extraction, ablation, SHAP — can be run and defended
end-to-end by anyone who clones the repo.

Design: a **latent-severity model**, tuned so that the ablation is informative
rather than saturated.

* every field of view draws one latent severity ``s`` from a class-conditional
  distribution (control and treated overlap heavily — real cohorts do too);
* the three biological cue families each depend on their own noisy copy of the
  severity (``s_morph = s + eps_morph`` etc., with *independent* noise):

  - **morphology**   — organelle length, curvature and count;
  - **texture**      — cristae stripe order/contrast inside the tubules;
  - **spatial**      — fraction of organelles recruited into clusters;

  Because the noise terms are independent, each cue family alone is only
  partially informative (AUC ≈ 0.70–0.78) while the fused feature set does
  markedly better — mirroring the paper's claim that the three families are
  complementary. A generator where one family solves the task at AUC 0.99
  would make every ablation table trivially flat.

Realism measures (this is not a toy generator):

* intensity is rendered as bright tubules with periodic cristae stripes whose
  contrast encodes "cristae order", plus optical edge halo, per-FOV gain and
  Poisson-like noise — the exact cues the features are supposed to pick up;
* per-organelle variability means the class difference is a shift in a
  distribution, never a threshold;
* FOVs are independent draws (a simplification vs. one coverslip per class —
  chosen deliberately so within-class correlation does not inflate AUC).

It is explicitly *not* a replacement for real data: the CLI prints a warning
and every figure produced from it is labelled SYNTHETIC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.draw import ellipse

from ..utils.logging import get_logger

log = get_logger("data.synthetic")

# Latent-severity mixing: class-conditional mean of s and the amount of
# overlap between the two class distributions. sigma is large on purpose.
CLASS_SEVERITY = {"control": 0.32, "treated": 0.68}
SEVERITY_SIGMA = 0.26
# Independent per-cue noise: how much each family's rendering deviates from
# the FOV's true severity. Larger => weaker single-family signal.
CUE_NOISE = {"morphology": 0.16, "texture": 0.16, "spatial": 0.16}


def _lerp(a: float, b: float, s: float) -> float:
    return float(a + (b - a) * np.clip(s, 0.0, 1.0))


def _cue(rng: np.random.Generator, s: float, family: str) -> float:
    """Noisy per-family copy of the latent severity."""
    eps = rng.normal(0.0, CUE_NOISE[family])
    return float(np.clip(s + eps, 0.02, 0.98))


def synthetic_mitochondrion(
    rng: np.random.Generator, length: float, width: float, curvature: float
) -> np.ndarray:
    """Rasterise one curved mitochondrion into a local square boolean patch."""
    pad = int(length * 0.8 + width * 3 + 14)
    side = 2 * pad
    canvas = np.zeros((side, side), dtype=bool)

    n_steps = 48
    t = np.linspace(-0.5, 0.5, n_steps)
    theta = rng.uniform(0, np.pi)
    phase = rng.uniform(0, 2 * np.pi)
    bend = curvature * rng.choice([-1.0, 1.0]) * length * 0.45
    xs = t * length * np.cos(theta) - bend * np.sin(theta) * np.sin(np.pi * t + phase)
    ys = t * length * np.sin(theta) + bend * np.cos(theta) * np.sin(np.pi * t + phase)
    xs += pad
    ys += pad
    radii = np.linspace(width / 2, width / 2 * rng.uniform(0.55, 1.0), n_steps)
    for x, y, r in zip(xs, ys, radii):
        rr, cc = ellipse(y, x, r * 1.02, r * rng.uniform(0.88, 1.12), shape=canvas.shape)
        canvas[rr, cc] = True
    return canvas


def _sample_positions(
    rng: np.random.Generator, n_mito: int, s_spatial: float, side: int
) -> np.ndarray:
    """Organelle centroids: higher severity -> more spatial clustering.

    A severity-dependent fraction of organelles is recruited into Gaussian
    clusters; the rest are uniformly scattered. This gives the spatial feature
    family (nearest-neighbour, Ripley, Voronoi) real signal instead of the
    weak residual it would get from the count alone.
    """
    cluster_frac = _lerp(0.04, 0.52, s_spatial)
    n_clustered = int(round(cluster_frac * n_mito))
    n_uniform = n_mito - n_clustered

    pts = []
    if n_clustered > 0:
        n_centers = int(rng.integers(5, 14))
        centers = rng.uniform(side * 0.1, side * 0.9, size=(n_centers, 2))
        idx = rng.integers(0, n_centers, size=n_clustered)
        pts.append(rng.normal(centers[idx], side * 0.045))
    if n_uniform > 0:
        pts.append(rng.uniform(0, side, size=(n_uniform, 2)))
    return np.concatenate(pts, axis=0) if pts else np.empty((0, 2))


def render_fov(
    rng: np.random.Generator,
    severity: float,
    size: int = 512,
    background: float = 0.08,
    noise: float = 0.035,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one field of view (intensity + mask) for a latent severity."""
    s_morph = _cue(rng, severity, "morphology")
    s_tex = _cue(rng, severity, "texture")
    s_spatial = _cue(rng, severity, "spatial")

    # --- morphology cue: tubule geometry -----------------------------------
    # per-512px-FOV counts mirror a real SR frame (~25-40 resolvable organelles
    # after overlap merging); higher severity -> more, shorter, more curved
    n_mito = int(_lerp(24, 42, s_morph) + rng.normal(0, 3.5))
    n_mito = max(10, n_mito)
    base_length = _lerp(92, 54, s_morph)
    base_curve = _lerp(0.16, 0.36, s_morph)

    mask = np.zeros((size, size), dtype=bool)
    positions = _sample_positions(rng, n_mito, s_spatial, size)
    for cy, cx in positions:
        length = float(np.clip(base_length * rng.uniform(0.7, 1.3), 22, 170))
        width = float(np.clip(8.4 * rng.uniform(0.8, 1.2), 5.5, 12.0))
        patch = synthetic_mitochondrion(rng, length, width, base_curve)
        mh, mw = patch.shape
        y0, x0 = int(cy - mh // 2), int(cx - mw // 2)
        ys0, ys1 = max(0, y0), min(size, y0 + mh)
        xs0, xs1 = max(0, x0), min(size, x0 + mw)
        if ys1 <= ys0 or xs1 <= xs0:
            continue
        mask[ys0:ys1, xs0:xs1] |= patch[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]

    # --- texture cue: cristae order inside the tubules ---------------------
    body = ndi.gaussian_filter(mask.astype(np.float32), 1.1)
    image = np.full(mask.shape, background, dtype=np.float32)

    orientation = rng.uniform(0, np.pi)
    period = _lerp(6.6, 8.4, s_tex) + rng.normal(0, 0.5)
    yy, xx = np.mgrid[0:size, 0:size]
    stripes = 0.5 + 0.5 * np.sin(
        2 * np.pi * (xx * np.cos(orientation) + yy * np.sin(orientation)) / max(period, 3.0)
    )
    # per-organelle variability: each tubule has its own cristae order, so the
    # class difference is a shift in a distribution, not a threshold
    local = ndi.gaussian_filter(rng.normal(0, 1, mask.shape).astype(np.float32), 2.5)
    local = (local - local.min()) / (np.ptp(local) + 1e-8) * 0.5 + 0.75   # 0.75..1.25
    order = np.clip(_lerp(0.86, 0.42, s_tex) * local, 0.12, 1.0)

    image += 0.50 * body
    image += 0.34 * body * order * stripes ** 1.6

    # disorganised cristae -> high-frequency speckle inside the organelle
    speckle = ndi.gaussian_filter(rng.normal(0, 1, mask.shape).astype(np.float32), 0.7)
    image += 0.16 * body * np.clip(1.0 - order, 0, None) * np.abs(speckle)

    # optical halo + per-FOV gain/shot noise
    halo = np.clip(body - ndi.gaussian_filter(mask.astype(np.float32), 2.4), 0, 1)
    image = np.clip(image + 0.20 * halo, 0, None)
    image *= float(rng.uniform(0.80, 1.22))
    image = np.clip(image + rng.normal(0, noise, mask.shape).astype(np.float32), 0, 1)
    return image.astype(np.float32), mask


def create_cohort(
    out_dir: str | Path,
    n_per_class: int = 30,
    size: int = 512,
    classes: tuple[str, ...] = ("control", "treated"),
    seed: int = 42,
) -> Path:
    """Write ``images/``, ``masks/`` and ``labels.csv`` for a demo cohort."""
    out_dir = Path(out_dir)
    image_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    rows = ["image_id,group,severity"]
    counter = {cls: 0 for cls in classes}
    for cls in classes:
        mu = CLASS_SEVERITY.get(cls, 0.5)
        for _ in range(n_per_class):
            severity = float(np.clip(rng.normal(mu, SEVERITY_SIGMA), 0.02, 0.98))
            image, mask = render_fov(rng, severity, size=size)
            name = f"{cls}_{counter[cls]:03d}"
            counter[cls] += 1
            Image.fromarray((image * 255).astype(np.uint8)).save(image_dir / f"{name}.png")
            Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_dir / f"{name}_mask.png")
            rows.append(f"{name},{cls},{severity:.3f}")

    (out_dir / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    log.warning(
        "generated a SYNTHETIC cohort (%d images) at %s - for demo/CI only, not real data",
        sum(counter.values()), out_dir,
    )
    return out_dir


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Generate a synthetic MitoSAM demo cohort")
    parser.add_argument("--out", default="examples/synthetic")
    parser.add_argument("--n", type=int, default=30, help="images per class")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--classes", nargs="+", default=["control", "treated"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_cohort(args.out, args.n, args.size, tuple(args.classes), args.seed)
