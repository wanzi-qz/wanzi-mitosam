"""Backbone bridge to the upstream ``segment_anything`` package.

MitoSAM deliberately does **not** vendor or re-implement SAM. It depends on the
official package and talks to it through this single module, so upgrading SAM
(or swapping in SAM2) only touches one file.

Resolution order for the SAM registry:

1. ``import segment_anything`` (installed, e.g. ``pip install git+...``)
2. ``SAMSAM_SRC`` env var pointing at a cloned ``segment-anything`` repo
3. ``third_party/segment_anything`` inside this repository
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..utils.logging import get_logger

log = get_logger("models.sam")

_CACHE: dict = {}


def _candidate_paths() -> list[Path]:
    here = Path(__file__).resolve()
    root = here.parents[2]  # MitoSAM/models/bridge.py -> repo root
    env = os.environ.get("MITOSAM_SAM_SRC") or os.environ.get("SAMSAM_SRC")
    candidates = [root / "third_party", root]
    if env:
        candidates.insert(0, Path(env))
    return candidates


def import_sam():
    """Import ``segment_anything`` and return the module (cached)."""
    if "mod" in _CACHE:
        return _CACHE["mod"]
    try:
        import segment_anything  # type: ignore

        _CACHE["mod"] = segment_anything
        return segment_anything
    except ImportError:
        pass

    for base in _candidate_paths():
        pkg = base / "segment_anything"
        if pkg.is_dir() and str(base) not in sys.path:
            sys.path.insert(0, str(base))
            try:
                import segment_anything  # type: ignore

                log.info("using vendored segment_anything from %s", base)
                _CACHE["mod"] = segment_anything
                return segment_anything
            except ImportError:
                sys.path.pop(0)

    raise ImportError(
        "segment_anything not found. Install it with\n"
        "    pip install git+https://github.com/facebookresearch/segment-anything.git\n"
        "or set MITOSAM_SAM_SRC=/path/to/segment-anything."
    )


def build_sam(sam_type: str = "vit_b", checkpoint: str | Path | None = None):
    """Instantiate a SAM model. ``checkpoint`` may be ``None`` for random init."""
    sam_mod = import_sam()
    registry = sam_mod.sam_model_registry
    if sam_type not in registry:
        raise KeyError(f"Unknown SAM type {sam_type!r}; choose from {list(registry)}")
    ckpt = str(checkpoint) if checkpoint and Path(checkpoint).exists() else None
    if checkpoint and ckpt is None:
        log.warning("checkpoint %s not found -> random init (fine-tuning from scratch)", checkpoint)
    return registry[sam_type](checkpoint=ckpt)
