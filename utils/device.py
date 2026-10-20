"""Device resolution: explicit -> cuda -> mps -> cpu."""

from __future__ import annotations


def resolve_device(preferred: str | None = None) -> str:
    """Pick the best available torch device.

    Args:
        preferred: "auto" / "cpu" / "cuda" / "cuda:1" / "mps" or ``None`` (= auto).
    """
    preferred = (preferred or "auto").lower()
    if preferred not in {"auto", "cuda", "mps", "cpu"} and not preferred.startswith("cuda"):
        raise ValueError(f"Unsupported device: {preferred!r}")

    if preferred == "cpu":
        return "cpu"

    try:
        import torch

        if preferred.startswith("cuda"):
            if not torch.cuda.is_available():
                return "cpu"
            return preferred if ":" in preferred else "cuda"
        if preferred == "mps":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        # auto
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:  # torch not installed -> feature/eval-only usage
        return "cpu"
