"""Tiny logging helper so every module reports the same way."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FMT = "[%(asctime)s] %(levelname)-7s %(name)-18s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    root = logging.getLogger("mitosam")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"mitosam.{name}")
