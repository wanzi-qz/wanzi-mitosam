"""Evaluation: segmentation metrics, classification + ablation, SHAP, figures."""

from .scoring import composite_score, generalization_gap, stability_score
from .feature_sets import SUBSET_DEFINITIONS, resolve_subset
from .models import SUPPORTED_MODELS, build_model, model_display_name

__all__ = [
    "composite_score",
    "generalization_gap",
    "stability_score",
    "SUBSET_DEFINITIONS",
    "resolve_subset",
    "SUPPORTED_MODELS",
    "build_model",
    "model_display_name",
]
