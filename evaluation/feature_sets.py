"""Feature-subset definitions used by the ablation study.

Keeping the subsets in one place guarantees the ablation table, the SHAP run and
the GUI all talk about the same columns.
"""

from __future__ import annotations

from ..features.extract import feature_family

SUBSET_DEFINITIONS: dict[str, list[str]] = {
    "morphology_only": ["morphology"],
    "texture_only": ["texture"],
    "spatial_only": ["spatial"],
    "morphology_texture": ["morphology", "texture"],
    "texture_spatial": ["texture", "spatial"],
    "morphology_spatial": ["morphology", "spatial"],
    "full_features": ["morphology", "texture", "spatial"],
}


def resolve_subset(subset: str, columns: list[str]) -> list[str]:
    """Return the columns of ``columns`` that belong to ``subset``."""
    families = SUBSET_DEFINITIONS.get(subset)
    if families is None:
        if subset in ("all", "full"):
            return list(columns)
        families = [subset]
    families_set = set(families)
    return [c for c in columns if feature_family(c) in families_set]
