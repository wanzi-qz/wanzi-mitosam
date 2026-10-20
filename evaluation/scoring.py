"""Scoring helpers shared by the ablation table and the report.

Reported alongside AUC because a single accuracy number hides the two failure
modes that reviewers (and interviewers) look for: over-fitting (big CV vs test
gap) and unstable models (large CV standard deviation).
"""

from __future__ import annotations


def generalization_gap(test_auc: float, cv_auc: float) -> float:
    """|test - CV|; small is trustworthy. NaN-safe."""
    try:
        return abs(float(test_auc) - float(cv_auc))
    except (TypeError, ValueError):
        return float("nan")


def stability_score(cv_mean: float, cv_std: float, test_auc: float) -> float:
    """0-1: rewards high CV AUC and penalises variance / test-CV disagreement."""
    try:
        cv_mean, cv_std, test_auc = float(cv_mean), float(cv_std), float(test_auc)
    except (TypeError, ValueError):
        return 0.0
    if any(v != v for v in (cv_mean, cv_std, test_auc)):  # NaN check
        return 0.0
    variance_penalty = min(cv_std / 0.2, 1.0) * 0.5
    gap_penalty = min(abs(test_auc - cv_mean) / 0.3, 1.0) * 0.5
    return round(max(0.0, cv_mean * (1 - variance_penalty - gap_penalty)), 4)


def composite_score(test_auc: float, cv_auc: float, cv_std: float,
                    biology_weight: float = 0.1, stability: float | None = None) -> float:
    """Weighted blend: test AUC (55%), CV AUC (30%), stability (15%).

    ``biology_weight`` is kept in the signature for parity with the thesis
    script, which included a hand-assigned biological-plausibility score. It is
    deliberately not used here: a manually assigned bonus is not reproducible.
    """
    try:
        test_auc, cv_auc, cv_std = float(test_auc), float(cv_auc), float(cv_std)
    except (TypeError, ValueError):
        return 0.0
    if any(v != v for v in (test_auc, cv_auc, cv_std)):
        return 0.0
    if stability is None:
        stability = stability_score(cv_auc, cv_std, test_auc)
    return round(0.55 * test_auc + 0.30 * cv_auc + 0.15 * stability, 4)
