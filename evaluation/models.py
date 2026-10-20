"""Model factory with graceful degradation when optional boosters are missing."""

from __future__ import annotations

SUPPORTED_MODELS = ["logreg", "random_forest", "extra_trees", "svm", "xgboost", "lightgbm"]

_DISPLAY = {
    "logreg": "LogisticRegression",
    "random_forest": "RandomForest",
    "extra_trees": "ExtraTrees",
    "svm": "SVM-RBF",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def model_display_name(key: str) -> str:
    return _DISPLAY.get(key, key)


def build_model(key: str, random_state: int = 42):
    """Instantiate an estimator by short key.

    Raises:
        ImportError: when an optional dependency (xgboost / lightgbm) is absent,
            so the caller can skip that row instead of failing the whole run.
    """
    if key == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
    if key == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
    if key == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced",
                                  random_state=random_state)
    if key == "svm":
        from sklearn.svm import SVC

        return SVC(C=1.0, kernel="rbf", probability=True,
                   class_weight="balanced", random_state=random_state)
    if key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError("xgboost not installed (pip install xgboost)") from exc
        return XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.8, eval_metric="logloss", random_state=random_state, n_jobs=-1,
        )
    if key == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError("lightgbm not installed (pip install lightgbm)") from exc
        return LGBMClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.9,
            random_state=random_state, n_jobs=-1, verbose=-1,
        )
    raise ValueError(f"Unknown model key: {key!r}. Choose from {SUPPORTED_MODELS}")
