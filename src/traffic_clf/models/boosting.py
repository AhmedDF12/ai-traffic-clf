"""CPU gradient-boosting models.

Imports stay inside the builder so the rest of the package loads when these
libraries are not installed. SHAP and the neural nets are not part of this
module. Sample weights balance the classes. The constructors do not also
set a class-weight argument, which would apply the correction twice.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import ClassifierMixin

from traffic_clf.errors import ConfigError

BOOST_MODELS = ("xgboost", "lightgbm", "catboost")
_INSTALL = 'pip install "xgboost>=2" "lightgbm>=4" "catboost>=1.2" "optuna>=3.6"'
_ALLOWED = {
    "xgboost": {"n_estimators", "learning_rate", "max_depth", "subsample"},
    "lightgbm": {"n_estimators", "learning_rate", "max_depth", "subsample"},
    "catboost": {"n_estimators", "learning_rate", "max_depth"},
}


def build_boosting(name: str, *, seed: int, n_jobs: int, params: dict[str, Any] | None = None) -> ClassifierMixin:
    """Return one boosting model. ``params`` overrides the fixed defaults."""
    if name not in BOOST_MODELS:
        raise ConfigError(f"Unknown boosting model {name!r}. Choose one of {', '.join(BOOST_MODELS)}.")
    chosen = _checked(name, params)
    if name == "xgboost":
        return _xgboost(chosen, seed=seed, n_jobs=n_jobs)
    if name == "lightgbm":
        return _lightgbm(chosen, seed=seed, n_jobs=n_jobs)
    return _catboost(chosen, seed=seed, n_jobs=n_jobs)


def _xgboost(params: dict[str, Any], *, seed: int, n_jobs: int) -> ClassifierMixin:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ConfigError(f"XGBoost is not installed. From the repository run: {_INSTALL}") from exc
    return XGBClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        max_depth=int(params.get("max_depth", 6)),
        subsample=float(params.get("subsample", 1.0)),
        tree_method="hist",
        device="cpu",
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=0,
    )


def _lightgbm(params: dict[str, Any], *, seed: int, n_jobs: int) -> ClassifierMixin:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ConfigError(f"LightGBM is not installed. From the repository run: {_INSTALL}") from exc
    return LGBMClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        max_depth=int(params.get("max_depth", 6)),
        subsample=float(params.get("subsample", 1.0)),
        subsample_freq=1,
        min_child_samples=1,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )


def _catboost(params: dict[str, Any], *, seed: int, n_jobs: int) -> ClassifierMixin:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ConfigError(f"CatBoost is not installed. From the repository run: {_INSTALL}") from exc
    return CatBoostClassifier(
        iterations=int(params.get("n_estimators", 100)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        depth=int(params.get("max_depth", 6)),
        random_seed=seed,
        thread_count=n_jobs,
        verbose=False,
        allow_writing_files=False,
    )


def _checked(name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    chosen = {} if params is None else dict(params)
    unknown = sorted(set(chosen) - _ALLOWED[name])
    if unknown:
        raise ConfigError(f"{name} does not accept parameter(s): {', '.join(unknown)}.")
    return chosen
