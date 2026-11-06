"""Dispatch a model name to a baseline or a boosting model."""

from __future__ import annotations

from typing import Any

from sklearn.base import ClassifierMixin

from traffic_clf.errors import ConfigError
from traffic_clf.models.baselines import BASELINE_MODELS, build_baseline
from traffic_clf.models.boosting import BOOST_MODELS, build_boosting

MODEL_NAMES = BASELINE_MODELS + BOOST_MODELS


def build_model(
    name: str,
    *,
    seed: int,
    n_jobs: int,
    params: dict[str, Any] | None = None,
) -> ClassifierMixin:
    """Build one classifier. Unknown parameter names are rejected."""
    if name in BASELINE_MODELS:
        return build_baseline(name, seed=seed, n_jobs=n_jobs, params=params)
    if name in BOOST_MODELS:
        return build_boosting(name, seed=seed, n_jobs=n_jobs, params=params)
    raise ConfigError(f"Unknown model {name!r}. Choose one of {', '.join(MODEL_NAMES)}.")
