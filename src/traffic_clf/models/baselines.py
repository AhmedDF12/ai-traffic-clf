"""Scikit-learn baselines.

Class weights stand in for resampling. SMOTE is not applied here, because
synthetic flows can invent combinations the network never produced.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from traffic_clf.errors import ConfigError

BASELINE_MODELS = ("logreg", "decision_tree", "random_forest")
_ALLOWED = {
    "logreg": {"C"},
    "decision_tree": {"max_depth", "min_samples_leaf"},
    "random_forest": {"n_estimators", "max_depth"},
}


def build_baseline(
    name: str,
    *,
    seed: int,
    n_jobs: int,
    params: dict[str, Any] | None = None,
) -> ClassifierMixin:
    """Return one baseline with the project seed and balanced class weights."""
    if name not in BASELINE_MODELS:
        raise ConfigError(f"Unknown model {name!r}. Choose one of {', '.join(BASELINE_MODELS)}.")
    chosen = _checked(name, params)
    if name == "logreg":
        return LogisticRegression(
            C=float(chosen.get("C", 1.0)),
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=_optional_int(chosen.get("max_depth")),
            min_samples_leaf=int(chosen.get("min_samples_leaf", 1)),
            class_weight="balanced",
            random_state=seed,
        )
    return RandomForestClassifier(
        n_estimators=int(chosen.get("n_estimators", 100)),
        max_depth=_optional_int(chosen.get("max_depth")),
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
    )


def _checked(name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    chosen = {} if params is None else dict(params)
    unknown = sorted(set(chosen) - _ALLOWED[name])
    if unknown:
        raise ConfigError(f"{name} does not accept parameter(s): {', '.join(unknown)}.")
    return chosen


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
