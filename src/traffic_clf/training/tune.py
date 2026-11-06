"""Choose hyperparameters on the validation split.

The search never reads the test split. Each trial is fit on the training
rows only. There is no row-wise cross-validation: a random fold would place
flows from the same capture on both sides of the score. After the search,
the winning parameters are fit once more on the training rows and the test
score is computed a single time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from traffic_clf.config import ProjectConfig
from traffic_clf.errors import ConfigError, SchemaError
from traffic_clf.models.build import MODEL_NAMES, build_model
from traffic_clf.seeding import seed_everything
from traffic_clf.training.metrics import score_split
from traffic_clf.training.run import (
    TrialRecord,
    TrainResult,
    fit_estimator,
    format_params,
    load_prepared,
    train_prepared,
)

_INSTALL = 'pip install "xgboost>=2" "lightgbm>=4" "catboost>=1.2" "optuna>=3.6"'


def tune_prepared(
    directory: Path,
    config: ProjectConfig,
    model_name: str,
    *,
    n_trials: int,
    out: Path | None = None,
) -> TrainResult:
    """Search ``n_trials`` settings, then train the winner on the training split."""
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ConfigError("n_trials must be a positive integer.")
    if model_name not in MODEL_NAMES:
        raise ConfigError(f"Unknown model {model_name!r}. Choose one of {', '.join(MODEL_NAMES)}.")
    if config.task is None:
        raise ConfigError("Tuning requires a task.")
    splits = load_prepared(directory)
    if splits["validation"][0].empty:
        raise SchemaError(
            "Tuning needs a validation split. The test split is not used to choose hyperparameters."
        )
    seed_everything(config.seed)
    optuna = _optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed))
    train_x, train_y = splits["train"]
    val_x, val_y = splits["validation"]

    def objective(trial: Any) -> float:
        params = suggest_params(trial, model_name)
        estimator = build_model(model_name, seed=config.seed, n_jobs=config.runtime.n_jobs, params=params)
        fitted = fit_estimator(estimator, model_name, train_x, train_y)
        classes = tuple(str(label) for label in fitted.classes_)
        score = score_split(
            "validation",
            val_y,
            fitted.predict(val_x),
            fitted.predict_proba(val_x),
            classes,
        )
        return score.metrics["f1_macro"]

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials = tuple(
        TrialRecord(trial.number, float(trial.value), format_params(trial.params))
        for trial in study.trials
        if trial.value is not None
    )
    return train_prepared(
        directory,
        config,
        model_name,
        out=out,
        params=dict(study.best_params),
        trials=trials,
    )


def suggest_params(trial: Any, model_name: str) -> dict[str, Any]:
    """Small CPU search space. The ranges stay inside the model builders."""
    if model_name == "logreg":
        return {"C": trial.suggest_float("C", 1e-2, 10.0, log=True)}
    if model_name == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 1, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 20, 60),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
        }
    if model_name in {"xgboost", "lightgbm"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 20, 60),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        }
    if model_name == "catboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 20, 60),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
        }
    raise ConfigError(f"Unknown model {model_name!r}. Choose one of {', '.join(MODEL_NAMES)}.")


def _optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise ConfigError(f"Optuna is not installed. From the repository run: {_INSTALL}") from exc
    return optuna
