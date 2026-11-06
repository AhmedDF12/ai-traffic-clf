"""Fit one model on the training parquet and score the held-out splits.

The scaler is already fitted. This module does not impute, scale, or read
captures. Validation can be empty for a single train. The test split cannot.
Monday's false-positive check is scored only when every row is benign.
Boosting models receive balanced sample weights. Baselines keep their own
class-weight argument instead, so the correction is not applied twice.
"""

from __future__ import annotations

import json
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from traffic_clf.config import ProjectConfig
from traffic_clf.data.schema import FORBIDDEN_FEATURES
from traffic_clf.errors import ConfigError, SchemaError, UnmappedLabelError
from traffic_clf.models.boosting import BOOST_MODELS
from traffic_clf.models.build import build_model
from traffic_clf.tracking import start_run
from traffic_clf.training.metrics import SplitScore, score_split

_REQUIRED_SPLITS = ("train", "validation", "test")


@dataclass
class TrainedModel:
    """A fitted baseline plus the column order it was trained on."""

    task: str
    model_name: str
    classes: tuple[str, ...]
    feature_columns: tuple[str, ...]
    estimator: object
    seed: int

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(self._align(features))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self._align(features))

    def _align(self, features: pd.DataFrame) -> pd.DataFrame:
        _reject_feature_frame(features)
        missing = [column for column in self.feature_columns if column not in features.columns]
        if missing:
            raise SchemaError("Prediction frame is missing features: " + ", ".join(missing))
        return features.loc[:, list(self.feature_columns)]


@dataclass(frozen=True)
class TrialRecord:
    """One Optuna trial. The value is validation macro-F1, never the test score."""

    number: int
    value: float
    params: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TrainResult:
    """One task, one model, and the scores that were not used for fitting."""

    model: TrainedModel
    train_rows: int
    classes_absent: tuple[str, ...]
    validation: SplitScore | None
    test: SplitScore
    fp_check: SplitScore | None
    false_positive_rate: float | None
    run_id: str
    params: tuple[tuple[str, str], ...] = ()
    trials: tuple[TrialRecord, ...] = ()

    def summary(self) -> str:
        lines = [
            f"model={self.model.model_name} task={self.model.task} seed={self.model.seed}",
            f"rows_train={self.train_rows} classes={','.join(self.model.classes)}",
        ]
        if self.classes_absent:
            lines.append("classes_absent=" + ",".join(self.classes_absent))
        if self.params:
            rendered = ",".join(f"{key}={value}" for key, value in self.params)
            lines.append(f"params={rendered}")
        if self.trials:
            lines.append(f"trials={len(self.trials)}")
        if self.validation is None:
            lines.append("f1_macro_validation=skipped")
        else:
            lines.append(f"f1_macro_validation={self.validation.metrics['f1_macro']:.4f}")
        lines.append(f"f1_macro_test={self.test.metrics['f1_macro']:.4f}")
        lines.append(f"f1_weighted_test={self.test.metrics['f1_weighted']:.4f}")
        lines.append(f"accuracy_test={self.test.metrics['accuracy']:.4f}")
        if self.false_positive_rate is not None:
            lines.append(f"false_positive_rate={self.false_positive_rate:.4f}")
        lines.append(f"run_id={self.run_id}")
        return "\n".join(lines)


def train_prepared(
    directory: Path,
    config: ProjectConfig,
    model_name: str,
    *,
    out: Path | None = None,
    params: dict[str, Any] | None = None,
    trials: tuple[TrialRecord, ...] = (),
) -> TrainResult:
    """Train on ``train_*.parquet`` and score validation, test, and fp_check."""
    if config.task is None:
        raise ConfigError("Training requires a task.")
    splits = load_prepared(directory)
    _validate_labels(splits, config.task.classes)
    train_x, train_y = splits["train"]
    if train_y.nunique() < 2:
        raise SchemaError("The training split has only one class, so a classifier cannot be fit.")
    if splits["test"][0].empty:
        raise SchemaError("The test split is empty, so this run would have no held-out score.")
    built = build_model(model_name, seed=config.seed, n_jobs=config.runtime.n_jobs, params=params)
    with start_run(config, f"{config.task.name}-{model_name}", tags={"model": model_name}) as run:
        estimator = fit_estimator(built, model_name, train_x, train_y)
        trained = TrainedModel(
            task=config.task.name,
            model_name=model_name,
            classes=tuple(str(label) for label in estimator.classes_),
            feature_columns=tuple(train_x.columns),
            estimator=estimator,
            seed=config.seed,
        )
        validation = _score_named(trained, splits["validation"], "validation")
        test = _score_named(trained, splits["test"], "test")
        if test is None:
            raise SchemaError("The test split is empty, so this run would have no held-out score.")
        fp_check = _score_named(trained, splits.get("fp_check"), "fp_check")
        false_positive_rate = _false_positive_rate(splits.get("fp_check"), trained)
        absent = tuple(label for label in config.task.classes if label not in trained.classes)
        result = TrainResult(
            model=trained,
            train_rows=int(len(train_x)),
            classes_absent=absent,
            validation=validation,
            test=test,
            fp_check=fp_check,
            false_positive_rate=false_positive_rate,
            run_id=run.info.run_id,
            params=format_params(params),
            trials=trials,
        )
        _log_result(result)
        if out is not None:
            write_train_result(result, out)
    return result


def load_trained_model(path: Path) -> TrainedModel:
    model = joblib.load(path)
    if not isinstance(model, TrainedModel):
        raise SchemaError(f"{path} is not a trained traffic-clf model.")
    return model


def load_prepared(directory: Path) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Read the feature and metadata parquet written by ``prepare``."""
    folder = Path(directory)
    if not folder.is_dir():
        raise SchemaError(f"Prepared directory not found: {folder}")
    loaded: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for name in (*_REQUIRED_SPLITS, "fp_check"):
        features_path = folder / f"{name}_features.parquet"
        meta_path = folder / f"{name}_meta.parquet"
        if name == "fp_check" and not features_path.is_file() and not meta_path.is_file():
            continue
        if not features_path.is_file() or not meta_path.is_file():
            raise SchemaError(f"{folder} is missing {name}_features.parquet or {name}_meta.parquet.")
        features = pd.read_parquet(features_path)
        meta = pd.read_parquet(meta_path)
        if len(features) != len(meta):
            raise SchemaError(f"{name} features and metadata have different numbers of rows.")
        if "label" not in meta.columns:
            raise SchemaError(f"{name} metadata has no label column.")
        _reject_feature_frame(features)
        labels = meta["label"].map(_label_text)
        loaded[name] = (features.reset_index(drop=True), labels.reset_index(drop=True))
    return loaded


def write_train_result(result: TrainResult, directory: Path) -> None:
    """Write the model, the metrics, and a short text report."""
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, directory / "model.joblib")
    payload = {
        "model": result.model.model_name,
        "task": result.model.task,
        "seed": result.model.seed,
        "run_id": result.run_id,
        "rows_train": result.train_rows,
        "classes": list(result.model.classes),
        "classes_absent": list(result.classes_absent),
        "feature_columns": list(result.model.feature_columns),
        "false_positive_rate": result.false_positive_rate,
        "params": dict(result.params),
        "trials": [
            {"number": trial.number, "value": trial.value, "params": dict(trial.params)}
            for trial in result.trials
        ],
        "splits": {
            score.name: {
                "rows": score.rows,
                "labels": list(score.labels),
                "metrics": score.metrics,
                "confusion": [list(row) for row in score.confusion],
            }
            for score in (result.validation, result.test, result.fp_check)
            if score is not None
        },
    }
    (directory / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (directory / "report.txt").write_text(result.summary() + "\n", encoding="utf-8")
    result.test.confusion_frame().to_csv(directory / "confusion_test.csv")
    if result.trials:
        pd.DataFrame(
            [
                {"number": trial.number, "value": trial.value, **dict(trial.params)}
                for trial in result.trials
            ]
        ).to_csv(directory / "trials.csv", index=False)


def _score_named(
    model: TrainedModel,
    split: tuple[pd.DataFrame, pd.Series] | None,
    name: str,
) -> SplitScore | None:
    if split is None or split[0].empty:
        return None
    features, labels = split
    return score_split(name, labels, model.predict(features), model.predict_proba(features), model.classes)


def _false_positive_rate(
    split: tuple[pd.DataFrame, pd.Series] | None,
    model: TrainedModel,
) -> float | None:
    if split is None or split[0].empty:
        return None
    features, labels = split
    if not bool((labels == "benign").all()):
        return None
    if "benign" not in model.classes:
        return None
    predicted = model.predict(features)
    return float(np.mean(predicted != "benign"))


def _validate_labels(splits: dict[str, tuple[pd.DataFrame, pd.Series]], classes: tuple[str, ...]) -> None:
    allowed = set(classes)
    for name, (_, labels) in splits.items():
        unknown = sorted(set(labels) - allowed)
        if unknown:
            raise UnmappedLabelError(
                f"{name} contains label(s) outside this task: {', '.join(unknown)}."
            )


def _log_result(result: TrainResult) -> None:
    import mlflow

    mlflow.log_params(
        {
            "model": result.model.model_name,
            "train_rows": str(result.train_rows),
            "classes_present": ",".join(result.model.classes),
            "classes_absent": ",".join(result.classes_absent),
            "tuned": "true" if result.trials else "false",
            **{f"hp_{key}": value for key, value in result.params},
        }
    )
    metrics: dict[str, float] = {}
    for score in (result.validation, result.test, result.fp_check):
        if score is None:
            continue
        for key, value in score.metrics.items():
            metrics[f"{key}_{score.name}"] = value
    if result.false_positive_rate is not None:
        metrics["false_positive_rate"] = result.false_positive_rate
    mlflow.log_metrics(metrics)
    mlflow.log_text(result.test.confusion_frame().to_csv(), "confusion_test.csv")
    if result.trials:
        mlflow.log_text(_trials_csv(result.trials), "trials.csv")
    # MLflow 3 saves scikit-learn models with skops and then refuses the tree
    # node array. The estimator was just fit in this process, so pickle is the copy we keep.
    stored = getattr(result.model.estimator, "estimator", result.model.estimator)
    mlflow.sklearn.log_model(stored, name="model", serialization_format="pickle")


class _LabelIndexedModel:
    """Map a booster's 0..N indexes back to the project class names.

    XGBoost rejects string labels. The saved training object still predicts
    names, and the probability columns stay in ``classes_`` order.
    """

    def __init__(self, estimator: object, classes: tuple[str, ...]) -> None:
        self.estimator = estimator
        self.classes_ = np.asarray(classes, dtype=object)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        indexes = np.asarray(self.estimator.predict(features)).astype(int).reshape(-1)
        return self.classes_[indexes]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(features)


def fit_estimator(estimator: object, model_name: str, features: pd.DataFrame, labels: pd.Series) -> object:
    """Fit on training rows. Boosting uses sample weights. Baselines do not."""
    y = labels.astype(str).to_numpy()
    if model_name in BOOST_MODELS:
        from sklearn.utils.class_weight import compute_sample_weight

        classes = tuple(sorted(set(y.tolist())))
        index = {label: position for position, label in enumerate(classes)}
        encoded = np.asarray([index[label] for label in y], dtype=int)
        estimator.fit(features, encoded, sample_weight=compute_sample_weight("balanced", y))
        return _LabelIndexedModel(estimator, classes)
    estimator.fit(features, y)
    return estimator


def format_params(params: dict[str, Any] | None) -> tuple[tuple[str, str], ...]:
    """Stable text form of a parameter dict for reports and MLflow."""
    if not params:
        return ()
    pairs = []
    for key in sorted(params):
        pairs.append((str(key), _param_text(params[key])))
    return tuple(pairs)


def _param_text(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return f"{float(value):.6g}"
    return str(value)


def _trials_csv(trials: tuple[TrialRecord, ...]) -> str:
    frame = pd.DataFrame(
        [{"number": trial.number, "value": trial.value, **dict(trial.params)} for trial in trials]
    )
    return frame.to_csv(index=False)


def _reject_feature_frame(features: pd.DataFrame) -> None:
    leaked = set(map(str, features.columns)) & FORBIDDEN_FEATURES
    if leaked:
        raise SchemaError(
            "Refusing to train on identifier columns: " + ", ".join(sorted(leaked))
        )
    if "protocol" in features.columns:
        raise SchemaError(
            "The protocol column is still numeric. Train on the prepared parquet, "
            "where protocol is protocol_tcp, protocol_udp, and protocol_other."
        )


def _label_text(value: object) -> str:
    if value is None or pd.isna(value):
        raise UnmappedLabelError("A prepared row has an empty label.")
    text = str(value).strip()
    if not text:
        raise UnmappedLabelError("A prepared row has an empty label.")
    return text
