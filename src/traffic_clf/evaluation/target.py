"""Stronger models chosen by the weakest class on the validation split.

The test split is scored once, after the model and the class weights are
fixed. A class that exists in only one capture source stays
environment-confounded, and a dominated split is not a headline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from traffic_clf.data.schema import FeatureSpec
from traffic_clf.data.splits import assign_grouped_split
from traffic_clf.errors import SplitError
from traffic_clf.evaluation.protocol import (
    _attach_host,
    _groups,
    _prepare_frames,
    _source_aware_dominance,
    cap_flows_per_capture,
    class_source_control,
    experiment_source,
)
from traffic_clf.models.build import build_model
from traffic_clf.preprocess.prepare import fit_preprocess
from traffic_clf.training.metrics import score_split
from traffic_clf.training.run import fit_estimator

TARGET_RECALL = 0.80
CANDIDATES: tuple[tuple[str, dict[str, float | int]], ...] = (
    ("xgboost", {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 8, "subsample": 0.9}),
    ("lightgbm", {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 8, "subsample": 0.9}),
    ("random_forest", {"n_estimators": 300, "max_depth": 24}),
    ("catboost", {"n_estimators": 200, "learning_rate": 0.08, "max_depth": 6}),
)


def choose_class_weights(
    proba: np.ndarray,
    y_true: np.ndarray,
    classes: tuple[str, ...],
    *,
    target_recall: float = TARGET_RECALL,
) -> np.ndarray:
    """Raise the weakest class on validation until the recall target is met or the search stops."""
    weights = np.ones(len(classes), dtype=float)
    truth = np.asarray(y_true).astype(str)
    best_weights = weights.copy()
    best_key = (-1.0, -1.0)
    for _ in range(24):
        predicted = _predict(proba, classes, weights)
        recalls = _recalls(truth, predicted, classes)
        if not recalls:
            break
        minimum = min(recalls.values())
        macro = float(f1_score(truth, predicted, average="macro", labels=list(classes), zero_division=0))
        if (minimum, macro) > best_key:
            best_key = (minimum, macro)
            best_weights = weights.copy()
        if minimum >= target_recall:
            break
        worst = min(recalls, key=recalls.get)
        weights[classes.index(worst)] *= 1.25
    return best_weights


def run_per_class_target(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    task: str,
    seed: int,
    n_jobs: int,
    max_flows_per_capture: int,
    out: Path,
    sessions: pd.DataFrame | None = None,
    candidates: tuple[tuple[str, dict[str, float | int]], ...] = CANDIDATES,
    target_recall: float = TARGET_RECALL,
) -> dict[str, object]:
    """Fit each source on its own capped file split and aim at the weakest class."""
    frame, labels = _prepare_frames(features, meta)
    labels = _attach_host(labels, sessions)
    labels["experiment_source"] = experiment_source(labels)
    control = class_source_control(labels)
    experiments = []
    for source_name in ("iscx", "vlc"):
        mask = labels["experiment_source"] == source_name
        if int(mask.sum()) == 0:
            experiments.append({"experiment": f"{source_name}_only_target", "status": "no rows", "headline": False})
            continue
        capped_features, capped_meta = cap_flows_per_capture(
            frame.loc[mask].reset_index(drop=True),
            labels.loc[mask].reset_index(drop=True),
            max_flows_per_capture,
            seed=seed,
            capture_column="capture_id",
        )
        experiments.append(
            _fit_source(
                capped_features,
                capped_meta,
                spec,
                task=task,
                source_name=source_name,
                seed=seed,
                n_jobs=n_jobs,
                candidates=candidates,
                target_recall=target_recall,
                confounded={str(row["class"]): bool(row["environment_confounded"]) for row in control},
            )
        )
    payload = {
        "task": task,
        "seed": seed,
        "tuned_on": "validation",
        "target_recall": target_recall,
        "max_flows_per_capture": max_flows_per_capture,
        "headline": None,
        "headline_reason": "Per-class scores stay inside one source. A class from one source is environment-confounded.",
        "class_source_control": control,
        "experiments": experiments,
    }
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{task}_target.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print_target(payload)
    return payload


def _fit_source(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    task: str,
    source_name: str,
    seed: int,
    n_jobs: int,
    candidates: tuple[tuple[str, dict[str, float | int]], ...],
    target_recall: float,
    confounded: dict[str, bool],
) -> dict[str, object]:
    experiment = f"{source_name}_only_target_cap"
    try:
        assignment = assign_grouped_split(_groups(meta), seed=seed)
    except SplitError as exc:
        return {"experiment": experiment, "source": source_name, "status": str(exc), "headline": False}
    labeled = meta.copy()
    labeled["split"] = labeled["capture_id"].map(assignment)
    train = labeled["split"] == "train"
    validation = labeled["split"] == "validation"
    test = labeled["split"] == "test"
    if int(train.sum()) == 0 or labeled.loc[train, "label"].nunique() < 2 or int(validation.sum()) == 0:
        return {"experiment": experiment, "source": source_name, "status": "training split cannot fit a classifier", "headline": False}
    state = fit_preprocess(features.loc[train], spec)
    transformed = state.transform(features)
    ranked = []
    for model_name, params in candidates:
        fitted = fit_estimator(
            build_model(model_name, seed=seed, n_jobs=n_jobs, params=params),
            model_name,
            transformed.loc[train],
            labeled.loc[train, "label"],
        )
        classes = tuple(str(label) for label in fitted.classes_)
        val_proba = np.asarray(fitted.predict_proba(transformed.loc[validation]))
        weights = choose_class_weights(
            val_proba,
            labeled.loc[validation, "label"].to_numpy(),
            classes,
            target_recall=target_recall,
        )
        val_pred = _predict(val_proba, classes, weights)
        val_recalls = _recalls(labeled.loc[validation, "label"].to_numpy(), val_pred, classes)
        minimum = min(val_recalls.values()) if val_recalls else 0.0
        macro = float(f1_score(labeled.loc[validation, "label"].astype(str), val_pred, average="macro", zero_division=0))
        ranked.append((minimum, macro, model_name, params, fitted, classes, weights))
    minimum, macro, model_name, params, fitted, classes, weights = max(ranked, key=lambda item: (item[0], item[1]))
    blocks = []
    for split_name, mask in (("validation", validation), ("test", test)):
        if int(mask.sum()) == 0:
            continue
        proba = np.asarray(fitted.predict_proba(transformed.loc[mask]))
        predicted = _predict(proba, classes, weights)
        scored = score_split(split_name, labeled.loc[mask, "label"], predicted, proba, classes)
        recalls = _recalls(labeled.loc[mask, "label"].to_numpy(), predicted, classes)
        per_class = {
            label: {
                "recall": recalls.get(label),
                "support": int((labeled.loc[mask, "label"].astype(str) == label).sum()),
                "meets_0.80": bool(recalls.get(label, 0.0) >= target_recall),
                "environment_confounded": bool(confounded.get(label, False)),
            }
            for label in classes
        }
        note = _source_aware_dominance(labeled, split_name)
        below = [label for label, row in per_class.items() if not row["meets_0.80"]]
        confounded_labels = [label for label, row in per_class.items() if row["environment_confounded"]]
        note = dict(note)
        reasons = [str(note.get("reason") or "")]
        if below:
            note["headline"] = False
            reasons.append("recall below 0.80: " + ", ".join(below))
        if confounded_labels:
            note["headline"] = False
            reasons.append("environment-confounded: " + ", ".join(confounded_labels))
        note["reason"] = "; ".join(part for part in reasons if part)
        blocks.append(
            {
                "split": split_name,
                "headline": bool(note.get("headline", False)) if split_name == "test" else False,
                "rows": scored.rows,
                "accuracy": scored.metrics["accuracy"],
                "macro_f1": scored.metrics["f1_macro"],
                "min_class_recall": min(recalls.values()) if recalls else None,
                "per_class": per_class,
                "dominance": note,
            }
        )
    return {
        "experiment": experiment,
        "task": task,
        "source": source_name,
        "model": model_name,
        "params": {key: value for key, value in params.items()},
        "validation_min_class_recall": minimum,
        "validation_macro_f1": macro,
        "class_weights": {label: float(weight) for label, weight in zip(classes, weights, strict=True)},
        "tuned_on": "validation",
        "headline": False,
        "splits": blocks,
    }


def _predict(proba: np.ndarray, classes: tuple[str, ...], weights: np.ndarray) -> np.ndarray:
    scaled = np.asarray(proba, dtype=float) * weights.reshape(1, -1)
    return np.asarray(classes, dtype=object)[scaled.argmax(axis=1)].astype(str)


def _recalls(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[str, ...]) -> dict[str, float]:
    truth = np.asarray(y_true).astype(str)
    predicted = np.asarray(y_pred).astype(str)
    recalls = {}
    for label in classes:
        mask = truth == label
        if int(mask.sum()) == 0:
            continue
        recalls[label] = float(np.mean(predicted[mask] == label))
    return recalls


def _print_target(payload: dict[str, object]) -> None:
    print("task", payload["task"], "per-class target", payload["target_recall"], "chosen on", payload["tuned_on"])
    for row in payload["class_source_control"]:
        flag = " environment-confounded" if row["environment_confounded"] else ""
        print(" class", row["class"], "sources", row["sources"], flag)
    for experiment in payload["experiments"]:
        print(
            "experiment",
            experiment.get("experiment"),
            "model",
            experiment.get("model"),
            "headline",
            experiment.get("headline", False),
        )
        if experiment.get("status"):
            print(" ", experiment["status"])
            continue
        print(
            " validation_min_class_recall",
            experiment.get("validation_min_class_recall"),
            "validation_macro_f1",
            experiment.get("validation_macro_f1"),
            "weights",
            experiment.get("class_weights"),
        )
        for block in experiment.get("splits", []):
            print(
                f"experiment={experiment.get('experiment')} split={block['split']} model={experiment.get('model')} "
                f"headline={block['headline']} rows={block['rows']} accuracy={block['accuracy']:.4f} "
                f"macro_f1={block['macro_f1']:.4f} min_class_recall={block['min_class_recall']}"
            )
            if block["dominance"].get("reason"):
                print(" dominance", block["dominance"]["reason"])
            for label, scores in block["per_class"].items():
                flag = " environment-confounded" if scores["environment_confounded"] else ""
                print(
                    f"  class={label} recall={scores['recall']:.4f} support={scores['support']} "
                    f"meets_0.80={scores['meets_0.80']}{flag}"
                )
