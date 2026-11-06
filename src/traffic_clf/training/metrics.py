"""Held-out scores for one split. The training rows are not scored here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class SplitScore:
    """Metrics for validation, test, or the Monday false-positive check."""

    name: str
    rows: int
    labels: tuple[str, ...]
    metrics: dict[str, float]
    confusion: tuple[tuple[int, ...], ...]

    def confusion_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.confusion, index=list(self.labels), columns=list(self.labels))


def score_split(
    name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    proba_classes: tuple[str, ...],
) -> SplitScore:
    """Score predictions. Labels that the model never saw stay in the report."""
    truth = y_true.astype(str).to_numpy()
    predicted = np.asarray(y_pred).astype(str)
    labels = _metric_labels(truth, proba_classes)
    scores = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision_macro": _average(precision_score, truth, predicted, labels),
        "recall_macro": _average(recall_score, truth, predicted, labels),
        "f1_macro": _average(f1_score, truth, predicted, labels),
        "precision_weighted": _average(precision_score, truth, predicted, labels, weighted=True),
        "recall_weighted": _average(recall_score, truth, predicted, labels, weighted=True),
        "f1_weighted": _average(f1_score, truth, predicted, labels, weighted=True),
    }
    roc = _roc_auc(truth, y_proba, proba_classes)
    pr = _pr_auc(truth, y_proba, proba_classes)
    if roc is not None:
        scores["roc_auc_macro"] = roc
    if pr is not None:
        scores["pr_auc_macro"] = pr
    matrix = confusion_matrix(truth, predicted, labels=list(labels))
    return SplitScore(
        name=name,
        rows=int(len(truth)),
        labels=labels,
        metrics=scores,
        confusion=tuple(tuple(int(value) for value in row) for row in matrix),
    )


def _metric_labels(y_true: np.ndarray, proba_classes: tuple[str, ...]) -> tuple[str, ...]:
    extra = sorted(set(y_true) - set(proba_classes))
    return tuple(dict.fromkeys([*proba_classes, *extra]))


def _average(metric, y_true, y_pred, labels: tuple[str, ...], *, weighted: bool = False) -> float:
    return float(
        metric(
            y_true,
            y_pred,
            labels=list(labels),
            average="weighted" if weighted else "macro",
            zero_division=0,
        )
    )


def _roc_auc(y_true: np.ndarray, proba: np.ndarray, classes: tuple[str, ...]) -> float | None:
    present = [label for label in classes if np.any(y_true == label)]
    if len(present) < 2:
        return None
    if len(present) == 2:
        positive = present[1]
        observed = (y_true == positive).astype(int)
        return float(roc_auc_score(observed, proba[:, classes.index(positive)]))
    indexes = [classes.index(label) for label in present]
    return float(
        roc_auc_score(
            y_true,
            proba[:, indexes],
            multi_class="ovr",
            average="macro",
            labels=present,
        )
    )


def _pr_auc(y_true: np.ndarray, proba: np.ndarray, classes: tuple[str, ...]) -> float | None:
    present = [label for label in classes if np.any(y_true == label)]
    if len(present) < 2:
        return None
    scores = []
    for index, label in enumerate(present):
        observed = (y_true == label).astype(int)
        scores.append(average_precision_score(observed, proba[:, classes.index(label)]))
    return float(np.mean(scores))
