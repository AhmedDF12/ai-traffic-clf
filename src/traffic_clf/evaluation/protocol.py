"""File-level evaluation that a single capture cannot decide.

The cap, the grouped folds, and the within-day security cut are separate
experiments. Every stored score names its experiment, split, and model.
A test score is not a headline when one capture or one class is more than
half of that split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier

from traffic_clf.data.schema import FeatureSpec
from traffic_clf.data.splits import Group, assign_grouped_split, load_named_split
from traffic_clf.errors import SchemaError, SplitError
from traffic_clf.models.build import build_model
from traffic_clf.preprocess.prepare import fit_preprocess
from traffic_clf.training.metrics import score_split
from traffic_clf.training.run import fit_estimator

COMPARE_MODELS = ("random_forest", "xgboost", "lightgbm", "catboost")
DOMINANCE_SHARE = 0.5
CHANCE_MARGIN = 0.10


def cap_flows_per_capture(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    max_flows: int,
    *,
    seed: int,
    capture_column: str = "capture_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep at most ``max_flows`` rows from each capture, in stable order."""
    if isinstance(max_flows, bool) or not isinstance(max_flows, int) or max_flows < 1:
        raise SchemaError("max_flows_per_capture must be a positive integer.")
    if capture_column not in meta.columns:
        raise SchemaError(f"Metadata is missing {capture_column!r}.")
    if len(features) != len(meta):
        raise SchemaError("Features and metadata have different numbers of rows.")
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for positions in meta.groupby(capture_column, sort=False).indices.values():
        positions = np.asarray(positions)
        if len(positions) > max_flows:
            positions = np.sort(rng.choice(positions, size=max_flows, replace=False))
        chosen.extend(int(position) for position in positions)
    keep = np.asarray(sorted(chosen), dtype=int)
    return features.iloc[keep].reset_index(drop=True), meta.iloc[keep].reset_index(drop=True)


def assign_group_kfold(
    groups: list[Group] | tuple[Group, ...],
    *,
    n_folds: int = 5,
    min_files: int = 5,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[str], str, int]:
    """Hold out each capture in turn, using only classes that have enough files.

    Returns the fold assignments, the excluded class names, the largest
    included capture, and the fold where that capture is held out.
    """
    if n_folds < 2:
        raise SplitError("GroupKFold needs at least 2 folds.")
    if min_files < n_folds:
        raise SplitError("A class needs at least one file per fold, so min_files must be at least n_folds.")
    by_stratum: dict[str, list[Group]] = {}
    for group in groups:
        by_stratum.setdefault(group.stratum, []).append(group)
    included = {name: items for name, items in by_stratum.items() if len(items) >= min_files}
    excluded = sorted(name for name, items in by_stratum.items() if len(items) < min_files)
    if not included:
        raise SplitError(f"No class has {min_files} capture files, so {n_folds}-fold GroupKFold was not built.")
    rng = np.random.default_rng(seed)
    folds: list[dict[str, str]] = [{} for _ in range(n_folds)]
    for name in sorted(included):
        items = list(included[name])
        order = rng.permutation(len(items))
        for position, index in enumerate(order):
            group = items[int(index)]
            test_fold = position % n_folds
            for fold in range(n_folds):
                folds[fold][group.group_id] = "test" if fold == test_fold else "train"
    largest = max((group for items in included.values() for group in items), key=lambda group: group.weight)
    largest_fold = next(index for index, fold in enumerate(folds) if fold[largest.group_id] == "test")
    for fold in folds:
        held = {group.stratum for group in groups if fold.get(group.group_id) == "test"}
        missing = sorted(set(included) - held)
        if missing:
            raise SplitError(f"A test fold is missing classes: {', '.join(missing)}.")
    return folds, excluded, largest.group_id, largest_fold


def assign_within_day(
    meta: pd.DataFrame,
    *,
    fp_check_files: set[str],
    file_column: str = "capture_file",
    label_column: str = "label",
    time_column: str = "timestamp",
) -> pd.DataFrame:
    """Cut each class inside a day by time. Monday stays the false-positive check.

    Earliest rows of a class go to train, the middle to validation, and the
    latest to test. A class that exists on only one day is still only that day.
    """
    required = {file_column, label_column, time_column}
    missing = sorted(required - set(meta.columns))
    if missing:
        raise SchemaError("Within-day split is missing columns: " + ", ".join(missing))
    assigned = meta.copy()
    assigned["split"] = pd.Series(pd.NA, index=assigned.index, dtype="object")
    names = assigned[file_column].map(lambda value: Path(str(value)).name)
    assigned.loc[names.isin(fp_check_files), "split"] = "fp_check"
    pending = assigned["split"].isna()
    times = _as_utc(assigned[time_column])
    for (_, _), part in assigned.loc[pending].groupby([file_column, label_column], sort=False):
        order = times.loc[part.index].sort_values(kind="mergesort").index.to_list()
        n_rows = len(order)
        if n_rows < 3:
            assigned.loc[order, "split"] = "train"
            continue
        n_train = max(1, int(round(n_rows * 0.6)))
        n_val = max(1, int(round(n_rows * 0.2)))
        if n_train + n_val >= n_rows:
            n_train = max(1, n_rows - 2)
            n_val = 1
        assigned.loc[order[:n_train], "split"] = "train"
        assigned.loc[order[n_train : n_train + n_val], "split"] = "validation"
        assigned.loc[order[n_train + n_val :], "split"] = "test"
    if assigned["split"].isna().any():
        raise SchemaError("Within-day split left rows without a split.")
    return assigned


def dominance(meta: pd.DataFrame, split_name: str) -> dict[str, object]:
    """Say whether one capture or one class is more than half of a split."""
    part = meta.loc[meta["split"] == split_name]
    if part.empty:
        return {"split": split_name, "rows": 0, "headline": False, "reason": "split is empty"}
    files = part["capture_file"].astype(str).value_counts(normalize=True)
    classes = part["label"].astype(str).value_counts(normalize=True)
    file_share = float(files.iloc[0])
    class_share = float(classes.iloc[0])
    reasons = []
    if file_share > DOMINANCE_SHARE:
        reasons.append(f"capture {files.index[0]} is {file_share:.3f} of {split_name} rows")
    if class_share > DOMINANCE_SHARE:
        reasons.append(f"class {classes.index[0]} is {class_share:.3f} of {split_name} rows")
    return {
        "split": split_name,
        "rows": int(len(part)),
        "largest_file": str(files.index[0]),
        "largest_file_share": file_share,
        "largest_class": str(classes.index[0]),
        "largest_class_share": class_share,
        "headline": not reasons,
        "reason": "; ".join(reasons),
    }


def environment_inventory(meta: pd.DataFrame, sessions: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per class and capture: source, host, date, and session count."""
    frame = meta.copy()
    frame["capture_name"] = frame["capture_file"].map(lambda value: Path(str(value)).name)
    frame["when"] = _as_utc(frame["timestamp"]) if "timestamp" in frame.columns else pd.Series(pd.NaT, index=frame.index)
    hosts = _hosts_by_file(sessions)
    rows = []
    for (label, capture), part in frame.groupby(["label", "capture_name"], sort=False):
        sources = sorted(set(part["source_dataset"].astype(str))) if "source_dataset" in part.columns else []
        dates = sorted({value.date().isoformat() for value in part["when"].dropna()})
        host = ""
        if "host" in part.columns and str(part["host"].iloc[0]):
            host = str(part["host"].iloc[0])
        elif str(capture) in hosts:
            host = hosts[str(capture)]
        rows.append(
            {
                "class": str(label),
                "capture_file": str(capture),
                "source": ",".join(sources) if sources else "unknown",
                "host": host,
                "capture_dates": ",".join(dates),
                "flows": int(len(part)),
            }
        )
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        return inventory
    sessions_per_class = inventory.groupby("class")["capture_file"].nunique()
    inventory["sessions_in_class"] = inventory["class"].map(sessions_per_class).astype(int)
    return inventory.sort_values(["class", "capture_file"]).reset_index(drop=True)


def run_capture_protocol(
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
    model_names: tuple[str, ...] = COMPARE_MODELS,
) -> dict[str, object]:
    """Score the capped and uncapped file splits, then the grouped folds.

    No model is tuned. Boosting and the forest use the library defaults.
    """
    frame, labels = _prepare_frames(features, meta)
    labels = _attach_host(labels, sessions)
    inventory = environment_inventory(labels, sessions)
    environment = _environment_audit(frame, labels, seed=seed)
    experiments = []
    for capped, experiment in ((False, "file_split_uncapped"), (True, f"file_split_cap_{max_flows_per_capture}")):
        used_features, used_meta = (frame, labels)
        if capped:
            used_features, used_meta = cap_flows_per_capture(
                frame, labels, max_flows_per_capture, seed=seed, capture_column="capture_id"
            )
        experiments.append(
            _score_file_split(
                used_features,
                used_meta,
                spec,
                task=task,
                experiment=experiment,
                model_names=model_names,
                seed=seed,
                n_jobs=n_jobs,
                out=out,
            )
        )
        experiments.append(
            _score_group_kfold(
                used_features,
                used_meta,
                spec,
                task=task,
                experiment=f"group_kfold_5_{'capped' if capped else 'uncapped'}",
                seed=seed,
                n_jobs=n_jobs,
            )
        )
    payload = {
        "task": task,
        "seed": seed,
        "max_flows_per_capture": max_flows_per_capture,
        "tuned": False,
        "environment_inventory": inventory.to_dict(orient="records"),
        "environment_audit": environment,
        "experiments": experiments,
    }
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{task}_protocol.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print_protocol(payload)
    return payload


def run_security_protocol(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    split_manifest: Path,
    seed: int,
    n_jobs: int,
    out: Path,
    unseen_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    """Label the day split as unseen-attack and fit one within-day LightGBM.

    Neither experiment is a headline. The within-day fit uses library defaults.
    """
    frame, labels = _prepare_frames(features, meta, require_single_label=False)
    manifest = load_named_split(split_manifest)
    fp_check_files = {name for name, split in manifest.assignments.items() if split == "fp_check"}
    within = assign_within_day(labels, fp_check_files=fp_check_files)
    within_score = _fit_split_models(
        frame,
        within,
        spec,
        model_names=("lightgbm",),
        seed=seed,
        n_jobs=n_jobs,
        out=out,
        task="security",
        experiment="within_day",
    )
    payload = {
        "task": "security",
        "headline": None,
        "headline_reason": "Unseen-attack and within-day are separate experiments. Neither is the headline.",
        "unseen_attack": {
            "experiment": "unseen_attack",
            "model": "lightgbm",
            "source": "stored_tuned_metrics" if unseen_metrics else "not_run",
            "metrics": unseen_metrics,
            "headline": False,
        },
        "within_day": within_score,
    }
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "security_protocol.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("experiment=unseen_attack model=lightgbm headline=false")
    if unseen_metrics is None:
        print("unseen_attack was not refit. The stored LightGBM metrics were not found.")
    else:
        _print_stored_security(unseen_metrics)
    print("experiment=within_day model=lightgbm headline=false tuned=false")
    _print_model_block(within_score)
    return payload


def _prepare_frames(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    require_single_label: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(features) != len(meta):
        raise SchemaError("Features and metadata have different numbers of rows.")
    labels = meta.copy().reset_index(drop=True)
    frame = features.reset_index(drop=True)
    if "label" not in labels.columns:
        raise SchemaError("Metadata has no label column.")
    if "capture_id" not in labels.columns:
        labels["capture_id"] = labels["capture_file"].map(lambda value: Path(str(value)).name)
    keep = labels["label"].notna()
    frame = frame.loc[keep].reset_index(drop=True)
    labels = labels.loc[keep].reset_index(drop=True)
    labels["label"] = labels["label"].astype(str)
    labels["capture_file"] = labels["capture_file"].map(lambda value: Path(str(value)).name)
    if require_single_label:
        for capture_id, part in labels.groupby("capture_id", sort=False):
            found = sorted(set(part["label"]))
            if len(found) != 1:
                raise SchemaError(f"Capture {capture_id} contains labels {found}.")
    return frame, labels


def _score_file_split(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    task: str,
    experiment: str,
    model_names: tuple[str, ...],
    seed: int,
    n_jobs: int,
    out: Path,
) -> dict[str, object]:
    groups = _groups(meta)
    try:
        assignment = assign_grouped_split(groups, seed=seed)
    except SplitError as exc:
        return {"experiment": experiment, "task": task, "error": str(exc), "headline": False}
    labeled = meta.copy()
    labeled["split"] = labeled["capture_id"].map(assignment)
    return _fit_split_models(
        features,
        labeled,
        spec,
        model_names=model_names,
        seed=seed,
        n_jobs=n_jobs,
        out=out,
        task=task,
        experiment=experiment,
    )


def _fit_split_models(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    model_names: tuple[str, ...],
    seed: int,
        n_jobs: int,
        out: Path | None,
        task: str,
        experiment: str,
    ) -> dict[str, object]:
    train_mask = meta["split"] == "train"
    if int(train_mask.sum()) == 0 or meta.loc[train_mask, "label"].nunique() < 2:
        return {"experiment": experiment, "task": task, "error": "training split cannot fit a classifier", "headline": False}
    state = fit_preprocess(features.loc[train_mask], spec)
    transformed = state.transform(features)
    blocks = []
    notes = {name: dominance(meta, name) for name in ("validation", "test")}
    for model_name in model_names:
        model = build_model(model_name, seed=seed, n_jobs=n_jobs)
        fitted = fit_estimator(model, model_name, transformed.loc[train_mask], meta.loc[train_mask, "label"])
        classes = tuple(str(label) for label in fitted.classes_)
        model_block: dict[str, object] = {"model": model_name, "tuned": False, "splits": {}}
        for split_name in ("validation", "test", "fp_check"):
            mask = meta["split"] == split_name
            if int(mask.sum()) == 0:
                continue
            pred = fitted.predict(transformed.loc[mask])
            proba = fitted.predict_proba(transformed.loc[mask])
            truth = meta.loc[mask, "label"]
            scored = score_split(split_name, truth, pred, proba, classes)
            per_class = classification_report(truth.astype(str), pred.astype(str), output_dict=True, zero_division=0)
            per_class.pop("accuracy", None)
            per_class.pop("macro avg", None)
            per_class.pop("weighted avg", None)
            file_recall = _file_mean_recall(meta.loc[mask], pred)
            note = notes.get(split_name, {"headline": False, "reason": "not a holdout split"})
            model_block["splits"][split_name] = {
                "experiment": experiment,
                "task": task,
                "model": model_name,
                "split": split_name,
                "headline": bool(note.get("headline", False)) if split_name == "test" else False,
                "dominance": note,
                "rows": scored.rows,
                "metrics": scored.metrics,
                "per_class": per_class,
                "file_mean_recall": file_recall,
                "confusion": [list(row) for row in scored.confusion],
                "confusion_labels": list(scored.labels),
            }
            if split_name == "test" and out is not None:
                directory = Path(out) / task / experiment / model_name
                directory.mkdir(parents=True, exist_ok=True)
                scored.confusion_frame().to_csv(directory / "confusion_test.csv")
        blocks.append(model_block)
    return {"experiment": experiment, "task": task, "tuned": False, "dominance": notes, "models": blocks}


def _score_group_kfold(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    task: str,
    experiment: str,
    seed: int,
    n_jobs: int,
) -> dict[str, object]:
    groups = _groups(meta)
    try:
        folds, excluded, largest_file, largest_fold = assign_group_kfold(groups, n_folds=5, min_files=5, seed=seed)
    except SplitError as exc:
        return {"experiment": experiment, "task": task, "model": "random_forest", "error": str(exc), "headline": False}
    scores = []
    for index, assignment in enumerate(folds):
        labeled = meta.copy()
        labeled["split"] = labeled["capture_id"].map(assignment)
        labeled = labeled.loc[labeled["split"].notna()].copy()
        mask = meta["capture_id"].isin(assignment)
        block = _fit_split_models(
            features.loc[mask].reset_index(drop=True),
            labeled.reset_index(drop=True),
            spec,
            model_names=("random_forest",),
            seed=seed,
            n_jobs=n_jobs,
            out=None,
            task=task,
            experiment=experiment,
        )
        # The helper writes a confusion file under the dummy path. The fold
        # score is what this experiment reports.
        test = block["models"][0]["splits"].get("test", {})
        scores.append(
            {
                "fold": index,
                "experiment": experiment,
                "split": "test",
                "model": "random_forest",
                "macro_f1": test.get("metrics", {}).get("f1_macro"),
                "accuracy": test.get("metrics", {}).get("accuracy"),
                "rows": test.get("rows"),
                "headline": False,
                "dominance": test.get("dominance"),
                "largest_file_held_out": index == largest_fold,
            }
        )
    values = [float(row["macro_f1"]) for row in scores if row["macro_f1"] is not None]
    return {
        "experiment": experiment,
        "task": task,
        "model": "random_forest",
        "tuned": False,
        "headline": False,
        "excluded_classes": excluded,
        "largest_file": largest_file,
        "largest_file_fold": largest_fold,
        "macro_f1_mean": float(np.mean(values)) if values else None,
        "macro_f1_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "largest_file_fold_macro_f1": next(row["macro_f1"] for row in scores if row["largest_file_held_out"]),
        "folds": scores,
    }


def _environment_audit(features: pd.DataFrame, meta: pd.DataFrame, *, seed: int) -> dict[str, object]:
    inventory_note = _class_sources(meta)
    audits = {
        "source": _predict_column(features, meta, "source_dataset", seed=seed),
        "host": _predict_column(features, meta, "host", seed=seed) if "host" in meta.columns else {"status": "no host column"},
    }
    if "flow_iat_min" in features.columns:
        reduced = features.drop(columns=["flow_iat_min"])
        audits["source_without_flow_iat_min"] = _predict_column(reduced, meta, "source_dataset", seed=seed)
    audits["port_only"] = _predict_metadata(meta, ["src_port", "dst_port"], "source_dataset", seed=seed)
    audits["ip_only"] = _predict_metadata(meta, ["src_ip", "dst_ip"], "source_dataset", seed=seed)
    separable = []
    source_result = audits["source"]
    above = bool(source_result.get("above_chance"))
    for row in inventory_note:
        if above and row["sources"] == 1:
            separable.append({"class": row["class"], "source": row["source_names"], "reason": "class comes from one capture source and source is predictable from features"})
    return {"classes": inventory_note, "models": audits, "separable_by_environment": separable}


def _predict_column(features: pd.DataFrame, meta: pd.DataFrame, column: str, *, seed: int) -> dict[str, object]:
    if column not in meta.columns:
        return {"target": column, "status": f"no {column} column"}
    target = meta[column].astype(str)
    if target.nunique() < 2:
        return {"target": column, "status": "only one value", "value": sorted(set(target))[:5]}
    groups = []
    for capture_id, part in meta.groupby("capture_id", sort=False):
        values = sorted(set(target.loc[part.index]))
        if len(values) == 1:
            groups.append(Group(str(capture_id), values[0], weight=int(len(part))))
    counts: dict[str, int] = {}
    for group in groups:
        counts[group.stratum] = counts.get(group.stratum, 0) + 1
    usable = {name for name, count in counts.items() if count >= 2}
    if len(usable) < 2:
        return {"target": column, "status": "fewer than two values have two captures"}
    kept = [group for group in groups if group.stratum in usable]
    n_folds = min(5, min(counts[name] for name in usable))
    if n_folds < 2:
        return {"target": column, "status": "not enough captures for a grouped fold"}
    try:
        folds, _excluded, _largest, _fold = assign_group_kfold(kept, n_folds=n_folds, min_files=n_folds, seed=seed)
    except SplitError as exc:
        return {"target": column, "status": str(exc)}
    return _tree_folds(features, meta, target, folds, column)


def _predict_metadata(meta: pd.DataFrame, columns: list[str], target_column: str, *, seed: int) -> dict[str, object]:
    present = [column for column in columns if column in meta.columns]
    if len(present) != len(columns) or target_column not in meta.columns:
        return {"target": target_column, "features": columns, "status": "metadata columns are absent"}
    frame = meta.loc[:, present].copy()
    for column in present:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype(str).astype("category").cat.codes
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(-1)
    return _predict_column(frame, meta, target_column, seed=seed)


def _tree_folds(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    target: pd.Series,
    folds: list[dict[str, str]],
    target_name: str,
) -> dict[str, object]:
    scores = []
    baselines = []
    for assignment in folds:
        labeled = meta["capture_id"].map(assignment)
        train = labeled == "train"
        test = labeled == "test"
        if int(train.sum()) == 0 or int(test.sum()) == 0 or target.loc[train].nunique() < 2:
            continue
        tree = DecisionTreeClassifier(max_depth=6, random_state=42)
        tree.fit(features.loc[train], target.loc[train].astype(str))
        pred = tree.predict(features.loc[test])
        truth = target.loc[test].astype(str)
        labels = sorted(set(truth) | set(target.loc[train].astype(str)))
        scores.append(float(f1_score(truth, pred, average="macro", labels=labels, zero_division=0)))
        majority = target.loc[train].astype(str).mode().iloc[0]
        baseline_pred = np.full(int(test.sum()), majority)
        baselines.append(float(f1_score(truth, baseline_pred, average="macro", labels=labels, zero_division=0)))
    if not scores:
        return {"target": target_name, "status": "no fold could be scored"}
    mean_score = float(np.mean(scores))
    mean_base = float(np.mean(baselines))
    return {
        "target": target_name,
        "status": "scored",
        "macro_f1_mean": mean_score,
        "macro_f1_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
        "majority_macro_f1_mean": mean_base,
        "above_chance": mean_score > mean_base + CHANCE_MARGIN,
        "folds": len(scores),
    }


def _class_sources(meta: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    source = meta["source_dataset"].astype(str) if "source_dataset" in meta.columns else pd.Series("unknown", index=meta.index)
    for label, part in meta.groupby("label", sort=False):
        names = sorted(set(source.loc[part.index]))
        rows.append(
            {
                "class": str(label),
                "sources": len(names),
                "source_names": names,
                "sessions": int(part["capture_id"].nunique()),
            }
        )
    return rows


def _groups(meta: pd.DataFrame) -> list[Group]:
    groups = []
    for capture_id, part in meta.groupby("capture_id", sort=False):
        groups.append(Group(str(capture_id), str(part["label"].iloc[0]), weight=int(len(part))))
    return groups


def _file_mean_recall(meta: pd.DataFrame, predicted: np.ndarray) -> float:
    truth = meta["label"].astype(str).to_numpy()
    pred = np.asarray(predicted).astype(str)
    files = meta["capture_file"].astype(str).to_numpy()
    recalls = []
    for capture in sorted(set(files)):
        mask = files == capture
        recalls.append(float(np.mean(pred[mask] == truth[mask])))
    return float(np.mean(recalls)) if recalls else 0.0


def _attach_host(meta: pd.DataFrame, sessions: pd.DataFrame | None) -> pd.DataFrame:
    hosts = _hosts_by_file(sessions)
    source = meta["source_dataset"].astype(str) if "source_dataset" in meta.columns else pd.Series("unknown", index=meta.index)
    names = meta["capture_file"].map(lambda value: Path(str(value)).name)
    attached = meta.copy()
    attached["host"] = [hosts.get(name) or f"{src}-no-sidecar-host" for name, src in zip(names, source, strict=True)]
    return attached


def _hosts_by_file(sessions: pd.DataFrame | None) -> dict[str, str]:
    if sessions is None or "file" not in sessions.columns or "host" not in sessions.columns:
        return {}
    hosts: dict[str, str] = {}
    for row in sessions.itertuples(index=False):
        hosts[Path(str(row.file)).name] = str(row.host)
    return hosts


def _as_utc(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if float(numeric.notna().mean()) > 0.8:
        median = float(numeric.dropna().median()) if numeric.notna().any() else 0.0
        unit = "us" if median > 1e14 else "ms" if median > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def _print_protocol(payload: dict[str, object]) -> None:
    print("task", payload["task"], "tuned", payload["tuned"], "max_flows_per_capture", payload["max_flows_per_capture"])
    audit = payload["environment_audit"]
    print("environment_classes")
    for row in audit["classes"]:
        print(" class", row["class"], "sources", row["source_names"], "sessions", row["sessions"])
    for name, result in audit["models"].items():
        print("environment", name, {key: result.get(key) for key in ("status", "macro_f1_mean", "macro_f1_std", "majority_macro_f1_mean", "above_chance")})
    if audit["separable_by_environment"]:
        print("separable_by_environment", audit["separable_by_environment"])
    else:
        print("separable_by_environment none")
    for experiment in payload["experiments"]:
        print("experiment", experiment.get("experiment"), "headline", experiment.get("headline", False))
        if "error" in experiment:
            print(" ", experiment["error"])
            continue
        if "folds" in experiment:
            print(
                " group_kfold macro_f1",
                _format_mean(experiment.get("macro_f1_mean"), experiment.get("macro_f1_std")),
                "largest_file",
                experiment.get("largest_file"),
                "fold",
                experiment.get("largest_file_fold"),
                "fold_macro_f1",
                experiment.get("largest_file_fold_macro_f1"),
            )
            print(" excluded_classes", experiment.get("excluded_classes"))
            continue
        _print_model_block(experiment)


def _print_model_block(experiment: dict[str, object]) -> None:
    for model in experiment.get("models", []):
        for split_name, block in model.get("splits", {}).items():
            metrics = block["metrics"]
            print(
                f"experiment={block['experiment']} split={split_name} model={block['model']} "
                f"headline={block['headline']} rows={block['rows']} "
                f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['f1_macro']:.4f} "
                f"file_mean_recall={block['file_mean_recall']:.4f}"
            )
            if block["dominance"].get("reason"):
                print(" dominance", block["dominance"]["reason"])
            for label, scores in block["per_class"].items():
                if isinstance(scores, dict) and "f1-score" in scores:
                    print(
                        f"  class={label} precision={scores['precision']:.4f} "
                        f"recall={scores['recall']:.4f} f1={scores['f1-score']:.4f} support={scores['support']}"
                    )


def _print_stored_security(metrics: dict[str, object]) -> None:
    splits = metrics.get("splits", {})
    if not isinstance(splits, dict):
        print(" stored metrics have no splits")
        return
    for split_name, block in splits.items():
        values = block.get("metrics", {})
        accuracy = values.get("accuracy")
        macro = values.get("f1_macro")
        if accuracy is None or macro is None:
            continue
        print(
            f"experiment=unseen_attack split={split_name} model=lightgbm headline=false "
            f"rows={block.get('rows')} accuracy={float(accuracy):.4f} macro_f1={float(macro):.4f}"
        )


def _format_mean(mean: object, std: object) -> str:
    if mean is None:
        return "not_scored"
    if std is None:
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} ± {float(std):.4f}"
