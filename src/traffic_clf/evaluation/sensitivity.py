"""Ablations, repeated seeds, latency, and SHAP on one source at a time.

No model is tuned. Idle-timeout changes flow boundaries, so that ablation
needs a fresh extract. Early packets and feature groups reuse the table.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_clf.data.schema import FeatureSpec
from traffic_clf.evaluation.protocol import (
    _attach_host,
    _prepare_frames,
    _score_file_split,
    class_source_control,
    experiment_source,
)
from traffic_clf.models.build import build_model
from traffic_clf.preprocess.prepare import fit_preprocess
from traffic_clf.training.run import fit_estimator

FEATURE_GROUPS = ("sizes", "iat", "flags", "rates", "active_idle", "flow", "sequence", "both")
_SIZE_MARKERS = ("pkt_len", "_bytes", "duration")
_FLAG_COLUMNS = ("syn_count", "ack_count", "fin_count", "rst_count", "psh_count", "urg_count")
_RATE_COLUMNS = ("bytes_per_s", "packets_per_s", "down_up_ratio")


def best_model_by_source(payload: dict[str, object], *, cap_label: str) -> dict[str, str]:
    """Pick the library default with the highest validation macro-F1 inside one source."""
    chosen: dict[str, tuple[float, str]] = {}
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        return {}
    for experiment in experiments:
        name = str(experiment.get("experiment", ""))
        if f"_file_split_{cap_label}" not in name:
            continue
        source = name.split("_only_", 1)[0]
        models = experiment.get("models")
        if not isinstance(models, list):
            continue
        for model in models:
            splits = model.get("splits", {})
            validation = splits.get("validation", {}) if isinstance(splits, dict) else {}
            metrics = validation.get("metrics", {}) if isinstance(validation, dict) else {}
            score = metrics.get("f1_macro")
            if score is None:
                continue
            current = chosen.get(source)
            if current is None or float(score) > current[0]:
                chosen[source] = (float(score), str(model.get("model")))
    return {source: pair[1] for source, pair in chosen.items()}


def select_feature_group(features: pd.DataFrame, group: str) -> pd.DataFrame:
    """Keep one feature family plus protocol. The other families are dropped."""
    if group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group {group!r}.")
    sequence = [
        column
        for column in features.columns
        if column.startswith(("size_", "direction_", "iat_")) or column == "seq_observed"
    ]
    if group == "both":
        return features.copy()
    if group == "sequence":
        columns = sequence + (["protocol"] if "protocol" in features.columns else [])
        return features.loc[:, columns].copy()
    if group == "flow":
        columns = [column for column in features.columns if column not in sequence]
        return features.loc[:, columns].copy()
    if group == "sizes":
        chosen = [column for column in features.columns if any(marker in column for marker in _SIZE_MARKERS)]
    elif group == "iat":
        chosen = [column for column in features.columns if "iat" in column and column not in sequence]
    elif group == "flags":
        chosen = [column for column in _FLAG_COLUMNS if column in features.columns]
    elif group == "rates":
        chosen = [column for column in _RATE_COLUMNS if column in features.columns]
    else:
        chosen = [
            column
            for column in features.columns
            if column.startswith(("active_", "idle_"))
        ]
    if "protocol" in features.columns and "protocol" not in chosen:
        chosen.append("protocol")
    return features.loc[:, chosen].copy()


def truncate_early_packets(features: pd.DataFrame, packets: int) -> pd.DataFrame:
    """Drop sequence columns past the first N packets. Flow statistics stay as extracted."""
    if packets < 1:
        raise ValueError("early packet count must be at least 1.")
    kept = []
    for column in features.columns:
        if _sequence_index(column) > packets:
            continue
        kept.append(column)
    frame = features.loc[:, kept].copy()
    if "seq_observed" in frame.columns:
        frame["seq_observed"] = np.minimum(frame["seq_observed"].astype(float), packets)
    return frame


def score_source_cap(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    task: str,
    source_name: str,
    experiment: str,
    model_name: str,
    seed: int,
    n_jobs: int,
    max_flows_per_capture: int,
    out: Path,
    sessions: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Score one capped file split for one source. Used by the idle-timeout ablation."""
    return _score_variant(
        features,
        meta,
        spec,
        task=task,
        source_name=source_name,
        experiment=experiment,
        model_name=model_name,
        seed=seed,
        n_jobs=n_jobs,
        max_flows_per_capture=max_flows_per_capture,
        out=out,
        sessions=sessions,
        group_spec=False,
    )


def run_feature_ablations(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    task: str,
    source_name: str,
    model_name: str,
    seed: int,
    n_jobs: int,
    max_flows_per_capture: int,
    out: Path,
    sessions: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    """Score each feature family on one source. The number is not a mixed-source headline."""
    return [
        _score_variant(
            select_feature_group(features, group),
            meta,
            spec,
            task=task,
            source_name=source_name,
            experiment=f"{source_name}_only_ablation_group_{group}",
            model_name=model_name,
            seed=seed,
            n_jobs=n_jobs,
            max_flows_per_capture=max_flows_per_capture,
            out=out,
            sessions=sessions,
            group_spec=group != "both",
        )
        for group in FEATURE_GROUPS
    ]


def run_early_packet_ablation(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    task: str,
    source_name: str,
    model_name: str,
    seed: int,
    n_jobs: int,
    max_flows_per_capture: int,
    out: Path,
    packets: tuple[int, ...] = (5, 10, 20, 30),
    sessions: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    """Score prefixes of the stored 30-packet sequence. Flow statistics are unchanged."""
    return [
        _score_variant(
            truncate_early_packets(features, count),
            meta,
            spec,
            task=task,
            source_name=source_name,
            experiment=f"{source_name}_only_ablation_early_{count}",
            model_name=model_name,
            seed=seed,
            n_jobs=n_jobs,
            max_flows_per_capture=max_flows_per_capture,
            out=out,
            sessions=sessions,
            group_spec=False,
        )
        for count in packets
    ]


def run_seed_repetition(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    task: str,
    source_name: str,
    model_name: str,
    n_jobs: int,
    max_flows_per_capture: int,
    out: Path,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    sessions: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Validation macro-F1 over five seeds, inside one source, as mean ± std."""
    scores = []
    for seed in seeds:
        scored = _score_variant(
            features,
            meta,
            spec,
            task=task,
            source_name=source_name,
            experiment=f"{source_name}_only_seed_{seed}",
            model_name=model_name,
            seed=seed,
            n_jobs=n_jobs,
            max_flows_per_capture=max_flows_per_capture,
            out=out,
            sessions=sessions,
            group_spec=False,
        )
        value = _validation_macro(scored)
        if value is not None:
            scores.append(value)
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else None
    return {
        "experiment": f"{source_name}_only_seeds",
        "task": task,
        "model": model_name,
        "source": source_name,
        "seeds": list(seeds),
        "validation_macro_f1": scores,
        "mean": float(np.mean(scores)) if scores else None,
        "std": std,
        "headline": False,
        "tuned": False,
    }


def measure_latency(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    task: str,
    source_name: str,
    model_name: str,
    seed: int,
    n_jobs: int,
    max_flows_per_capture: int,
    sessions: pd.DataFrame | None = None,
    sample_rows: int = 1000,
) -> dict[str, object]:
    """Time one already-fitted predict call. The fit itself is not included."""
    prepared = _slice_source(
        features,
        meta,
        source_name=source_name,
        sessions=sessions,
        max_flows_per_capture=max_flows_per_capture,
        seed=seed,
    )
    if prepared is None:
        return {"experiment": f"{source_name}_only_latency", "status": "no rows", "headline": False}
    _frame, labels, state_features, _confounded = prepared
    from traffic_clf.data.splits import assign_grouped_split
    from traffic_clf.errors import SplitError
    from traffic_clf.evaluation.protocol import _groups

    try:
        assignment = assign_grouped_split(_groups(labels), seed=seed)
    except SplitError as exc:
        return {"experiment": f"{source_name}_only_latency", "status": str(exc), "headline": False}
    labels = labels.copy()
    labels["split"] = labels["capture_id"].map(assignment)
    train = labels["split"] == "train"
    state = fit_preprocess(state_features.loc[train], spec)
    transformed = state.transform(state_features)
    model = fit_estimator(
        build_model(model_name, seed=seed, n_jobs=n_jobs),
        model_name,
        transformed.loc[train],
        labels.loc[train, "label"],
    )
    test = labels["split"] == "test"
    sample = transformed.loc[test]
    if sample.empty:
        sample = transformed.loc[labels["split"] == "validation"]
    sample = sample.head(sample_rows)
    if sample.empty:
        return {"experiment": f"{source_name}_only_latency", "status": "no scored rows", "headline": False}
    model.predict(sample.head(1))
    started = time.perf_counter()
    model.predict(sample)
    elapsed = time.perf_counter() - started
    return {
        "experiment": f"{source_name}_only_latency",
        "task": task,
        "model": model_name,
        "source": source_name,
        "rows": int(len(sample)),
        "latency_ms_per_flow": (elapsed / len(sample)) * 1000,
        "throughput_flows_per_s": (len(sample) / elapsed) if elapsed else None,
        "headline": False,
        "tuned": False,
    }


def write_shap(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    task: str,
    source_name: str,
    model_name: str,
    seed: int,
    n_jobs: int,
    max_flows_per_capture: int,
    out: Path,
    sessions: pd.DataFrame | None = None,
    sample_rows: int = 200,
) -> dict[str, object]:
    """Write mean absolute SHAP values for the selected model. A missing install is recorded."""
    try:
        import shap
    except ImportError:
        return {
            "experiment": f"{source_name}_only_shap",
            "status": "shap is not installed",
            "headline": False,
        }
    prepared = _slice_source(
        features,
        meta,
        source_name=source_name,
        sessions=sessions,
        max_flows_per_capture=max_flows_per_capture,
        seed=seed,
    )
    if prepared is None:
        return {"experiment": f"{source_name}_only_shap", "status": "no rows", "headline": False}
    _frame, labels, state_features, _confounded = prepared
    from traffic_clf.data.splits import assign_grouped_split
    from traffic_clf.errors import SplitError
    from traffic_clf.evaluation.protocol import _groups

    try:
        assignment = assign_grouped_split(_groups(labels), seed=seed)
    except SplitError as exc:
        return {"experiment": f"{source_name}_only_shap", "status": str(exc), "headline": False}
    labels = labels.copy()
    labels["split"] = labels["capture_id"].map(assignment)
    train = labels["split"] == "train"
    state = fit_preprocess(state_features.loc[train], spec)
    transformed = state.transform(state_features)
    fitted = fit_estimator(
        build_model(model_name, seed=seed, n_jobs=n_jobs),
        model_name,
        transformed.loc[train],
        labels.loc[train, "label"],
    )
    validation = transformed.loc[labels["split"] == "validation"].head(sample_rows)
    if validation.empty:
        return {"experiment": f"{source_name}_only_shap", "status": "validation split is empty", "headline": False}
    estimator = getattr(fitted, "estimator", fitted)
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(validation)
    importance = _mean_abs_shap(values, list(validation.columns))
    directory = Path(out) / task / f"{source_name}_only_shap"
    directory.mkdir(parents=True, exist_ok=True)
    table = importance.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})
    table_path = directory / f"{model_name}.csv"
    table.to_csv(table_path, index=False)
    return {
        "experiment": f"{source_name}_only_shap",
        "task": task,
        "model": model_name,
        "source": source_name,
        "rows": int(len(validation)),
        "path": str(table_path),
        "top": table.head(15).to_dict(orient="records"),
        "headline": False,
        "tuned": False,
    }


def _score_variant(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    task: str,
    source_name: str,
    experiment: str,
    model_name: str,
    seed: int,
    n_jobs: int,
    max_flows_per_capture: int,
    out: Path,
    sessions: pd.DataFrame | None,
    group_spec: bool,
) -> dict[str, object]:
    prepared = _slice_source(
        features,
        meta,
        source_name=source_name,
        sessions=sessions,
        max_flows_per_capture=max_flows_per_capture,
        seed=seed,
    )
    if prepared is None:
        return {"experiment": experiment, "status": "no rows", "headline": False, "tuned": False}
    _frame, labels, state_features, confounded = prepared
    used_spec = _group_spec(spec, state_features) if group_spec else spec
    try:
        scored = _score_file_split(
            state_features,
            labels,
            used_spec,
            task=task,
            experiment=experiment,
            model_names=(model_name,),
            seed=seed,
            n_jobs=n_jobs,
            out=out,
            confounded=confounded,
        )
    except Exception as exc:
        return {"experiment": experiment, "status": str(exc), "headline": False, "tuned": False}
    scored["headline"] = False
    scored["source"] = source_name
    return scored


def _slice_source(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    source_name: str,
    sessions: pd.DataFrame | None,
    max_flows_per_capture: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, bool]] | None:
    from traffic_clf.evaluation.protocol import cap_flows_per_capture

    frame, labels = _prepare_frames(features, meta)
    labels = _attach_host(labels, sessions)
    labels["experiment_source"] = experiment_source(labels)
    confounded = {str(row["class"]): bool(row["environment_confounded"]) for row in class_source_control(labels)}
    mask = labels["experiment_source"] == source_name
    if int(mask.sum()) == 0:
        return None
    part_features = frame.loc[mask].reset_index(drop=True)
    part_meta = labels.loc[mask].reset_index(drop=True)
    capped_features, capped_meta = cap_flows_per_capture(
        part_features,
        part_meta,
        max_flows_per_capture,
        seed=seed,
        capture_column="capture_id",
    )
    return part_features, capped_meta, capped_features, confounded


def _group_spec(spec: FeatureSpec, features: pd.DataFrame) -> FeatureSpec:
    numeric = tuple(column for column in spec.numeric if column in features.columns)
    return FeatureSpec(
        version=spec.version,
        numeric=numeric,
        categorical=tuple(column for column in spec.categorical if column in features.columns),
        identifiers=spec.identifiers,
    )


def _validation_macro(scored: dict[str, object]) -> float | None:
    models = scored.get("models")
    if not isinstance(models, list) or not models:
        return None
    splits = models[0].get("splits", {})
    if not isinstance(splits, dict):
        return None
    validation = splits.get("validation", {})
    if not isinstance(validation, dict):
        return None
    metrics = validation.get("metrics", {})
    if not isinstance(metrics, dict) or metrics.get("f1_macro") is None:
        return None
    return float(metrics["f1_macro"])


def _sequence_index(column: str) -> int:
    for prefix in ("size_", "direction_", "iat_"):
        if column.startswith(prefix):
            suffix = column.removeprefix(prefix)
            if suffix.isdigit():
                return int(suffix)
    return 0


def _mean_abs_shap(values: object, columns: list[str]) -> pd.Series:
    array = np.asarray(values)
    if isinstance(values, list):
        array = np.mean([np.abs(np.asarray(item)) for item in values], axis=0)
    elif array.ndim == 3:
        array = np.mean(np.abs(array), axis=-1)
    else:
        array = np.abs(array)
    return pd.Series(array.mean(axis=0), index=columns).sort_values(ascending=False)
