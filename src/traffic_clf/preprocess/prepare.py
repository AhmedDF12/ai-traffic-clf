"""Train-only imputation and scaling.

The imputer, the scaler, and the duplicate removal see the training split
and no other split. Protocol is one-hot encoded. It is not scaled as a
number, because 6 and 17 are names, not magnitudes. Direction columns stay
at -1, 0, or 1. Size and inter-arrival columns are scaled with the other
numeric features.

Outlier clipping is intentionally absent. Clipping byte and rate columns
on CIC-IDS would pull DDoS flows toward the benign bulk of the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from traffic_clf.config import ProjectConfig
from traffic_clf.data.labels import load_iscx_rules
from traffic_clf.data.loaders import load_cic_csv
from traffic_clf.data.schema import FORBIDDEN_FEATURES, FeatureSpec
from traffic_clf.data.splits import Group, NamedSplit, assign_grouped_split, attach_split, load_named_split
from traffic_clf.errors import SchemaError, SplitError
from traffic_clf.features.extract import FlowTable, sequence_column_names
from traffic_clf.flows.pipeline import extract_custom_pcap, extract_iscx_pcap

_PROTOCOL_CATEGORIES = ["other", "tcp", "udp"]


@dataclass
class PreprocessState:
    """Transform fitted on training rows only."""

    numeric_columns: tuple[str, ...]
    passthrough_columns: tuple[str, ...]
    imputer: SimpleImputer
    scaler: StandardScaler
    encoder: OneHotEncoder
    output_columns: tuple[str, ...]

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        _reject_identifiers(features)
        numeric = self.imputer.transform(features.loc[:, list(self.numeric_columns)])
        scaled = self.scaler.transform(numeric)
        scaled_frame = pd.DataFrame(scaled, columns=list(self.numeric_columns))
        pieces = [scaled_frame.reset_index(drop=True)]
        if self.passthrough_columns:
            pieces.append(features.loc[:, list(self.passthrough_columns)].reset_index(drop=True))
        names = [_protocol_name(value) for value in features["protocol"]]
        encoded = self.encoder.transform(np.asarray(names, dtype=object).reshape(-1, 1))
        encoded_frame = pd.DataFrame(encoded, columns=[f"protocol_{name}" for name in _PROTOCOL_CATEGORIES])
        pieces.append(encoded_frame)
        output = pd.concat(pieces, axis=1)
        output = output.loc[:, list(self.output_columns)]
        return output.reset_index(drop=True)


@dataclass(frozen=True)
class PreparedPart:
    features: pd.DataFrame
    meta: pd.DataFrame


@dataclass(frozen=True)
class PrepareReport:
    rows_in: int
    dropped_unlabeled: int
    dropped_duplicates: int
    rows_train: int
    rows_validation: int
    rows_test: int
    rows_fp_check: int
    output_features: int

    def summary(self) -> str:
        return (
            f"rows_in={self.rows_in} dropped_unlabeled={self.dropped_unlabeled} "
            f"dropped_duplicates={self.dropped_duplicates} train={self.rows_train} "
            f"validation={self.rows_validation} test={self.rows_test} "
            f"fp_check={self.rows_fp_check} output_features={self.output_features}"
        )


@dataclass
class PreparedDataset:
    train: PreparedPart
    validation: PreparedPart
    test: PreparedPart
    fp_check: PreparedPart | None
    state: PreprocessState
    report: PrepareReport


def fit_preprocess(features: pd.DataFrame, spec: FeatureSpec) -> PreprocessState:
    """Fit imputation and scaling on one frame. That frame must be the training split."""
    _reject_identifiers(features)
    missing = [column for column in spec.model_columns if column not in features.columns]
    if missing:
        raise SchemaError("Cannot fit preprocessing without features: " + ", ".join(missing))
    numeric, passthrough = _column_roles(features, spec)
    unknown = [
        column
        for column in features.columns
        if column not in numeric and column not in passthrough and column != "protocol"
    ]
    if unknown:
        raise SchemaError("Unexpected feature columns: " + ", ".join(unknown))
    imputer = SimpleImputer(strategy="median")
    imputer.fit(features.loc[:, list(numeric)])
    statistics = np.asarray(imputer.statistics_, dtype=float)
    imputer.statistics_ = np.where(np.isnan(statistics), 0.0, statistics)
    imputed = imputer.transform(features.loc[:, list(numeric)])
    scaler = StandardScaler()
    scaler.fit(imputed)
    encoder = OneHotEncoder(
        categories=[_PROTOCOL_CATEGORIES],
        handle_unknown="ignore",
        sparse_output=False,
    )
    names = np.asarray([_protocol_name(value) for value in features["protocol"]], dtype=object).reshape(-1, 1)
    encoder.fit(names)
    output = tuple(numeric) + tuple(passthrough) + tuple(f"protocol_{name}" for name in _PROTOCOL_CATEGORIES)
    return PreprocessState(
        numeric_columns=tuple(numeric),
        passthrough_columns=tuple(passthrough),
        imputer=imputer,
        scaler=scaler,
        encoder=encoder,
        output_columns=output,
    )


def prepare_cic_csvs(
    paths: list[Path],
    config: ProjectConfig,
    *,
    max_rows: int | None = None,
) -> PreparedDataset:
    """Load CIC flow CSVs, apply the day split, and fit preprocessing on train."""
    if config.task is None or config.task.name != "security":
        raise SchemaError("CIC preparation requires the security task.")
    if not paths:
        raise SchemaError("No CIC CSV files were given.")
    frames = [load_cic_csv(Path(path), dataset="cic_ids2017", root=config.root, max_rows=max_rows) for path in paths]
    features = pd.concat([frame.features for frame in frames], ignore_index=True)
    meta = pd.concat([frame.meta for frame in frames], ignore_index=True)
    meta["label"] = meta["label"].astype(str)
    split_table = load_named_split(config.splits.cic_ids2017)
    return _prepare_named_split(features, meta, split_table, config.features.flow, label_column="label")


def prepare_iscx_pcaps(paths: list[Path], config: ProjectConfig) -> PreparedDataset:
    """Extract ISCX captures, split by file, and fit preprocessing on train."""
    if config.task is None or config.task.name not in {"service", "application"}:
        raise SchemaError("ISCX preparation requires the service or application task.")
    rules = load_iscx_rules(config.labels.iscx_vpn)
    tables = [extract_iscx_pcap(Path(path), config, rules) for path in paths]
    return _prepare_capture_tables(tables, config)


def prepare_custom_pcaps(paths: list[Path], sidecar: Path, config: ProjectConfig) -> PreparedDataset:
    """Extract self-captures with a sidecar, then split by file."""
    if config.task is None or config.task.name not in {"service", "application"}:
        raise SchemaError("Custom preparation requires the service or application task.")
    spec = load_custom_spec_for(config)
    sessions = load_custom_sidecar(sidecar, spec)
    tables = [extract_custom_pcap(Path(path), config, sessions, spec) for path in paths]
    return _prepare_capture_tables(tables, config)


def load_custom_spec_for(config: ProjectConfig):
    from traffic_clf.data.labels import load_custom_spec

    return load_custom_spec(config.labels.custom)


def write_prepared(dataset: PreparedDataset, directory: Path) -> None:
    """Write train, validation, test, and the fitted state. Features stay separate from metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, part in (
        ("train", dataset.train),
        ("validation", dataset.validation),
        ("test", dataset.test),
    ):
        part.features.to_parquet(directory / f"{name}_features.parquet", index=False)
        part.meta.to_parquet(directory / f"{name}_meta.parquet", index=False)
    if dataset.fp_check is not None:
        dataset.fp_check.features.to_parquet(directory / "fp_check_features.parquet", index=False)
        dataset.fp_check.meta.to_parquet(directory / "fp_check_meta.parquet", index=False)
    joblib.dump(dataset.state, directory / "preprocess.joblib")
    (directory / "report.txt").write_text(dataset.report.summary() + "\n", encoding="utf-8")


def load_preprocess_state(path: Path) -> PreprocessState:
    state = joblib.load(path)
    if not isinstance(state, PreprocessState):
        raise SchemaError(f"{path} is not a preprocessing state.")
    return state


def _prepare_capture_tables(tables: list[FlowTable], config: ProjectConfig) -> PreparedDataset:
    assert config.task is not None
    features = pd.concat([table.features for table in tables], ignore_index=True)
    sequence = pd.concat([table.sequence for table in tables], ignore_index=True)
    meta = pd.concat([table.meta for table in tables], ignore_index=True)
    combined = pd.concat([features, sequence], axis=1)
    label_column = config.task.name
    unlabeled = meta[label_column].isna()
    dropped_unlabeled = int(unlabeled.sum())
    combined = combined.loc[~unlabeled].reset_index(drop=True)
    meta = meta.loc[~unlabeled].reset_index(drop=True)
    if combined.empty:
        raise SchemaError("Every flow was unlabeled after applying the task filter.")
    meta = meta.copy()
    meta["label"] = meta[label_column].astype(str)
    groups = _one_label_per_capture(meta)
    assignment = assign_grouped_split(groups, seed=config.seed)
    meta["split"] = meta["capture_id"].map(assignment)
    if meta["split"].isna().any():
        raise SplitError("A capture file did not receive a split.")
    return _fit_assigned(combined, meta, config.features.flow, dropped_unlabeled=dropped_unlabeled)


def _prepare_named_split(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    table: NamedSplit,
    spec: FeatureSpec,
    *,
    label_column: str,
) -> PreparedDataset:
    assigned = attach_split(meta, table)
    assigned["label"] = assigned[label_column].astype(str)
    return _fit_assigned(features, assigned, spec, dropped_unlabeled=0)


def _fit_assigned(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    dropped_unlabeled: int,
) -> PreparedDataset:
    rows_in = int(len(features)) + dropped_unlabeled
    parts: dict[str, PreparedPart] = {}
    dropped_duplicates = 0
    raw_train: pd.DataFrame | None = None
    for split_name in ("train", "validation", "test", "fp_check"):
        mask = meta["split"] == split_name
        split_features = features.loc[mask].reset_index(drop=True)
        split_meta = meta.loc[mask].reset_index(drop=True)
        split_features, split_meta, removed = _drop_duplicates(split_features, split_meta)
        dropped_duplicates += removed
        parts[split_name] = PreparedPart(features=split_features, meta=split_meta)
        if split_name == "train":
            raw_train = split_features
    if raw_train is None or raw_train.empty:
        raise SchemaError("The training split is empty, so the scaler cannot be fit.")
    state = fit_preprocess(raw_train, spec)
    transformed = {
        name: PreparedPart(features=state.transform(part.features), meta=part.meta)
        if not part.features.empty
        else PreparedPart(features=pd.DataFrame(columns=list(state.output_columns)), meta=part.meta)
        for name, part in parts.items()
    }
    report = PrepareReport(
        rows_in=rows_in,
        dropped_unlabeled=dropped_unlabeled,
        dropped_duplicates=dropped_duplicates,
        rows_train=int(len(transformed["train"].features)),
        rows_validation=int(len(transformed["validation"].features)),
        rows_test=int(len(transformed["test"].features)),
        rows_fp_check=int(len(transformed["fp_check"].features)),
        output_features=len(state.output_columns),
    )
    return PreparedDataset(
        train=transformed["train"],
        validation=transformed["validation"],
        test=transformed["test"],
        fp_check=transformed["fp_check"],
        state=state,
        report=report,
    )


def _one_label_per_capture(meta: pd.DataFrame) -> list[Group]:
    groups: list[Group] = []
    for capture_id, part in meta.groupby("capture_id", sort=False):
        labels = sorted(set(part["label"].astype(str)))
        if len(labels) != 1:
            raise SplitError(
                f"Capture {capture_id} contains labels {labels}. "
                "Split the file so each capture has one application."
            )
        groups.append(Group(str(capture_id), labels[0], weight=int(len(part))))
    return groups


def _drop_duplicates(features: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if features.empty:
        return features, meta, 0
    marked = features.copy()
    marked["__label"] = meta["label"].to_numpy()
    duplicate = marked.duplicated()
    removed = int(duplicate.sum())
    return (
        features.loc[~duplicate].reset_index(drop=True),
        meta.loc[~duplicate].reset_index(drop=True),
        removed,
    )


def _column_roles(features: pd.DataFrame, spec: FeatureSpec) -> tuple[list[str], list[str]]:
    width = _sequence_width(features.columns)
    allowed_sequence = set(sequence_column_names(width)) if width else set()
    numeric = [column for column in spec.numeric if column in features.columns]
    numeric.extend(
        column
        for column in sequence_column_names(width)
        if column in features.columns and not column.startswith("direction_")
    )
    passthrough = [
        column
        for column in sequence_column_names(width)
        if column.startswith("direction_") and column in features.columns
    ]
    extra = set(features.columns) - set(numeric) - set(passthrough) - {"protocol"} - allowed_sequence
    if extra:
        raise SchemaError("Unexpected feature columns: " + ", ".join(sorted(extra)))
    return numeric, passthrough


def _sequence_width(columns: pd.Index) -> int:
    indexes = []
    for column in columns:
        if isinstance(column, str) and column.startswith("size_"):
            indexes.append(int(column.removeprefix("size_")))
    if not indexes:
        return 0
    width = max(indexes)
    expected = {f"size_{index:02d}" for index in range(1, width + 1)}
    if expected - set(columns):
        raise SchemaError("Sequence size columns must be contiguous from size_01.")
    return width


def _protocol_name(value: object) -> str:
    if pd.isna(value):
        return "other"
    number = int(float(value))
    if number == 6:
        return "tcp"
    if number == 17:
        return "udp"
    return "other"


def _reject_identifiers(features: pd.DataFrame) -> None:
    leaked = set(features.columns) & FORBIDDEN_FEATURES
    if leaked:
        raise SchemaError(
            "Refusing to fit a scaler on identifier columns: " + ", ".join(sorted(leaked))
        )
