"""Load CIC flow CSVs and custom session sidecars into the project schema.

Scaling, imputation, and SMOTE belong to a later module. This loader only
renames columns, separates identifiers from features, and maps labels.
Infinite rates become missing values and are counted. They are not filled.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_clf.config import load_config
from traffic_clf.data.labels import (
    CustomLabelSpec,
    load_custom_spec,
    load_security_map,
    normalize_raw_label,
    to_binary_security,
)
from traffic_clf.data.schema import FEATURE_ALIASES, META_ALIASES, FeatureSpec, load_feature_spec
from traffic_clf.errors import SchemaError, UnmappedLabelError

META_COLUMNS = (
    "capture_file",
    "source_dataset",
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "timestamp",
    "label_raw",
    "label",
)
_SIDECAR_COLUMNS = ("file", "t_start", "t_end", "application", "service", "vpn", "host", "notes")
_TRUE = {"1", "true", "yes", "y", "vpn"}
_FALSE = {"0", "false", "no", "n", "nonvpn", "non-vpn"}


@dataclass(frozen=True)
class LoadReport:
    source_dataset: str
    capture_file: str
    dialect: str
    rows_loaded: int
    inf_replaced: int
    missing_feature_cells: int
    duplicate_feature_rows: int
    class_counts: dict[str, int]
    binary_counts: dict[str, int]
    unused_columns: tuple[str, ...]

    def summary(self) -> str:
        return (
            f"file={self.capture_file} dataset={self.source_dataset} dialect={self.dialect} "
            f"rows={self.rows_loaded} inf_replaced={self.inf_replaced} "
            f"missing_cells={self.missing_feature_cells} "
            f"duplicate_feature_rows={self.duplicate_feature_rows}"
        )


@dataclass(frozen=True)
class LoadedFlows:
    features: pd.DataFrame
    meta: pd.DataFrame
    report: LoadReport


def load_cic_csv(
    path: Path,
    *,
    dataset: str,
    root: Path | None = None,
    max_rows: int | None = None,
) -> LoadedFlows:
    """Load one CIC-IDS CSV and map its labels onto the security taxonomy.

    ``max_rows`` reads a prefix of the file. Use it to inspect a multi-gigabyte
    CSV on a machine that cannot hold the whole day in memory. Only the columns
    in the feature contract and the identifier set are parsed.
    """
    if dataset not in {"cic_ids2017", "cic_ids2018"}:
        raise SchemaError("dataset must be cic_ids2017 or cic_ids2018.")
    if max_rows is not None and (isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1):
        raise SchemaError("max_rows must be a positive integer.")
    csv_path = Path(path)
    if not csv_path.is_file():
        raise SchemaError(f"CIC CSV not found: {csv_path}")

    config = load_config(root=root)
    spec = config.features.flow
    label_map = load_security_map(getattr(config.labels, dataset))
    header = _read_csv(csv_path, nrows=0)
    columns = [str(column) for column in header.columns]
    norms = [_norm_header(column) for column in columns]
    if len(set(norms)) != len(norms):
        raise SchemaError(f"{csv_path.name} has two headers that normalize to the same name.")

    feature_aliases = dict(FEATURE_ALIASES)
    for name in spec.model_columns:
        feature_aliases.setdefault(_norm_header(name), name)
    present = {feature_aliases[norm] for norm in norms if norm in feature_aliases}
    missing = [name for name in spec.model_columns if name not in present]
    # The published CIC-IDS2017 MachineLearningCSV has every flow statistic but
    # no Protocol column. Leave it missing so it becomes protocol_other later.
    # Do not invent 6 or 17 from the port number.
    protocol_absent = dataset == "cic_ids2017" and missing == ["protocol"]
    if protocol_absent:
        missing = []
    if missing:
        raise SchemaError(
            f"{csv_path.name} is missing features: {', '.join(missing)}. "
            "Expected CICFlowMeter headers such as 'Flow Duration' / 'Tot Fwd Pkts'."
        )
    if "label" not in norms:
        raise SchemaError(f"{csv_path.name} has no Label column.")

    dialect = _dialect(set(norms))
    unused = tuple(column for column, norm in zip(columns, norms, strict=True) if norm not in feature_aliases and norm not in META_ALIASES and norm != "")
    selected = [column for column, norm in zip(columns, norms, strict=True) if norm in feature_aliases or norm in META_ALIASES]
    frame = _read_csv(csv_path, usecols=selected, nrows=max_rows)
    frame.columns = [_norm_header(column) for column in frame.columns]

    renamed_features = pd.DataFrame(index=frame.index)
    for norm, canonical in feature_aliases.items():
        if norm in frame.columns and canonical in spec.model_columns:
            renamed_features[canonical] = frame[norm]
    if "protocol" not in renamed_features.columns:
        if not protocol_absent:
            raise SchemaError(f"{csv_path.name} is missing features: protocol.")
        renamed_features["protocol"] = float("nan")
        print(
            f"protocol_absent {csv_path.name}: CIC-IDS2017 MachineLearningCSV has no Protocol column. "
            "The value stays missing and is encoded as protocol_other. No protocol number was filled in."
        )
    features = _coerce_features(renamed_features, spec)
    raw_labels = frame["label"].map(lambda value: "" if pd.isna(value) else str(value).strip())
    if (raw_labels == "").any():
        raise UnmappedLabelError(f"{csv_path.name} contains an empty label.")
    normalized = raw_labels.map(normalize_raw_label)
    unknown_mask = ~normalized.isin(label_map.lookup)
    if unknown_mask.any():
        unknown = ", ".join(sorted(set(raw_labels[unknown_mask])))
        raise UnmappedLabelError(
            f"Unmapped {dataset} label(s) in {csv_path.name}: {unknown}. "
            "Add them to the label map before training."
        )
    mapped = normalized.map(label_map.lookup)

    meta = pd.DataFrame(index=features.index)
    meta["capture_file"] = csv_path.name
    meta["source_dataset"] = dataset
    for canonical in ("flow_id", "src_ip", "dst_ip", "src_port", "dst_port", "timestamp"):
        candidates = [
            header
            for header, target in META_ALIASES.items()
            if target == canonical and header in frame.columns
        ]
        meta[canonical] = frame[candidates[0]] if candidates else pd.NA
    meta["label_raw"] = raw_labels.to_numpy()
    meta["label"] = mapped.to_numpy()
    meta = meta.loc[:, list(META_COLUMNS)].reset_index(drop=True)
    features = features.reset_index(drop=True)

    class_counts = {str(label): int(count) for label, count in meta["label"].value_counts().items()}
    binary_counts = {
        "benign": int((meta["label"] == "benign").sum()),
        "malicious": int((meta["label"].map(to_binary_security) == "malicious").sum()),
    }
    values = features.to_numpy(dtype="float64")
    missing_cells = int(np.isnan(values).sum())
    inf_replaced = int(np.isinf(values).sum())
    # Recount missing before replacement. The array above already includes inf as non-nan.
    stored = features.mask(np.isinf(features))
    report = LoadReport(
        source_dataset=dataset,
        capture_file=csv_path.name,
        dialect=dialect,
        rows_loaded=int(len(stored)),
        inf_replaced=inf_replaced,
        missing_feature_cells=missing_cells,
        duplicate_feature_rows=int(stored.duplicated().sum()),
        class_counts=class_counts,
        binary_counts=binary_counts,
        unused_columns=unused,
    )
    return LoadedFlows(features=stored, meta=meta, report=report)


def load_custom_sidecar(path: Path, spec: CustomLabelSpec | None = None, *, root: Path | None = None) -> pd.DataFrame:
    """Validate a self-capture label file. This does not read packets."""
    sidecar = Path(path)
    if not sidecar.is_file():
        raise SchemaError(f"Sidecar not found: {sidecar}")
    if spec is None:
        config = load_config(root=root)
        spec = load_custom_spec(config.labels.custom)
    frame = _read_csv(sidecar)
    missing = [column for column in _SIDECAR_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError(f"{sidecar.name} is missing column(s): {', '.join(missing)}.")
    if frame.empty:
        raise SchemaError(f"{sidecar.name} has no sessions.")

    cleaned = pd.DataFrame()
    cleaned["file"] = frame["file"].map(_required_text)
    cleaned["application"] = frame["application"].map(lambda value: _required_text(value).casefold())
    cleaned["service"] = frame["service"].map(lambda value: _required_text(value).casefold())
    cleaned["vpn"] = frame["vpn"].map(_parse_vpn)
    cleaned["host"] = frame["host"].map(_required_text)
    cleaned["notes"] = frame["notes"].map(lambda value: "" if pd.isna(value) else str(value).strip())
    cleaned["t_start"] = _parse_times(frame["t_start"], column="t_start", filename=sidecar.name)
    cleaned["t_end"] = _parse_times(frame["t_end"], column="t_end", filename=sidecar.name)
    if (cleaned["t_end"] <= cleaned["t_start"]).any():
        raise SchemaError(f"{sidecar.name} has a session whose end is not after its start.")
    for row in cleaned.itertuples(index=False):
        allowed = spec.applications.get(row.application)
        if allowed is None:
            known = ", ".join(sorted(spec.applications))
            raise SchemaError(
                f"{sidecar.name} application {row.application!r} is not in the custom label map. "
                f"Known applications: {known}."
            )
        if row.service not in allowed:
            raise SchemaError(
                f"{sidecar.name} labels {row.application} as {row.service}, "
                f"which is outside {sorted(allowed)}."
            )
    _reject_overlaps(cleaned, filename=sidecar.name)
    return cleaned


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace", **kwargs)
    except pd.errors.ParserError as exc:
        raise SchemaError(f"Could not parse {path.name}: {exc}") from exc
    except UnicodeError as exc:
        raise SchemaError(f"Could not decode {path.name}: {exc}") from exc


def _coerce_features(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    coerced = pd.DataFrame(index=frame.index)
    for column in spec.model_columns:
        coerced[column] = pd.to_numeric(frame[column], errors="coerce")
    return coerced


def _dialect(norms: set[str]) -> str:
    has_v3 = "total fwd packets" in norms
    has_v4 = "tot fwd pkts" in norms
    if has_v3 and has_v4:
        raise SchemaError("CSV mixes CICFlowMeter 3 and 4 headers. Use one extractor per file.")
    if has_v3:
        return "cicflowmeter3"
    if has_v4:
        return "cicflowmeter4"
    return "canonical"


def _norm_header(name: object) -> str:
    return re.sub(r"\s+", " ", str(name).replace("\ufeff", "").strip().lower())


def _required_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        raise SchemaError("A required sidecar cell is empty.")
    text = str(value).strip()
    if not text:
        raise SchemaError("A required sidecar cell is empty.")
    return text


def _parse_vpn(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().casefold()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise SchemaError(f"vpn must be yes/no, true/false, or 1/0. Got {value!r}.")


def _parse_times(values: pd.Series, *, column: str, filename: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise SchemaError(f"{filename} has a value in {column} that is not a timestamp.")
    return parsed


def _reject_overlaps(frame: pd.DataFrame, *, filename: str) -> None:
    """One host may run one labeled application at a time."""
    for host, group in frame.groupby("host", sort=False):
        ordered = group.sort_values(["t_start", "t_end"])
        active_end = None
        active_file = None
        for row in ordered.itertuples(index=False):
            if active_end is not None and row.t_start < active_end:
                raise SchemaError(
                    f"{filename} has overlapping sessions on host {host!r}: "
                    f"{active_file!r} and {row.file!r}. Keep one application open per host."
                )
            if active_end is None or row.t_end > active_end:
                active_end = row.t_end
                active_file = row.file
