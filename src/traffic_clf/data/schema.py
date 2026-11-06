"""Frozen flow-feature contract and CICFlowMeter column names.

The model never sees ports, addresses, timestamps, or the flow id. Those
columns stay in the metadata frame so a dashboard can still name a talker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from traffic_clf.errors import SchemaError
from traffic_clf.yaml_util import as_string_tuple, expect_version, read_yaml_mapping, require_keys

FORBIDDEN_FEATURES = frozenset(
    {
        "src_port",
        "dst_port",
        "src_ip",
        "dst_ip",
        "timestamp",
        "flow_id",
        "sni",
        "ja3",
        "ja3s",
        "label",
        "label_raw",
        "vpn",
    }
)
REQUIRED_IDENTIFIERS = (
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "timestamp",
)

# Header text is matched after stripping, lower-casing, and collapsing spaces.
CIC2017_FEATURE_ALIASES: dict[str, str] = {
    "flow duration": "duration",
    "total fwd packets": "fwd_packets",
    "total backward packets": "bwd_packets",
    "total length of fwd packets": "fwd_bytes",
    "total length of bwd packets": "bwd_bytes",
    "fwd packet length min": "fwd_pkt_len_min",
    "fwd packet length max": "fwd_pkt_len_max",
    "fwd packet length mean": "fwd_pkt_len_mean",
    "fwd packet length std": "fwd_pkt_len_std",
    "bwd packet length min": "bwd_pkt_len_min",
    "bwd packet length max": "bwd_pkt_len_max",
    "bwd packet length mean": "bwd_pkt_len_mean",
    "bwd packet length std": "bwd_pkt_len_std",
    "min packet length": "pkt_len_min",
    "max packet length": "pkt_len_max",
    "packet length mean": "pkt_len_mean",
    "packet length std": "pkt_len_std",
    "flow iat min": "flow_iat_min",
    "flow iat max": "flow_iat_max",
    "flow iat mean": "flow_iat_mean",
    "flow iat std": "flow_iat_std",
    "fwd iat min": "fwd_iat_min",
    "fwd iat max": "fwd_iat_max",
    "fwd iat mean": "fwd_iat_mean",
    "fwd iat std": "fwd_iat_std",
    "bwd iat min": "bwd_iat_min",
    "bwd iat max": "bwd_iat_max",
    "bwd iat mean": "bwd_iat_mean",
    "bwd iat std": "bwd_iat_std",
    "syn flag count": "syn_count",
    "ack flag count": "ack_count",
    "fin flag count": "fin_count",
    "rst flag count": "rst_count",
    "psh flag count": "psh_count",
    "urg flag count": "urg_count",
    "flow bytes/s": "bytes_per_s",
    "flow packets/s": "packets_per_s",
    "down/up ratio": "down_up_ratio",
    "active min": "active_min",
    "active max": "active_max",
    "active mean": "active_mean",
    "active std": "active_std",
    "idle min": "idle_min",
    "idle max": "idle_max",
    "idle mean": "idle_mean",
    "idle std": "idle_std",
    "protocol": "protocol",
}

CIC2018_FEATURE_ALIASES: dict[str, str] = {
    "flow duration": "duration",
    "tot fwd pkts": "fwd_packets",
    "tot bwd pkts": "bwd_packets",
    "totlen fwd pkts": "fwd_bytes",
    "totlen bwd pkts": "bwd_bytes",
    "fwd pkt len min": "fwd_pkt_len_min",
    "fwd pkt len max": "fwd_pkt_len_max",
    "fwd pkt len mean": "fwd_pkt_len_mean",
    "fwd pkt len std": "fwd_pkt_len_std",
    "bwd pkt len min": "bwd_pkt_len_min",
    "bwd pkt len max": "bwd_pkt_len_max",
    "bwd pkt len mean": "bwd_pkt_len_mean",
    "bwd pkt len std": "bwd_pkt_len_std",
    "pkt len min": "pkt_len_min",
    "pkt len max": "pkt_len_max",
    "pkt len mean": "pkt_len_mean",
    "pkt len std": "pkt_len_std",
    "flow iat min": "flow_iat_min",
    "flow iat max": "flow_iat_max",
    "flow iat mean": "flow_iat_mean",
    "flow iat std": "flow_iat_std",
    "fwd iat min": "fwd_iat_min",
    "fwd iat max": "fwd_iat_max",
    "fwd iat mean": "fwd_iat_mean",
    "fwd iat std": "fwd_iat_std",
    "bwd iat min": "bwd_iat_min",
    "bwd iat max": "bwd_iat_max",
    "bwd iat mean": "bwd_iat_mean",
    "bwd iat std": "bwd_iat_std",
    "syn flag cnt": "syn_count",
    "ack flag cnt": "ack_count",
    "fin flag cnt": "fin_count",
    "rst flag cnt": "rst_count",
    "psh flag cnt": "psh_count",
    "urg flag cnt": "urg_count",
    "flow byts/s": "bytes_per_s",
    "flow pkts/s": "packets_per_s",
    "down/up ratio": "down_up_ratio",
    "active min": "active_min",
    "active max": "active_max",
    "active mean": "active_mean",
    "active std": "active_std",
    "idle min": "idle_min",
    "idle max": "idle_max",
    "idle mean": "idle_mean",
    "idle std": "idle_std",
    "protocol": "protocol",
}

CIC2017_META_ALIASES: dict[str, str] = {
    "flow id": "flow_id",
    "source ip": "src_ip",
    "destination ip": "dst_ip",
    "source port": "src_port",
    "destination port": "dst_port",
    "timestamp": "timestamp",
    "label": "label_raw",
}

CIC2018_META_ALIASES: dict[str, str] = {
    "flow id": "flow_id",
    "src ip": "src_ip",
    "dst ip": "dst_ip",
    "src port": "src_port",
    "dst port": "dst_port",
    "timestamp": "timestamp",
    "label": "label_raw",
}


def merge_aliases(*parts: dict[str, str]) -> dict[str, str]:
    """Merge header maps and fail if one header would mean two columns."""
    merged: dict[str, str] = {}
    for part in parts:
        for header, canonical in part.items():
            current = merged.get(header)
            if current is not None and current != canonical:
                raise SchemaError(f"Alias {header!r} maps to both {current!r} and {canonical!r}.")
            merged[header] = canonical
    return merged


FEATURE_ALIASES = merge_aliases(CIC2017_FEATURE_ALIASES, CIC2018_FEATURE_ALIASES)
META_ALIASES = merge_aliases(CIC2017_META_ALIASES, CIC2018_META_ALIASES)


@dataclass(frozen=True)
class FeatureSpec:
    """Columns a model may see, and identifiers that must stay beside it."""

    version: int
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    identifiers: tuple[str, ...]

    @property
    def model_columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical


@dataclass(frozen=True)
class SequenceSpec:
    """Contract for the first-N packet view. Extraction reads ``flow.early_packets``."""

    version: int
    packets: int
    fields: tuple[str, ...]


def load_feature_spec(path: Path) -> FeatureSpec:
    data = read_yaml_mapping(path)
    source = str(path)
    expect_version(data, source=source)
    require_keys(
        data,
        required={"version", "numeric", "categorical", "identifiers"},
        optional=set(),
        source=source,
    )
    spec = FeatureSpec(
        version=1,
        numeric=as_string_tuple(data["numeric"], name=f"{source} numeric"),
        categorical=as_string_tuple(data["categorical"], name=f"{source} categorical"),
        identifiers=as_string_tuple(data["identifiers"], name=f"{source} identifiers"),
    )
    _validate_feature_spec(spec)
    return spec


def load_sequence_spec(path: Path) -> SequenceSpec:
    data = read_yaml_mapping(path)
    source = str(path)
    expect_version(data, source=source)
    require_keys(
        data,
        required={"version", "packets", "fields"},
        optional=set(),
        source=source,
    )
    packets = data["packets"]
    if isinstance(packets, bool) or not isinstance(packets, int) or not 1 <= packets <= 100:
        raise SchemaError(f"{source} packets must be an integer from 1 to 100.")
    fields = as_string_tuple(data["fields"], name=f"{source} fields")
    if fields != ("size", "direction", "iat"):
        raise SchemaError("Sequence fields must be size, direction, iat, in that order.")
    return SequenceSpec(version=1, packets=packets, fields=fields)


def _validate_feature_spec(spec: FeatureSpec) -> None:
    model = spec.model_columns
    if len(set(model)) != len(model):
        raise SchemaError("Feature columns must be unique.")
    overlap = set(model) & set(spec.identifiers)
    if overlap:
        raise SchemaError(
            "Columns cannot be both features and identifiers: " + ", ".join(sorted(overlap))
        )
    forbidden = set(model) & FORBIDDEN_FEATURES
    if forbidden:
        raise SchemaError(
            "These columns leak identity or the label and cannot be features: "
            + ", ".join(sorted(forbidden))
        )
    missing = set(REQUIRED_IDENTIFIERS) - set(spec.identifiers)
    if missing:
        raise SchemaError("Feature spec is missing identifiers: " + ", ".join(sorted(missing)))
    if spec.categorical != ("protocol",):
        raise SchemaError("protocol is the only categorical flow feature in this version.")
