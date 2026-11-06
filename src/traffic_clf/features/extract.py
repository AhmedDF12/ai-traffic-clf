"""Flow statistics and the first-N packet sequence.

Duration and inter-arrival times are microseconds, matching CICFlowMeter.
Packet sizes are bytes. Rates are per second. A zero-duration flow gets a
missing rate instead of infinity. Active and idle times use the same idle
threshold that closes the flow, so a closed flow does not contain an idle gap.
Sample standard deviation is zero when a direction has fewer than two values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from traffic_clf.data.schema import FORBIDDEN_FEATURES, FeatureSpec
from traffic_clf.errors import SchemaError
from traffic_clf.flows.builder import BuiltFlow

_FIN, _SYN, _RST, _PSH, _ACK, _URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20


@dataclass(frozen=True)
class FlowTable:
    features: pd.DataFrame
    sequence: pd.DataFrame
    meta: pd.DataFrame
    skipped_packets: int = 0
    truncated_packets: int = 0
    dropped_flows: int = 0


def sequence_column_names(packets: int) -> tuple[str, ...]:
    """Column names for one early-classification prefix."""
    names: list[str] = []
    for prefix in ("size", "direction", "iat"):
        names.extend(f"{prefix}_{index:02d}" for index in range(1, packets + 1))
    names.append("seq_observed")
    return tuple(names)


def flows_to_tables(flows: list[BuiltFlow], spec: FeatureSpec, *, early_packets: int) -> FlowTable:
    """Convert closed flows into the feature contract and a sequence matrix."""
    if early_packets < 1:
        raise SchemaError("early_packets must be at least 1.")
    feature_rows: list[dict[str, float]] = []
    sequence_rows: list[dict[str, float]] = []
    meta_rows: list[dict[str, object]] = []
    for flow in flows:
        feature_rows.append(_flow_features(flow, spec))
        sequence_rows.append(_sequence_features(flow, early_packets))
        meta_rows.append(_meta(flow))
    features = pd.DataFrame(feature_rows, columns=list(spec.model_columns))
    sequence = pd.DataFrame(sequence_rows, columns=list(sequence_column_names(early_packets)))
    meta = pd.DataFrame(meta_rows)
    leaked = set(features.columns) & FORBIDDEN_FEATURES
    if leaked:
        raise SchemaError("Flow features include identifier columns: " + ", ".join(sorted(leaked)))
    return FlowTable(features=features, sequence=sequence, meta=meta)


def _flow_features(flow: BuiltFlow, spec: FeatureSpec) -> dict[str, float]:
    lengths = np.asarray([packet.length for packet in flow.packets], dtype=float)
    directions = np.asarray(flow.directions, dtype=int)
    times = np.asarray([packet.timestamp for packet in flow.packets], dtype=float)
    forward = directions == 1
    backward = ~forward
    duration_us = float((times[-1] - times[0]) * 1_000_000) if len(times) else 0.0
    duration_s = duration_us / 1_000_000
    fwd_lengths = lengths[forward]
    bwd_lengths = lengths[backward]
    fwd_iat = _iat_us(times[forward])
    bwd_iat = _iat_us(times[backward])
    flow_iat = _iat_us(times)
    flags = [packet.tcp_flags for packet in flow.packets]
    total_bytes = float(lengths.sum()) if len(lengths) else 0.0
    fwd_packets = int(forward.sum())
    bwd_packets = int(backward.sum())
    active, idle = _active_idle_us(times)
    values = {
        "duration": duration_us,
        "fwd_packets": float(fwd_packets),
        "bwd_packets": float(bwd_packets),
        "fwd_bytes": float(fwd_lengths.sum()) if len(fwd_lengths) else 0.0,
        "bwd_bytes": float(bwd_lengths.sum()) if len(bwd_lengths) else 0.0,
        "bytes_per_s": total_bytes / duration_s if duration_s > 0 else np.nan,
        "packets_per_s": len(flow.packets) / duration_s if duration_s > 0 else np.nan,
        "down_up_ratio": (bwd_packets / fwd_packets) if fwd_packets else 0.0,
        "syn_count": _flag_count(flags, _SYN),
        "ack_count": _flag_count(flags, _ACK),
        "fin_count": _flag_count(flags, _FIN),
        "rst_count": _flag_count(flags, _RST),
        "psh_count": _flag_count(flags, _PSH),
        "urg_count": _flag_count(flags, _URG),
        "protocol": float(flow.protocol),
    }
    _put_stats(values, "fwd_pkt_len", fwd_lengths)
    _put_stats(values, "bwd_pkt_len", bwd_lengths)
    _put_stats(values, "pkt_len", lengths)
    _put_stats(values, "fwd_iat", fwd_iat)
    _put_stats(values, "bwd_iat", bwd_iat)
    _put_stats(values, "flow_iat", flow_iat)
    _put_stats(values, "active", active)
    _put_stats(values, "idle", idle)
    return {column: float(values[column]) for column in spec.model_columns}


def _sequence_features(flow: BuiltFlow, early_packets: int) -> dict[str, float]:
    observed = min(len(flow.packets), early_packets)
    row: dict[str, float] = {"seq_observed": float(observed)}
    previous = flow.packets[0].timestamp if flow.packets else 0.0
    for index in range(1, early_packets + 1):
        if index <= observed:
            packet = flow.packets[index - 1]
            gap = 0.0 if index == 1 else (packet.timestamp - previous) * 1_000_000
            previous = packet.timestamp
            row[f"size_{index:02d}"] = float(packet.length)
            row[f"direction_{index:02d}"] = float(flow.directions[index - 1])
            row[f"iat_{index:02d}"] = float(gap)
        else:
            row[f"size_{index:02d}"] = 0.0
            row[f"direction_{index:02d}"] = 0.0
            row[f"iat_{index:02d}"] = 0.0
    return row


def _meta(flow: BuiltFlow) -> dict[str, object]:
    start_us = int(round(flow.start_time * 1_000_000))
    flow_id = f"{flow.protocol}-{flow.src_ip}-{flow.src_port}-{flow.dst_ip}-{flow.dst_port}-{start_us}"
    return {
        "flow_id": flow_id,
        "src_ip": flow.src_ip,
        "dst_ip": flow.dst_ip,
        "src_port": flow.src_port,
        "dst_port": flow.dst_port,
        "timestamp": flow.start_time,
    }


def _iat_us(times: np.ndarray) -> np.ndarray:
    if len(times) < 2:
        return np.asarray([], dtype=float)
    return np.diff(times) * 1_000_000


def _active_idle_us(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One active period covering the flow. Idle gaps are removed by the builder."""
    if len(times) == 0:
        return np.asarray([0.0]), np.asarray([], dtype=float)
    return np.asarray([(times[-1] - times[0]) * 1_000_000]), np.asarray([], dtype=float)


def _put_stats(target: dict[str, float], prefix: str, values: np.ndarray) -> None:
    if len(values) == 0:
        minimum = maximum = mean = std = 0.0
    else:
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    target[f"{prefix}_min"] = minimum
    target[f"{prefix}_max"] = maximum
    target[f"{prefix}_mean"] = mean
    target[f"{prefix}_std"] = std


def _flag_count(flags: list[int], bit: int) -> float:
    return float(sum(1 for value in flags if value & bit))
