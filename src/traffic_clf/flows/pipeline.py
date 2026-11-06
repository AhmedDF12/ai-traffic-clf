"""Label captures and turn them into flow tables.

ISCX files inherit one label from the filename. Self-captures inherit the
sidecar interval that contains the flow start. Flows outside every interval
are dropped and counted, because a real file also contains setup traffic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from traffic_clf.config import ProjectConfig
from traffic_clf.data.labels import (
    CaptureLabel,
    CustomLabelSpec,
    IscxRuleSet,
    load_custom_spec,
    load_iscx_rules,
    parse_iscx_capture,
)
from traffic_clf.data.loaders import load_custom_sidecar
from traffic_clf.errors import SchemaError, UnmappedLabelError
from traffic_clf.features.extract import FlowTable, flows_to_tables
from traffic_clf.flows.builder import build_flows
from traffic_clf.flows.pcap import read_packets


def extract_iscx_pcap(path: Path, config: ProjectConfig, rules: IscxRuleSet | None = None) -> FlowTable:
    """Read one ISCX capture and label every flow from its filename."""
    if rules is None:
        rules = load_iscx_rules(config.labels.iscx_vpn)
    label = parse_iscx_capture(path, rules)
    table = _extract(path, config, source_dataset="iscx_vpn")
    return _with_capture_label(table, label)


def extract_custom_pcap(
    path: Path,
    config: ProjectConfig,
    sidecar: pd.DataFrame | None = None,
    spec: CustomLabelSpec | None = None,
) -> FlowTable:
    """Read one self-capture and label flows from the session sidecar."""
    if spec is None:
        spec = load_custom_spec(config.labels.custom)
    if sidecar is None:
        sidecar = load_custom_sidecar(config.paths.raw / "custom" / "sessions.csv", spec)
    table = _extract(path, config, source_dataset="custom")
    return _apply_sidecar(table, sidecar, filename=Path(path).name)


def _extract(path: Path, config: ProjectConfig, *, source_dataset: str) -> FlowTable:
    batch = read_packets(path)
    flows = build_flows(
        batch.packets,
        active_timeout_s=config.flow.active_timeout_s,
        idle_timeout_s=config.flow.idle_timeout_s,
    )
    table = flows_to_tables(flows, config.features.flow, early_packets=config.flow.early_packets)
    table.meta["capture_file"] = Path(path).name
    table.meta["capture_id"] = Path(path).as_posix()
    table.meta["source_dataset"] = source_dataset
    return FlowTable(
        features=table.features,
        sequence=table.sequence,
        meta=table.meta,
        skipped_packets=batch.skipped,
        truncated_packets=batch.truncated,
    )


def _with_capture_label(table: FlowTable, label: CaptureLabel) -> FlowTable:
    meta = table.meta.copy()
    meta["application"] = label.application
    meta["service"] = label.service
    meta["vpn"] = label.vpn
    meta["excluded_service"] = label.excluded_service
    meta["label_raw"] = label.application
    return FlowTable(
        features=table.features,
        sequence=table.sequence,
        meta=meta,
        skipped_packets=table.skipped_packets,
        truncated_packets=table.truncated_packets,
    )


def _apply_sidecar(table: FlowTable, sidecar: pd.DataFrame, *, filename: str) -> FlowTable:
    sessions = sidecar.loc[sidecar["file"] == filename]
    if sessions.empty:
        raise UnmappedLabelError(
            f"No sidecar session names {filename!r}. The file column must match the capture name."
        )
    keep: list[int] = []
    applications: list[str] = []
    services: list[str] = []
    vpn_flags: list[bool] = []
    for index, timestamp in table.meta["timestamp"].items():
        moment = pd.Timestamp(float(timestamp), unit="s", tz="UTC")
        hits = sessions[(sessions["t_start"] <= moment) & (moment < sessions["t_end"])]
        if len(hits) > 1:
            raise SchemaError(f"{filename} flow at {moment.isoformat()} matches more than one session.")
        if len(hits) == 1:
            row = hits.iloc[0]
            keep.append(index)
            applications.append(str(row["application"]))
            services.append(str(row["service"]))
            vpn_flags.append(bool(row["vpn"]))
    if not keep:
        raise UnmappedLabelError(
            f"No flow in {filename} starts inside a sidecar interval. Check the clock and the trim window."
        )
    dropped = int(len(table.meta) - len(keep))
    meta = table.meta.loc[keep].copy().reset_index(drop=True)
    meta["application"] = applications
    meta["service"] = services
    meta["vpn"] = vpn_flags
    meta["excluded_service"] = pd.NA
    meta["label_raw"] = applications
    return FlowTable(
        features=table.features.loc[keep].reset_index(drop=True),
        sequence=table.sequence.loc[keep].reset_index(drop=True),
        meta=meta,
        skipped_packets=table.skipped_packets,
        truncated_packets=table.truncated_packets,
        dropped_flows=dropped,
    )
