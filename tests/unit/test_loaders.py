import math
from pathlib import Path

import pandas as pd
import pytest

from traffic_clf.config import load_config
from traffic_clf.data.distribution import class_distribution
from traffic_clf.data.labels import load_custom_spec
from traffic_clf.data.loaders import load_cic_csv, load_custom_sidecar
from traffic_clf.data.schema import (
    CIC2017_FEATURE_ALIASES,
    CIC2017_META_ALIASES,
    CIC2018_FEATURE_ALIASES,
    CIC2018_META_ALIASES,
    FEATURE_ALIASES,
    META_ALIASES,
)
from traffic_clf.errors import SchemaError, UnmappedLabelError


def test_both_cic_dialects_cover_the_feature_contract(repo_root: Path) -> None:
    spec = load_config(root=repo_root).features.flow
    assert set(FEATURE_ALIASES).isdisjoint(META_ALIASES)
    for column in spec.model_columns:
        assert column in set(CIC2017_FEATURE_ALIASES.values())
        assert column in set(CIC2018_FEATURE_ALIASES.values())


def test_cic2017_csv_drops_identifiers_and_maps_labels(repo_root: Path, tmp_path: Path) -> None:
    spec = load_config(root=repo_root).features.flow
    path = tmp_path / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    _write_flows(
        path,
        spec,
        CIC2017_FEATURE_ALIASES,
        CIC2017_META_ALIASES,
        [
            {
                "label": "BENIGN",
                "fwd_packets": 1,
                "bytes_per_s": math.inf,
                "src_ip": "10.0.0.8",
                "dst_port": 443,
            },
            {"label": "DoS Hulk", "fwd_packets": 2, "fwd_bytes": None},
            {"label": "Web Attack \u2013 Brute Force", "fwd_packets": 3},
            {"label": "PortScan", "fwd_packets": 4},
            {"label": "PortScan", "fwd_packets": 4},
        ],
        label_header=" label",
        junk={"Fwd Avg Bytes/Bulk": 9},
    )
    loaded = load_cic_csv(path, dataset="cic_ids2017", root=repo_root)
    assert list(loaded.features.columns) == list(spec.model_columns)
    assert "src_ip" not in loaded.features.columns
    assert "dst_port" not in loaded.features.columns
    assert "label" not in loaded.features.columns
    assert loaded.meta.loc[0, "src_ip"] == "10.0.0.8"
    assert int(loaded.meta.loc[0, "dst_port"]) == 443
    assert loaded.meta["label"].tolist() == ["benign", "ddos", "brute_force", "port_scan", "port_scan"]
    assert loaded.report.dialect == "cicflowmeter3"
    assert loaded.report.inf_replaced == 1
    assert loaded.report.missing_feature_cells == 1
    assert loaded.report.duplicate_feature_rows == 1
    assert loaded.report.binary_counts == {"benign": 1, "malicious": 4}
    assert pd.isna(loaded.features.loc[0, "bytes_per_s"])
    assert pd.isna(loaded.features.loc[1, "fwd_bytes"])
    assert "Fwd Avg Bytes/Bulk" in loaded.report.unused_columns


def test_cic2018_short_headers_and_the_infilteration_typo(repo_root: Path, tmp_path: Path) -> None:
    spec = load_config(root=repo_root).features.flow
    path = tmp_path / "Friday-02-03-2018.csv"
    _write_flows(
        path,
        spec,
        CIC2018_FEATURE_ALIASES,
        CIC2018_META_ALIASES,
        [
            {"label": "Infilteration", "fwd_packets": 8, "dst_port": 80},
            {"label": "Benign", "fwd_packets": 1, "dst_port": 443},
        ],
        label_header="Label",
    )
    loaded = load_cic_csv(path, dataset="cic_ids2018", root=repo_root)
    assert loaded.report.dialect == "cicflowmeter4"
    assert loaded.meta["label"].tolist() == ["malware", "benign"]
    assert pd.isna(loaded.meta.loc[0, "src_ip"])
    assert "dst_port" not in loaded.features.columns


def test_max_rows_and_unknown_labels(repo_root: Path, tmp_path: Path) -> None:
    spec = load_config(root=repo_root).features.flow
    identity = {column: column for column in spec.model_columns}
    path = tmp_path / "subset.csv"
    _write_flows(
        path,
        spec,
        identity,
        {},
        [{"label": "BENIGN", "fwd_packets": index} for index in range(4)],
        label_header="label",
    )
    loaded = load_cic_csv(path, dataset="cic_ids2017", root=repo_root, max_rows=2)
    assert loaded.report.rows_loaded == 2
    assert loaded.report.dialect == "canonical"

    unknown = tmp_path / "unknown.csv"
    _write_flows(
        unknown,
        spec,
        identity,
        {},
        [{"label": "Mystery", "fwd_packets": 1}],
        label_header="label",
    )
    with pytest.raises(UnmappedLabelError, match="Mystery"):
        load_cic_csv(unknown, dataset="cic_ids2017", root=repo_root)


def test_cic2017_without_protocol_column_stays_missing(repo_root: Path, tmp_path: Path) -> None:
    spec = load_config(root=repo_root).features.flow
    path = tmp_path / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    _write_flows(
        path,
        spec,
        CIC2017_FEATURE_ALIASES,
        CIC2017_META_ALIASES,
        [{"label": "DDoS", "fwd_packets": 4, "dst_port": 80}],
        label_header=" Label",
        omit=("protocol",),
    )
    loaded = load_cic_csv(path, dataset="cic_ids2017", root=repo_root)
    assert pd.isna(loaded.features.loc[0, "protocol"])
    assert loaded.meta.loc[0, "label"] == "ddos"
    with pytest.raises(SchemaError, match="protocol"):
        load_cic_csv(path, dataset="cic_ids2018", root=repo_root)


def test_class_distribution_counts_missing_labels() -> None:
    frame = class_distribution(pd.Series(["b", "a", "a", None]))
    assert set(frame["class"]) == {"a", "b", "(missing)"}
    assert math.isclose(float(frame["fraction"].sum()), 1.0)
    assert class_distribution(pd.Series([], dtype=object)).empty


def test_custom_sidecar_accepts_the_example_and_rejects_bad_sessions(repo_root: Path, tmp_path: Path) -> None:
    spec = load_custom_spec(load_config(root=repo_root).labels.custom)
    loaded = load_custom_sidecar(repo_root / "configs" / "examples" / "custom_sessions.csv", spec)
    assert loaded["application"].tolist() == ["youtube", "browser", "zoom"]
    assert bool(loaded.loc[2, "vpn"]) is True
    assert loaded["service"].tolist() == ["streaming", "browsing", "voip"]

    overlap = tmp_path / "overlap.csv"
    overlap.write_text(
        "file,t_start,t_end,application,service,vpn,host,notes\n"
        "a.pcap,2026-11-06T12:00:00Z,2026-11-06T12:30:00Z,youtube,streaming,no,lab,one\n"
        "b.pcap,2026-11-06T12:10:00Z,2026-11-06T12:20:00Z,browser,browsing,no,lab,two\n",
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="overlapping"):
        load_custom_sidecar(overlap, spec)

    other_host = tmp_path / "hosts.csv"
    other_host.write_text(
        "file,t_start,t_end,application,service,vpn,host,notes\n"
        "a.pcap,2026-11-06T12:00:00Z,2026-11-06T12:10:00Z,youtube,streaming,no,lab-a,\n"
        "b.pcap,2026-11-06T12:00:00Z,2026-11-06T12:10:00Z,zoom,voip,yes,lab-b,\n",
        encoding="utf-8",
    )
    both = load_custom_sidecar(other_host, spec)
    assert list(both["host"]) == ["lab-a", "lab-b"]

    bad_service = tmp_path / "zoom.csv"
    bad_service.write_text(
        "file,t_start,t_end,application,service,vpn,host,notes\n"
        "zoom.pcap,2026-11-06T12:00:00Z,2026-11-06T12:10:00Z,zoom,streaming,no,lab,\n",
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="zoom"):
        load_custom_sidecar(bad_service, spec)

    folded = tmp_path / "case.csv"
    folded.write_text(
        "file,t_start,t_end,application,service,vpn,host,notes\n"
        "yt.pcap,2026-11-06T12:00:00Z,2026-11-06T12:10:00Z,YouTube,streaming,no,lab,call\n",
        encoding="utf-8",
    )
    assert load_custom_sidecar(folded, spec).loc[0, "application"] == "youtube"


def _write_flows(path, spec, feature_aliases, meta_aliases, rows, *, label_header, junk=None, omit=()) -> None:
    feature_header = {canonical: header for header, canonical in feature_aliases.items()}
    meta_header = {canonical: header for header, canonical in meta_aliases.items()}
    records = []
    for row in rows:
        record = {
            feature_header[column]: row.get(column, 0)
            for column in spec.model_columns
            if column not in omit
        }
        for canonical in ("flow_id", "src_ip", "dst_ip", "src_port", "dst_port", "timestamp"):
            if canonical in row:
                record[meta_header[canonical]] = row[canonical]
        if junk:
            record.update(junk)
        record[label_header] = row["label"]
        records.append(record)
    pd.DataFrame.from_records(records).to_csv(path, index=False)
