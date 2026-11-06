import socket
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_clf.cli import main
from traffic_clf.config import load_config
from traffic_clf.data.schema import CIC2017_FEATURE_ALIASES, CIC2017_META_ALIASES
from traffic_clf.errors import SchemaError
from traffic_clf.flows.pcap import write_pcap
from traffic_clf.preprocess.prepare import (
    fit_preprocess,
    load_preprocess_state,
    prepare_cic_csvs,
    prepare_iscx_pcaps,
    write_prepared,
)


def test_scaler_is_fit_on_the_training_rows_only(repo_root: Path) -> None:
    spec = load_config(root=repo_root).features.flow
    train = _matrix(spec, [0.0, 10.0, np.nan], [6, 6, 17], direction=-1)
    test = _matrix(spec, [100.0], [17], direction=-1)
    state = fit_preprocess(train, spec)
    duration = list(state.numeric_columns).index("duration")
    assert state.scaler.mean_[duration] == pytest.approx(5.0)
    transformed = state.transform(test)
    scale = state.scaler.scale_[duration]
    assert transformed.loc[0, "duration"] == pytest.approx((100.0 - 5.0) / scale)
    assert "protocol" not in transformed.columns
    assert "dst_port" not in transformed.columns
    assert transformed.loc[0, "protocol_udp"] == 1
    assert transformed.loc[0, "protocol_tcp"] == 0
    assert transformed.loc[0, "direction_01"] == -1

    leaked = train.copy()
    leaked["dst_port"] = 443
    with pytest.raises(SchemaError, match="identifier"):
        fit_preprocess(leaked, spec)


def test_cic_day_split_keeps_monday_out_of_training(repo_root: Path, tmp_path: Path) -> None:
    config = load_config(root=repo_root, task="security")
    spec = config.features.flow
    tuesday = tmp_path / "Tuesday-WorkingHours.pcap_ISCX.csv"
    friday = tmp_path / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    monday = tmp_path / "Monday-WorkingHours.pcap_ISCX.csv"
    _write_cic(tuesday, spec, [{"label": "BENIGN", "duration": 10.0, "fwd_packets": 1}, {"label": "BENIGN", "duration": 10.0, "fwd_packets": 1}])
    _write_cic(friday, spec, [{"label": "DoS Hulk", "duration": 100.0, "fwd_packets": 4}])
    _write_cic(monday, spec, [{"label": "BENIGN", "duration": 1.0, "fwd_packets": 2}])
    prepared = prepare_cic_csvs([tuesday, friday, monday], config)
    assert prepared.report.rows_train == 1
    assert prepared.report.dropped_duplicates == 1
    assert prepared.report.rows_validation == 0
    assert prepared.report.rows_test == 1
    assert prepared.report.rows_fp_check == 1
    assert prepared.train.meta["capture_file"].iloc[0].startswith("Tuesday")
    assert prepared.test.meta["label"].iloc[0] == "ddos"
    assert "src_ip" not in prepared.train.features.columns
    assert "protocol_tcp" in prepared.train.features.columns
    write_prepared(prepared, tmp_path / "out")
    restored = load_preprocess_state(tmp_path / "out" / "preprocess.joblib")
    raw = pd.read_parquet(tmp_path / "out" / "train_features.parquet")
    assert list(raw.columns) == list(restored.output_columns)
    assert "src_ip" in pd.read_parquet(tmp_path / "out" / "train_meta.parquet").columns


def test_iscx_file_split_drops_email_and_keeps_a_capture_together(repo_root: Path, tmp_path: Path) -> None:
    overlay = tmp_path / "early.yaml"
    overlay.write_text("flow:\n  early_packets: 2\n", encoding="utf-8")
    config = load_config(root=repo_root, task="service", overlay=overlay)
    gap = tmp_path / "youtube_gap.pcap"
    write_pcap(
        gap,
        [
            (1_700_000_000.0, _tcp("10.0.0.2", "1.1.1.1", 1000, 443, 0x02, 0)),
            (1_700_000_000.5, _tcp("1.1.1.1", "10.0.0.2", 443, 1000, 0x12, 0)),
            (1_700_000_010.0, _tcp("10.0.0.2", "1.1.1.1", 1000, 443, 0x02, 0)),
        ],
    )
    other = []
    for index in (2, 3):
        path = tmp_path / f"youtube{index}.pcap"
        write_pcap(path, [(1_700_000_100.0, _tcp("10.0.0.3", "1.1.1.1", 2000, 443, 0x02, 0))])
        other.append(path)
    email = tmp_path / "email2b.pcap"
    write_pcap(email, [(1_700_000_200.0, _tcp("10.0.0.4", "1.1.1.1", 3000, 443, 0x02, 0))])
    prepared = prepare_iscx_pcaps([gap, *other, email], config)
    assert prepared.report.dropped_unlabeled == 1
    assert prepared.report.rows_train >= 1
    assert prepared.report.rows_validation >= 1
    assert prepared.report.rows_test >= 1
    found = [
        part
        for part in (prepared.train, prepared.validation, prepared.test)
        if (part.meta["capture_file"] == "youtube_gap.pcap").any()
    ]
    assert len(found) == 1
    assert (found[0].meta["capture_file"] == "youtube_gap.pcap").sum() == 2
    assert "email" not in set(prepared.train.meta["application"]) | set(prepared.test.meta["application"])
    directions = pd.concat(
        [
            prepared.train.features["direction_02"],
            prepared.validation.features["direction_02"],
            prepared.test.features["direction_02"],
        ]
    )
    assert set(directions.tolist()) <= {-1.0, 0.0, 1.0}
    assert -1.0 in set(directions.tolist())


def test_extract_command_prints_the_iscx_label(repo_root: Path, tmp_path: Path, capsys) -> None:
    path = tmp_path / "vpn_netflix_A.pcap"
    write_pcap(path, [(1_700_000_000.0, _tcp("10.0.0.2", "1.1.1.1", 1111, 443, 0x02, 0))])
    assert main(["extract-pcap", "--root", str(repo_root), "--task", "application", str(path)]) == 0
    output = capsys.readouterr().out
    assert "application=netflix" in output
    assert "service=streaming" in output
    assert "vpn=true" in output
    assert "flows=1" in output


def _matrix(spec, durations, protocols, *, direction: float) -> pd.DataFrame:
    rows = []
    for duration, protocol in zip(durations, protocols, strict=True):
        row = {column: 0.0 for column in spec.numeric}
        row["duration"] = duration
        row["protocol"] = protocol
        row["size_01"] = 40.0
        row["direction_01"] = direction
        row["iat_01"] = 0.0
        row["seq_observed"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def _write_cic(path: Path, spec, rows: list[dict]) -> None:
    feature_header = {canonical: header for header, canonical in CIC2017_FEATURE_ALIASES.items()}
    meta_header = {canonical: header for header, canonical in CIC2017_META_ALIASES.items()}
    records = []
    for row in rows:
        record = {feature_header[column]: row.get(column, 0) for column in spec.model_columns}
        record[meta_header["src_ip"]] = "10.0.0.8"
        record[meta_header["dst_port"]] = 443
        record[" label"] = row["label"]
        records.append(record)
    pd.DataFrame.from_records(records).to_csv(path, index=False)


def _tcp(src: str, dst: str, sport: int, dport: int, flags: int, payload_len: int) -> bytes:
    tcp = struct.pack("!HHIIBBHHH", sport, dport, 0, 0, 5 << 4, flags, 65535, 0, 0) + bytes(payload_len)
    ip = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        20 + len(tcp),
        1,
        0,
        64,
        6,
        0,
        socket.inet_aton(src),
        socket.inet_aton(dst),
    )
    return bytes(12) + struct.pack("!H", 0x0800) + ip + tcp
