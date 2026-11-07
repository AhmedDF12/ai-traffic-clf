from pathlib import Path

import pandas as pd
import pytest

from traffic_clf.config import load_config
from traffic_clf.data.splits import Group
from traffic_clf.errors import SplitError
from traffic_clf.evaluation.protocol import (
    assign_group_kfold,
    assign_within_day,
    cap_flows_per_capture,
    dominance,
    run_capture_protocol,
    run_source_protocol,
)
from traffic_clf.evaluation.sensitivity import select_feature_group, truncate_early_packets


def test_cap_limits_each_capture_and_keeps_small_files() -> None:
    features = pd.DataFrame({"duration": range(7)})
    meta = pd.DataFrame({"capture_id": ["big"] * 5 + ["small"] * 2, "label": ["voip"] * 7})
    capped, capped_meta = cap_flows_per_capture(features, meta, 2, seed=1)
    assert capped_meta["capture_id"].value_counts().to_dict() == {"big": 2, "small": 2}
    again, _ = cap_flows_per_capture(features, meta, 2, seed=1)
    assert capped["duration"].tolist() == again["duration"].tolist()


def test_group_kfold_keeps_every_included_class_in_every_fold() -> None:
    groups = [Group(f"a{i}", "streaming", weight=10) for i in range(5)]
    groups.append(Group("huge", "streaming", weight=5000))
    groups.extend(Group(f"b{i}", "chat", weight=3) for i in range(2))
    folds, excluded, largest, largest_fold = assign_group_kfold(groups, n_folds=5, min_files=5, seed=3)
    assert excluded == ["chat"]
    assert largest == "huge"
    assert folds[largest_fold]["huge"] == "test"
    assert len(folds) == 5
    for fold in folds:
        held = [group.group_id for group in groups if fold.get(group.group_id) == "test" and group.stratum == "streaming"]
        assert held
        assert all(fold[group.group_id] == "train" for group in groups if group.stratum == "streaming" and group.group_id not in held)


def test_group_kfold_rejects_a_short_class_set() -> None:
    groups = [Group(f"a{i}", "chat", weight=1) for i in range(4)]
    with pytest.raises(SplitError, match="5"):
        assign_group_kfold(groups, n_folds=5, min_files=5, seed=1)


def test_within_day_orders_each_class_and_leaves_monday_alone() -> None:
    rows = []
    for label, start in (("ddos", 10), ("benign", 100)):
        for offset in range(6):
            rows.append({"capture_file": "Friday.csv", "label": label, "timestamp": start + offset})
    rows.append({"capture_file": "Monday-WorkingHours.pcap_ISCX.csv", "label": "benign", "timestamp": 1})
    assigned = assign_within_day(pd.DataFrame(rows), fp_check_files={"Monday-WorkingHours.pcap_ISCX.csv"})
    monday = assigned.loc[assigned["capture_file"].str.startswith("Monday"), "split"]
    assert set(monday) == {"fp_check"}
    ddos = assigned.loc[assigned["label"] == "ddos"].sort_values("timestamp")
    assert ddos["split"].tolist()[:3] == ["train", "train", "train"]
    assert set(ddos["split"]) == {"train", "validation", "test"}
    assert ddos.loc[ddos["split"] == "test", "timestamp"].min() > ddos.loc[ddos["split"] == "train", "timestamp"].max()


def test_dominance_blocks_a_headline_when_one_file_is_most_of_the_split() -> None:
    meta = pd.DataFrame(
        {
            "split": ["test"] * 4,
            "capture_file": ["huge", "huge", "huge", "other"],
            "label": ["voip"] * 4,
        }
    )
    note = dominance(meta, "test")
    assert note["headline"] is False
    assert "huge" in str(note["reason"])


def test_protocol_reports_cap_kfold_and_environment(repo_root: Path, tmp_path: Path) -> None:
    spec = load_config(root=repo_root, task="service").features.flow
    rows = []
    for source, marker, label in (("iscx_vpn", 0.0, "chat"), ("custom", 50.0, "voip")):
        for file_index in range(5):
            for flow_index in range(4):
                row = {column: 0.0 for column in spec.numeric}
                row["protocol"] = 6
                row["flow_iat_min"] = marker
                row["capture_id"] = f"{source}-{file_index}"
                row["capture_file"] = f"{source}-{file_index}.pcap"
                row["label"] = label
                row["source_dataset"] = source
                row["timestamp"] = 1_700_000_000 + file_index * 10 + flow_index
                row["src_ip"] = "10.0.0.1" if source == "iscx_vpn" else "10.0.0.2"
                row["dst_ip"] = "1.1.1.1"
                row["src_port"] = 40000 + file_index
                row["dst_port"] = 443
                rows.append(row)
    frame = pd.DataFrame(rows)
    meta_columns = ["capture_id", "capture_file", "label", "source_dataset", "timestamp", "src_ip", "dst_ip", "src_port", "dst_port"]
    payload = run_capture_protocol(
        frame.drop(columns=meta_columns),
        frame.loc[:, meta_columns],
        spec,
        task="service",
        seed=1,
        n_jobs=1,
        max_flows_per_capture=2,
        out=tmp_path,
        sessions=pd.DataFrame({"file": [f"custom-{i}.pcap" for i in range(5)], "host": ["lab"] * 5}),
        model_names=("random_forest",),
    )
    names = [item["experiment"] for item in payload["experiments"]]
    assert "file_split_uncapped" in names
    assert "file_split_cap_2" in names
    assert "group_kfold_5_uncapped" in names
    assert "group_kfold_5_capped" in names
    kfold = next(item for item in payload["experiments"] if item["experiment"] == "group_kfold_5_uncapped")
    assert kfold["macro_f1_mean"] is not None
    assert kfold["largest_file_fold_macro_f1"] is not None
    uncapped = next(item for item in payload["experiments"] if item["experiment"] == "file_split_uncapped")
    mixed_test = uncapped["models"][0]["splits"]["test"]
    assert mixed_test["headline"] is False
    assert "mixes capture sources" in mixed_test["dominance"]["reason"]
    assert any(row["environment_confounded"] for row in payload["class_source_control"])
    source_audit = payload["environment_audit"]["models"]["source"]
    assert source_audit["above_chance"] is True
    without = payload["environment_audit"]["models"]["source_without_flow_iat_min"]
    assert without["above_chance"] is False
    assert (tmp_path / "service_protocol.json").is_file()


def test_source_experiments_stay_separate_and_mark_one_source_classes(repo_root: Path, tmp_path: Path) -> None:
    spec = load_config(root=repo_root, task="service").features.flow
    rows = []
    plan = (
        ("iscx_vpn", "iscx", "chat", 0.0),
        ("iscx_vpn", "iscx", "voip", 1.0),
        ("custom", "vlc", "chat", 10.0),
        ("custom", "vlc", "voip", 11.0),
        ("custom", "vlc", "streaming", 20.0),
    )
    for source, kind, label, marker in plan:
        for file_index in range(3):
            for flow_index in range(3):
                row = {column: marker for column in spec.numeric}
                row["protocol"] = 6
                row["capture_id"] = f"{kind}-{label}-{file_index}"
                row["capture_file"] = f"{kind}-{label}-{file_index}.pcap"
                row["label"] = label
                row["source_dataset"] = source
                row["timestamp"] = file_index * 10 + flow_index
                row["src_ip"] = "10.0.0.1"
                row["dst_ip"] = "1.1.1.1"
                row["src_port"] = 1
                row["dst_port"] = 443
                rows.append(row)
    frame = pd.DataFrame(rows)
    meta_columns = ["capture_id", "capture_file", "label", "source_dataset", "timestamp", "src_ip", "dst_ip", "src_port", "dst_port"]
    sessions = pd.DataFrame(
        {
            "file": [f"vlc-{label}-{file_index}.pcap" for label in ("chat", "voip", "streaming") for file_index in range(3)],
            "host": ["vlc-lab"] * 9,
        }
    )
    payload = run_source_protocol(
        frame.drop(columns=meta_columns),
        frame.loc[:, meta_columns],
        spec,
        task="service",
        seed=1,
        n_jobs=1,
        max_flows_per_capture=2,
        out=tmp_path,
        sessions=sessions,
        model_names=("random_forest",),
    )
    flags = {row["class"]: row["environment_confounded"] for row in payload["class_source_control"]}
    assert flags["streaming"] is True
    assert flags["chat"] is False
    assert flags["voip"] is False
    names = [item["experiment"] for item in payload["experiments"]]
    assert "iscx_only_file_split_uncapped" in names
    assert "vlc_only_file_split_cap_2" in names
    assert "cross_source_iscx_train_vlc_test" in names
    assert "adversarial_source_balanced" in names
    cross = next(item for item in payload["experiments"] if item["experiment"] == "cross_source_iscx_train_vlc_test")
    assert cross["headline"] is False
    assert "streaming" not in cross["shared_classes"]
    adversarial = next(item for item in payload["experiments"] if item["experiment"] == "adversarial_source_balanced")
    assert adversarial["headline"] is False
    assert adversarial.get("status") is None
    assert (tmp_path / "service_source.json").is_file()


def test_adversarial_reports_when_no_class_has_two_sources(repo_root: Path, tmp_path: Path) -> None:
    spec = load_config(root=repo_root, task="service").features.flow
    rows = []
    for source, kind, label in (("iscx_vpn", "iscx", "chat"), ("custom", "vlc", "streaming")):
        for file_index in range(3):
            row = {column: 0.0 for column in spec.numeric}
            row["protocol"] = 6
            row["capture_id"] = f"{kind}-{file_index}"
            row["capture_file"] = f"{kind}-{label}-{file_index}.pcap"
            row["label"] = label
            row["source_dataset"] = source
            row["timestamp"] = file_index
            row["src_ip"] = "10.0.0.1"
            row["dst_ip"] = "1.1.1.1"
            row["src_port"] = 1
            row["dst_port"] = 443
            rows.append(row)
    frame = pd.DataFrame(rows)
    meta_columns = ["capture_id", "capture_file", "label", "source_dataset", "timestamp", "src_ip", "dst_ip", "src_port", "dst_port"]
    sessions = pd.DataFrame({"file": [f"vlc-streaming-{i}.pcap" for i in range(3)], "host": ["vlc-lab"] * 3})
    payload = run_source_protocol(
        frame.drop(columns=meta_columns),
        frame.loc[:, meta_columns],
        spec,
        task="service",
        seed=1,
        n_jobs=1,
        max_flows_per_capture=2,
        out=tmp_path,
        sessions=sessions,
        model_names=("random_forest",),
    )
    cross = next(item for item in payload["experiments"] if item["experiment"] == "cross_source_iscx_train_vlc_test")
    adversarial = next(item for item in payload["experiments"] if item["experiment"] == "adversarial_source_balanced")
    assert cross["status"].startswith("no class")
    assert adversarial["status"] == "none exists"
    assert all(row["environment_confounded"] for row in payload["class_source_control"])


def test_feature_groups_and_early_packets_drop_the_other_columns() -> None:
    frame = pd.DataFrame(
        {
            "duration": [1.0],
            "fwd_bytes": [2.0],
            "flow_iat_min": [3.0],
            "syn_count": [1.0],
            "bytes_per_s": [4.0],
            "active_mean": [5.0],
            "idle_mean": [6.0],
            "protocol": [6],
            "size_01": [10.0],
            "size_02": [11.0],
            "direction_01": [1.0],
            "direction_02": [-1.0],
            "iat_01": [7.0],
            "iat_02": [8.0],
            "seq_observed": [2.0],
        }
    )
    sizes = select_feature_group(frame, "sizes")
    assert "flow_iat_min" not in sizes.columns
    assert "fwd_bytes" in sizes.columns
    assert "protocol" in sizes.columns
    early = truncate_early_packets(frame, 1)
    assert "size_02" not in early.columns
    assert early["size_01"].tolist() == [10.0]
    assert early["seq_observed"].tolist() == [1.0]
