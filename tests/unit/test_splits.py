from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from traffic_clf.config import load_config
from traffic_clf.data.splits import (
    Group,
    assign_grouped_split,
    attach_split,
    load_named_split,
    resolve_named_split,
)
from traffic_clf.errors import SplitError


def test_cic_days_are_not_split_at_random(repo_root: Path) -> None:
    table = load_named_split(load_config(root=repo_root).splits.cic_ids2017)
    assert resolve_named_split(table, "Monday-WorkingHours.pcap_ISCX.csv") == "fp_check"
    assert resolve_named_split(table, "Tuesday-WorkingHours.pcap_ISCX.csv") == "train"
    assert resolve_named_split(table, "Wednesday-workingHours.pcap_ISCX.csv") == "train"
    assert resolve_named_split(table, "Wednesday-WorkingHours.pcap_ISCX.csv") == "train"
    assert resolve_named_split(table, "Friday-WorkingHours-Morning.pcap_ISCX.csv") == "validation"
    assert resolve_named_split(table, "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv") == "test"
    assert resolve_named_split(table, "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv") == "test"
    with pytest.raises(SplitError):
        resolve_named_split(table, "not-a-day.csv")


def test_attach_split_rejects_an_unknown_file(repo_root: Path) -> None:
    table = load_named_split(load_config(root=repo_root).splits.cic_ids2017)
    meta = pd.DataFrame({"capture_file": ["Tuesday-WorkingHours.pcap_ISCX.csv", "nope.csv"]})
    with pytest.raises(SplitError):
        attach_split(meta, table)


def test_grouped_split_is_by_file_and_keeps_every_class_in_every_split() -> None:
    groups = [Group(f"a{i}", "youtube") for i in range(5)]
    groups += [Group(f"b{i}", "netflix") for i in range(3)]
    assigned = assign_grouped_split(groups, seed=7)
    again = assign_grouped_split(list(reversed(groups)), seed=7)
    assert assigned == again
    assert Counter(split for key, split in assigned.items() if key.startswith("a")) == Counter(
        train=3, validation=1, test=1
    )
    assert {assigned[f"b{i}"] for i in range(3)} == {"train", "validation", "test"}
    assert len(assigned) == len(groups)


def test_large_capture_goes_to_train_and_every_split_keeps_a_file() -> None:
    groups = [
        Group("huge", "voip", weight=1000),
        Group("a", "voip", weight=10),
        Group("b", "voip", weight=10),
        Group("c", "voip", weight=10),
        Group("d", "voip", weight=10),
    ]
    assigned = assign_grouped_split(groups, seed=0)
    assert assigned["huge"] == "train"
    assert set(assigned.values()) == {"train", "validation", "test"}


def test_grouped_split_rejects_a_class_with_too_few_files() -> None:
    groups = [Group("one", "gaming"), Group("two", "gaming")]
    with pytest.raises(SplitError, match="gaming"):
        assign_grouped_split(groups, seed=1)
    with pytest.raises(SplitError, match="Duplicate"):
        assign_grouped_split([Group("same", "youtube"), Group("same", "netflix")], seed=1)
