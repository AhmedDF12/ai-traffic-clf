"""Split assignments that follow capture files, not individual flow rows.

A random row split puts pieces of the same capture on both sides of the
evaluation and inflates the score. Callers either use the CIC day manifest
or :func:`assign_grouped_split`, then copy that one decision onto every flow
from the capture.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from traffic_clf.errors import SplitError
from traffic_clf.yaml_util import expect_version, read_yaml_mapping, require_keys

ALLOWED_SPLITS = ("train", "validation", "test", "fp_check")


@dataclass(frozen=True)
class Group:
    """One capture file and the class used to stratify the file split.

    ``weight`` is the number of flows in the file. Equal weights keep the
    original file-count split. Unequal weights place large files in the split
    that still has room, so one capture cannot fill validation by itself.
    """

    group_id: str
    stratum: str
    weight: int = 1


@dataclass(frozen=True)
class NamedSplit:
    dataset: str
    unit: str
    assignments: dict[str, str]
    aliases: dict[str, str]


def load_named_split(path: Path) -> NamedSplit:
    data = read_yaml_mapping(path)
    source = str(path)
    expect_version(data, source=source)
    require_keys(
        data,
        required={"version", "dataset", "unit", "assignments"},
        optional={"aliases"},
        source=source,
    )
    if data["unit"] != "capture_file":
        raise SplitError(f"{source} only supports unit capture_file.")
    raw_assignments = data["assignments"]
    if not isinstance(raw_assignments, dict) or not raw_assignments:
        raise SplitError(f"{source} assignments must be a non-empty mapping.")
    assignments: dict[str, str] = {}
    for filename, split in raw_assignments.items():
        if not isinstance(filename, str) or not filename:
            raise SplitError(f"{source} has an empty capture filename.")
        if split not in ALLOWED_SPLITS:
            raise SplitError(f"{source} split for {filename} must be one of {ALLOWED_SPLITS}.")
        assignments[filename] = str(split)
    raw_aliases = data.get("aliases", {})
    if not isinstance(raw_aliases, dict):
        raise SplitError(f"{source} aliases must be a mapping.")
    aliases: dict[str, str] = {}
    for alias, target in raw_aliases.items():
        if alias in assignments:
            raise SplitError(f"{source} alias {alias} already has its own assignment.")
        if target not in assignments:
            raise SplitError(f"{source} alias {alias} points at unknown file {target}.")
        aliases[str(alias)] = str(target)
    return NamedSplit(
        dataset=str(data["dataset"]),
        unit="capture_file",
        assignments=assignments,
        aliases=aliases,
    )


def resolve_named_split(table: NamedSplit, capture_file: object) -> str:
    """Return the split for one official CIC CSV name."""
    name = Path(str(capture_file)).name
    if name in table.assignments:
        return table.assignments[name]
    target = table.aliases.get(name)
    if target is not None:
        return table.assignments[target]
    raise SplitError(
        f"No {table.dataset} split for {name!r}. "
        "Add the filename to the split manifest before using the file."
    )


def attach_split(
    meta: pd.DataFrame,
    table: NamedSplit,
    *,
    column: str = "capture_file",
) -> pd.DataFrame:
    """Copy a file-level split onto every row. Unknown files raise."""
    if column not in meta.columns:
        raise SplitError(f"Metadata is missing column {column!r}.")
    assigned = meta.copy()
    assigned["split"] = [resolve_named_split(table, value) for value in meta[column]]
    return assigned


def assign_grouped_split(
    groups: list[Group] | tuple[Group, ...],
    *,
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> dict[str, str]:
    """Assign each capture group to one split.

    Stratification is by ``Group.stratum``. A class with fewer than three
    capture files cannot appear in train, validation, and test, so it raises
    instead of disappearing from the test score.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SplitError("seed must be a non-negative integer.")
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0):
        raise SplitError("train, validation, and test ratios must be positive and sum to 1.")
    if not groups:
        raise SplitError("No capture groups were provided.")

    by_stratum: dict[str, list[str]] = defaultdict(list)
    weights: dict[str, int] = {}
    seen: set[str] = set()
    for group in groups:
        if not group.group_id or not group.stratum:
            raise SplitError("Every group needs a group id and a stratum.")
        if isinstance(group.weight, bool) or not isinstance(group.weight, int) or group.weight < 1:
            raise SplitError(f"Capture {group.group_id!r} needs a positive flow weight.")
        if group.group_id in seen:
            raise SplitError(f"Duplicate capture group {group.group_id!r}.")
        seen.add(group.group_id)
        by_stratum[group.stratum].append(group.group_id)
        weights[group.group_id] = group.weight

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    balance_by_weight = any(weight != 1 for weight in weights.values())
    for stratum in sorted(by_stratum):
        ids = sorted(by_stratum[stratum])
        if len(ids) < 3:
            raise SplitError(
                f"Stratum {stratum!r} has {len(ids)} capture group(s). "
                "A file-level train/validation/test split needs at least 3 files "
                "so the class is present in every split. Collect another capture "
                "or move this class to the qualitative set."
            )
        if balance_by_weight:
            assignment.update(_assign_by_weight([(group_id, weights[group_id]) for group_id in ids], ratios, rng))
            continue
        rng.shuffle(ids)
        n_train, n_val, _n_test = _split_counts(len(ids), ratios)
        for group_id in ids[:n_train]:
            assignment[group_id] = "train"
        for group_id in ids[n_train : n_train + n_val]:
            assignment[group_id] = "validation"
        for group_id in ids[n_train + n_val :]:
            assignment[group_id] = "test"
    return assignment


def _assign_by_weight(
    items: list[tuple[str, int]],
    ratios: tuple[float, float, float],
    rng: random.Random,
) -> dict[str, str]:
    """Give each split at least one file and keep flow mass near 60/20/20."""
    ordered = list(items)
    rng.shuffle(ordered)
    ordered.sort(key=lambda item: item[1], reverse=True)
    total = sum(weight for _, weight in ordered)
    targets = [total * ratio for ratio in ratios]
    names = ("train", "validation", "test")
    loads = [0, 0, 0]
    counts = [0, 0, 0]
    assignment: dict[str, str] = {}
    for index, (group_id, weight) in enumerate(ordered):
        remaining = len(ordered) - index
        empty = [slot for slot, count in enumerate(counts) if count == 0]
        if empty and remaining <= len(empty):
            choice = empty[0]
        else:
            candidates = [slot for slot in range(3) if counts[slot] > 0 or remaining > len(empty)]
            choice = max(candidates, key=lambda slot: (targets[slot] - loads[slot], -counts[slot]))
        assignment[group_id] = names[choice]
        loads[choice] += weight
        counts[choice] += 1
    if min(counts) < 1:
        raise SplitError("Weight balancing left a split empty.")
    return assignment


def _split_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if n < 3:
        raise SplitError(f"Need at least 3 capture groups, found {n}.")
    raw = [n * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    for index, count in enumerate(counts):
        if count < 1:
            counts[index] = 1
    while sum(counts) > n:
        donors = [index for index, count in enumerate(counts) if count > 1]
        if not donors:
            raise SplitError("Ratios cannot fill train, validation, and test.")
        donor = max(donors, key=lambda index: counts[index] - raw[index])
        counts[donor] -= 1
    leftover = n - sum(counts)
    priority = (0, 1, 2)
    order = sorted(
        range(3),
        key=lambda index: (raw[index] - math.floor(raw[index]), priority[index]),
        reverse=True,
    )
    for index in order:
        if leftover == 0:
            break
        counts[index] += 1
        leftover -= 1
    if leftover or sum(counts) != n or min(counts) < 1:
        raise SplitError("Could not assign every group while keeping all three splits non-empty.")
    return counts[0], counts[1], counts[2]
