"""YAML reads shared by config and label maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from traffic_clf.errors import ConfigError


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML file that must contain a mapping."""
    if not path.is_file():
        raise ConfigError(f"Missing file: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a YAML mapping.")
    return loaded


def require_keys(
    data: dict[str, Any],
    *,
    required: set[str],
    optional: set[str],
    source: str,
) -> None:
    """Reject missing and unknown keys. ``notes`` is always optional."""
    allowed = required | optional | {"notes"}
    missing = required - set(data)
    unknown = set(data) - allowed
    if missing or unknown:
        parts: list[str] = []
        if missing:
            parts.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            parts.append("unknown " + ", ".join(sorted(unknown)))
        raise ConfigError(f"{source}: {'; '.join(parts)}.")


def expect_version(data: dict[str, Any], *, source: str) -> None:
    if data.get("version") != 1:
        raise ConfigError(f"{source} must set version: 1.")


def as_string_tuple(
    value: object,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ConfigError(f"{name} must be a list of non-empty strings.")
    cleaned = tuple(item.strip() for item in value)
    if not cleaned and not allow_empty:
        raise ConfigError(f"{name} must not be empty.")
    if len(set(cleaned)) != len(cleaned):
        raise ConfigError(f"{name} contains duplicate entries.")
    return cleaned
