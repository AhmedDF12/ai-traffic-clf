from pathlib import Path

import pytest

from traffic_clf.config import early_packet_warning, find_repo_root, load_config
from traffic_clf.data.schema import load_feature_spec
from traffic_clf.errors import ConfigError, SchemaError


def test_tasks_load_with_the_agreed_priority_and_guards(repo_root: Path) -> None:
    service = load_config(root=repo_root, task="service")
    application = load_config(root=repo_root, task="application")
    security = load_config(root=repo_root, task="security")

    assert service.task is not None and service.task.secondary is False
    assert application.task is not None and "email" in application.task.classes
    assert "gmail" not in application.task.classes
    assert security.task is not None and security.task.secondary is True
    assert security.task.evaluation_sources == ("cic_ids2018",)
    assert service.features.use_ports is False
    assert service.features.use_sni is False
    assert service.privacy.allow_tls_metadata is False
    assert service.privacy.strip_payload is True
    assert service.seed == 42
    assert service.flow.early_packets == service.features.sequence.packets == 30
    assert "src_port" not in service.features.flow.model_columns
    assert "dst_port" in service.features.flow.identifiers
    assert find_repo_root(repo_root) == repo_root


def test_live_preset_changes_idle_timeout_without_enabling_ports(repo_root: Path) -> None:
    config = load_config(root=repo_root, task="service", overlay=repo_root / "configs" / "presets" / "live.yaml")
    assert config.flow.idle_timeout_s == 15
    assert config.flow.active_timeout_s == 120
    assert config.features.use_ports is False


def test_colab_overlay_selects_cuda_and_keeps_the_seed(repo_root: Path) -> None:
    config = load_config(root=repo_root, task="application", overlay=repo_root / "configs" / "colab.yaml")
    assert config.runtime.device == "cuda"
    assert config.seed == 42
    assert "content" in config.paths.data_root.as_posix()
    assert config.features.use_ports is False


def test_port_and_payload_guards_cannot_be_turned_off(repo_root: Path, tmp_path: Path) -> None:
    ports = tmp_path / "ports.yaml"
    ports.write_text("features:\n  use_ports: true\n", encoding="utf-8")
    payload = tmp_path / "payload.yaml"
    payload.write_text("privacy:\n  strip_payload: false\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="use_ports"):
        load_config(root=repo_root, task="service", overlay=ports)
    with pytest.raises(ConfigError, match="strip_payload"):
        load_config(root=repo_root, task="service", overlay=payload)


def test_early_packet_overlay_is_allowed_and_warned(repo_root: Path, tmp_path: Path) -> None:
    overlay = tmp_path / "n.yaml"
    overlay.write_text("flow:\n  early_packets: 20\n", encoding="utf-8")
    config = load_config(root=repo_root, task="service", overlay=overlay)
    assert config.flow.early_packets == 20
    warning = early_packet_warning(config)
    assert warning is not None and "early_packets override" in warning


def test_unknown_task_and_bad_seed_are_rejected(repo_root: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown task"):
        load_config(root=repo_root, task="malware")
    overlay = tmp_path / "seed.yaml"
    overlay.write_text("seed: -1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="seed"):
        load_config(root=repo_root, overlay=overlay)


def test_feature_spec_rejects_a_port_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "version: 1\n"
        "numeric: [duration, dst_port]\n"
        "categorical: [protocol]\n"
        "identifiers: [flow_id, src_ip, dst_ip, src_port, dst_port, timestamp]\n",
        encoding="utf-8",
    )
    with pytest.raises(SchemaError):
        load_feature_spec(path)
