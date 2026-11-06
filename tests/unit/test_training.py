import json
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from traffic_clf.cli import main
from traffic_clf.config import load_config
from traffic_clf.errors import SchemaError, UnmappedLabelError
from traffic_clf.training.run import load_trained_model, train_prepared


def test_tree_learns_the_training_rows_only(repo_root: Path, tmp_path: Path) -> None:
    data = tmp_path / "prepared"
    _write_service(data, test_signals=(100.0, 1.0))
    config = _config(repo_root, tmp_path, "service")
    result = train_prepared(data, config, "decision_tree", out=tmp_path / "run")
    tree = result.model.estimator.tree_
    assert int(tree.n_node_samples[0]) == result.train_rows
    assert result.test.metrics["f1_macro"] < 1.0
    assert result.validation is not None
    assert result.validation.metrics["f1_macro"] == pytest.approx(1.0)
    assert "gaming" in result.classes_absent
    stored = load_trained_model(tmp_path / "run" / "model.joblib")
    assert stored.predict(pd.read_parquet(data / "validation_features.parquet")).tolist() == [
        "streaming",
        "voip",
    ]
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_id"] == result.run_id
    logged = MlflowClient(tracking_uri=config.mlflow.tracking_uri).get_run(result.run_id)
    assert logged.data.params["model"] == "decision_tree"
    assert float(logged.data.metrics["f1_macro_validation"]) == pytest.approx(1.0)


def test_logreg_and_forest_separate_an_easy_split(repo_root: Path, tmp_path: Path) -> None:
    data = tmp_path / "prepared"
    _write_service(data)
    config = _config(repo_root, tmp_path, "service")
    for name in ("logreg", "random_forest"):
        result = train_prepared(data, config, name)
        assert result.test.metrics["accuracy"] == pytest.approx(1.0)
        assert result.test.metrics["roc_auc_macro"] == pytest.approx(1.0)
        assert result.test.metrics["pr_auc_macro"] == pytest.approx(1.0)


def test_training_refuses_identifiers_raw_protocol_and_unknown_labels(repo_root: Path, tmp_path: Path) -> None:
    config = _config(repo_root, tmp_path, "service")
    leaked = tmp_path / "leaked"
    _write_service(leaked)
    features = pd.read_parquet(leaked / "train_features.parquet")
    features["dst_port"] = 443
    features.to_parquet(leaked / "train_features.parquet", index=False)
    with pytest.raises(SchemaError, match="identifier"):
        train_prepared(leaked, config, "logreg")

    raw = tmp_path / "raw-protocol"
    _write_service(raw)
    features = pd.read_parquet(raw / "train_features.parquet")
    features["protocol"] = 6
    features.to_parquet(raw / "train_features.parquet", index=False)
    with pytest.raises(SchemaError, match="protocol"):
        train_prepared(raw, config, "logreg")

    foreign = tmp_path / "foreign"
    _write_service(foreign)
    meta = pd.read_parquet(foreign / "test_meta.parquet")
    meta.loc[0, "label"] = "ddos"
    meta.to_parquet(foreign / "test_meta.parquet", index=False)
    with pytest.raises(UnmappedLabelError, match="ddos"):
        train_prepared(foreign, config, "logreg")


def test_security_false_positive_check_uses_monday_only(repo_root: Path, tmp_path: Path) -> None:
    data = tmp_path / "security"
    _write_security(data)
    config = _config(repo_root, tmp_path, "security")
    result = train_prepared(data, config, "decision_tree", out=tmp_path / "security-run")
    assert result.false_positive_rate == pytest.approx(0.0)
    assert result.fp_check is not None
    assert result.fp_check.rows == 2
    assert "false_positive_rate=0.0000" in result.summary()


def test_train_command_prints_the_held_out_score(repo_root: Path, tmp_path: Path, capsys) -> None:
    data = tmp_path / "prepared"
    _write_service(data)
    overlay = _overlay(tmp_path)
    out = tmp_path / "cli-run"
    assert (
        main(
            [
                "train",
                "--root",
                str(repo_root),
                "--task",
                "service",
                "--overlay",
                str(overlay),
                "--data",
                str(data),
                "--model",
                "decision_tree",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "f1_macro_test=1.0000" in output
    assert "classes_absent=browsing,file_transfer,chat,gaming" in output
    assert (out / "confusion_test.csv").is_file()


def _config(repo_root: Path, tmp_path: Path, task: str):
    return load_config(root=repo_root, task=task, overlay=_overlay(tmp_path))


def _overlay(tmp_path: Path) -> Path:
    overlay = tmp_path / "tracking.yaml"
    overlay.write_text(
        "mlflow:\n"
        f"  tracking_uri: '{(tmp_path / 'mlruns').as_posix()}'\n"
        "  experiment_name: unit-train\n",
        encoding="utf-8",
    )
    return overlay


def _write_service(directory: Path, *, test_signals: tuple[float, float] = (0.0, 1.0)) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_split(directory, "train", [0.0, 0.0, 1.0, 1.0], ["streaming", "streaming", "voip", "voip"])
    _write_split(directory, "validation", [0.0, 1.0], ["streaming", "voip"])
    _write_split(directory, "test", list(test_signals), ["streaming", "voip"])
    _write_split(directory, "fp_check", [], [])


def _write_security(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_split(directory, "train", [0.0, 0.0, 1.0, 1.0], ["benign", "benign", "ddos", "ddos"])
    _write_split(directory, "validation", [0.0, 1.0], ["benign", "ddos"])
    _write_split(directory, "test", [0.0, 1.0], ["benign", "ddos"])
    _write_split(directory, "fp_check", [0.0, 0.0], ["benign", "benign"])


def _write_split(directory: Path, name: str, signals: list[float], labels: list[str]) -> None:
    features = pd.DataFrame(
        {
            "signal": signals,
            "protocol_tcp": [1.0] * len(signals),
            "protocol_udp": [0.0] * len(signals),
            "protocol_other": [0.0] * len(signals),
        }
    )
    meta = pd.DataFrame({"label": labels, "capture_file": [f"{name}.pcap"] * len(labels)})
    features.to_parquet(directory / f"{name}_features.parquet", index=False)
    meta.to_parquet(directory / f"{name}_meta.parquet", index=False)
