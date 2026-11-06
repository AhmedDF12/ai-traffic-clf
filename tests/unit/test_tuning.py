from pathlib import Path

import pandas as pd
import pytest

from traffic_clf.cli import main
from traffic_clf.config import load_config
from traffic_clf.errors import ConfigError, SchemaError
from traffic_clf.training.run import train_prepared
from traffic_clf.training.tune import tune_prepared


def test_search_uses_validation_and_leaves_the_flipped_test_alone(repo_root: Path, tmp_path: Path) -> None:
    data = tmp_path / "prepared"
    _write(data, test_labels=("voip", "streaming"))
    config = _config(repo_root, tmp_path)
    result = tune_prepared(data, config, "decision_tree", n_trials=4, out=tmp_path / "run")
    assert result.validation is not None
    assert result.validation.metrics["f1_macro"] == pytest.approx(1.0)
    assert result.test.metrics["f1_macro"] == pytest.approx(0.0)
    assert max(trial.value for trial in result.trials) == pytest.approx(result.validation.metrics["f1_macro"])
    assert (tmp_path / "run" / "trials.csv").is_file()
    assert result.summary().count("trials=4") == 1


def test_tuning_refuses_an_empty_validation_split(repo_root: Path, tmp_path: Path) -> None:
    data = tmp_path / "prepared"
    _write(data, validation_rows=0)
    config = _config(repo_root, tmp_path)
    with pytest.raises(SchemaError, match="validation"):
        tune_prepared(data, config, "logreg", n_trials=1)


def test_unknown_parameter_is_rejected(repo_root: Path, tmp_path: Path) -> None:
    data = tmp_path / "prepared"
    _write(data)
    config = _config(repo_root, tmp_path)
    with pytest.raises(ConfigError, match="n_estimators"):
        train_prepared(data, config, "logreg", params={"n_estimators": 10})


@pytest.mark.parametrize("model_name", ["xgboost", "lightgbm", "catboost"])
def test_boosting_models_separate_an_easy_split(repo_root: Path, tmp_path: Path, model_name: str) -> None:
    data = tmp_path / model_name
    _write(data)
    config = _config(repo_root, tmp_path)
    result = train_prepared(data, config, model_name)
    assert result.test.metrics["accuracy"] == pytest.approx(1.0)
    assert result.params == ()


def test_xgboost_search_reports_the_validation_score(repo_root: Path, tmp_path: Path) -> None:
    data = tmp_path / "prepared"
    _write(data)
    config = _config(repo_root, tmp_path)
    result = tune_prepared(data, config, "xgboost", n_trials=2, out=tmp_path / "run")
    assert result.validation is not None
    assert result.validation.metrics["f1_macro"] == pytest.approx(max(trial.value for trial in result.trials))
    assert any(key == "learning_rate" for key, _value in result.params)


def test_tune_command_prints_the_validation_score(repo_root: Path, tmp_path: Path, capsys) -> None:
    data = tmp_path / "prepared"
    _write(data)
    overlay = _overlay(tmp_path)
    out = tmp_path / "cli-run"
    assert (
        main(
            [
                "tune",
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
                "--trials",
                "2",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "f1_macro_validation=1.0000" in output
    assert "trials=2" in output


def _config(repo_root: Path, tmp_path: Path):
    return load_config(root=repo_root, task="service", overlay=_overlay(tmp_path))


def _overlay(tmp_path: Path) -> Path:
    overlay = tmp_path / "tracking.yaml"
    overlay.write_text(
        "mlflow:\n"
        f"  tracking_uri: '{(tmp_path / 'mlruns').as_posix()}'\n"
        "  experiment_name: unit-tune\n",
        encoding="utf-8",
    )
    return overlay


def _write(
    directory: Path,
    *,
    test_labels: tuple[str, str] = ("streaming", "voip"),
    validation_rows: int = 4,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _split(directory, "train", [0.0] * 8 + [1.0] * 8, ["streaming"] * 8 + ["voip"] * 8)
    if validation_rows == 0:
        _split(directory, "validation", [], [])
    else:
        _split(directory, "validation", [0.0] * 4 + [1.0] * 4, ["streaming"] * 4 + ["voip"] * 4)
    _split(directory, "test", [0.0] * 4 + [1.0] * 4, [test_labels[0]] * 4 + [test_labels[1]] * 4)
    _split(directory, "fp_check", [], [])


def _split(directory: Path, name: str, signals: list[float], labels: list[str]) -> None:
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
