from pathlib import Path

from mlflow.tracking import MlflowClient

from traffic_clf.config import load_config
from traffic_clf.tracking import start_run


def test_local_run_records_the_seed_and_the_port_guard(repo_root: Path, tmp_path: Path) -> None:
    overlay = tmp_path / "tracking.yaml"
    uri = (tmp_path / "mlruns").as_posix()
    overlay.write_text(
        "mlflow:\n"
        f"  tracking_uri: '{uri}'\n"
        "  experiment_name: unit-test\n",
        encoding="utf-8",
    )
    config = load_config(root=repo_root, task="service", overlay=overlay)
    with start_run(config, "unit", tags={"purpose": "test"}) as run:
        run_id = run.info.run_id
    stored = MlflowClient(tracking_uri=config.mlflow.tracking_uri).get_run(run_id)
    assert stored.data.params["seed"] == "42"
    assert stored.data.params["use_ports"] == "false"
    assert stored.data.params["use_sni"] == "false"
    assert stored.data.params["task"] == "service"
    assert stored.data.params["priority"] == "primary"
    assert stored.data.tags["purpose"] == "test"
