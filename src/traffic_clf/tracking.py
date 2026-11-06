"""Local MLflow tracking. Runs stay on disk under the configured ``mlruns`` directory."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from traffic_clf.config import ProjectConfig
from traffic_clf.errors import TrackingError
from traffic_clf.seeding import seed_everything


def configure_tracking(config: ProjectConfig) -> None:
    """Point MLflow at the local SQLite file and select the experiment."""
    mlflow = _mlflow()
    uri = config.mlflow.tracking_uri
    if config.mlflow.database_path is not None:
        config.mlflow.database_path.parent.mkdir(parents=True, exist_ok=True)
    if config.mlflow.artifact_dir is not None:
        config.mlflow.artifact_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient(tracking_uri=uri)
    name = config.mlflow.experiment_name
    existing = client.get_experiment_by_name(name)
    if existing is None:
        client.create_experiment(name, artifact_location=config.mlflow.artifact_location or None)
    elif existing.lifecycle_stage == "deleted":
        raise TrackingError(f"MLflow experiment {name!r} is deleted. Choose another experiment_name.")
    mlflow.set_experiment(name)


@contextmanager
def start_run(
    config: ProjectConfig,
    run_name: str,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Start a run and record the seed, the task, and the leakage guards."""
    mlflow = _mlflow()
    configure_tracking(config)
    seed_everything(config.seed)
    if mlflow.active_run() is not None:
        mlflow.end_run()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(tracking_params(config))
        mlflow.set_tags({"project": config.project_name, **(tags or {})})
        yield run


def tracking_params(config: ProjectConfig) -> dict[str, str]:
    """Parameters stored with every run. Feature names are counted, not listed."""
    params = {
        "seed": str(config.seed),
        "early_packets": str(config.flow.early_packets),
        "active_timeout_s": str(config.flow.active_timeout_s),
        "idle_timeout_s": str(config.flow.idle_timeout_s),
        "use_ports": "false",
        "use_sni": "false",
        "allow_tls_metadata": str(config.privacy.allow_tls_metadata).lower(),
        "strip_payload": "true",
        "device": config.runtime.device,
        "n_features": str(len(config.features.flow.model_columns)),
        "feature_spec_version": str(config.features.flow.version),
    }
    task = config.task
    if task is not None:
        params["task"] = task.name
        params["dataset"] = task.dataset
        params["priority"] = "secondary" if task.secondary else "primary"
        params["classes"] = ",".join(task.classes)
    return params


def _mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:
        raise TrackingError(
            'MLflow is not installed. From the repository run: pip install -e ".[dev]"'
        ) from exc
    return mlflow
