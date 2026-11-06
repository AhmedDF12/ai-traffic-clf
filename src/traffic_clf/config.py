"""Load the project configuration and reject settings that would leak labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from traffic_clf.data.labels import (
    SECURITY_CLASSES,
    SERVICE_CLASSES,
    load_custom_spec,
    load_iscx_rules,
    load_security_map,
)
from traffic_clf.data.schema import FeatureSpec, SequenceSpec, load_feature_spec, load_sequence_spec
from traffic_clf.errors import ConfigError
from traffic_clf.yaml_util import as_string_tuple, expect_version, read_yaml_mapping, require_keys

ALLOWED_TASKS = ("service", "application", "security")
_TOP_LEVEL = {
    "version",
    "project_name",
    "seed",
    "paths",
    "flow",
    "privacy",
    "features",
    "mlflow",
    "runtime",
    "labels",
    "splits",
}
_REQUIRED_CUSTOM_APPS = frozenset({"zoom", "whatsapp", "telegram", "file_download", "gaming"})


@dataclass(frozen=True)
class PathsConfig:
    data_root: Path
    raw: Path
    interim: Path
    processed: Path
    reports: Path
    mlruns: Path


@dataclass(frozen=True)
class FlowConfig:
    active_timeout_s: float
    idle_timeout_s: float
    early_packets: int


@dataclass(frozen=True)
class PrivacyConfig:
    allow_tls_metadata: bool
    strip_payload: bool


@dataclass(frozen=True)
class FeatureConfig:
    flow_spec_path: Path
    sequence_spec_path: Path
    flow: FeatureSpec
    sequence: SequenceSpec
    use_ports: bool
    use_sni: bool


@dataclass(frozen=True)
class MlflowConfig:
    """Local tracking uses SQLite. MLflow 3 rejects the old ``./mlruns`` folder store."""

    tracking_uri: str
    artifact_location: str
    artifact_dir: Path | None
    database_path: Path | None
    experiment_name: str


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    n_jobs: int


@dataclass(frozen=True)
class LabelPaths:
    iscx_vpn: Path
    cic_ids2017: Path
    cic_ids2018: Path
    custom: Path


@dataclass(frozen=True)
class SplitPaths:
    cic_ids2017: Path


@dataclass(frozen=True)
class TaskConfig:
    name: str
    secondary: bool
    dataset: str
    label_map: Path
    label_field: str
    classes: tuple[str, ...]
    custom_only_classes: tuple[str, ...]
    excluded_from_task: tuple[str, ...]
    extra_sources: tuple[str, ...]
    evaluation_sources: tuple[str, ...]


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    project_name: str
    seed: int
    paths: PathsConfig
    flow: FlowConfig
    privacy: PrivacyConfig
    features: FeatureConfig
    mlflow: MlflowConfig
    runtime: RuntimeConfig
    labels: LabelPaths
    splits: SplitPaths
    task: TaskConfig | None


def find_repo_root(start: Path | None = None) -> Path:
    """Find the directory that contains ``pyproject.toml`` and ``configs/default.yaml``."""
    seeds: list[Path] = []
    if start is not None:
        seeds.append(Path(start).resolve())
    seeds.append(Path.cwd().resolve())
    seeds.append(Path(__file__).resolve())
    checked: set[Path] = set()
    for seed in seeds:
        for candidate in (seed, *seed.parents):
            if candidate in checked:
                continue
            checked.add(candidate)
            if (candidate / "pyproject.toml").is_file() and (candidate / "configs" / "default.yaml").is_file():
                return candidate
    raise ConfigError(
        "Could not find the project root (pyproject.toml and configs/default.yaml). "
        "Run from the repository or pass --root."
    )


def load_config(
    root: Path | None = None,
    task: str | None = None,
    overlay: Path | None = None,
) -> ProjectConfig:
    """Load ``configs/default.yaml``, an optional overlay, and an optional task file."""
    project_root = Path(root).resolve() if root is not None else find_repo_root()
    default_path = project_root / "configs" / "default.yaml"
    if not default_path.is_file():
        raise ConfigError(f"{project_root} is not the project root.")
    data = read_yaml_mapping(default_path)
    if overlay is not None:
        overlay_path = Path(overlay).resolve()
        data = deep_merge(data, read_yaml_mapping(overlay_path))
    require_keys(data, required=_TOP_LEVEL, optional=set(), source="config")
    expect_version(data, source="config")

    config = ProjectConfig(
        root=project_root,
        project_name=_text(data["project_name"], "project_name"),
        seed=_nonnegative_int(data["seed"], "seed"),
        paths=_paths(data["paths"], project_root),
        flow=_flow(data["flow"]),
        privacy=_privacy(data["privacy"]),
        features=_features(data["features"], project_root),
        mlflow=_mlflow(data["mlflow"], project_root),
        runtime=_runtime(data["runtime"]),
        labels=_label_paths(data["labels"], project_root),
        splits=_split_paths(data["splits"], project_root),
        task=None,
    )
    if task is None:
        return config
    if task not in ALLOWED_TASKS:
        raise ConfigError(f"Unknown task {task!r}. Choose one of {', '.join(ALLOWED_TASKS)}.")
    task_path = project_root / "configs" / "tasks" / f"{task}.yaml"
    task_config = _load_task(task_path, expected_name=task, root=project_root)
    _validate_task(task_config, config.labels)
    return ProjectConfig(
        root=config.root,
        project_name=config.project_name,
        seed=config.seed,
        paths=config.paths,
        flow=config.flow,
        privacy=config.privacy,
        features=config.features,
        mlflow=config.mlflow,
        runtime=config.runtime,
        labels=config.labels,
        splits=config.splits,
        task=task_config,
    )


def early_packet_warning(config: ProjectConfig) -> str | None:
    """Warn when an overlay changes N without editing the checked-in sequence contract."""
    running = config.flow.early_packets
    contract = config.features.sequence.packets
    if running == contract:
        return None
    return (
        "early_packets override differs from configs/features/sequence_v1.yaml "
        f"({running} vs {contract}). Training must use flow.early_packets."
    )


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _load_task(path: Path, *, expected_name: str, root: Path) -> TaskConfig:
    data = read_yaml_mapping(path)
    source = str(path)
    expect_version(data, source=source)
    require_keys(
        data,
        required={
            "version",
            "name",
            "secondary",
            "dataset",
            "label_map",
            "label_field",
            "classes",
            "custom_only_classes",
            "excluded_from_task",
            "extra_sources",
            "evaluation_sources",
        },
        optional=set(),
        source=source,
    )
    name = _text(data["name"], "name")
    if name != expected_name:
        raise ConfigError(f"{source} name is {name!r}; the filename requires {expected_name!r}.")
    if not isinstance(data["secondary"], bool):
        raise ConfigError(f"{source} secondary must be true or false.")
    return TaskConfig(
        name=name,
        secondary=data["secondary"],
        dataset=_text(data["dataset"], "dataset"),
        label_map=_existing_file(_resolve(root, data["label_map"], "label_map"), "label_map"),
        label_field=_text(data["label_field"], "label_field"),
        classes=as_string_tuple(data["classes"], name=f"{source} classes"),
        custom_only_classes=as_string_tuple(
            data["custom_only_classes"],
            name=f"{source} custom_only_classes",
            allow_empty=True,
        ),
        excluded_from_task=as_string_tuple(
            data["excluded_from_task"],
            name=f"{source} excluded_from_task",
            allow_empty=True,
        ),
        extra_sources=as_string_tuple(
            data["extra_sources"],
            name=f"{source} extra_sources",
            allow_empty=True,
        ),
        evaluation_sources=as_string_tuple(
            data["evaluation_sources"],
            name=f"{source} evaluation_sources",
            allow_empty=True,
        ),
    )


def _validate_task(task: TaskConfig, labels: LabelPaths) -> None:
    if task.label_field != task.name:
        raise ConfigError(f"Task {task.name} must use label_field {task.name!r}.")
    if not set(task.custom_only_classes) <= set(task.classes):
        raise ConfigError(f"{task.name} lists a custom-only class that is not in classes.")
    load_custom_spec(labels.custom)
    if task.name == "service":
        _validate_primary(task, labels)
        if task.classes != SERVICE_CLASSES:
            raise ConfigError("Service classes must stay streaming, voip, browsing, file_transfer, chat, gaming.")
        rules = load_iscx_rules(task.label_map)
        emitted = {rule.service for rule in rules.rules if rule.service is not None}
        excluded = {rule.exclude for rule in rules.rules if rule.exclude is not None}
        if not emitted <= set(task.classes):
            raise ConfigError("ISCX rules emit a service that is not in the service task.")
        if excluded != set(task.excluded_from_task):
            raise ConfigError(
                "Service exclusions must match the ISCX rules. "
                f"Rules exclude {sorted(excluded)}; the task excludes {list(task.excluded_from_task)}."
            )
        if set(task.custom_only_classes) & emitted:
            raise ConfigError("A service emitted by an ISCX rule cannot be custom-only.")
        if "gaming" not in task.custom_only_classes:
            raise ConfigError("gaming is custom-only. ISCX VPN-nonVPN has no gaming captures.")
        return
    if task.name == "application":
        _validate_primary(task, labels)
        rules = load_iscx_rules(task.label_map)
        emitted_apps = {rule.application for rule in rules.rules}
        missing = emitted_apps - set(task.classes)
        if missing:
            raise ConfigError(
                "ISCX rules emit applications missing from the application task: "
                + ", ".join(sorted(missing))
            )
        if set(task.custom_only_classes) & emitted_apps:
            raise ConfigError("An application emitted by an ISCX rule cannot be custom-only.")
        if not _REQUIRED_CUSTOM_APPS <= set(task.custom_only_classes):
            raise ConfigError(
                "The application task must mark zoom, whatsapp, telegram, file_download, and gaming as custom-only."
            )
        return
    _validate_security(task, labels)


def _validate_primary(task: TaskConfig, labels: LabelPaths) -> None:
    if task.secondary:
        raise ConfigError("Application and service are the primary tasks.")
    if task.dataset != "iscx_vpn":
        raise ConfigError("Application and service train on iscx_vpn.")
    if task.label_map != labels.iscx_vpn:
        raise ConfigError("Application and service must use configs/labels/iscx_vpn.yaml.")
    if task.extra_sources != ("custom",):
        raise ConfigError("Application and service accept only custom captures as an extra source.")
    if task.evaluation_sources:
        raise ConfigError("Application and service have no second public corpus in this milestone.")


def _validate_security(task: TaskConfig, labels: LabelPaths) -> None:
    if not task.secondary:
        raise ConfigError("Security stays the secondary task.")
    if task.dataset != "cic_ids2017":
        raise ConfigError("The security task trains on cic_ids2017.")
    if task.label_map != labels.cic_ids2017:
        raise ConfigError("The security task must use the CIC-IDS2017 label map.")
    if task.extra_sources:
        raise ConfigError("The security task does not mix in custom captures.")
    if task.evaluation_sources != ("cic_ids2018",):
        raise ConfigError("CSE-CIC-IDS2018 is the security evaluation source.")
    if task.custom_only_classes or task.excluded_from_task:
        raise ConfigError("The security task does not use custom-only classes or ISCX exclusions.")
    if task.classes != SECURITY_CLASSES:
        raise ConfigError(
            "Security classes must stay benign, ddos, brute_force, port_scan, botnet, malware."
        )
    table = load_security_map(task.label_map)
    missing = [name for name in task.classes if name not in table.classes]
    if missing:
        raise ConfigError("CIC-IDS2017 label map does not emit: " + ", ".join(missing))
    evaluation = load_security_map(labels.cic_ids2018)
    if "port_scan" not in evaluation.absent_classes:
        raise ConfigError("CIC-IDS2018 has no dedicated port-scan class. Mark port_scan absent.")


def _paths(value: object, root: Path) -> PathsConfig:
    data = _section(value, "paths")
    require_keys(
        data,
        required={"data_root", "raw", "interim", "processed", "reports", "mlruns"},
        optional=set(),
        source="paths",
    )
    return PathsConfig(
        data_root=_resolve(root, data["data_root"], "paths.data_root"),
        raw=_resolve(root, data["raw"], "paths.raw"),
        interim=_resolve(root, data["interim"], "paths.interim"),
        processed=_resolve(root, data["processed"], "paths.processed"),
        reports=_resolve(root, data["reports"], "paths.reports"),
        mlruns=_resolve(root, data["mlruns"], "paths.mlruns"),
    )


def _flow(value: object) -> FlowConfig:
    data = _section(value, "flow")
    require_keys(
        data,
        required={"active_timeout_s", "idle_timeout_s", "early_packets"},
        optional=set(),
        source="flow",
    )
    active = _positive_number(data["active_timeout_s"], "flow.active_timeout_s")
    idle = _positive_number(data["idle_timeout_s"], "flow.idle_timeout_s")
    if idle > active:
        raise ConfigError("flow.idle_timeout_s cannot be longer than flow.active_timeout_s.")
    early = data["early_packets"]
    if isinstance(early, bool) or not isinstance(early, int) or not 1 <= early <= 100:
        raise ConfigError("flow.early_packets must be an integer from 1 to 100.")
    return FlowConfig(active_timeout_s=active, idle_timeout_s=idle, early_packets=early)


def _privacy(value: object) -> PrivacyConfig:
    data = _section(value, "privacy")
    require_keys(
        data,
        required={"allow_tls_metadata", "strip_payload"},
        optional=set(),
        source="privacy",
    )
    if data["strip_payload"] is not True:
        raise ConfigError("privacy.strip_payload must stay true. Payloads are not stored.")
    if not isinstance(data["allow_tls_metadata"], bool):
        raise ConfigError("privacy.allow_tls_metadata must be true or false.")
    return PrivacyConfig(
        allow_tls_metadata=data["allow_tls_metadata"],
        strip_payload=True,
    )


def _features(value: object, root: Path) -> FeatureConfig:
    data = _section(value, "features")
    require_keys(
        data,
        required={"flow_spec", "sequence_spec", "use_ports", "use_sni"},
        optional=set(),
        source="features",
    )
    if data["use_ports"] is not False:
        raise ConfigError("features.use_ports must stay false. Ports are not model inputs.")
    if data["use_sni"] is not False:
        raise ConfigError("features.use_sni must stay false. SNI is a label leak for application class.")
    flow_path = _existing_file(_resolve(root, data["flow_spec"], "features.flow_spec"), "features.flow_spec")
    sequence_path = _existing_file(
        _resolve(root, data["sequence_spec"], "features.sequence_spec"),
        "features.sequence_spec",
    )
    return FeatureConfig(
        flow_spec_path=flow_path,
        sequence_spec_path=sequence_path,
        flow=load_feature_spec(flow_path),
        sequence=load_sequence_spec(sequence_path),
        use_ports=False,
        use_sni=False,
    )


def _mlflow(value: object, root: Path) -> MlflowConfig:
    data = _section(value, "mlflow")
    require_keys(data, required={"tracking_uri", "experiment_name"}, optional=set(), source="mlflow")
    tracking_uri, artifact_location, artifact_dir, database_path = _tracking_target(
        _text(data["tracking_uri"], "mlflow.tracking_uri"),
        root,
    )
    return MlflowConfig(
        tracking_uri=tracking_uri,
        artifact_location=artifact_location,
        artifact_dir=artifact_dir,
        database_path=database_path,
        experiment_name=_text(data["experiment_name"], "mlflow.experiment_name"),
    )


def _tracking_target(raw: str, root: Path) -> tuple[str, str, Path | None, Path | None]:
    """Return a tracking URI, an artifact URI, the artifact directory, and the database file."""
    if raw.startswith(("http://", "https://")):
        return raw, "", None, None
    database = _database_path(raw, root)
    artifact_dir = database.parent / "artifacts"
    return _sqlite_uri(database), artifact_dir.as_uri(), artifact_dir, database


def _database_path(raw: str, root: Path) -> Path:
    if raw.startswith("sqlite:"):
        if not raw.startswith("sqlite:///"):
            raise ConfigError("mlflow.tracking_uri sqlite paths must start with sqlite:///")
        location = raw[len("sqlite:///") :]
        path = Path(location)
        if not _is_absolute_location(location):
            path = root / location
    else:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        if path.suffix != ".db":
            path = path / "mlflow.db"
    return path.resolve()


def _is_absolute_location(location: str) -> bool:
    path = Path(location)
    if path.is_absolute():
        return True
    return len(location) >= 2 and location[1] == ":"


def _sqlite_uri(path: Path) -> str:
    return "sqlite:///" + quote(path.as_posix(), safe="/:")


def _runtime(value: object) -> RuntimeConfig:
    data = _section(value, "runtime")
    require_keys(data, required={"device", "n_jobs"}, optional=set(), source="runtime")
    device = _text(data["device"], "runtime.device")
    if device not in {"cpu", "cuda"}:
        raise ConfigError("runtime.device must be cpu or cuda.")
    return RuntimeConfig(device=device, n_jobs=_positive_int(data["n_jobs"], "runtime.n_jobs"))


def _label_paths(value: object, root: Path) -> LabelPaths:
    data = _section(value, "labels")
    require_keys(
        data,
        required={"iscx_vpn", "cic_ids2017", "cic_ids2018", "custom"},
        optional=set(),
        source="labels",
    )
    return LabelPaths(
        iscx_vpn=_existing_file(_resolve(root, data["iscx_vpn"], "labels.iscx_vpn"), "labels.iscx_vpn"),
        cic_ids2017=_existing_file(
            _resolve(root, data["cic_ids2017"], "labels.cic_ids2017"),
            "labels.cic_ids2017",
        ),
        cic_ids2018=_existing_file(
            _resolve(root, data["cic_ids2018"], "labels.cic_ids2018"),
            "labels.cic_ids2018",
        ),
        custom=_existing_file(_resolve(root, data["custom"], "labels.custom"), "labels.custom"),
    )


def _split_paths(value: object, root: Path) -> SplitPaths:
    data = _section(value, "splits")
    require_keys(data, required={"cic_ids2017"}, optional=set(), source="splits")
    return SplitPaths(
        cic_ids2017=_existing_file(
            _resolve(root, data["cic_ids2017"], "splits.cic_ids2017"),
            "splits.cic_ids2017",
        )
    )


def _section(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping.")
    return value


def _resolve(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a path.")
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _existing_file(path: Path, name: str) -> Path:
    if not path.is_file():
        raise ConfigError(f"{name} does not exist: {path}")
    return path


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string.")
    return value.strip()


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer.")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive integer.")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{name} must be a positive number.")
    return float(value)
