"""Label maps for ISCX VPN-nonVPN, CIC-IDS, and self-captured sessions.

Application and service labels come from the capture filename for ISCX,
because that dataset is one application per file. Security labels come from
the CIC flow CSV. A row is never given both a YouTube label and a DDoS label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from traffic_clf.errors import LabelMapError, UnmappedLabelError
from traffic_clf.yaml_util import as_string_tuple, expect_version, read_yaml_mapping, require_keys

SECURITY_CLASSES = ("benign", "ddos", "brute_force", "port_scan", "botnet", "malware")
SERVICE_CLASSES = ("streaming", "voip", "browsing", "file_transfer", "chat", "gaming")

_DASHES = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\ufffd": "-",
        "\u0096": "-",
    }
)


def normalize_raw_label(value: object) -> str:
    """Make CIC label spelling comparable across the 2017 and 2018 files.

    Unicode dashes become ASCII hyphens. A trailing ``- Attempted`` is removed
    so an unsuccessful attack uses the same class as the successful one.
    Hyphens inside a tool name, such as ``FTP-Patator``, are left in place.
    """
    text = str(value).strip().replace("\ufeff", "").translate(_DASHES)
    text = re.sub(r"\s+", " ", text).casefold()
    text = re.sub(r"\s*-\s*attempted$", "", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class SecurityLabelMap:
    version: int
    dataset: str
    lookup: dict[str, str]
    absent_classes: tuple[str, ...]

    @property
    def classes(self) -> frozenset[str]:
        return frozenset(self.lookup.values())


@dataclass(frozen=True)
class IscxRule:
    id: str
    pattern: re.Pattern[str]
    application: str
    service: str | None
    exclude: str | None


@dataclass(frozen=True)
class IscxRuleSet:
    version: int
    dataset: str
    rules: tuple[IscxRule, ...]


@dataclass(frozen=True)
class CaptureLabel:
    """Label read from an ISCX capture path. The file does not have to exist."""

    capture_id: str
    capture_file: str
    vpn: bool
    application: str
    service: str | None
    excluded_service: str | None


@dataclass(frozen=True)
class CustomLabelSpec:
    services: tuple[str, ...]
    applications: dict[str, frozenset[str]]


def load_security_map(path: Path) -> SecurityLabelMap:
    data = read_yaml_mapping(path)
    source = str(path)
    expect_version(data, source=source)
    require_keys(
        data,
        required={"version", "dataset", "map"},
        optional={"absent_classes"},
        source=source,
    )
    if data["dataset"] not in {"cic_ids2017", "cic_ids2018"}:
        raise LabelMapError(f"{source} dataset must be cic_ids2017 or cic_ids2018.")
    raw_map = data["map"]
    if not isinstance(raw_map, dict):
        raise LabelMapError(f"{source} map must be a mapping of project class to raw labels.")
    lookup: dict[str, str] = {}
    for project_label, raw_labels in raw_map.items():
        if project_label not in SECURITY_CLASSES:
            raise LabelMapError(f"{source} uses unknown security class {project_label!r}.")
        if not isinstance(raw_labels, list) or not raw_labels:
            raise LabelMapError(f"{source} class {project_label} needs a list of raw labels.")
        for raw in raw_labels:
            if not isinstance(raw, str) or not raw.strip():
                raise LabelMapError(f"{source} class {project_label} has an empty raw label.")
            key = normalize_raw_label(raw)
            current = lookup.get(key)
            if current is not None and current != project_label:
                raise LabelMapError(
                    f"{source} raw label {raw!r} maps to both {current!r} and {project_label!r}."
                )
            lookup[key] = project_label
    absent = as_string_tuple(
        data.get("absent_classes", []),
        name=f"{source} absent_classes",
        allow_empty=True,
    )
    emitted = set(lookup.values())
    for name in absent:
        if name not in SECURITY_CLASSES:
            raise LabelMapError(f"{source} absent class {name!r} is not a security class.")
        if name in emitted:
            raise LabelMapError(f"{source} lists {name} as absent but the map also emits it.")
    return SecurityLabelMap(
        version=1,
        dataset=str(data["dataset"]),
        lookup=lookup,
        absent_classes=absent,
    )


def map_security_label(raw: object, table: SecurityLabelMap) -> str:
    """Map one raw CIC label. Raises when the label is not in the map."""
    key = normalize_raw_label(raw)
    try:
        return table.lookup[key]
    except KeyError as exc:
        raise UnmappedLabelError(
            f"Unmapped {table.dataset} label {raw!r}. Add it to the label map before training."
        ) from exc


def to_binary_security(label: str) -> str:
    """Collapse the six-way security taxonomy to benign or malicious."""
    if label == "benign":
        return "benign"
    if label in SECURITY_CLASSES:
        return "malicious"
    raise UnmappedLabelError(f"Unknown security label {label!r}.")


def load_iscx_rules(path: Path) -> IscxRuleSet:
    data = read_yaml_mapping(path)
    source = str(path)
    expect_version(data, source=source)
    require_keys(data, required={"version", "dataset", "rules"}, optional=set(), source=source)
    if data["dataset"] != "iscx_vpn":
        raise LabelMapError(f"{source} dataset must be iscx_vpn.")
    raw_rules = data["rules"]
    if not isinstance(raw_rules, list) or not raw_rules:
        raise LabelMapError(f"{source} needs a non-empty rules list.")
    rules: list[IscxRule] = []
    seen: set[str] = set()
    for entry in raw_rules:
        rules.append(_parse_iscx_rule(entry, source=source, seen=seen))
    return IscxRuleSet(version=1, dataset="iscx_vpn", rules=tuple(rules))


def parse_iscx_capture(path: str | Path, rules: IscxRuleSet) -> CaptureLabel:
    """Map a capture path to an application and, when allowed, a service.

    The first matching rule wins. ``vpn_youtube_A.pcap`` and a file inside a
    ``VPN-PCAPs`` directory are both treated as VPN. A ``NonVPN`` directory is not.
    """
    capture = Path(path)
    stem = _rule_stem(capture)
    for rule in rules.rules:
        if rule.pattern.search(stem):
            return CaptureLabel(
                capture_id=capture.as_posix(),
                capture_file=capture.name,
                vpn=_path_is_vpn(capture),
                application=rule.application,
                service=rule.service,
                excluded_service=rule.exclude,
            )
    raise UnmappedLabelError(
        f"No ISCX rule matched {capture.name!r} (stem {stem!r}). "
        "Add a rule, or keep this file out of the labeled set."
    )


def load_custom_spec(path: Path) -> CustomLabelSpec:
    data = read_yaml_mapping(path)
    source = str(path)
    expect_version(data, source=source)
    require_keys(
        data,
        required={"version", "services", "applications"},
        optional=set(),
        source=source,
    )
    services = as_string_tuple(data["services"], name=f"{source} services")
    if services != SERVICE_CLASSES:
        raise LabelMapError(f"{source} services must be exactly {', '.join(SERVICE_CLASSES)}.")
    raw_apps = data["applications"]
    if not isinstance(raw_apps, dict) or not raw_apps:
        raise LabelMapError(f"{source} applications must be a non-empty mapping.")
    applications: dict[str, frozenset[str]] = {}
    for name, allowed in raw_apps.items():
        if not isinstance(name, str) or name != name.casefold() or not name.strip():
            raise LabelMapError(f"{source} application names must be lowercase.")
        allowed_services = as_string_tuple(allowed, name=f"{source} application {name}")
        unknown = set(allowed_services) - set(services)
        if unknown:
            raise LabelMapError(
                f"{source} application {name} uses unknown service(s): {', '.join(sorted(unknown))}."
            )
        applications[name] = frozenset(allowed_services)
    return CustomLabelSpec(services=services, applications=applications)


def _parse_iscx_rule(entry: object, *, source: str, seen: set[str]) -> IscxRule:
    if not isinstance(entry, dict):
        raise LabelMapError(f"{source} rules must be mappings.")
    require_keys(
        entry,
        required={"id", "pattern", "application", "service"},
        optional={"exclude"},
        source=f"{source} rule",
    )
    rule_id = entry["id"]
    pattern_text = entry["pattern"]
    application = entry["application"]
    if not isinstance(rule_id, str) or not rule_id:
        raise LabelMapError(f"{source} has a rule with an empty id.")
    if rule_id in seen:
        raise LabelMapError(f"{source} repeats rule id {rule_id!r}.")
    seen.add(rule_id)
    if not isinstance(pattern_text, str) or not pattern_text:
        raise LabelMapError(f"{source} rule {rule_id} needs a pattern.")
    if pattern_text != pattern_text.casefold():
        raise LabelMapError(f"{source} rule {rule_id} pattern must be lowercase.")
    if not isinstance(application, str) or application != application.casefold():
        raise LabelMapError(f"{source} rule {rule_id} application must be a lowercase string.")
    service = entry["service"]
    exclude = entry.get("exclude")
    if service is not None and (not isinstance(service, str) or service not in SERVICE_CLASSES):
        raise LabelMapError(f"{source} rule {rule_id} has a service outside the six-class list.")
    if exclude is not None and exclude not in {"email", "p2p"}:
        raise LabelMapError(f"{source} rule {rule_id} exclude must be email or p2p.")
    if service is None and exclude is None:
        raise LabelMapError(f"{source} rule {rule_id} must set a service or an exclusion.")
    if service is not None and exclude is not None:
        raise LabelMapError(f"{source} rule {rule_id} cannot set both a service and an exclusion.")
    try:
        pattern = re.compile(pattern_text)
    except re.error as exc:
        raise LabelMapError(f"{source} rule {rule_id} has an invalid pattern: {exc}") from exc
    return IscxRule(
        id=rule_id,
        pattern=pattern,
        application=application,
        service=service,
        exclude=exclude,
    )


def _rule_stem(path: Path) -> str:
    stem = path.stem.casefold()
    for prefix in ("vpn_", "vpn-"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def _path_is_vpn(path: Path) -> bool:
    stem = path.stem.casefold()
    if stem.startswith(("vpn_", "vpn-")):
        return True
    for part in path.parts[:-1]:
        compact = re.sub(r"[\s_\-]", "", part.casefold())
        if "nonvpn" in compact:
            continue
        if compact.startswith("vpn"):
            return True
    return False
