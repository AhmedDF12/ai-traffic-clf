"""Command line for checks, capture preparation, and baseline training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from traffic_clf.config import early_packet_warning, load_config
from traffic_clf.data.distribution import class_distribution
from traffic_clf.data.labels import (
    load_iscx_rules,
    load_security_map,
    map_security_label,
    normalize_raw_label,
    parse_iscx_capture,
    to_binary_security,
)
from traffic_clf.data.loaders import load_cic_csv
from traffic_clf.errors import ProjectError
from traffic_clf.flows.pipeline import extract_custom_pcap, extract_iscx_pcap
from traffic_clf.preprocess.prepare import prepare_cic_csvs, prepare_custom_pcaps, prepare_iscx_pcaps, write_prepared
from traffic_clf.models.build import MODEL_NAMES
from traffic_clf.seeding import seed_everything

_TASKS = ("service", "application", "security")


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand. Returns 0 on success and 2 on a user-fixable error."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ProjectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def console_main() -> None:
    """Entry point for the ``traffic-clf`` script."""
    sys.exit(main())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="traffic-clf", description="Traffic classification utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Load a task config, seed RNGs, and print the resolved settings.")
    _add_root(check)
    check.add_argument("--task", choices=_TASKS, required=True)
    check.add_argument("--overlay", type=Path)
    check.set_defaults(handler=_check)

    iscx = sub.add_parser("map-iscx", help="Map ISCX capture filenames to application and service labels.")
    _add_root(iscx)
    iscx.add_argument("paths", nargs="+")
    iscx.set_defaults(handler=_map_iscx)

    cic = sub.add_parser("map-cic", help="Map a raw CIC-IDS label onto the security taxonomy.")
    _add_root(cic)
    cic.add_argument("labels", nargs="+")
    cic.add_argument("--dataset", choices=("cic_ids2017", "cic_ids2018"), default="cic_ids2017")
    cic.set_defaults(handler=_map_cic)

    load = sub.add_parser("load-cic", help="Load a CIC flow CSV and print its class distribution.")
    _add_root(load)
    load.add_argument("csv", type=Path)
    load.add_argument("--dataset", choices=("cic_ids2017", "cic_ids2018"), required=True)
    load.add_argument("--max-rows", type=int)
    load.set_defaults(handler=_load_cic)

    smoke = sub.add_parser("track-smoke", help="Write one local MLflow run to prove tracking works.")
    _add_root(smoke)
    smoke.add_argument("--task", choices=_TASKS)
    smoke.add_argument("--overlay", type=Path)
    smoke.set_defaults(handler=_track_smoke)

    extract = sub.add_parser("extract-pcap", help="Turn captures into labeled flows and print a summary.")
    _add_root(extract)
    extract.add_argument("--task", choices=("service", "application"), required=True)
    extract.add_argument("--overlay", type=Path)
    extract.add_argument("--sidecar", type=Path, help="Session log for self-captures. Omit for ISCX filenames.")
    extract.add_argument("pcaps", nargs="+", type=Path)
    extract.set_defaults(handler=_extract_pcap)

    prepare_pcap = sub.add_parser("prepare-pcap", help="Extract captures, split by file, and fit scaling on train.")
    _add_root(prepare_pcap)
    prepare_pcap.add_argument("--task", choices=("service", "application"), required=True)
    prepare_pcap.add_argument("--overlay", type=Path)
    prepare_pcap.add_argument("--sidecar", type=Path)
    prepare_pcap.add_argument("--out", type=Path, required=True)
    prepare_pcap.add_argument("pcaps", nargs="+", type=Path)
    prepare_pcap.set_defaults(handler=_prepare_pcap)

    prepare_cic = sub.add_parser("prepare-cic", help="Apply the CIC day split and fit scaling on the training days.")
    _add_root(prepare_cic)
    prepare_cic.add_argument("--overlay", type=Path)
    prepare_cic.add_argument("--max-rows", type=int)
    prepare_cic.add_argument("--out", type=Path, required=True)
    prepare_cic.add_argument("csvs", nargs="+", type=Path)
    prepare_cic.set_defaults(handler=_prepare_cic)

    train = sub.add_parser("train", help="Fit one model on a prepared directory and score the test split.")
    _add_root(train)
    train.add_argument("--task", choices=_TASKS, required=True)
    train.add_argument("--overlay", type=Path)
    train.add_argument("--data", type=Path, required=True, help="Directory written by prepare-pcap or prepare-cic.")
    train.add_argument("--model", choices=MODEL_NAMES, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.set_defaults(handler=_train)

    tune = sub.add_parser("tune", help="Search hyperparameters on the validation split, then score the test split once.")
    _add_root(tune)
    tune.add_argument("--task", choices=_TASKS, required=True)
    tune.add_argument("--overlay", type=Path)
    tune.add_argument("--data", type=Path, required=True)
    tune.add_argument("--model", choices=MODEL_NAMES, required=True)
    tune.add_argument("--trials", type=int, default=20)
    tune.add_argument("--out", type=Path, required=True)
    tune.set_defaults(handler=_tune)
    return parser


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, help="Project root. Defaults to an upward search from the working directory.")


def _root(args: argparse.Namespace) -> Path | None:
    if args.root is None:
        return None
    return Path(args.root)


def _check(args: argparse.Namespace) -> int:
    config = load_config(root=_root(args), task=args.task, overlay=args.overlay)
    seed_everything(config.seed)
    task = config.task
    if task is None:
        raise RuntimeError("check requires a task.")
    priority = "secondary" if task.secondary else "primary"
    tls = "on" if config.privacy.allow_tls_metadata else "off"
    print(f"project={config.project_name}")
    print(f"task={task.name}")
    print(f"priority={priority}")
    print(f"dataset={task.dataset}")
    print(f"classes={','.join(task.classes)}")
    print(f"seed={config.seed}")
    print(
        "timeouts "
        f"active={config.flow.active_timeout_s:g} "
        f"idle={config.flow.idle_timeout_s:g} "
        f"early_packets={config.flow.early_packets}"
    )
    print(f"guards ports=off sni=off tls={tls} payload=stripped")
    print(f"n_features={len(config.features.flow.model_columns)}")
    print(f"device={config.runtime.device}")
    print(f"mlflow={config.mlflow.tracking_uri}")
    warning = early_packet_warning(config)
    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _map_iscx(args: argparse.Namespace) -> int:
    config = load_config(root=_root(args))
    rules = load_iscx_rules(config.labels.iscx_vpn)
    for raw_path in args.paths:
        label = parse_iscx_capture(raw_path, rules)
        vpn = "true" if label.vpn else "false"
        service = label.service or ""
        excluded = label.excluded_service or ""
        print(
            f"file={label.capture_file} vpn={vpn} application={label.application} "
            f"service={service} excluded={excluded}"
        )
    return 0


def _map_cic(args: argparse.Namespace) -> int:
    config = load_config(root=_root(args))
    table = load_security_map(getattr(config.labels, args.dataset))
    for raw in args.labels:
        mapped = map_security_label(raw, table)
        print(
            f"raw={raw} normalized={normalize_raw_label(raw)} "
            f"label={mapped} binary={to_binary_security(mapped)}"
        )
    return 0


def _load_cic(args: argparse.Namespace) -> int:
    loaded = load_cic_csv(args.csv, dataset=args.dataset, root=_root(args), max_rows=args.max_rows)
    print(loaded.report.summary())
    print(class_distribution(loaded.meta["label"]).to_string(index=False))
    return 0


def _extract_pcap(args: argparse.Namespace) -> int:
    config = load_config(root=_root(args), task=args.task, overlay=args.overlay)
    for path in args.pcaps:
        table = _extract_one(path, args.sidecar, config)
        service = ""
        application = ""
        vpn = ""
        if not table.meta.empty:
            application = str(table.meta["application"].iloc[0])
            raw_service = table.meta["service"].iloc[0]
            service = "" if pd.isna(raw_service) else str(raw_service)
            vpn = "true" if bool(table.meta["vpn"].iloc[0]) else "false"
        print(
            f"file={path.name} flows={len(table.features)} skipped={table.skipped_packets} "
            f"truncated={table.truncated_packets} dropped={table.dropped_flows} "
            f"application={application} service={service} vpn={vpn}"
        )
    return 0


def _prepare_pcap(args: argparse.Namespace) -> int:
    config = load_config(root=_root(args), task=args.task, overlay=args.overlay)
    if args.sidecar is None:
        prepared = prepare_iscx_pcaps(args.pcaps, config)
    else:
        prepared = prepare_custom_pcaps(args.pcaps, args.sidecar, config)
    write_prepared(prepared, args.out)
    print(prepared.report.summary())
    print(f"out={args.out}")
    return 0


def _prepare_cic(args: argparse.Namespace) -> int:
    config = load_config(root=_root(args), task="security", overlay=args.overlay)
    prepared = prepare_cic_csvs(args.csvs, config, max_rows=args.max_rows)
    write_prepared(prepared, args.out)
    print(prepared.report.summary())
    print(f"out={args.out}")
    return 0


def _extract_one(path: Path, sidecar: Path | None, config):
    if sidecar is None:
        return extract_iscx_pcap(path, config)
    from traffic_clf.data.labels import load_custom_spec
    from traffic_clf.data.loaders import load_custom_sidecar

    spec = load_custom_spec(config.labels.custom)
    sessions = load_custom_sidecar(sidecar, spec)
    return extract_custom_pcap(path, config, sessions, spec)


def _train(args: argparse.Namespace) -> int:
    from traffic_clf.training.run import train_prepared

    config = load_config(root=_root(args), task=args.task, overlay=args.overlay)
    result = train_prepared(args.data, config, args.model, out=args.out)
    print(result.summary())
    print(f"out={args.out}")
    return 0


def _tune(args: argparse.Namespace) -> int:
    from traffic_clf.training.tune import tune_prepared

    config = load_config(root=_root(args), task=args.task, overlay=args.overlay)
    result = tune_prepared(args.data, config, args.model, n_trials=args.trials, out=args.out)
    print(result.summary())
    print(f"out={args.out}")
    return 0


def _track_smoke(args: argparse.Namespace) -> int:
    from traffic_clf.tracking import start_run

    config = load_config(root=_root(args), task=args.task, overlay=args.overlay)
    with start_run(config, "smoke", tags={"purpose": "smoke"}) as run:
        print(f"run_id={run.info.run_id}")
        print(f"tracking_uri={config.mlflow.tracking_uri}")
    return 0
