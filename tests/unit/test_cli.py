from pathlib import Path

from traffic_clf.cli import main


def test_check_prints_the_primary_and_secondary_tasks(repo_root: Path, capsys) -> None:
    assert main(["check", "--task", "service", "--root", str(repo_root)]) == 0
    service = capsys.readouterr().out
    assert "priority=primary" in service
    assert "guards ports=off sni=off tls=off payload=stripped" in service

    assert main(["check", "--task", "security", "--root", str(repo_root)]) == 0
    assert "priority=secondary" in capsys.readouterr().out


def test_map_commands(repo_root: Path, capsys) -> None:
    assert main(["map-iscx", "--root", str(repo_root), "vpn_youtube_A.pcap", "email2b.pcap"]) == 0
    mapped = capsys.readouterr().out
    assert "application=youtube" in mapped
    assert "service=streaming" in mapped
    assert "application=email" in mapped
    assert "excluded=email" in mapped
    assert main(["map-iscx", "--root", str(repo_root), "not_a_capture.pcap"]) == 2

    assert main(["map-cic", "--root", str(repo_root), "DoS Hulk", "BENIGN"]) == 0
    cic = capsys.readouterr().out
    assert "label=ddos" in cic
    assert "binary=malicious" in cic
    assert "label=benign" in cic
    assert "binary=benign" in cic


def test_track_smoke_command(repo_root: Path, tmp_path: Path, capsys) -> None:
    overlay = tmp_path / "tracking.yaml"
    uri = (tmp_path / "mlruns").as_posix()
    overlay.write_text(
        "mlflow:\n"
        f"  tracking_uri: '{uri}'\n"
        "  experiment_name: cli-test\n",
        encoding="utf-8",
    )
    assert main(
        [
            "track-smoke",
            "--root",
            str(repo_root),
            "--task",
            "application",
            "--overlay",
            str(overlay),
        ]
    ) == 0
    assert "run_id=" in capsys.readouterr().out
