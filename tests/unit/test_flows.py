import socket
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from traffic_clf.config import load_config
from traffic_clf.features.extract import flows_to_tables
from traffic_clf.flows.builder import build_flows
from traffic_clf.flows.pcap import read_packets, write_pcap
from traffic_clf.flows.pipeline import extract_custom_pcap, extract_iscx_pcap


def test_bidirectional_flow_keeps_sizes_and_drops_payload(repo_root: Path, tmp_path: Path) -> None:
    spec = load_config(root=repo_root).features.flow
    path = tmp_path / "sample.pcap"
    write_pcap(
        path,
        [
            (1_700_000_000.0, _tcp("10.0.0.2", "93.184.216.34", 40000, 443, 0x02, 0)),
            (1_700_000_000.5, _tcp("93.184.216.34", "10.0.0.2", 443, 40000, 0x12, 10, marker=b"\xab")),
            (1_700_000_001.0, _tcp("10.0.0.2", "93.184.216.34", 40000, 443, 0x18, 20)),
        ],
    )
    batch = read_packets(path)
    assert batch.skipped == 0
    assert all(not hasattr(packet, "payload") for packet in batch.packets)
    flows = build_flows(batch.packets, active_timeout_s=120, idle_timeout_s=5)
    assert len(flows) == 1
    table = flows_to_tables(flows, spec, early_packets=2)
    row = table.features.iloc[0]
    assert row["fwd_packets"] == 2
    assert row["bwd_packets"] == 1
    assert row["fwd_bytes"] == 40 + 60
    assert row["bwd_bytes"] == 50
    assert row["syn_count"] == 2
    assert row["ack_count"] == 2
    assert row["duration"] == 1_000_000
    assert row["pkt_len_max"] == 60
    assert table.sequence.loc[0, "size_01"] == 40
    assert table.sequence.loc[0, "direction_01"] == 1
    assert table.sequence.loc[0, "iat_01"] == 0
    assert table.sequence.loc[0, "size_02"] == 50
    assert table.sequence.loc[0, "direction_02"] == -1
    assert table.sequence.loc[0, "iat_02"] == 500_000
    assert table.sequence.loc[0, "seq_observed"] == 2
    assert "size_03" not in table.sequence.columns
    assert "src_ip" not in table.features.columns
    assert table.meta.loc[0, "src_ip"] == "10.0.0.2"
    assert table.meta.loc[0, "dst_port"] == 443
    assert table.features.loc[0, "bytes_per_s"] == 150


def test_timeouts_close_a_flow_before_the_late_packet() -> None:
    early = _packet(0.0)
    middle = _packet(1.0)
    late = _packet(130.0)
    idle_split = build_flows([early, _packet(10.0)], active_timeout_s=120, idle_timeout_s=5)
    active_split = build_flows([early, middle, late], active_timeout_s=120, idle_timeout_s=1000)
    assert len(idle_split) == 2
    assert len(active_split) == 2
    assert len(active_split[0].packets) == 2
    assert active_split[1].packets[0].timestamp == 130.0


def test_iscx_filename_labels_the_flow_and_email_has_no_service(repo_root: Path, tmp_path: Path) -> None:
    config = load_config(root=repo_root, task="service")
    youtube = tmp_path / "vpn_youtube_A.pcap"
    email = tmp_path / "email2b.pcap"
    write_pcap(youtube, [(1_700_000_000.0, _tcp("10.0.0.2", "1.1.1.1", 1111, 443, 0x02, 0))])
    write_pcap(email, [(1_700_000_000.0, _tcp("10.0.0.2", "1.1.1.1", 1111, 443, 0x02, 0))])
    labeled = extract_iscx_pcap(youtube, config)
    mail = extract_iscx_pcap(email, config)
    assert labeled.meta.loc[0, "application"] == "youtube"
    assert labeled.meta.loc[0, "service"] == "streaming"
    assert bool(labeled.meta.loc[0, "vpn"]) is True
    assert mail.meta.loc[0, "application"] == "email"
    assert pd.isna(mail.meta.loc[0, "service"])


def test_sidecar_drops_flows_outside_the_session(repo_root: Path, tmp_path: Path) -> None:
    config = load_config(root=repo_root, task="service")
    start = datetime(2025, 11, 1, 8, 53, 20, tzinfo=timezone.utc)
    end = start + timedelta(seconds=60)
    path = tmp_path / "youtube_a.pcap"
    write_pcap(
        path,
        [
            (start.timestamp(), _tcp("10.0.0.2", "1.1.1.1", 1111, 443, 0x02, 0)),
            (start.timestamp() + 120, _tcp("10.0.0.2", "1.1.1.1", 1111, 443, 0x02, 0)),
        ],
    )
    sidecar = tmp_path / "sessions.csv"
    sidecar.write_text(
        "file,t_start,t_end,application,service,vpn,host,notes\n"
        f"youtube_a.pcap,{start.strftime('%Y-%m-%dT%H:%M:%SZ')},{end.strftime('%Y-%m-%dT%H:%M:%SZ')},youtube,streaming,no,lab,inside\n",
        encoding="utf-8",
    )
    from traffic_clf.data.labels import load_custom_spec
    from traffic_clf.data.loaders import load_custom_sidecar

    sessions = load_custom_sidecar(sidecar, load_custom_spec(config.labels.custom))
    table = extract_custom_pcap(path, config, sessions)
    assert len(table.features) == 1
    assert table.dropped_flows == 1
    assert table.meta.loc[0, "service"] == "streaming"


def test_pcapng_enhanced_packet_is_read(tmp_path: Path) -> None:
    frame = _tcp("10.0.0.2", "1.1.1.1", 1111, 443, 0x02, 0)
    path = tmp_path / "one.pcapng"
    path.write_bytes(_pcapng(frame, timestamp_us=2_000_000))
    batch = read_packets(path)
    assert len(batch.packets) == 1
    assert batch.packets[0].length == 40
    assert batch.packets[0].timestamp == 2.0


def test_non_ip_frames_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "arp.pcap"
    write_pcap(path, [(1.0, bytes(12) + struct.pack("!H", 0x0806) + bytes(28))])
    batch = read_packets(path)
    assert batch.packets == []
    assert batch.skipped == 1


def _tcp(
    src: str,
    dst: str,
    sport: int,
    dport: int,
    flags: int,
    payload_len: int,
    *,
    marker: bytes = b"",
) -> bytes:
    payload = (marker + bytes(payload_len))[:payload_len]
    tcp = struct.pack("!HHIIBBHHH", sport, dport, 0, 0, 5 << 4, flags, 65535, 0, 0) + payload
    ip_len = 20 + len(tcp)
    ip = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        ip_len,
        1,
        0,
        64,
        6,
        0,
        socket.inet_aton(src),
        socket.inet_aton(dst),
    )
    return bytes(12) + struct.pack("!H", 0x0800) + ip + tcp


def _packet(timestamp: float):
    from traffic_clf.flows.pcap import Packet

    return Packet(timestamp, "10.0.0.2", "1.1.1.1", 1111, 443, 6, 40, 0x02)


def _pcapng(frame: bytes, *, timestamp_us: int) -> bytes:
    def block(block_type: int, body: bytes) -> bytes:
        padded = body + bytes((-len(body)) % 4)
        total = 12 + len(padded)
        return struct.pack("<II", block_type, total) + padded + struct.pack("<I", total)

    section = struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1)
    interface = struct.pack("<HHI", 1, 0, 65535)
    high, low = timestamp_us >> 32, timestamp_us & 0xFFFFFFFF
    packet = struct.pack("<IIIII", 0, high, low, len(frame), len(frame)) + frame
    return block(0x0A0D0D0A, section) + block(1, interface) + block(6, packet)
