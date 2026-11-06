"""Classic PCAP and a minimal PCAPNG reader.

Payload bytes are discarded as soon as the header lengths are known. The
returned packets carry sizes, ports, and TCP flags only.
"""

from __future__ import annotations

import ipaddress
import socket
import struct
from dataclasses import dataclass
from pathlib import Path

from traffic_clf.errors import SchemaError

_PCAP_LE = 0xA1B2C3D4
_PCAP_BE = 0xD4C3B2A1
_PCAP_NSEC_LE = 0xA1B23C4D
_PCAP_NSEC_BE = 0x4D3CB2A1
_LINK_ETHERNET = 1
_LINK_RAW = 101
_LINK_LINUX_SLL = 113
_LINK_IPV4 = 228
_IPV6_EXTENSIONS = {0, 43, 44, 51, 60}


@dataclass(frozen=True)
class Packet:
    """One transport packet. There is no payload field."""

    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    length: int
    tcp_flags: int


@dataclass(frozen=True)
class PacketBatch:
    packets: list[Packet]
    skipped: int
    truncated: int


def read_packets(path: Path) -> PacketBatch:
    """Read a PCAP or a single-section PCAPNG file."""
    data = Path(path).read_bytes()
    if len(data) < 24:
        raise SchemaError(f"{path} is too small to be a packet capture.")
    if data[:4] == b"\x0a\x0d\x0d\x0a":
        return _read_pcapng(data, path)
    return _read_classic(data, path)


def write_pcap(path: Path, frames: list[tuple[float, bytes]]) -> None:
    """Write a little-endian microsecond PCAP. Used by tests and small fixtures."""
    header = struct.pack("<IHHIIII", _PCAP_LE, 2, 4, 0, 0, 65535, _LINK_ETHERNET)
    body = bytearray(header)
    for timestamp, frame in frames:
        seconds = int(timestamp)
        micros = int(round((timestamp - seconds) * 1_000_000))
        if micros == 1_000_000:
            seconds += 1
            micros = 0
        body.extend(struct.pack("<IIII", seconds, micros, len(frame), len(frame)))
        body.extend(frame)
    Path(path).write_bytes(body)


def _read_classic(data: bytes, path: Path) -> PacketBatch:
    magic = struct.unpack("<I", data[:4])[0]
    if magic in {_PCAP_LE, _PCAP_NSEC_LE}:
        endian = "<"
    elif magic in {_PCAP_BE, _PCAP_NSEC_BE}:
        endian = ">"
    else:
        raise SchemaError(f"{path} is not a classic PCAP or PCAPNG file.")
    header = struct.unpack(endian + "IHHIIII", data[:24])
    linktype = header[6]
    nanoseconds = magic in {_PCAP_NSEC_LE, _PCAP_NSEC_BE}
    packets: list[Packet] = []
    skipped = 0
    truncated = 0
    offset = 24
    while offset + 16 <= len(data):
        ts_sec, ts_frac, incl_len, _orig_len = struct.unpack_from(endian + "IIII", data, offset)
        offset += 16
        if offset + incl_len > len(data):
            truncated += 1
            break
        frame = data[offset : offset + incl_len]
        offset += incl_len
        timestamp = ts_sec + ts_frac / (1_000_000_000 if nanoseconds else 1_000_000)
        parsed, was_truncated = _parse_frame(frame, linktype, timestamp)
        truncated += was_truncated
        if parsed is None:
            skipped += 1
        else:
            packets.append(parsed)
    packets.sort(key=lambda packet: packet.timestamp)
    return PacketBatch(packets=packets, skipped=skipped, truncated=truncated)


def _read_pcapng(data: bytes, path: Path) -> PacketBatch:
    if len(data) < 12:
        raise SchemaError(f"{path} is too small to be PCAPNG.")
    block_type, block_length = struct.unpack_from("<II", data, 0)
    if block_type != 0x0A0D0D0A or block_length < 28 or block_length > len(data):
        raise SchemaError(f"{path} has an unsupported PCAPNG section header.")
    byte_order = struct.unpack_from("<I", data, 8)[0]
    endian = "<" if byte_order == 0x1A2B3C4D else ">"
    if byte_order not in {0x1A2B3C4D, 0x4D3C2B1A}:
        raise SchemaError(f"{path} has an unknown PCAPNG byte-order magic.")
    linktype = _LINK_ETHERNET
    ts_divisor = 1_000_000.0
    packets: list[Packet] = []
    skipped = 0
    truncated = 0
    offset = 0
    while offset + 8 <= len(data):
        block_type, block_length = struct.unpack_from(endian + "II", data, offset)
        if block_length < 12 or offset + block_length > len(data):
            truncated += 1
            break
        body = data[offset + 8 : offset + block_length - 4]
        if block_type == 1 and len(body) >= 8:
            linktype = struct.unpack_from(endian + "H", body, 0)[0]
            ts_divisor = _tsresol(body[8:], endian)
        elif block_type == 6 and len(body) >= 20:
            _interface, ts_high, ts_low, caplen, _orig = struct.unpack_from(endian + "IIIII", body, 0)
            timestamp = ((ts_high << 32) | ts_low) / ts_divisor
            if 20 + caplen > len(body):
                truncated += 1
            else:
                parsed, was_truncated = _parse_frame(body[20 : 20 + caplen], linktype, timestamp)
                truncated += was_truncated
                if parsed is None:
                    skipped += 1
                else:
                    packets.append(parsed)
        offset += block_length
    packets.sort(key=lambda packet: packet.timestamp)
    return PacketBatch(packets=packets, skipped=skipped, truncated=truncated)


def _tsresol(options: bytes, endian: str) -> float:
    """PCAPNG default resolution is microseconds. Option code 9 overrides it."""
    offset = 0
    while offset + 4 <= len(options):
        code, length = struct.unpack_from(endian + "HH", options, offset)
        offset += 4
        if code == 0:
            break
        value = options[offset : offset + length]
        padded = (length + 3) & ~3
        offset += padded
        if code == 9 and value:
            raw = value[0]
            if raw & 0x80:
                return float(2 ** (raw & 0x7F))
            return float(10 ** (raw & 0x7F))
    return 1_000_000.0


def _parse_frame(frame: bytes, linktype: int, timestamp: float) -> tuple[Packet | None, int]:
    if linktype == _LINK_ETHERNET:
        return _parse_ethernet(frame, timestamp)
    if linktype == _LINK_LINUX_SLL:
        if len(frame) < 16:
            return None, 1
        protocol = struct.unpack("!H", frame[14:16])[0]
        return _parse_network(frame[16:], protocol, timestamp)
    if linktype in {_LINK_RAW, _LINK_IPV4}:
        return _parse_ip(frame, timestamp)
    raise SchemaError(f"Link type {linktype} is not supported. Use Ethernet, Linux SLL, or raw IP.")


def _parse_ethernet(frame: bytes, timestamp: float) -> tuple[Packet | None, int]:
    if len(frame) < 14:
        return None, 1
    ethertype = struct.unpack("!H", frame[12:14])[0]
    payload = frame[14:]
    if ethertype == 0x8100:
        if len(payload) < 4:
            return None, 1
        ethertype = struct.unpack("!H", payload[2:4])[0]
        payload = payload[4:]
    return _parse_network(payload, ethertype, timestamp)


def _parse_network(payload: bytes, ethertype: int, timestamp: float) -> tuple[Packet | None, int]:
    if ethertype == 0x0800 or (payload and payload[0] >> 4 == 4 and ethertype in {0x0800, 0x0008}):
        return _parse_ipv4(payload, timestamp)
    if ethertype == 0x86DD:
        return _parse_ipv6(payload, timestamp)
    if payload and payload[0] >> 4 in {4, 6}:
        return _parse_ip(payload, timestamp)
    return None, 0


def _parse_ip(payload: bytes, timestamp: float) -> tuple[Packet | None, int]:
    if not payload:
        return None, 1
    version = payload[0] >> 4
    if version == 4:
        return _parse_ipv4(payload, timestamp)
    if version == 6:
        return _parse_ipv6(payload, timestamp)
    return None, 0


def _parse_ipv4(payload: bytes, timestamp: float) -> tuple[Packet | None, int]:
    if len(payload) < 20:
        return None, 1
    version = payload[0] >> 4
    header_len = (payload[0] & 0x0F) * 4
    if version != 4 or header_len < 20 or len(payload) < header_len:
        return None, 1
    total_length = struct.unpack("!H", payload[2:4])[0]
    fragment = struct.unpack("!H", payload[6:8])[0] & 0x1FFF
    if fragment:
        return None, 0
    protocol = payload[9]
    src_ip = socket.inet_ntoa(payload[12:16])
    dst_ip = socket.inet_ntoa(payload[16:20])
    usable = min(total_length, len(payload)) if total_length >= header_len else len(payload)
    return _parse_transport(
        payload[header_len:usable],
        protocol,
        total_length if total_length >= header_len else usable,
        src_ip,
        dst_ip,
        timestamp,
    )


def _parse_ipv6(payload: bytes, timestamp: float) -> tuple[Packet | None, int]:
    if len(payload) < 40:
        return None, 1
    if payload[0] >> 4 != 6:
        return None, 0
    payload_length = struct.unpack("!H", payload[4:6])[0]
    next_header = payload[6]
    src_ip = str(ipaddress.IPv6Address(payload[8:24]))
    dst_ip = str(ipaddress.IPv6Address(payload[24:40]))
    offset = 40
    hops = 0
    while next_header in _IPV6_EXTENSIONS and hops < 8:
        if next_header == 44:
            if offset + 8 > len(payload):
                return None, 1
            next_header = payload[offset]
            offset += 8
        else:
            if offset + 2 > len(payload):
                return None, 1
            next_header = payload[offset]
            extension_len = (payload[offset + 1] + 1) * 8
            if offset + extension_len > len(payload):
                return None, 1
            offset += extension_len
        hops += 1
    wire_length = 40 + payload_length
    return _parse_transport(
        payload[offset:],
        next_header,
        wire_length,
        src_ip,
        dst_ip,
        timestamp,
    )


def _parse_transport(
    segment: bytes,
    protocol: int,
    length: int,
    src_ip: str,
    dst_ip: str,
    timestamp: float,
) -> tuple[Packet | None, int]:
    if protocol == 6:
        if len(segment) < 20:
            return None, 1
        src_port, dst_port = struct.unpack("!HH", segment[:4])
        flags = segment[13]
    elif protocol == 17:
        if len(segment) < 8:
            return None, 1
        src_port, dst_port = struct.unpack("!HH", segment[:4])
        flags = 0
    else:
        return None, 0
    return (
        Packet(timestamp, src_ip, dst_ip, src_port, dst_port, protocol, length, flags),
        0,
    )
