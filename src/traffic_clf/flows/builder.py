"""Bidirectional flow assembly.

The first packet of a flow defines the forward direction. A later packet is
forward when it repeats that source and destination, and backward otherwise.
The idle timeout and the active timeout both close the flow before the packet
that crossed them is added, so that packet opens the next flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from traffic_clf.flows.pcap import Packet


@dataclass
class BuiltFlow:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    start_time: float
    last_time: float
    packets: list[Packet] = field(default_factory=list)
    directions: list[int] = field(default_factory=list)

    def add(self, packet: Packet) -> None:
        forward = packet.src_ip == self.src_ip and packet.src_port == self.src_port
        self.packets.append(packet)
        self.directions.append(1 if forward else -1)
        self.last_time = packet.timestamp


def build_flows(
    packets: list[Packet],
    *,
    active_timeout_s: float,
    idle_timeout_s: float,
) -> list[BuiltFlow]:
    """Group packets into bidirectional flows."""
    if active_timeout_s <= 0 or idle_timeout_s <= 0:
        raise ValueError("Flow timeouts must be positive.")
    ordered = sorted(packets, key=lambda packet: packet.timestamp)
    open_flows: dict[tuple, BuiltFlow] = {}
    closed: list[BuiltFlow] = []
    for packet in ordered:
        key = _flow_key(packet)
        current = open_flows.get(key)
        if current is not None and _expired(current, packet, active_timeout_s, idle_timeout_s):
            closed.append(current)
            current = None
        if current is None:
            current = BuiltFlow(
                src_ip=packet.src_ip,
                dst_ip=packet.dst_ip,
                src_port=packet.src_port,
                dst_port=packet.dst_port,
                protocol=packet.protocol,
                start_time=packet.timestamp,
                last_time=packet.timestamp,
            )
            open_flows[key] = current
        current.add(packet)
    closed.extend(open_flows.values())
    closed.sort(key=lambda flow: flow.start_time)
    return closed


def _flow_key(packet: Packet) -> tuple:
    left = (packet.src_ip, packet.src_port)
    right = (packet.dst_ip, packet.dst_port)
    return (packet.protocol, *sorted((left, right)))


def _expired(flow: BuiltFlow, packet: Packet, active_timeout_s: float, idle_timeout_s: float) -> bool:
    if packet.timestamp - flow.last_time > idle_timeout_s:
        return True
    return packet.timestamp - flow.start_time > active_timeout_s
