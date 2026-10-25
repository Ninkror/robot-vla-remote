from __future__ import annotations

import json
import socket
import struct


MAX_PAYLOAD_BYTES = 32 * 1024 * 1024


def _read_exact(stream: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        block = stream.recv(size - len(result))
        if not block:
            raise ConnectionError("worker connection closed before the payload completed")
        result.extend(block)
    return bytes(result)


def receive(stream: socket.socket) -> dict:
    size = struct.unpack("!I", _read_exact(stream, 4))[0]
    if size < 2 or size > MAX_PAYLOAD_BYTES:
        raise ValueError(f"invalid worker payload length: {size}")
    value = json.loads(_read_exact(stream, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("worker payload must be an object")
    return value


def send(stream: socket.socket, value: dict) -> None:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("worker payload exceeds the size limit")
    stream.sendall(struct.pack("!I", len(payload)) + payload)
