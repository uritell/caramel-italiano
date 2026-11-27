#!/usr/bin/env python3
"""CPU test: fake model, one live frame, then a websocket that returns audio.

No pod is created and no API key is sent.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chat"))
sys.path.insert(0, str(ROOT / "runpod"))

import chat_launch  # noqa: E402
import server  # noqa: E402
from live import FRAME, LiveLoop  # noqa: E402


class Cfg:
    def __init__(self, context: int):
        self.card = 32
        self.n_q = 16
        self.dep_q = 8
        self.context = context
        self.delays = [0] + [1] * 7 + [0] + [1] * 7


class FakeModel:
    def __init__(self, context: int = 8):
        self.cfg = Cfg(context)

    def init_state(self, batch, tmax=None):
        return {"pos": torch.zeros((), dtype=torch.long)}

    def step(self, st, inp, forced, temp=1.1, top_k=50):
        st["pos"] += 1
        if int(forced.max()) >= 0:
            raise AssertionError("the person's codes are inputs, not forced outputs")
        return (inp[:, 8:16] + 1) % self.cfg.card


class FakeMimi:
    def __init__(self):
        self.streaming = False

    def encode(self, input_values, num_quantizers=8, padding_cache=None, use_streaming=None, encoder_cache=None):
        if use_streaming:
            self.streaming = True
        return SimpleNamespace(
            audio_codes=torch.full((1, 8, 1), 4, dtype=torch.long),
            padding_cache={"enc": 1},
            encoder_cache={"ok": True},
        )

    def decode(self, audio_codes, padding_cache=None, use_streaming=None, decoder_cache=None):
        if not use_streaming:
            raise AssertionError("decode should stream")
        return SimpleNamespace(
            audio_values=torch.full((1, 1, FRAME), 0.25),
            padding_cache={"dec": 1},
            decoder_cache={"ok": True},
        )


def client_frame(opcode: int, data: bytes) -> bytes:
    mask = os.urandom(4)
    n = len(data)
    if n < 126:
        hdr = bytes([0x80 | opcode, 0x80 | n])
    else:
        hdr = bytes([0x80 | opcode, 0x80 | 126]) + n.to_bytes(2, "big")
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    return hdr + mask + masked


async def read_http(reader: asyncio.StreamReader) -> tuple[int, dict, bytes]:
    data = b""
    while b"\r\n\r\n" not in data:
        data += await reader.readexactly(1)
    head, rest = data.split(b"\r\n\r\n", 1)
    lines = head.decode().split("\r\n")
    code = int(lines[0].split(" ", 2)[1])
    headers = {}
    for line in lines[1:]:
        k, v = line.split(":", 1)
        headers[k.strip().lower()] = v.strip()
    length = int(headers.get("content-length") or "0")
    while len(rest) < length:
        rest += await reader.readexactly(length - len(rest))
    return code, headers, rest[:length]


def check_loop() -> None:
    mimi = FakeMimi()
    loop = LiveLoop(FakeModel(), mimi)
    first = loop.push(np.zeros(FRAME, dtype=np.float32))
    if first.pcm.size != 0 or first.reset:
        raise SystemExit(f"first frame should be silent, got {first}")
    second = loop.push(np.full(FRAME, 0.2, dtype=np.float32))
    if second.pcm.size != FRAME or not mimi.streaming:
        raise SystemExit(f"second frame did not return streamed audio: {second.pcm.size} streaming={mimi.streaming}")
    if abs(float(second.pcm[0]) - 0.25) > 1e-5:
        raise SystemExit("decoded audio was not the fake codec output")
    short = LiveLoop(FakeModel(context=2), FakeMimi())
    short.push(np.zeros(FRAME, dtype=np.float32))
    short.push(np.zeros(FRAME, dtype=np.float32))
    reset = short.push(np.zeros(FRAME, dtype=np.float32))
    if not reset.reset or reset.frame_index != 1:
        raise SystemExit(f"context did not restart: reset={reset.reset} index={reset.frame_index}")
    print("live loop ok")


def check_plan() -> None:
    picked = chat_launch.choose_gpu([
        ("NVIDIA A40", {"securePrice": 0.59, "lowestPrice": {"stockStatus": "High"}}),
        ("NVIDIA GeForce RTX 4090", {"securePrice": 0.89, "lowestPrice": {"stockStatus": "High"}}),
        ("NVIDIA L40S", {"securePrice": 0.86, "lowestPrice": {"stockStatus": "High"}}),
    ])
    if picked != ("NVIDIA A40", 59):
        raise SystemExit(f"expected the A40, got {picked}")
    fallback = chat_launch.choose_gpu([
        ("NVIDIA A40", {"securePrice": 1.21, "lowestPrice": {"stockStatus": "High"}}),
        ("NVIDIA GeForce RTX 4090", {"securePrice": 0.89, "lowestPrice": {"stockStatus": "None"}}),
        ("NVIDIA L40S", {"securePrice": 0.86, "lowestPrice": {"stockStatus": "Low"}}),
    ])
    if fallback != ("NVIDIA L40S", 86):
        raise SystemExit(f"expected L40S after a high price and empty stock, got {fallback}")
    if chat_launch.choose_gpu([
        ("NVIDIA A40", {"securePrice": 1.50, "lowestPrice": {"stockStatus": "High"}}),
    ]) is not None:
        raise SystemExit("a price above $1.20 was accepted")
    body = chat_launch.build_create_input(
        "NVIDIA A40", 59, 1_700_000_000, "tok",
        "https://github.com/uritell/caramel-italiano.git", "main", 3600, 150,
    )
    problems = chat_launch.validate(body)
    if problems:
        raise SystemExit("chat create input rejected: " + "; ".join(problems))
    keys = {item["key"] for item in body["env"]}
    if "RUNPOD_API_KEY" in keys:
        raise SystemExit("chat pod env contains RUNPOD_API_KEY")
    if body["containerDiskInGb"] != 40 or body["gpuCount"] != 1:
        raise SystemExit("disk or gpu count is wrong")
    if "EU-RO-1" not in body["dataCenterIds"]:
        raise SystemExit("EU datacenters were not requested")
    if cost_half_hour() != 30:
        raise SystemExit("30 min at $0.59/h should be 30 cents")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "runpod" / "chat_launch.py"), "--dry-run"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0 or "No pod was created" not in proc.stdout:
        raise SystemExit(proc.stdout + proc.stderr)
    if '"key": "RUNPOD_API_KEY"' in proc.stdout:
        raise SystemExit("dry run put the API key in the pod env")
    shell = subprocess.run(["bash", "-n", str(ROOT / "runpod" / "chat_bootstrap.sh")], capture_output=True, text=True)
    if shell.returncode != 0:
        raise SystemExit(shell.stderr)
    print("launch plan ok")


def cost_half_hour() -> int:
    import cost
    return cost.cost_cents(30 * 60, 59)


async def check_server() -> None:
    reasons = []
    chat = server.ChatServer(
        "secret", Path("/tmp/caramel-chat-model"),
        LiveLoop(FakeModel(), FakeMimi()),
        lambda reason: reasons.append(reason),
        idle_s=30,
    )
    running = await server.serve(chat, "127.0.0.1", 0)
    port = running.sockets[0].getsockname()[1]
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write((
            "GET /ws?token=secret HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode())
        await writer.drain()
        await read_http(reader)
        tone = np.full(FRAME, 1000, dtype="<i2").tobytes()
        got_audio = b""
        for _ in range(2):
            writer.write(client_frame(2, tone))
            await writer.drain()
            opcode, data = await asyncio.wait_for(server._read_ws(reader), timeout=5)
            if opcode != 1:
                raise SystemExit(f"expected a status text frame, got opcode {opcode}")
            msg = json.loads(data)
            if msg.get("samples"):
                opcode, data = await asyncio.wait_for(server._read_ws(reader), timeout=5)
                if opcode == 2 and data:
                    got_audio = data
                    break
        writer.close()
        await writer.wait_closed()
        if len(got_audio) < 100:
            raise SystemExit(f"websocket returned no audio ({len(got_audio)} bytes)")

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write((
            "POST /frame?token=secret HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            f"Content-Length: {len(tone)}\r\n\r\n"
        ).encode() + tone)
        await writer.drain()
        code, _headers, body = await read_http(reader)
        writer.close()
        await writer.wait_closed()
        if code != 200 or len(body) < 100:
            raise SystemExit(f"HTTP frame failed: {code} {len(body)} bytes")

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        await writer.drain()
        code, _, _ = await read_http(reader)
        writer.close()
        await writer.wait_closed()
        if code != 401:
            raise SystemExit(f"missing token returned {code}")

        bad = b"POST /upload HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Chat-Token: secret\r\nX-Pilot-Path: ../secret\r\nContent-Length: 1\r\n\r\nx"
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(bad)
        await writer.drain()
        code, _, _ = await read_http(reader)
        writer.close()
        await writer.wait_closed()
        if code != 400:
            raise SystemExit(f"bad upload path returned {code}")

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"POST /stop?token=secret HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        code, _, body = await read_http(reader)
        writer.close()
        await writer.wait_closed()
        if code != 200 or "stop button" not in reasons:
            raise SystemExit(f"stop did not terminate: {code} {reasons} {body!r}")
    finally:
        running.close()
        await running.wait_closed()

    idle_reasons = []
    idle_done = asyncio.Event()

    def on_idle(reason):
        idle_reasons.append(reason)
        idle_done.set()

    idle = server.ChatServer("secret", Path("/tmp/caramel-chat-model"), LiveLoop(FakeModel(), FakeMimi()), on_idle, idle_s=0.4)
    idle_server = await server.serve(idle, "127.0.0.1", 0)
    try:
        await asyncio.wait_for(idle_done.wait(), timeout=3)
    finally:
        idle_server.close()
        await idle_server.wait_closed()
    if idle_reasons != ["idle"]:
        raise SystemExit(f"idle shutdown did not fire: {idle_reasons}")
    print("server ok")


def main() -> int:
    check_loop()
    check_plan()
    asyncio.run(check_server())
    print("chat test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
