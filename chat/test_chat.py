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
        self.calls = []

    def init_state(self, batch, tmax=None):
        return {"pos": torch.zeros((), dtype=torch.long)}

    def step(self, st, inp, forced, temp=1.1, top_k=50):
        st["pos"] += 1
        self.calls.append({
            "temp": float(temp),
            "top_k": int(top_k),
            "forced": forced.detach().cpu().clone(),
            "human": inp[:, 8:16].detach().cpu().clone(),
        })
        if int(forced.max()) >= 0:
            return forced.clone()
        return (inp[:, 8:16] + 1) % self.cfg.card


class FakeMimi:
    def __init__(self):
        self.streaming = False

    def encode(self, input_values, num_quantizers=8, padding_cache=None, use_streaming=None, encoder_cache=None):
        if use_streaming:
            self.streaming = True
        level = float(input_values.detach().float().abs().mean())
        code = 2 if level < 1e-4 else min(30, 10 + int(level * 10))
        return SimpleNamespace(
            audio_codes=torch.full((1, 8, 1), code, dtype=torch.long),
            padding_cache={"enc": 1},
            encoder_cache={"ok": True},
        )

    def decode(self, audio_codes, padding_cache=None, use_streaming=None, decoder_cache=None):
        frames = int(audio_codes.shape[-1])
        return SimpleNamespace(
            audio_values=torch.full((1, 1, frames * FRAME), 0.25),
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
    from live import DECODE_CONTEXT

    mimi = FakeMimi()
    loop = LiveLoop(FakeModel(context=32), mimi, gate=False)
    first = loop.push(np.zeros(FRAME, dtype=np.float32))
    if first.pcm.size != 0 or first.reset:
        raise SystemExit(f"first frame should be silent, got {first}")
    heard = None
    for _ in range(DECODE_CONTEXT + 2):
        heard = loop.push(np.full(FRAME, 0.001, dtype=np.float32))
        if heard.pcm.size:
            break
    if heard is None or heard.pcm.size != FRAME or not mimi.streaming:
        raise SystemExit(f"windowed decode did not return audio: {None if heard is None else heard.pcm.size}")
    if abs(float(heard.pcm[0]) - 0.25) > 1e-5:
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
    values = {item["key"]: item["value"] for item in body["env"]}
    if values.get("CHAT_TEMP") != "0.75" or values.get("CHAT_TOP_K") != "25":
        raise SystemExit(f"launcher sampling defaults drifted: {values.get('CHAT_TEMP')} {values.get('CHAT_TOP_K')}")
    if values.get("CHAT_IDLE_S") != "600":
        raise SystemExit(f"chat idle env is not 10 min: {values.get('CHAT_IDLE_S')}")
    if values.get("PILOT_MAX_USD_CENTS") != "150" or values.get("PILOT_MAX_SECONDS") != "3600":
        raise SystemExit("chat caps changed")
    if values.get("PILOT_WATCH_JOB"):
        raise SystemExit("chat create input arms the training pull-grace terminate")
    if chat_launch.dollar_stop_seconds(59, 150) <= 8 * 60:
        raise SystemExit("an A40 at $0.59/h would hit the $1.50 cap during a short chat")
    if chat_launch.dollar_stop_seconds(1200, 150) != 448:
        raise SystemExit(
            f"$1.50 at $12/h should die at 448s, got {chat_launch.dollar_stop_seconds(1200, 150)}"
        )
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

    idle = server.ChatServer("secret", Path("/tmp/caramel-chat-model"), None, on_idle, idle_s=0.35)
    idle_server = await server.serve(idle, "127.0.0.1", 0)
    try:
        await asyncio.sleep(0.6)
        if idle_reasons:
            raise SystemExit(f"idle fired before any browser: {idle_reasons}")
        if idle.seen_browser:
            raise SystemExit("the idle clock started with no page view")
        reader, writer = await asyncio.open_connection("127.0.0.1", idle_server.sockets[0].getsockname()[1])
        writer.write(b"GET /?token=secret HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        await writer.drain()
        code, _, body = await read_http(reader)
        writer.close()
        await writer.wait_closed()
        if code != 200 or b"Talk" not in body:
            raise SystemExit(f"page did not load: {code}")
        if not idle.seen_browser or idle.last_client is None:
            raise SystemExit("GET / did not start the idle clock")
        await asyncio.sleep(0.15)
        if idle_reasons:
            raise SystemExit("idle fired in the same moment the page was opened")
        await asyncio.wait_for(idle_done.wait(), timeout=2)
    finally:
        idle.stopping = True
        idle_server.close()
        await idle_server.wait_closed()
    if idle_reasons != ["idle"]:
        raise SystemExit(f"idle did not fire after the visitor left: {idle_reasons}")
    print("server ok")


def check_gate() -> None:
    model = FakeModel()
    loop = LiveLoop(model, FakeMimi(), hangover_ms=80)
    if loop.temp != 0.75 or loop.top_k != 25 or not loop.gate_enabled:
        raise SystemExit("defaults are not temperature 0.75, top_k 25, gate on")
    loop.push(np.zeros(FRAME, dtype=np.float32))
    quiet = model.calls[-1]
    if quiet["temp"] != 0.75 or quiet["top_k"] != 25 or int(quiet["forced"].max()) >= 0:
        raise SystemExit(f"quiet frame was not sampled at 0.75/25: {quiet['temp']} {quiet['top_k']}")
    loud = np.full(FRAME, 0.5, dtype=np.float32)
    heard = loop.push(loud)
    if heard.pcm.size != FRAME or float(np.max(np.abs(heard.pcm))) != 0.0:
        raise SystemExit("speech above the threshold was still audible")
    silence = loop._silence.detach().cpu().reshape(-1)
    if not torch.equal(model.calls[-1]["forced"].reshape(-1), silence):
        raise SystemExit("the gate did not force silence tokens")
    loop.push(loud)
    human = model.calls[-1]["human"].reshape(-1)
    if int(human[0]) == int(silence[0]):
        raise SystemExit("the person's codes were replaced by the silence tokens")
    if int(model.calls[-1]["forced"].min()) < 0 or float(np.max(np.abs(loop.push(np.zeros(FRAME, dtype=np.float32)).pcm))) != 0.0:
        raise SystemExit("the hangover did not keep the model quiet")
    # The line above consumed the hangover frame. This one should sample again.
    loop.push(np.zeros(FRAME, dtype=np.float32))
    if int(model.calls[-1]["forced"].max()) >= 0:
        raise SystemExit("sampling did not resume after the hangover")
    loop.set_sampling(temp=0.4, top_k=10)
    loop.push(np.zeros(FRAME, dtype=np.float32))
    if model.calls[-1]["temp"] != 0.4 or model.calls[-1]["top_k"] != 10:
        raise SystemExit("a sampling change was not passed into Model.step")
    if len(model.calls) < 5:
        raise SystemExit("Model.step was skipped while the gate was closed")
    os.environ["CHAT_TEMP"] = "0.55"
    os.environ["CHAT_TOP_K"] = "12"
    try:
        got = server.settings_from_env()
    finally:
        os.environ.pop("CHAT_TEMP", None)
        os.environ.pop("CHAT_TOP_K", None)
    if got["temp"] != 0.55 or got["top_k"] != 12 or got["hangover_ms"] != 500:
        raise SystemExit(f"env settings were not read: {got}")
    print("turn taking ok")


def check_upload_fail_fast() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import chat_upload

    posts = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b'{"ok":true,"have":[],"complete":false}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or "0")
            if length:
                self.rfile.read(length)
            posts["n"] += 1
            body = b"too large"
            self.send_response(413)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        folder = Path("/tmp/caramel-chunk-src")
        folder.mkdir(exist_ok=True)
        src = folder / "weights.bin"
        src.write_bytes(b"weights-that-must-not-be-retried-as-one-body")
        try:
            chat_upload.upload_file(
                f"http://127.0.0.1:{port}", "secret", src, "model/model.safetensors",
                chunk_bytes=16, attempts=5,
            )
            raise SystemExit("HTTP 413 did not abort the upload")
        except chat_upload.UploadFailed as exc:
            if exc.status != 413:
                raise SystemExit(f"expected status 413, got {exc.status}: {exc}")
        if posts["n"] != 1:
            raise SystemExit(f"413 was sent again ({posts['n']} posts); the launch would keep billing")
    finally:
        httpd.shutdown()

    listed = {"ids": ["pod-chat", "someone-else"]}
    calls = []

    def terminate(pod_id, api_key):
        calls.append(pod_id)
        listed["ids"] = [item for item in listed["ids"] if item != pod_id]
        return {"via": "test"}

    def list_ids(api_key):
        return list(listed["ids"])

    left = chat_upload.confirm_pod_gone("pod-chat", "k", terminate, list_ids, pause=lambda _s: None)
    if calls != ["pod-chat"] or left != ["someone-else"]:
        raise SystemExit(f"terminate did not stop only the chat pod: calls={calls} left={left}")
    text = (ROOT / "runpod" / "chat_launch.py").read_text(encoding="utf-8")
    if "upload will retry" in text:
        raise SystemExit("a failed upload still loops inside the cost watch")
    if "DEFAULT_CAP_CENTS = 150" not in text or "DEFAULT_MINUTES = 60" not in text:
        raise SystemExit("chat caps changed")
    print("upload fail-fast ok")


async def check_chunked_upload() -> None:
    import chat_upload

    payload = bytes((i * 17) % 256 for i in range(100))
    src_dir = Path("/tmp/caramel-chunk-src")
    src_dir.mkdir(exist_ok=True)
    src = src_dir / "model.safetensors"
    src.write_bytes(payload)
    model_dir = Path("/tmp/caramel-chunk-dst")
    if model_dir.exists():
        import shutil
        shutil.rmtree(model_dir)
    model_dir.mkdir()
    chat = server.ChatServer("secret", model_dir, None, lambda reason: None, idle_s=60)
    running = await server.serve(chat, "127.0.0.1", 0)
    port = running.sockets[0].getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    try:
        sha = chat_upload.file_sha256(src)
        headers = {
            "X-Pilot-Path": "model/model.safetensors",
            "X-Chunk-Index": "0",
            "X-Chunk-Offset": "0",
            "X-Chunk-Bytes": "40",
            "X-Total-Size": str(len(payload)),
            "X-File-Sha256": sha,
            "Content-Type": "application/octet-stream",
        }
        code, _raw = await asyncio.to_thread(
            chat_upload._call, "POST", base + "/upload", "secret", payload[:40], headers,
        )
        if code != 200:
            raise SystemExit(f"first chunk failed: {code}")
        await asyncio.to_thread(
            chat_upload.upload_file, base, "secret", src, "model/model.safetensors", 40, 3,
        )
        stored = (model_dir / "model.safetensors").read_bytes()
        if stored != payload:
            raise SystemExit("assembled file does not match the upload")
        if chat.upload_writes != 3:
            raise SystemExit(f"resume re-sent stored chunks (writes={chat.upload_writes})")
        await asyncio.to_thread(
            chat_upload.upload_file, base, "secret", src, "model/model.safetensors", 40, 3,
        )
        if chat.upload_writes != 3:
            raise SystemExit("a finished file was uploaded again")
    finally:
        chat.stopping = True
        running.close()
        await running.wait_closed()
    print("chunked upload ok")


def main() -> int:
    check_loop()
    check_gate()
    check_plan()
    check_upload_fail_fast()
    asyncio.run(check_server())
    asyncio.run(check_chunked_upload())
    print("chat test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
