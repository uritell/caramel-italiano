#!/usr/bin/env python3
"""Browser chat server. One page, a websocket, and an HTTP frame fallback.

Listens on port 8080 so RunPod's proxy
https://<pod>-8080.proxy.runpod.net can reach it. Every request needs the
random token from the launcher, in ?token= or X-Chat-Token.

The model folder can be uploaded first, the same headers as the pilot pull
server: POST /upload (X-Pilot-Path under model/) then POST /ready.

With no browser for 10 minutes the process terminates the pod using the
pod-scoped RUNPOD_API_KEY. The Stop button calls POST /stop and does the
same. The user's key is never read from anywhere but the environment
RunPod injects.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from live import FRAME, FrameResult, LiveLoop  # noqa: E402

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
IDLE_S = 600
ALLOWED = {"model/config.json", "model/model.safetensors", "model/SHA256SUMS"}


def token_ok(got: str, expected: str) -> bool:
    if not got or not expected:
        return False
    return hmac.compare_digest(got.encode(), expected.encode())


def pcm_to_i16(pcm: np.ndarray) -> bytes:
    if pcm.size == 0:
        return b""
    clipped = np.clip(pcm.astype(np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def i16_to_f32(raw: bytes) -> np.ndarray:
    if len(raw) % 2:
        raw = raw[: len(raw) - 1]
    if not raw:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0


def result_json(result: FrameResult) -> dict:
    return {
        "status": result.status,
        "ms": round(result.ms, 1),
        "minutes_left": round(result.minutes_left, 2),
        "reset": result.reset,
        "samples": int(result.pcm.size),
    }


def terminate_pod(reason: str) -> dict:
    """Delete this pod. Uses the key RunPod injected. Tests pass a stub instead."""
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not pod_id or not key:
        print(f"terminate skipped ({reason}): no pod-scoped key", flush=True)
        return {"ok": False, "reason": "no pod-scoped key"}
    body = json.dumps({
        "query": "mutation($input:PodTerminateInput!){podTerminate(input:$input)}",
        "variables": {"input": {"podId": pod_id}},
    }).encode()
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "caramel-italiano-chat",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        print(f"terminate failed ({reason}): {exc}", flush=True)
        return {"ok": False, "reason": str(exc)}
    print(f"terminate sent ({reason})", flush=True)
    return {"ok": True, "payload": payload}


class ChatServer:
    def __init__(self, token: str, model_dir: Path, loop: LiveLoop | None, on_terminate, idle_s: int = IDLE_S):
        self.token = token
        self.model_dir = model_dir
        self.loop = loop
        self.on_terminate = on_terminate
        self.idle_s = idle_s
        self.clients = 0
        self.last_client = time.monotonic()
        self.ready = asyncio.Event()
        self.lock = asyncio.Lock()
        self.stopping = False
        if loop is not None:
            self.ready.set()

    def touch(self) -> None:
        self.last_client = time.monotonic()

    async def idle_watch(self) -> None:
        while not self.stopping:
            await asyncio.sleep(min(5.0, max(0.1, self.idle_s / 4)))
            if self.clients > 0:
                self.touch()
                continue
            if time.monotonic() - self.last_client >= self.idle_s:
                print("idle: no client, terminating", flush=True)
                self.stopping = True
                await asyncio.to_thread(self.on_terminate, "idle")
                return

    def _auth(self, headers: dict, query: str) -> bool:
        got = ""
        for part in query.split("&"):
            if part.startswith("token="):
                got = part.split("=", 1)[1]
        got = got or headers.get("x-chat-token", "") or headers.get("x-pilot-token", "")
        return token_ok(got, self.token)

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await self._read_request(reader)
            if request is None:
                return
            method, path, query, headers, body = request
            if not self._auth(headers, query):
                await self._send(writer, 401, b"token required", "text/plain")
                return
            route = path.rstrip("/") or "/"
            if headers.get("upgrade", "").lower() == "websocket" and route == "/ws":
                await self._websocket(reader, writer, headers)
                return
            if method == "GET" and route == "/":
                await self._send(writer, 200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
                return
            if method == "GET" and route == "/worklet.js":
                await self._send(writer, 200, (HERE / "worklet.js").read_bytes(), "text/javascript; charset=utf-8")
                return
            if method == "GET" and route == "/health":
                self.touch()
                await self._send(writer, 200, b'{"ok":true}', "application/json")
                return
            if method == "GET" and route == "/idle":
                idle = 0 if self.clients else int(time.monotonic() - self.last_client)
                await self._send(writer, 200, json.dumps({"idle_s": idle, "limit_s": self.idle_s}).encode(), "application/json")
                return
            if method == "POST" and route == "/upload":
                await self._upload(writer, headers, body)
                return
            if method == "POST" and route == "/ready":
                await self._mark_ready(writer)
                return
            if method == "POST" and route == "/stop":
                self.stopping = True
                result = await asyncio.to_thread(self.on_terminate, "stop button")
                await self._send(writer, 200, json.dumps(result).encode(), "application/json")
                return
            if method == "POST" and route == "/frame":
                await self._frame_http(writer, body)
                return
            await self._send(writer, 404, b"not found", "text/plain")
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_request(self, reader: asyncio.StreamReader):
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = await reader.read(4096)
            if not chunk:
                return None
            data += chunk
            if len(data) > 1 << 20:
                return None
        head, rest = data.split(b"\r\n\r\n", 1)
        lines = head.decode("latin1").split("\r\n")
        method, target, _ = lines[0].split(" ", 2)
        path, _, query = target.partition("?")
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        length = int(headers.get("content-length") or "0")
        while len(rest) < length:
            rest += await reader.read(min(1 << 20, length - len(rest)))
        return method, path, query, headers, rest[:length]

    async def _send(self, writer, code: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 409: "Conflict", 503: "Unavailable"}.get(code, "OK")
        head = [f"HTTP/1.1 {code} {reason}", f"Content-Type: {content_type}", f"Content-Length: {len(body)}", "Connection: close"]
        for k, v in (extra or {}).items():
            head.append(f"{k}: {v}")
        writer.write(("\r\n".join(head) + "\r\n\r\n").encode() + body)
        await writer.drain()

    def _safe_model_path(self, rel: str) -> Path | None:
        rel = rel.lstrip("/")
        if rel not in ALLOWED:
            return None
        path = (self.model_dir / rel.split("/", 1)[1]).resolve()
        try:
            path.relative_to(self.model_dir.resolve())
        except ValueError:
            return None
        return path

    async def _upload(self, writer, headers, body: bytes) -> None:
        rel = headers.get("x-pilot-path", "")
        path = self._safe_model_path(rel)
        if path is None:
            await self._send(writer, 400, b"path not allowed", "text/plain")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        self.touch()
        await self._send(writer, 200, json.dumps({"ok": True, "bytes": len(body), "path": rel}).encode(), "application/json")

    async def _mark_ready(self, writer) -> None:
        if not (self.model_dir / "config.json").is_file() or not (self.model_dir / "model.safetensors").is_file():
            await self._send(writer, 409, b"model files missing", "text/plain")
            return
        (self.model_dir / "READY").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        if self.loop is None:
            self.loop = await asyncio.to_thread(load_live, self.model_dir)
        self.ready.set()
        self.touch()
        await self._send(writer, 200, b'{"ok":true}', "application/json")

    async def _frame_http(self, writer, body: bytes) -> None:
        if self.loop is None:
            await self._send(writer, 503, b"model not loaded", "text/plain")
            return
        self.touch()
        pcm, result = await self._run_samples(i16_to_f32(body))
        extra = {
            "X-Chat-Status": result.status if result else "warming up",
            "X-Chat-Ms": str(round(result.ms, 1) if result else 0),
            "X-Chat-Minutes-Left": str(round(result.minutes_left, 2) if result else 4),
            "X-Chat-Reset": "1" if result and result.reset else "0",
        }
        await self._send(writer, 200, pcm_to_i16(pcm), "application/octet-stream", extra)

    async def _run_samples(self, samples: np.ndarray) -> tuple[np.ndarray, FrameResult | None]:
        """Buffer to whole frames. Returns concatenated PCM and the last result."""
        async with self.lock:
            pending = getattr(self, "_pending", np.zeros(0, dtype=np.float32))
            pending = np.concatenate([pending, samples]) if samples.size else pending
            pieces = []
            last = None
            while pending.size >= FRAME:
                frame, pending = pending[:FRAME], pending[FRAME:]
                last = await asyncio.to_thread(self.loop.push, frame)
                if last.pcm.size:
                    pieces.append(last.pcm)
            self._pending = pending
            if not pieces:
                return np.zeros(0, dtype=np.float32), last
            return np.concatenate(pieces), last

    async def _websocket(self, reader, writer, headers) -> None:
        key = headers.get("sec-websocket-key", "")
        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        writer.write((
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode())
        await writer.drain()
        self.clients += 1
        self.touch()
        pending = np.zeros(0, dtype=np.float32)
        try:
            while not self.stopping:
                opcode, data = await _read_ws(reader)
                if opcode == 8:
                    break
                if opcode == 9:
                    writer.write(_ws_frame(10, data))
                    await writer.drain()
                    continue
                if opcode == 1:
                    samples = i16_to_f32(base64.b64decode(json.loads(data.decode())["pcm_b64"]))
                elif opcode == 2:
                    samples = i16_to_f32(data)
                else:
                    continue
                self.touch()
                if self.loop is None:
                    await _send_text(writer, {"status": "The model is still loading.", "ms": 0, "minutes_left": 4, "reset": False, "samples": 0})
                    continue
                pending = np.concatenate([pending, samples]) if samples.size else pending
                while pending.size >= FRAME:
                    frame, pending = pending[:FRAME], pending[FRAME:]
                    async with self.lock:
                        result = await asyncio.to_thread(self.loop.push, frame.copy())
                    await _send_text(writer, result_json(result))
                    if result.pcm.size:
                        writer.write(_ws_frame(2, pcm_to_i16(result.pcm)))
                        await writer.drain()
        finally:
            self.clients = max(0, self.clients - 1)
            self.touch()


async def _send_text(writer, payload: dict) -> None:
    writer.write(_ws_frame(1, json.dumps(payload).encode()))
    await writer.drain()


async def _read_ws(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    hdr = await reader.readexactly(2)
    opcode = hdr[0] & 0x0F
    masked = hdr[1] & 0x80
    length = hdr[1] & 0x7F
    if length == 126:
        length = int.from_bytes(await reader.readexactly(2), "big")
    elif length == 127:
        length = int.from_bytes(await reader.readexactly(8), "big")
    mask = await reader.readexactly(4) if masked else b""
    data = await reader.readexactly(length) if length else b""
    if mask:
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    return opcode, data


def _ws_frame(opcode: int, data: bytes) -> bytes:
    n = len(data)
    if n < 126:
        hdr = bytes([0x80 | opcode, n])
    elif n < 65536:
        hdr = bytes([0x80 | opcode, 126]) + n.to_bytes(2, "big")
    else:
        hdr = bytes([0x80 | opcode, 127]) + n.to_bytes(8, "big")
    return hdr + data


def load_live(model_dir: Path) -> LiveLoop:
    root = HERE.parent
    sys.path.insert(0, str(root / "third_party" / "caramel"))
    import torch
    from model import load
    from transformers import MimiModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(str(model_dir), dev)
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(dev).eval()
    return LiveLoop(model, mimi)


async def serve(chat: ChatServer, host: str, port: int):
    server = await asyncio.start_server(chat.handle, host, port)
    sock = server.sockets[0].getsockname()
    print(f"chat listening on {sock[0]}:{sock[1]}", flush=True)
    asyncio.create_task(chat.idle_watch())
    return server


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path(os.environ.get("CHAT_MODEL_DIR", "/workspace/chat-model")))
    p.add_argument("--token", default=os.environ.get("CHAT_TOKEN") or os.environ.get("PILOT_PULL_TOKEN") or "")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("CHAT_PORT", "8080")))
    p.add_argument("--wait-upload", action="store_true")
    p.add_argument("--idle-s", type=int, default=IDLE_S)
    args = p.parse_args(argv)
    if not args.token:
        raise SystemExit("CHAT_TOKEN is not set")
    loop = None
    if not args.wait_upload:
        if not (args.model_dir / "config.json").is_file():
            raise SystemExit(f"no model in {args.model_dir}")
        loop = load_live(args.model_dir)
    chat = ChatServer(args.token, args.model_dir, loop, terminate_pod, args.idle_s)

    async def run():
        server = await serve(chat, args.host, args.port)
        async with server:
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
