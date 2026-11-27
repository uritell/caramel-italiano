#!/usr/bin/env python3
"""Serve pilot-out to the launcher. Every request needs X-Pilot-Token.

The token is created by the launcher and passed in the pod environment.
It is not written into the output directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(os.environ["PILOT_OUT"]).resolve()
TOKEN = os.environ["PILOT_PULL_TOKEN"]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("pull " + (fmt % args) + "\n")

    def _unauthorized(self) -> None:
        self.send_error(401)

    def _path(self) -> Path | None:
        rel = self.path.split("?", 1)[0].lstrip("/")
        if rel in ("", "."):
            return None
        path = (ROOT / rel).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            return None
        if not path.is_file():
            return None
        return path

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _restore_path(self, rel: str) -> Path | None:
        rel = rel.lstrip("/")
        parts = Path(rel).parts
        if not parts or ".." in parts:
            return None
        if parts[0] not in ("stores", "checkpoints"):
            return None
        path = (ROOT / "restore" / rel).resolve()
        try:
            path.relative_to((ROOT / "restore").resolve())
        except ValueError:
            return None
        return path

    def _upload(self) -> None:
        if self.headers.get("X-Chunk-Index") is not None:
            self._upload_chunk()
            return
        rel = self.headers.get("X-Pilot-Path") or ""
        path = self._restore_path(rel)
        if path is None:
            self.send_error(400)
            return
        length = int(self.headers.get("Content-Length") or "0")
        path.parent.mkdir(parents=True, exist_ok=True)
        remaining = length
        with path.open("wb") as fh:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)
        self._json(200, {"ok": True, "bytes": length - remaining, "path": rel})

    def _partial(self, rel: str) -> Path:
        return ROOT / "restore" / ".partial" / rel

    def _upload_status(self) -> None:
        rel = (parse_qs(self.path.split("?", 1)[-1]).get("path") or [""])[0]
        dest = self._restore_path(rel)
        if dest is None:
            self.send_error(400)
            return
        if dest.is_file():
            self._json(200, {"ok": True, "path": rel, "complete": True, "bytes": dest.stat().st_size, "have": []})
            return
        manifest_path = self._partial(rel) / "manifest.json"
        if not manifest_path.is_file():
            self._json(200, {"ok": True, "path": rel, "complete": False, "have": []})
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        total, stride = int(manifest["total"]), int(manifest["chunk_bytes"])
        n = 1 if total <= 0 else (total + stride - 1) // stride
        have = []
        for index in range(n):
            part = self._partial(rel) / f"{index:06d}.part"
            expect = 0 if total <= 0 else min(stride, total - index * stride)
            if part.is_file() and part.stat().st_size == expect:
                have.append(index)
        self._json(200, {"ok": True, "path": rel, "complete": False, "have": have, "total": total, "sha256": manifest.get("sha256")})

    def _upload_chunk(self) -> None:
        rel = self.headers.get("X-Pilot-Path") or ""
        dest = self._restore_path(rel)
        if dest is None:
            self.send_error(400)
            return
        try:
            index = int(self.headers["X-Chunk-Index"])
            offset = int(self.headers["X-Chunk-Offset"])
            total = int(self.headers["X-Total-Size"])
            stride = int(self.headers["X-Chunk-Bytes"])
            sha = self.headers["X-File-Sha256"].strip().lower()
        except (KeyError, ValueError):
            self.send_error(400)
            return
        length = int(self.headers.get("Content-Length") or "0")
        if stride < 1 or stride > 50 * 1024 * 1024 or length > 50 * 1024 * 1024:
            self.send_error(400)
            return
        body = self.rfile.read(length)
        n = 1 if total <= 0 else (total + stride - 1) // stride
        expect = 0 if total <= 0 else min(stride, total - index * stride)
        if index < 0 or index >= n or offset != index * stride or len(body) != expect or len(sha) != 64:
            self.send_error(400)
            return
        if dest.is_file() and dest.stat().st_size == total and _sha(dest) == sha:
            self._json(200, {"ok": True, "complete": True, "skipped": True, "path": rel, "bytes": total})
            return
        partial = self._partial(rel)
        partial.mkdir(parents=True, exist_ok=True)
        manifest = {"total": total, "sha256": sha, "chunk_bytes": stride}
        manifest_path = partial / "manifest.json"
        if manifest_path.is_file():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("sha256") != sha or int(old.get("total", -1)) != total:
                self._json(409, {"ok": False, "error": "different file already started"})
                return
        else:
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        part = partial / f"{index:06d}.part"
        tmp = partial / f"{index:06d}.part.tmp"
        tmp.write_bytes(body)
        tmp.replace(part)
        have = [i for i in range(n) if (partial / f"{i:06d}.part").is_file()]
        if len(have) != n:
            self._json(200, {"ok": True, "complete": False, "have": have, "path": rel})
            return
        digest = hashlib.sha256()
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = dest.with_name(dest.name + ".uploading")
        written = 0
        with blob.open("wb") as out:
            for i in range(n):
                data = (partial / f"{i:06d}.part").read_bytes()
                digest.update(data)
                out.write(data)
                written += len(data)
        if written != total or digest.hexdigest() != sha:
            blob.unlink(missing_ok=True)
            self._json(400, {"ok": False, "error": "size or sha256 mismatch"})
            return
        blob.replace(dest)
        shutil.rmtree(partial, ignore_errors=True)
        self._json(200, {"ok": True, "complete": True, "path": rel, "bytes": total, "have": have})

    def _ack(self) -> None:
        # Routine pulls must not create PULLED. Only an ack after DONE or
        # FAILED, and only if it is newer than that file, ends the wait.
        marks = [p for p in (ROOT / "DONE", ROOT / "FAILED") if p.is_file()]
        if not marks:
            self._json(200, {"ok": False, "reason": "no DONE or FAILED yet"})
            return
        newest = max(p.stat().st_mtime for p in marks)
        if time.time() <= newest:
            self._json(409, {"ok": False, "reason": "ack is not newer than DONE or FAILED"})
            return
        (ROOT / "PULLED").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        self._json(200, {"ok": True})

    def do_HEAD(self) -> None:  # noqa: N802
        if self.headers.get("X-Pilot-Token") != TOKEN:
            self._unauthorized()
            return
        path = self._path()
        if path is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def _dispatch(self) -> None:
        if self.headers.get("X-Pilot-Token") != TOKEN:
            self._unauthorized()
            return
        route = self.path.split("?", 1)[0]
        if route in ("/ack", "/ack/"):
            self._ack()
            return
        if route == "/upload":
            if self.command == "GET":
                self._upload_status()
                return
            if self.command != "POST":
                self.send_error(405)
                return
            self._upload()
            return
        if route == "/ready":
            if self.command != "POST":
                self.send_error(405)
                return
            (ROOT / "restore").mkdir(parents=True, exist_ok=True)
            (ROOT / "restore" / "READY").write_text(
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n"
            )
            self._json(200, {"ok": True})
            return
        if route in ("/", ""):
            files = []
            for dirpath, _dirs, names in os.walk(ROOT):
                for name in names:
                    full = Path(dirpath) / name
                    files.append(str(full.relative_to(ROOT)))
            body = json.dumps({"files": sorted(files)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        path = self._path()
        if path is None:
            self.send_error(404)
            return
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                self.wfile.write(chunk)


def main() -> int:
    port = int(os.environ.get("PILOT_PULL_PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"pull server on 0.0.0.0:{port} root={ROOT}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
