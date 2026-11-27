#!/usr/bin/env python3
"""Serve pilot-out to the launcher. Every request needs X-Pilot-Token.

The token is created by the launcher and passed in the pod environment.
It is not written into the output directory.
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(os.environ["PILOT_OUT"]).resolve()
TOKEN = os.environ["PILOT_PULL_TOKEN"]


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

    def do_GET(self) -> None:  # noqa: N802
        if self.headers.get("X-Pilot-Token") != TOKEN:
            self._unauthorized()
            return
        if self.path.split("?", 1)[0] in ("/", ""):
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
