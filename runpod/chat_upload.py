"""Upload the model folder through the RunPod HTTPS proxy in small chunks.

The proxy rejects a single 826 MB POST with HTTP 413. Each chunk stays
under that limit. The server remembers which chunks it has, so a dropped
chunk is sent again by itself.

A real 4xx from our server, including 401, 403, and 413, is fatal: the caller
must terminate the pod. A proxy flap is not our server. An empty 404, a
502/503/504, or a dropped connection is retried on the same chunk. The pull
server must answer GET / twice in a row before the first byte is sent.
"""
from __future__ import annotations

import hashlib
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# 32 MB is well under the body size that produced HTTP 413. --chunk-mb can
# lower it. The server refuses anything above 50 MB.
DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024
MAX_CHUNK_BYTES = 50 * 1024 * 1024
CHUNK_ATTEMPTS = 8
RETRY_DELAY_S = 2.0
RETRY_DELAY_MAX_S = 60.0
PULL_READY_TIMEOUT_S = 180


class UploadFailed(Exception):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


def chunk_count(total: int, chunk_bytes: int) -> int:
    if total <= 0:
        return 1
    return (total + chunk_bytes - 1) // chunk_bytes


def chunk_len(total: int, chunk_bytes: int, index: int) -> int:
    if total <= 0:
        return 0
    start = index * chunk_bytes
    return min(chunk_bytes, total - start)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _call(method: str, url: str, token: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 180):
    hdrs = {
        "X-Chat-Token": token,
        "X-Pilot-Token": token,
        "User-Agent": "caramel-italiano-chat",
    }
    if headers:
        hdrs.update(headers)
    if data is not None:
        hdrs["Content-Length"] = str(len(data))
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _proxy_failure(code: int, raw: bytes) -> bool:
    """True for a RunPod/Cloudflare flap, not for a reply our server composed.

    The proxy returns 404 with an empty body while nothing is listening, and
    502/503/504 while it is restarting. Our own 404 includes an error body.
    """
    if code in (0, 500, 502, 503, 504):
        return True
    if code == 404 and not raw.strip():
        return True
    return False


def _file_list(raw: bytes) -> bool:
    try:
        payload = json.loads(raw.decode() or "")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("files"), list)


def wait_for_pull_server(
    base: str, token: str, timeout: float = PULL_READY_TIMEOUT_S,
    pause=time.sleep, gap: float = RETRY_DELAY_S,
) -> None:
    """Return once GET / is the pull server's file list, twice in a row."""
    deadline = time.monotonic() + timeout
    streak = 0
    last = "no response"
    url = base.rstrip("/") + "/"
    while True:
        try:
            code, raw = _call("GET", url, token, timeout=20)
        except (TimeoutError, urllib.error.URLError, ssl.SSLError, ConnectionError, OSError) as exc:
            code, raw = 0, str(exc).encode()
        if code == 200 and _file_list(raw):
            streak += 1
            last = "HTTP 200 file list"
            if streak >= 2:
                print("pull server answered OK twice", flush=True)
                return
            print("pull server answered once; waiting for a second OK", flush=True)
        else:
            streak = 0
            last = f"HTTP {code}: {raw[:180]!r}"
            if 400 <= code < 500 and not _proxy_failure(code, raw):
                raise UploadFailed(f"pull server {last}", status=code)
            print(f"pull server not steady ({last}); waiting", flush=True)
        if time.monotonic() >= deadline:
            raise UploadFailed(
                f"pull server did not answer OK twice within {timeout:.0f}s: {last}",
                status=0,
            )
        pause(gap)


def _request(method, url, token, data=None, headers=None, timeout=180, attempts=CHUNK_ATTEMPTS, what="request", pause=time.sleep):
    """Return (status, body). A real 4xx raises. Proxy flaps retry the same call."""
    delay = RETRY_DELAY_S
    last = "no response"
    for attempt in range(attempts):
        try:
            code, raw = _call(method, url, token, data, headers, timeout)
        except (TimeoutError, urllib.error.URLError, ssl.SSLError, ConnectionError, OSError) as exc:
            code, raw = 0, str(exc).encode()
        if code == 200:
            return code, raw
        if 400 <= code < 500 and not _proxy_failure(code, raw):
            raise UploadFailed(f"{what} HTTP {code}: {raw[:180]!r}", status=code)
        last = f"HTTP {code}: {raw[:180]!r}"
        if attempt + 1 == attempts:
            break
        print(f"{what} {last}; retry {attempt + 1}/{attempts}", flush=True)
        pause(delay)
        delay = min(delay * 2, RETRY_DELAY_MAX_S)
    raise UploadFailed(f"{what} failed {attempts} times: {last}", status=0)


def _status(base: str, token: str, rel: str, attempts: int, pause=time.sleep) -> dict:
    url = base.rstrip("/") + "/upload?path=" + urllib.parse.quote(rel, safe="")
    _code, raw = _request("GET", url, token, attempts=attempts, what=f"status {rel}", pause=pause)
    try:
        payload = json.loads(raw.decode() or "{}")
    except json.JSONDecodeError as exc:
        raise UploadFailed(f"status {rel} was not JSON") from exc
    return payload


def upload_file(base: str, token: str, src: Path, rel: str, chunk_bytes: int = DEFAULT_CHUNK_BYTES, attempts: int = CHUNK_ATTEMPTS, pause=time.sleep) -> None:
    if chunk_bytes < 1 or chunk_bytes > MAX_CHUNK_BYTES:
        raise UploadFailed(f"chunk size {chunk_bytes} is outside 1..{MAX_CHUNK_BYTES}")
    total = src.stat().st_size
    sha = file_sha256(src)
    n = chunk_count(total, chunk_bytes)
    state = _status(base, token, rel, attempts, pause=pause)
    if state.get("complete") and int(state.get("bytes") or -1) == total:
        print(f"{rel} already on the pod ({total} bytes)", flush=True)
        return
    have = {int(i) for i in (state.get("have") or [])}
    url = base.rstrip("/") + "/upload"
    with src.open("rb") as fh:
        for index in range(n):
            if state.get("complete") and int(state.get("bytes") or -1) == total:
                return
            if index in have:
                print(f"{rel} chunk {index + 1}/{n} already stored", flush=True)
                continue
            offset = index * chunk_bytes
            length = chunk_len(total, chunk_bytes, index)
            fh.seek(offset)
            blob = fh.read(length)
            headers = {
                "X-Pilot-Path": rel,
                "X-Chunk-Index": str(index),
                "X-Chunk-Offset": str(offset),
                "X-Chunk-Bytes": str(chunk_bytes),
                "X-Total-Size": str(total),
                "X-File-Sha256": sha,
                "Content-Type": "application/octet-stream",
            }
            delay = RETRY_DELAY_S
            done = False
            last = "no response"
            for attempt in range(attempts):
                try:
                    code, raw = _call("POST", url, token, blob, headers, timeout=180)
                except (TimeoutError, urllib.error.URLError, ssl.SSLError, ConnectionError, OSError) as exc:
                    code, raw = 0, str(exc).encode()
                if code == 200:
                    print(f"uploaded {rel} chunk {index + 1}/{n} ({length} bytes)", flush=True)
                    try:
                        state = json.loads(raw.decode() or "{}")
                    except json.JSONDecodeError:
                        state = {}
                    have = {int(i) for i in (state.get("have") or [])}
                    done = True
                    break
                if 400 <= code < 500 and not _proxy_failure(code, raw):
                    raise UploadFailed(f"{rel} chunk {index} HTTP {code}: {raw[:180]!r}", status=code)
                last = f"HTTP {code}: {raw[:180]!r}"
                if attempt + 1 == attempts:
                    break
                print(f"{rel} chunk {index} {last}; retry {attempt + 1}/{attempts}", flush=True)
                pause(delay)
                delay = min(delay * 2, RETRY_DELAY_MAX_S)
                state = _status(base, token, rel, attempts, pause=pause)
                have = {int(i) for i in (state.get("have") or [])}
                if index in have or (state.get("complete") and int(state.get("bytes") or -1) == total):
                    print(f"{rel} chunk {index} landed after the retry", flush=True)
                    done = True
                    break
            if not done:
                raise UploadFailed(f"{rel} chunk {index} failed {attempts} times: {last}", status=0)
    state = _status(base, token, rel, attempts, pause=pause)
    if not state.get("complete") or int(state.get("bytes") or -1) != total:
        raise UploadFailed(f"{rel} was not complete after {n} chunks")


def upload_files(base: str, token: str, pairs: list[tuple[Path, str]], chunk_bytes: int = DEFAULT_CHUNK_BYTES, attempts: int = CHUNK_ATTEMPTS, pause=time.sleep) -> None:
    for src, rel in pairs:
        upload_file(base, token, src, rel, chunk_bytes, attempts, pause=pause)


def notify_ready(base: str, token: str, attempts: int = CHUNK_ATTEMPTS, pause=time.sleep) -> None:
    # Loading the weights can take several minutes. A real 4xx aborts; a proxy flap retries.
    _request("POST", base.rstrip("/") + "/ready", token, data=b"", timeout=600, attempts=attempts, what="ready", pause=pause)


def confirm_pod_gone(pod_id: str, api_key: str, terminate, list_ids, tries: int = 3, pause=time.sleep) -> list[str]:
    """Terminate until myself.pods no longer lists this pod. Does not touch other pods."""
    last: list[str] = []
    for _attempt in range(tries):
        try:
            last = [str(item) for item in list_ids(api_key)]
        except Exception as exc:  # noqa: BLE001
            print(f"pod list failed: {exc}", flush=True)
            pause(2)
            continue
        if pod_id not in last:
            shown = "empty" if not last else ",".join(last)
            print(f"confirmed {pod_id} is gone; myself.pods={shown}", flush=True)
            return last
        print(f"{pod_id} still listed, terminating again", flush=True)
        try:
            terminate(pod_id, api_key)
        except Exception as exc:  # noqa: BLE001
            print(f"terminate failed: {exc}", flush=True)
        pause(2)
    print(f"{pod_id} still in myself.pods after terminate: {last}", flush=True)
    return last
