"""RunPod calls for the pilot launcher.

Create, status, stop, and terminate go to the GraphQL endpoint
https://api.runpod.io/graphql with Authorization: Bearer $RUNPOD_API_KEY.
That is the endpoint that answers from the machine this launcher runs on.

If a GraphQL call fails, the same operation is tried once against the REST
API on https://api.runpod.io/v2. Nothing in this file calls any other host.
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_BASE = "https://api.runpod.io/v2"

# Fields a normal account key can read. Host inventory and daemon fields
# (availableGpuCount, availableGpuCounts, maxUnreservedGpuCount, rentedCount,
# pool ids, datacenter stock) need CSR_READ / HOST_DAEMON and are not selected.
# Pod selections also omit apiKey, env, and volumeKey.
POD_MACHINE = "podHostId gpuDisplayName secureCloud currentPricePerGpu"

POD_CORE = f"""
id
desiredStatus
costPerHr
gpuCount
imageName
machineId
machine {{ {POD_MACHINE} }}
""".strip()

POD_STATUS = f"""
id
name
desiredStatus
costPerHr
gpuCount
imageName
lastStatusChange
ports
runtime {{
  uptimeInSeconds
  ports {{ ip isIpPublic privatePort publicPort type }}
}}
machine {{ {POD_MACHINE} }}
""".strip()

CREATE = """
mutation Deploy($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    %s
  }
}
""" % POD_CORE

STATUS = """
query PodStatus($input: PodFilter!) {
  pod(input: $input) {
    %s
  }
}
""" % POD_STATUS

STOP = """
mutation StopPod($input: PodStopInput!) {
  podStop(input: $input) { id desiredStatus }
}
"""

# podTerminate returns a scalar. A selection set here is a GraphQL error.
TERMINATE = """
mutation TerminatePod($input: PodTerminateInput!) {
  podTerminate(input: $input)
}
"""

# securePrice is the catalog rate. lowestPrice.uninterruptablePrice and
# stockStatus are the same public fields the RunPod user docs select.
# Live stock counts are not requested.
GPU_QUOTE = """
query GpuQuote($id: String!) {
  gpuTypes(input: { id: $id }) {
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
    securePrice
    maxGpuCountSecureCloud
    lowestPrice(input: { gpuCount: 1, secureCloud: true, minCudaVersion: "12.8" }) {
      uninterruptablePrice
      stockStatus
    }
  }
}
"""

# If a future field under this query is rejected, the quote still stands
# when securePrice came back. An error on securePrice itself does not.
_HOST_ONLY_FIELDS = (
    "availableGpuCount",
    "availableGpuCounts",
    "maxUnreservedGpuCount",
    "rentedCount",
    "totalCount",
    "rentalPercentage",
    "nodeGroupDatacenters",
    "nodeGroupGpuSizes",
    "gpuTypeDatacenters",
    "defaultQuotaAbsolute",
)


class ApiError(RuntimeError):
    pass


def _urlopen_retry(req: urllib.request.Request, timeout: int, attempts: int = 5):
    """Retry timeouts and TLS errors. A 4xx other than 429 is returned to the caller."""
    delay = 2.0
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and 400 <= exc.code < 500:
                raise
            last = exc
        except (TimeoutError, urllib.error.URLError, ssl.SSLError, ConnectionError, OSError) as exc:
            last = exc
        if attempt + 1 == attempts:
            break
        time.sleep(delay)
        delay = min(delay * 2, 30)
    if isinstance(last, urllib.error.HTTPError):
        raise last
    raise ApiError(f"network failed after {attempts} tries: {last}")


def graphql_payload(query: str, variables: dict, api_key: str, timeout: int = 90) -> dict:
    """POST one GraphQL document. HTTP failures raise. Field errors stay in the body."""
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "caramel-italiano-pilot",
        },
    )
    try:
        with _urlopen_retry(req, timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        raise ApiError(f"GraphQL HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ApiError(f"GraphQL request failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise ApiError("GraphQL response was not an object")
    return payload


def graphql(query: str, variables: dict, api_key: str, timeout: int = 90) -> dict:
    payload = graphql_payload(query, variables, api_key, timeout=timeout)
    if payload.get("errors"):
        raise ApiError("GraphQL errors: " + json.dumps(payload["errors"])[:800])
    if "data" not in payload:
        raise ApiError("GraphQL response had no data")
    return payload["data"]


def _secure_price(quote: dict):
    price = quote.get("securePrice")
    if isinstance(price, (int, float)) and not isinstance(price, bool) and price > 0:
        return price
    return None


def _quote_errors_optional(errors) -> bool:
    """True when every error is a gpuTypes field other than securePrice."""
    if not isinstance(errors, list) or not errors:
        return False
    for err in errors:
        if not isinstance(err, dict):
            return False
        path = err.get("path") or []
        if not path or path[0] != "gpuTypes":
            return False
        if "securePrice" in path:
            return False
    return True


def quote_from_payload(payload: dict) -> tuple[dict, list]:
    """Return (gpu type, optional field errors).

    A field error is ignored only when securePrice is present in the same
    payload and the error path is some other gpuTypes field. Missing
    securePrice, or an error on securePrice, is fatal.
    """
    if not isinstance(payload, dict):
        raise ApiError("GraphQL quote payload was not an object")
    data = payload.get("data") or {}
    types = data.get("gpuTypes") if isinstance(data, dict) else None
    quote = types[0] if isinstance(types, list) and types else {}
    if not isinstance(quote, dict):
        quote = {}
    errors = payload.get("errors") or []
    if not errors:
        return quote, []
    if _secure_price(quote) is not None and _quote_errors_optional(errors):
        return quote, errors
    raise ApiError("GraphQL errors: " + json.dumps(errors)[:800])


def _rest(method: str, path: str, api_key: str, payload: dict | None = None, timeout: int = 90):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        REST_BASE + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "caramel-italiano-pilot",
        },
    )
    try:
        with _urlopen_retry(req, timeout) as resp:
            raw = resp.read().decode()
            if not raw:
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        raise ApiError(f"REST {method} {path} HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ApiError(f"REST {method} {path} failed: {exc.reason}") from exc


def create_pod(pod_input: dict, api_key: str) -> dict:
    """podFindAndDeployOnDemand. Falls back to POST /v2/pods."""
    try:
        data = graphql(CREATE, {"input": pod_input}, api_key)
    except ApiError as exc:
        print(f"GraphQL create failed ({exc}); trying REST fallback", flush=True)
        return {"pod": _rest("POST", "/pods", api_key, _rest_create_body(pod_input)), "via": "rest"}
    pod = data.get("podFindAndDeployOnDemand")
    if not pod or not pod.get("id"):
        raise ApiError(f"create returned no pod id: {data}")
    return {"pod": pod, "via": "graphql"}


def pod_status(pod_id: str, api_key: str) -> dict:
    try:
        data = graphql(STATUS, {"input": {"podId": pod_id}}, api_key)
    except ApiError as exc:
        print(f"GraphQL status failed ({exc}); trying REST fallback", flush=True)
        return {"pod": _rest("GET", f"/pods/{pod_id}", api_key), "via": "rest"}
    return {"pod": data.get("pod"), "via": "graphql"}


def stop_pod(pod_id: str, api_key: str) -> dict:
    """podStop. The pilot cap does not call this; stop can keep a disk charge."""
    try:
        data = graphql(STOP, {"input": {"podId": pod_id}}, api_key)
    except ApiError as exc:
        print(f"GraphQL stop failed ({exc}); trying REST fallback", flush=True)
        return {"pod": _rest("POST", f"/pods/{pod_id}/action", api_key, {"action": "stop"}), "via": "rest"}
    return {"pod": data.get("podStop"), "via": "graphql"}


def terminate_pod(pod_id: str, api_key: str) -> dict:
    """podTerminate. This deletes the pod so container disk stops billing."""
    try:
        data = graphql(TERMINATE, {"input": {"podId": pod_id}}, api_key)
    except ApiError as exc:
        print(f"GraphQL terminate failed ({exc}); trying REST fallback", flush=True)
        _rest("POST", f"/pods/{pod_id}/action", api_key, {"action": "terminate"})
        return {"via": "rest"}
    return {"podTerminate": data.get("podTerminate"), "via": "graphql"}


LIST_PODS = """
query MyPods {
  myself { pods { id desiredStatus } }
}
"""


def list_pod_ids(api_key: str) -> list[str]:
    data = graphql(LIST_PODS, {}, api_key)
    pods = ((data.get("myself") or {}).get("pods")) or []
    return [str(p.get("id")) for p in pods if isinstance(p, dict) and p.get("id")]


def quote_gpu(gpu_id: str, api_key: str) -> dict:
    payload = graphql_payload(GPU_QUOTE, {"id": gpu_id}, api_key)
    quote, optional = quote_from_payload(payload)
    for err in optional:
        path = ".".join(str(part) for part in (err.get("path") or []))
        print(f"quote ignored optional field error at {path or 'unknown'}", flush=True)
    return quote


def _rest_create_body(pod_input: dict) -> dict:
    """Map the GraphQL create input onto REST v2. No network volume."""
    env = {item["key"]: item["value"] for item in pod_input.get("env") or []}
    gpu = {
        "id": pod_input["gpuTypeId"],
        "count": pod_input.get("gpuCount", 1),
    }
    if pod_input.get("minCudaVersion"):
        gpu["minCudaVersion"] = pod_input["minCudaVersion"]
    # Same argv as GraphQL dockerArgs. Arrays, so nothing is split on spaces.
    # `args` repeats that JSON for a server that only reads the string form.
    spec = json.loads(pod_input["dockerArgs"])
    body = {
        "name": pod_input.get("name") or "caramel-it-pilot",
        "cloud": pod_input.get("cloudType") or "SECURE",
        "image": pod_input["imageName"],
        "gpu": gpu,
        "disk": pod_input.get("containerDiskInGb") or 120,
        "ports": ["22/tcp", "8080/http"],
        "startSsh": True,
        "env": env,
        "entrypoint": spec["entrypoint"],
        "cmd": spec["cmd"],
        "args": pod_input["dockerArgs"],
    }
    return body
