"""RunPod calls for the pilot launcher.

Create, status, stop, and terminate go to the GraphQL endpoint
https://api.runpod.io/graphql with Authorization: Bearer $RUNPOD_API_KEY.
That is the endpoint that answers from the machine this launcher runs on.

If a GraphQL call fails, the same operation is tried once against the REST
API on https://api.runpod.io/v2. Nothing in this file calls any other host.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_BASE = "https://api.runpod.io/v2"

# Selection used after create. No apiKey, no env (those echo secrets).
POD_CORE = """
id
desiredStatus
costPerHr
gpuCount
imageName
machineId
machine { podHostId gpuDisplayName secureCloud currentPricePerGpu }
""".strip()

POD_STATUS = """
id
name
desiredStatus
costPerHr
gpuCount
imageName
lastStatusChange
ports
runtime {
  uptimeInSeconds
  ports { ip isIpPublic privatePort publicPort type }
}
machine { podHostId gpuDisplayName secureCloud currentPricePerGpu }
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

GPU_QUOTE = """
query GpuQuote($id: String!) {
  gpuTypes(input: { id: $id }) {
    id
    displayName
    secureCloud
    securePrice
    maxGpuCountSecureCloud
    memoryInGb
    lowestPrice(input: { gpuCount: 2, secureCloud: true, minCudaVersion: "12.8" }) {
      uninterruptablePrice
      stockStatus
      maxGpuCount
      availableGpuCount
    }
  }
}
"""


class ApiError(RuntimeError):
    pass


def graphql(query: str, variables: dict, api_key: str, timeout: int = 90) -> dict:
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        raise ApiError(f"GraphQL HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ApiError(f"GraphQL request failed: {exc.reason}") from exc
    if payload.get("errors"):
        raise ApiError("GraphQL errors: " + json.dumps(payload["errors"])[:800])
    if "data" not in payload:
        raise ApiError("GraphQL response had no data")
    return payload["data"]


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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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


def quote_gpu(gpu_id: str, api_key: str) -> dict:
    data = graphql(GPU_QUOTE, {"id": gpu_id}, api_key)
    types = data.get("gpuTypes") or []
    return types[0] if types else {}


def _rest_create_body(pod_input: dict) -> dict:
    """Map the GraphQL create input onto REST v2. No network volume."""
    env = {item["key"]: item["value"] for item in pod_input.get("env") or []}
    gpu = {
        "id": pod_input["gpuTypeId"],
        "count": pod_input.get("gpuCount", 1),
    }
    if pod_input.get("minCudaVersion"):
        gpu["minCudaVersion"] = pod_input["minCudaVersion"]
    body = {
        "name": pod_input.get("name") or "caramel-it-pilot",
        "cloud": pod_input.get("cloudType") or "SECURE",
        "image": pod_input["imageName"],
        "gpu": gpu,
        "disk": pod_input.get("containerDiskInGb") or 120,
        "ports": ["22/tcp", "8080/http"],
        "startSsh": True,
        "env": env,
        "args": json.dumps({
            "entrypoint": ["/bin/bash", "-lc"],
            "cmd": [pod_input.get("dockerArgs") or "true"],
        }),
    }
    return body
