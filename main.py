import json
import os
import threading
import time
from collections import deque
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import errors, types
from kubernetes import client, config
from pydantic import BaseModel

NAMESPACES = {"monitoring", "fluent-operator", "ping-demo1"}
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090")
MAX_GEMINI_REQUESTS_PER_MINUTE = int(os.getenv("MAX_GEMINI_REQUESTS_PER_MINUTE", "4"))
MAX_MODEL_CALLS_PER_CHAT = int(os.getenv("MAX_MODEL_CALLS_PER_CHAT", "4"))
REQUEST_WINDOW_SECONDS = 60

config.load_incluster_config()
core = client.CoreV1Api()
apps = client.AppsV1Api()
client_gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
gemini_request_times: deque[float] = deque()
gemini_rate_lock = threading.Lock()

SYSTEM = """You are a read-only Kubernetes incident investigator. Analyze evidence, cite namespace/pod/index names, distinguish facts from hypotheses, and suggest commands or fixes without executing changes. You may inspect only monitoring, fluent-operator, and ping-demo1. Never claim you changed the cluster."""

class ChatRequest(BaseModel):
    messages: list[dict[str, Any]]


def namespace(value: str) -> str:
    if value not in NAMESPACES:
        raise ValueError("namespace is outside the read-only allowlist")
    return value


def run_tool(name: str, args: dict[str, Any]) -> Any:
    ns = namespace(args.get("namespace", "")) if "namespace" in args else None
    if name == "list_pods":
        return [{"name": p.metadata.name, "phase": p.status.phase, "ready": all(c.ready for c in (p.status.container_statuses or []))} for p in core.list_namespaced_pod(ns).items]
    if name == "get_events":
        events = core.list_namespaced_event(ns).items
        return [{"reason": e.reason, "type": e.type, "message": e.message, "object": e.involved_object.name, "last": str(e.last_timestamp or e.event_time)} for e in sorted(events, key=lambda x: str(x.last_timestamp or x.event_time))[-50:]]
    if name == "pod_logs":
        return core.read_namespaced_pod_log(args["pod"], ns, container=args.get("container"), tail_lines=300)
    if name == "prometheus_query":
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": args["query"]}, timeout=15)
        response.raise_for_status()
        return response.json()
    raise ValueError(f"unsupported tool: {name}")


def list_pods(namespace: str) -> Any:
    """List pods in an allowed namespace."""
    return run_tool("list_pods", {"namespace": namespace})


def get_events(namespace: str) -> Any:
    """Get recent Kubernetes events in an allowed namespace."""
    return run_tool("get_events", {"namespace": namespace})


def pod_logs(namespace: str, pod: str, container: str | None = None) -> Any:
    """Read logs from a pod in an allowed namespace."""
    return run_tool("pod_logs", {"namespace": namespace, "pod": pod, "container": container})


def prometheus_query(query: str) -> Any:
    """Run a read-only PromQL instant query."""
    return run_tool("prometheus_query", {"query": query})


TOOLS = [list_pods, get_events, pod_logs, prometheus_query]


def reserve_gemini_request() -> None:
    now = time.monotonic()
    with gemini_rate_lock:
        while gemini_request_times and now - gemini_request_times[0] >= REQUEST_WINDOW_SECONDS:
            gemini_request_times.popleft()
        if len(gemini_request_times) >= MAX_GEMINI_REQUESTS_PER_MINUTE:
            retry_after = int(REQUEST_WINDOW_SECONDS - (now - gemini_request_times[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Gemini request limit reached. Retry in about {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )
        gemini_request_times.append(now)


def as_gemini_content(role: str, text: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part.from_text(text=text)])


app = FastAPI(title="Kubernetes AI Investigator")

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><title>Kubernetes Investigator</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;background:#10151c;color:#e7edf5}textarea{width:100%;height:120px;background:#18212b;color:#fff;border:1px solid #526273;padding:12px}button{margin-top:12px;padding:10px 18px;background:#56b4d3;border:0;cursor:pointer}pre{white-space:pre-wrap;background:#18212b;padding:16px}</style></head><body><h1>Kubernetes Investigator</h1><p>Read-only investigation across monitoring, fluent-operator, and ping-demo1.</p><textarea id='q' placeholder='Why are logs not reaching Elasticsearch?'></textarea><br><button onclick='ask()'>Investigate</button><pre id='out'></pre><script>async function ask(){out.textContent='Investigating...';const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:[{role:'user',content:q.value}]})});out.textContent=JSON.stringify(await r.json(),null,2)}</script></body></html>"""

@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    history: list[types.Content] = [
        as_gemini_content("user", message["content"]) if message["role"] == "user" else as_gemini_content("model", message["content"])
        for message in request.messages
    ]

    for _ in range(MAX_MODEL_CALLS_PER_CHAT):
        reserve_gemini_request()
        try:
            response = client_gemini.models.generate_content(
                model=MODEL,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    tools=TOOLS,
                ),
            )
        except errors.APIError as exc:
            if getattr(exc, "code", None) == 429 or getattr(exc, "status_code", None) == 429:
                raise HTTPException(
                    status_code=429,
                    detail="Gemini quota is exhausted. Retry after the quota window resets.",
                    headers={"Retry-After": str(REQUEST_WINDOW_SECONDS)},
                ) from exc
            raise HTTPException(status_code=502, detail=f"Gemini API error: {exc}") from exc

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return {"answer": "No answer returned."}

        candidate = candidates[0]
        parts = getattr(candidate.content, "parts", []) or []
        function_calls = [part.function_call for part in parts if getattr(part, "function_call", None)]

        if not function_calls:
            answer = "".join(getattr(part, "text", "") for part in parts)
            return {"answer": answer or "No answer returned."}

        history.append(candidate.content)

        for call in function_calls:
            try:
                output = run_tool(call.name, dict(call.args))
            except Exception as exc:
                output = {"error": str(exc)}
            history.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=call.name, response={"result": output})],
                )
            )

    raise HTTPException(502, "The model requested too many investigation steps; try a narrower question")