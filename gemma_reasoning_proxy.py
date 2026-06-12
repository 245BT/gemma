import argparse
import copy
import json
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gemma_response_proxy import clean_channel_markers


HOST = "127.0.0.1"
PORT = 8082
UPSTREAM = "http://127.0.0.1:8081"
UPSTREAM_TIMEOUT_SECONDS = 120
FAST_PATH_MAX_OUTPUT_TOKENS = 32
FAST_PATH_MAX_PROMPT_CHARS = 200
FAST_PATH_EXACT_OUTPUT_RE = re.compile(
    r"\b(?:reply|respond|answer|output|return|print|say)\s+"
    r"(?:with\s+)?(?:only|exactly)\b",
    re.IGNORECASE,
)
FAST_PATH_COMPLEX_RE = re.compile(
    r"\b("
    r"research|search|browse|latest|cite|citation|source|"
    r"edit|debug|fix|modify|implement|refactor|test|"
    r"tool|tools|subagent|sub-agent|"
    r"reasoning|reason|think|deep|deeply|plan|planning|verify|verification|"
    r"benchmark|measure|optimize|compare"
    r")\b",
    re.IGNORECASE,
)
PUBLIC_ERROR_MESSAGE = "Reasoning proxy request failed."
FORCE_REASONING_FIELD = "x_gemma_force_reasoning"


def build_response_payload(final_text, response_id=None, model=None, metadata=None):
    text = clean_channel_markers(final_text)
    payload = {
        "id": response_id or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "output_text": text,
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": text,
                    }
                ],
            }
        ],
    }
    if model:
        payload["model"] = model
    if isinstance(metadata, dict):
        for key in ("usage", "model_calls", "context_length", "cache_hit", "cache_hit_rate"):
            if key in metadata:
                payload[key] = metadata[key]
    return payload


def _sse_event(name, payload):
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def build_sse_payload(final_text, model=None):
    response = build_response_payload(final_text, model=model)
    response_id = response["id"]
    item = response["output"][0]
    item_id = item["id"]
    text = item["content"][0]["text"]
    in_progress = {"id": response_id, "object": "response", "status": "in_progress"}
    events = [
        (
            "response.created",
            {"type": "response.created", "response": in_progress},
        ),
        (
            "response.in_progress",
            {"type": "response.in_progress", "response": in_progress},
        ),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "item": {
                    "content": [],
                    "id": item_id,
                    "role": "assistant",
                    "status": "in_progress",
                    "type": "message",
                },
            },
        ),
        (
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": item_id,
                "part": {"type": "output_text", "text": ""},
            },
        ),
        (
            "response.output_text.delta",
            {"type": "response.output_text.delta", "item_id": item_id, "delta": text},
        ),
        (
            "response.output_text.done",
            {"type": "response.output_text.done", "item_id": item_id, "text": text},
        ),
        (
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": item_id,
                "part": {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": text,
                },
            },
        ),
        (
            "response.output_item.done",
            {"type": "response.output_item.done", "item": item},
        ),
        (
            "response.completed",
            {"type": "response.completed", "response": response},
        ),
    ]
    return "".join(_sse_event(name, payload) for name, payload in events).encode("utf-8")


def build_error_payload(_error=None):
    return {
        "error": {
            "message": PUBLIC_ERROR_MESSAGE,
            "type": "reasoning_proxy_error",
        }
    }


def _content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = _content_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        if "text" in content:
            return _content_to_text(content["text"])
        if "content" in content:
            return _content_to_text(content["content"])
    return ""


def latest_user_task_text(payload):
    input_value = payload.get("input", payload.get("prompt", ""))
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, dict):
        return _content_to_text(input_value)
    if isinstance(input_value, list):
        fallback = ""
        for item in reversed(input_value):
            if isinstance(item, dict):
                text = _content_to_text(item.get("content", item))
                if text and not fallback:
                    fallback = text
                if item.get("role") == "user" and text:
                    return text
            elif isinstance(item, str):
                if item and not fallback:
                    fallback = item
        return fallback
    return ""


def _small_output_cap(payload):
    token_cap = payload.get("max_output_tokens", payload.get("max_tokens"))
    if token_cap is None:
        return False
    try:
        token_cap = int(token_cap)
    except (TypeError, ValueError):
        return False
    return 0 < token_cap <= FAST_PATH_MAX_OUTPUT_TOKENS


def should_use_direct_fast_path(payload):
    if not isinstance(payload, dict):
        return False
    if payload.get(FORCE_REASONING_FIELD):
        return False
    return _is_simple_exact_output_payload(payload)


def should_use_lightweight_reasoning(payload):
    if not isinstance(payload, dict):
        return False
    if not payload.get(FORCE_REASONING_FIELD):
        return False
    return _is_simple_exact_output_payload(payload)


def _is_simple_exact_output_payload(payload):
    if payload.get("stream"):
        return False
    if payload.get("tool_choice") or payload.get("parallel_tool_calls"):
        return False
    if not _small_output_cap(payload):
        return False

    task = latest_user_task_text(payload).strip()
    if not task or len(task) > FAST_PATH_MAX_PROMPT_CHARS:
        return False

    lower_task = task.lower()
    if FAST_PATH_COMPLEX_RE.search(lower_task):
        return False
    return bool(FAST_PATH_EXACT_OUTPUT_RE.search(lower_task))


def fast_path_payload(payload):
    forwarded = copy.deepcopy(payload)
    for key in ("tools", "parallel_tool_calls", FORCE_REASONING_FIELD):
        forwarded.pop(key, None)
    return forwarded


def forward_response_to_upstream(request_payload, upstream=UPSTREAM):
    body = json.dumps(request_payload).encode("utf-8")
    request = Request(
        upstream.rstrip("/") + "/v1/responses",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=UPSTREAM_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def run_reasoning_request(request_payload, graph_runner=None, upstream=UPSTREAM):
    if graph_runner is None:
        from gemma_reasoning.dspy_programs import LocalHeuristicPrograms
        from gemma_reasoning.graph import run_reasoning_graph
        from gemma_reasoning.graph import ReasoningConfig
        from gemma_reasoning.upstream import UpstreamResponsesClient

        client = UpstreamResponsesClient(upstream)
        if should_use_lightweight_reasoning(request_payload):
            programs = LocalHeuristicPrograms()
            config = ReasoningConfig(use_langgraph=False)
            graph_runner = lambda payload: run_reasoning_graph(
                payload,
                client=client,
                programs=programs,
                config=config,
            )
        else:
            graph_runner = lambda payload: run_reasoning_graph(payload, client=client)
    result = graph_runner(request_payload)
    return build_response_payload(
        result.get("final_text", ""),
        model=request_payload.get("model"),
        metadata=result,
    )


class ReasoningProxyHandler(BaseHTTPRequestHandler):
    upstream = UPSTREAM

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        self.forward_to_upstream()

    def do_POST(self):
        if self.path.startswith("/v1/responses"):
            self.handle_reasoning_response()
            return
        self.forward_to_upstream()

    def read_json_body(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def handle_reasoning_response(self):
        try:
            payload = self.read_json_body()
            if should_use_direct_fast_path(payload):
                response = forward_response_to_upstream(fast_path_payload(payload), upstream=self.upstream)
            else:
                response = run_reasoning_request(payload, upstream=self.upstream)
            if payload.get("stream"):
                body = build_sse_payload(response["output"][0]["content"][0]["text"], model=payload.get("model"))
                self.send_bytes(200, body, "text/event-stream")
            else:
                self.send_json(200, response)
        except Exception as exc:
            self.send_json(500, build_error_payload(exc))

    def forward_to_upstream(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "accept-encoding"}
        }
        request = Request(
            self.upstream.rstrip("/") + self.path,
            data=body if body else None,
            headers=headers,
            method=self.command,
        )
        try:
            with urlopen(request, timeout=UPSTREAM_TIMEOUT_SECONDS) as response:
                status = response.status
                response_body = response.read()
                content_type = response.headers.get("Content-Type", "application/json")
        except HTTPError as exc:
            status = exc.code
            response_body = exc.read()
            content_type = exc.headers.get("Content-Type", "application/json")
        self.send_bytes(status, response_body, content_type)

    def send_json(self, status, payload):
        self.send_bytes(status, json.dumps(payload).encode("utf-8"), "application/json")

    def send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_proxy(host=HOST, port=PORT, upstream=UPSTREAM):
    handler = type(
        "ConfiguredReasoningProxyHandler",
        (ReasoningProxyHandler,),
        {"upstream": upstream.rstrip("/")},
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Gemma reasoning proxy listening on http://{host}:{port}, upstream {upstream}", flush=True)
    server.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local Gemma LangGraph/DSPy reasoning proxy.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--upstream", default=UPSTREAM)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_proxy(args.host, args.port, args.upstream)
