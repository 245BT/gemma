import argparse
import copy
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen


UPSTREAM = "http://127.0.0.1:8080"
HOST = "127.0.0.1"
PORT = 8081
UPSTREAM_TIMEOUT_SECONDS = 120
CHANNEL_RE = re.compile(r"^\s*<\|channel\>(?:thought|final)\s*<channel\|>\s*", re.IGNORECASE)
STREAM_TEXT_KEYS = {"content", "delta", "text"}
STREAM_CHANNEL_MARKERS = tuple(
    f"<|channel>{role}{separator}<channel|>"
    for role in ("thought", "final")
    for separator in ("", "\n", "\r\n", " ")
)


def clean_channel_markers(text):
    cleaned, count = CHANNEL_RE.subn("", text, count=1)
    return cleaned if count else text


def is_channel_marker_prefix(text):
    candidate = text.lstrip().lower()
    return not candidate or any(marker.startswith(candidate) for marker in STREAM_CHANNEL_MARKERS)


class StreamChannelCleaner:
    def __init__(self):
        self.buffer = ""
        self.pending = True

    def clean(self, text):
        if not self.pending:
            return clean_channel_markers(text)

        self.buffer += text
        cleaned = clean_channel_markers(self.buffer)
        if cleaned != self.buffer:
            self.buffer = ""
            self.pending = False
            return cleaned

        if is_channel_marker_prefix(self.buffer):
            return ""

        self.pending = False
        result = self.buffer
        self.buffer = ""
        return result


def clean_response_payload(payload):
    cleaned = copy.deepcopy(payload)
    for item in cleaned.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                content["text"] = clean_channel_markers(content["text"])
    return cleaned


def clean_json_strings(value):
    if isinstance(value, str):
        return clean_channel_markers(value)
    if isinstance(value, list):
        return [clean_json_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_json_strings(item) for key, item in value.items()}
    return value


def clean_stream_json_strings(value, cleaner, key=None):
    if isinstance(value, str):
        if key in STREAM_TEXT_KEYS:
            return cleaner.clean(value)
        return clean_channel_markers(value)
    if isinstance(value, list):
        return [clean_stream_json_strings(item, cleaner) for item in value]
    if isinstance(value, dict):
        return {
            item_key: clean_stream_json_strings(item, cleaner, item_key)
            for item_key, item in value.items()
        }
    return value


def clean_sse_payload(response_body):
    cleaned_lines = []
    stream_cleaner = StreamChannelCleaner()
    for line in response_body.decode("utf-8").splitlines(keepends=True):
        if not line.startswith("data: "):
            cleaned_lines.append(line)
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            cleaned_lines.append(line)
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            cleaned_lines.append(line)
            continue
        cleaned_lines.append("data: " + json.dumps(clean_stream_json_strings(payload, stream_cleaner)) + "\n")
    return "".join(cleaned_lines).encode("utf-8")


class ProxyHandler(BaseHTTPRequestHandler):
    upstream = UPSTREAM

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
        url = self.upstream + self.path
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "accept-encoding"}
        }
        request = Request(url, data=body if body else None, headers=headers, method=self.command)
        try:
            with urlopen(request, timeout=UPSTREAM_TIMEOUT_SECONDS) as response:
                status = response.status
                response_body = response.read()
                content_type = response.headers.get("Content-Type", "application/json")
        except HTTPError as exc:
            status = exc.code
            response_body = exc.read()
            content_type = exc.headers.get("Content-Type", "application/json")

        if self.path.startswith("/v1/responses"):
            if content_type.startswith("application/json"):
                try:
                    payload = json.loads(response_body.decode("utf-8"))
                    response_body = json.dumps(clean_response_payload(payload)).encode("utf-8")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            elif content_type.startswith("text/event-stream"):
                try:
                    response_body = clean_sse_payload(response_body)
                except UnicodeDecodeError:
                    pass

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


def run_proxy(host=HOST, port=PORT, upstream=UPSTREAM):
    handler = type("ConfiguredProxyHandler", (ProxyHandler,), {"upstream": upstream.rstrip("/")})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Gemma response proxy listening on http://{host}:{port}, upstream {upstream}", flush=True)
    server.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Gemma channel markers from llama.cpp responses.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--upstream", default=UPSTREAM)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_proxy(args.host, args.port, args.upstream)
