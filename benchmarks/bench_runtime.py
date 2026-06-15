import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import psutil
except ImportError:  # pragma: no cover - depends on local environment
    psutil = None


DEFAULT_MODEL = "gemma-4-26b-a4b-it-uncensored-q4-k-m"
MARKDOWN_HEADER = (
    "| Run | Phase | Workloads | First byte ms | First text ms | Total ms | "
    "Tok/s | Prompt tok | Completion tok | Model calls | Peak RAM MB | "
    "Peak VRAM MB | Context | Cache hit | Cache hit rate | Test pass | "
    "Task success | Hallucinated tools | Failed JSON/tools | Failed tools | Summary |"
)
MARKDOWN_SEPARATOR = (
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
    "---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
)


def load_workloads(path):
    workloads = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Workload at {path}:{line_number} must be a JSON object")
            if "id" not in item:
                item["id"] = f"workload-{line_number}"
            if not isinstance(item.get("request"), dict):
                raise ValueError(f"Workload {item['id']} must include a request object")
            workloads.append(item)
    if not workloads:
        raise ValueError(f"No workloads found in {path}")
    return workloads


def default_resource_sampler(target_patterns=None):
    ram_mb = None
    if psutil is not None and target_patterns:
        ram_mb = sample_process_ram_mb(psutil.process_iter, target_patterns)
    elif psutil is not None:
        ram_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return {"ram_mb": ram_mb, "vram_mb": sample_vram_mb()}


def process_patterns_for_endpoint(endpoint):
    patterns = ["llama-server.exe"]
    if ":8081" in endpoint:
        patterns.append("gemma_response_proxy.py")
    if ":8082" in endpoint:
        patterns.extend(["gemma_response_proxy.py", "gemma_reasoning_proxy.py"])
    return patterns


def sample_process_ram_mb(process_iter=None, command_patterns=None):
    if psutil is None and process_iter is None:
        return None
    process_iter = psutil.process_iter if process_iter is None else process_iter
    patterns = [pattern.lower() for pattern in (command_patterns or []) if pattern]
    if not patterns:
        return None
    total_bytes = 0
    matched = False
    try:
        processes = process_iter(attrs=["cmdline"])
    except TypeError:
        processes = process_iter()
    for process in processes:
        try:
            info = getattr(process, "info", {}) or {}
            cmdline = info.get("cmdline") or process.cmdline()
            command_text = " ".join(str(part) for part in cmdline).lower()
            if any(pattern in command_text for pattern in patterns):
                total_bytes += process.memory_info().rss
                matched = True
        except Exception:
            continue
    if not matched:
        return None
    return total_bytes / (1024 * 1024)


def parse_nvidia_smi_memory_used_mb(output):
    values = []
    for line in output.splitlines():
        match = re.search(r"(\d+(?:\.\d+)?)", line)
        if match:
            values.append(float(match.group(1)))
    if not values:
        return None
    return sum(values)


def sample_vram_mb(command_runner=None):
    if command_runner is None:
        command_runner = subprocess.run
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = command_runner(
            command,
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(completed, "returncode", 1) != 0:
        return None
    return parse_nvidia_smi_memory_used_mb(getattr(completed, "stdout", ""))


class EventWriter:
    def __init__(self, handle, clock):
        self.handle = handle
        self.clock = clock

    def emit(self, event, **fields):
        timestamp = fields.pop("timestamp", None)
        if timestamp is None:
            timestamp = self.clock()
        fields = sanitize_event_fields(fields)
        payload = {"timestamp": timestamp, "event": event}
        payload.update(fields)
        self.handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self.handle.flush()
        return payload


def sanitize_event_fields(fields):
    sanitized = dict(fields)
    text = sanitized.pop("text", None)
    if isinstance(text, str):
        sanitized["text_length"] = len(text)
        sanitized["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return sanitized


def build_endpoint_payload(endpoint, payload):
    payload = dict(payload)
    prompt = payload.pop("prompt", None)
    model = payload.setdefault("model", DEFAULT_MODEL)

    if "/v1/responses" in endpoint:
        if prompt is not None and "input" not in payload:
            payload["input"] = prompt
        payload["model"] = model
        return payload

    if prompt is not None and "messages" not in payload:
        payload["messages"] = [{"role": "user", "content": prompt}]
    payload["model"] = model
    return payload


def make_http_request(endpoint, payload, emit_event, timeout=120):
    request_payload = build_endpoint_payload(endpoint, payload)
    body = json.dumps(request_payload).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            first_chunk = response.read(1)
            if first_chunk:
                emit_event("first_byte")
            remainder = response.read()
            response_body = first_chunk + remainder
    except HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {body_text}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach {endpoint}: {exc}") from exc

    payload = json.loads(response_body.decode("utf-8"))
    text = extract_text(payload)
    if text:
        emit_event("first_text", text=text[:80])
    return {
        "text": text,
        "usage": extract_usage(payload),
        "context_length": extract_context_length(payload),
        "cache_hit": extract_cache_hit(payload),
        "cache_hit_rate": extract_cache_hit_rate(payload),
        "raw_response": payload,
    }


def extract_text(payload):
    if not isinstance(payload, dict):
        return ""
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    output = payload.get("output")
    if isinstance(output, list):
        pieces = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    pieces.append(content["text"])
        if pieces:
            return "".join(pieces)
    content = payload.get("content")
    if isinstance(content, str):
        return content
    return ""


def extract_usage(payload):
    if not isinstance(payload, dict):
        return {}
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    timings = payload.get("timings")
    if isinstance(timings, dict):
        usage = dict(usage)
        if "predicted_per_second" in timings:
            usage["tokens_per_second"] = timings["predicted_per_second"]
        if "prompt_n" in timings and "prompt_tokens" not in usage:
            usage["prompt_tokens"] = timings["prompt_n"]
        if "predicted_n" in timings and "completion_tokens" not in usage:
            usage["completion_tokens"] = timings["predicted_n"]
    return usage


def extract_context_length(payload):
    for key in ("context_length", "context_length_used", "n_ctx", "ctx_size"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, (int, float)):
            return value
    return None


def extract_cache_hit(payload):
    if not isinstance(payload, dict):
        return None
    for key in ("cache_hit", "cache_used"):
        if isinstance(payload.get(key), bool):
            return payload[key]
    return None


def extract_cache_hit_rate(payload):
    if not isinstance(payload, dict):
        return None
    value = payload.get("cache_hit_rate")
    return value if isinstance(value, (int, float)) else None


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _average(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _max_number(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values)


def _usage_value(usage, *keys):
    if not isinstance(usage, dict):
        return None
    for key in keys:
        value = _number(usage.get(key))
        if value is not None:
            return value
    return None


def run_benchmark(
    workload_path,
    endpoint,
    phase,
    run_label,
    events_path,
    summary_path,
    markdown_path,
    request_func=None,
    resource_sampler=None,
    clock=None,
):
    if request_func is None:
        request_func = make_http_request
    if resource_sampler is None:
        target_patterns = process_patterns_for_endpoint(endpoint)
        resource_sampler = lambda: default_resource_sampler(target_patterns)
    if clock is None:
        clock = time.perf_counter

    workloads = load_workloads(workload_path)
    events_path = Path(events_path)
    summary_path = Path(summary_path)
    markdown_path = Path(markdown_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    first_byte_latencies = []
    first_text_latencies = []
    total_times = []
    tool_latencies = []
    prompt_tokens = 0
    completion_tokens = 0
    token_rates = []
    model_calls = 0
    ram_samples = []
    vram_samples = []
    context_lengths = []
    cache_hits = []
    cache_hit_rates = []
    test_results = []
    task_results = []
    hallucinated_tool_calls = 0
    failed_json_tool_calls = 0
    failed_tool_calls = 0
    catalog_drift_results = []
    syntax_preflight_pass_rates = []
    hallucinated_tool_claim_rates = []
    raw_untrusted_bytes_in_prompt = 0
    raw_untrusted_bytes_seen = False
    failed_json_tool_call_rates = []

    run_started_at = clock()
    with events_path.open("w", encoding="utf-8") as handle:
        writer = EventWriter(handle, clock)
        writer.emit(
            "run_start",
            timestamp=run_started_at,
            run_label=run_label,
            phase=phase,
            endpoint=endpoint,
            workload_path=str(workload_path),
        )

        for workload in workloads:
            workload_id = str(workload["id"])
            workload_started_at = clock()
            writer.emit(
                "workload_start",
                timestamp=workload_started_at,
                workload_id=workload_id,
            )
            workload_model_calls = 0
            first_byte_latency = None
            first_text_latency = None

            def sample_resources(stage):
                sample = resource_sampler() or {}
                ram_mb = _number(sample.get("ram_mb"))
                vram_mb = _number(sample.get("vram_mb"))
                if ram_mb is not None:
                    ram_samples.append(ram_mb)
                if vram_mb is not None:
                    vram_samples.append(vram_mb)
                writer.emit(
                    "resource_sample",
                    workload_id=workload_id,
                    stage=stage,
                    ram_mb=ram_mb,
                    vram_mb=vram_mb,
                )

            def emit_event(event, **fields):
                nonlocal first_byte_latency
                nonlocal first_text_latency
                nonlocal workload_model_calls
                nonlocal failed_tool_calls
                nonlocal failed_json_tool_calls
                emitted = writer.emit(event, workload_id=workload_id, **fields)
                if event == "first_byte" and first_byte_latency is None:
                    first_byte_latency = (emitted["timestamp"] - request_started_at) * 1000
                if event == "first_text" and first_text_latency is None:
                    first_text_latency = (emitted["timestamp"] - request_started_at) * 1000
                if event == "tool_call":
                    latency = _number(fields.get("latency_ms"))
                    if latency is not None:
                        tool_latencies.append(latency)
                    if fields.get("failed"):
                        failed_tool_calls += 1
                if event == "failed_json_tool_call":
                    failed_json_tool_calls += 1
                if event == "model_call":
                    workload_model_calls += 1
                return emitted

            sample_resources("before")
            request_started_at = clock()
            result = request_func(endpoint, workload["request"], emit_event) or {}
            finished_at = clock()
            sample_resources("after")

            total_time_ms = (finished_at - request_started_at) * 1000
            total_times.append(total_time_ms)
            first_byte_latencies.append(first_byte_latency)
            first_text_latencies.append(first_text_latency)
            explicit_model_calls = _number(result.get("model_calls"))
            if explicit_model_calls is not None and explicit_model_calls >= 0:
                model_calls += int(explicit_model_calls)
            else:
                model_calls += max(1, workload_model_calls)

            usage = result.get("usage", {})
            workload_prompt_tokens = _usage_value(
                usage, "prompt_tokens", "input_tokens", "tokens_evaluated"
            )
            workload_completion_tokens = _usage_value(
                usage, "completion_tokens", "output_tokens", "tokens_predicted"
            )
            if workload_prompt_tokens is not None:
                prompt_tokens += workload_prompt_tokens
            if workload_completion_tokens is not None:
                completion_tokens += workload_completion_tokens
            explicit_rate = _usage_value(usage, "tokens_per_second", "tok_per_s")
            if explicit_rate is not None:
                token_rates.append(explicit_rate)
            elif workload_completion_tokens is not None and total_time_ms > 0:
                token_rates.append(workload_completion_tokens / (total_time_ms / 1000))

            context_length = _number(result.get("context_length"))
            if context_length is not None:
                context_lengths.append(context_length)
            cache_hit = result.get("cache_hit")
            if isinstance(cache_hit, bool):
                cache_hits.append(cache_hit)
            cache_hit_rate = _number(result.get("cache_hit_rate"))
            if cache_hit_rate is not None:
                cache_hit_rates.append(cache_hit_rate)

            hallucinated_tool_calls += int(result.get("hallucinated_tool_calls", 0) or 0)
            failed_json_tool_calls += int(result.get("failed_json_tool_calls", 0) or 0)
            failed_tool_calls += int(result.get("failed_tool_calls", 0) or 0)
            catalog_drift = result.get("catalog_drift")
            if isinstance(catalog_drift, bool):
                catalog_drift_results.append(catalog_drift)
            syntax_preflight_pass_rate = _number(result.get("syntax_preflight_pass_rate"))
            if syntax_preflight_pass_rate is not None:
                syntax_preflight_pass_rates.append(syntax_preflight_pass_rate)
            hallucinated_tool_claim_rate = _number(result.get("hallucinated_tool_claim_rate"))
            if hallucinated_tool_claim_rate is not None:
                hallucinated_tool_claim_rates.append(hallucinated_tool_claim_rate)
            raw_untrusted_bytes = _number(result.get("raw_untrusted_bytes_in_prompt"))
            if raw_untrusted_bytes is not None:
                raw_untrusted_bytes_in_prompt += raw_untrusted_bytes
                raw_untrusted_bytes_seen = True
            failed_json_tool_call_rate = _number(result.get("failed_json_tool_call_rate"))
            if failed_json_tool_call_rate is not None:
                failed_json_tool_call_rates.append(failed_json_tool_call_rate)

            text = result.get("text", "")
            expected_exact = workload.get("expected_exact")
            expected_contains = workload.get("expected_contains")
            task_success = None
            if isinstance(expected_exact, str):
                task_success = text.strip() == expected_exact.strip()
                task_results.append(task_success)
            elif isinstance(expected_contains, str):
                task_success = expected_contains in text
                task_results.append(task_success)
            explicit_test_pass_rate = _number(result.get("test_pass_rate"))
            if explicit_test_pass_rate is not None:
                test_results.append(explicit_test_pass_rate)

            writer.emit(
                "workload_result",
                workload_id=workload_id,
                first_byte_latency_ms=first_byte_latency,
                first_text_latency_ms=first_text_latency,
                total_time_ms=total_time_ms,
                prompt_tokens=workload_prompt_tokens,
                completion_tokens=workload_completion_tokens,
                task_success=task_success,
            )

        run_finished_at = clock()
        summary = {
            "run_label": run_label,
            "phase": phase,
            "endpoint": endpoint,
            "workload_path": str(workload_path),
            "workload_count": len(workloads),
            "started_at": run_started_at,
            "finished_at": run_finished_at,
            "run_elapsed_ms": (run_finished_at - run_started_at) * 1000,
            "first_byte_latency_ms": _average(first_byte_latencies),
            "first_text_latency_ms": _average(first_text_latencies),
            "total_time_ms": sum(total_times),
            "tokens_per_second": _average(token_rates),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "tool_latency_ms": _average(tool_latencies),
            "model_calls": model_calls,
            "peak_ram_mb": _max_number(ram_samples),
            "peak_vram_mb": _max_number(vram_samples),
            "context_length_used": _max_number(context_lengths),
            "cache_hit": any(cache_hits) if cache_hits else None,
            "cache_hit_rate": _average(cache_hit_rates),
            "test_pass_rate": _average(test_results),
            "task_success_rate": _average([1.0 if item else 0.0 for item in task_results]),
            "hallucinated_tool_calls": hallucinated_tool_calls,
            "failed_json_tool_calls": failed_json_tool_calls,
            "failed_tool_calls": failed_tool_calls,
            "events_path": str(events_path),
            "summary_path": str(summary_path),
        }
        if catalog_drift_results:
            summary["catalog_drift"] = any(catalog_drift_results)
        if syntax_preflight_pass_rates:
            summary["syntax_preflight_pass_rate"] = _average(syntax_preflight_pass_rates)
        if hallucinated_tool_claim_rates:
            summary["hallucinated_tool_claim_rate"] = _average(hallucinated_tool_claim_rates)
        if raw_untrusted_bytes_seen:
            summary["raw_untrusted_bytes_in_prompt"] = int(raw_untrusted_bytes_in_prompt)
        if failed_json_tool_call_rates:
            summary["failed_json_tool_call_rate"] = _average(failed_json_tool_call_rates)
        writer.emit("run_summary", **summary)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_markdown_table(markdown_path, summary)
    return summary


def _format_cell(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def markdown_row(summary):
    cells = [
        summary.get("run_label"),
        summary.get("phase"),
        summary.get("workload_count"),
        summary.get("first_byte_latency_ms"),
        summary.get("first_text_latency_ms"),
        summary.get("total_time_ms"),
        summary.get("tokens_per_second"),
        summary.get("prompt_tokens"),
        summary.get("completion_tokens"),
        summary.get("model_calls"),
        summary.get("peak_ram_mb"),
        summary.get("peak_vram_mb"),
        summary.get("context_length_used"),
        summary.get("cache_hit"),
        summary.get("cache_hit_rate"),
        summary.get("test_pass_rate"),
        summary.get("task_success_rate"),
        summary.get("hallucinated_tool_calls"),
        summary.get("failed_json_tool_calls"),
        summary.get("failed_tool_calls"),
        summary.get("summary_path"),
    ]
    return "| " + " | ".join(_format_cell(cell) for cell in cells) + " |"


def update_markdown_table(markdown_path, summary):
    markdown_path = Path(markdown_path)
    row = markdown_row(summary)
    if not markdown_path.exists():
        markdown_path.write_text(
            "# Gemma Runtime Benchmarks\n\n" + MARKDOWN_HEADER + "\n" + MARKDOWN_SEPARATOR + "\n" + row + "\n",
            encoding="utf-8",
        )
        return

    content = markdown_path.read_text(encoding="utf-8")
    if MARKDOWN_HEADER not in content:
        if content and not content.endswith("\n"):
            content += "\n"
        content += "\n" + MARKDOWN_HEADER + "\n" + MARKDOWN_SEPARATOR + "\n" + row + "\n"
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        content += row + "\n"
    markdown_path.write_text(content, encoding="utf-8")


def safe_label(label):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip())
    return cleaned.strip("-") or "benchmark"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run local Gemma benchmark workloads.")
    parser.add_argument("--workload", type=Path, default=Path("benchmarks/workloads/smoke.jsonl"))
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--phase", choices=("before", "after", "baseline", "candidate"), default="before")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/runs"))
    parser.add_argument("--markdown", type=Path, default=Path("benchmarks/BENCHMARKS.md"))
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_label = args.run_label or f"{args.phase}-{time.strftime('%Y%m%d-%H%M%S')}"
    label = safe_label(run_label)
    events_path = args.out_dir / f"{label}.events.jsonl"
    summary_path = args.out_dir / f"{label}.summary.json"

    def http_request(endpoint, payload, emit_event):
        payload = dict(payload)
        payload.setdefault("model", args.model)
        return make_http_request(endpoint, payload, emit_event, timeout=args.timeout)

    summary = run_benchmark(
        workload_path=args.workload,
        endpoint=args.endpoint,
        phase=args.phase,
        run_label=run_label,
        events_path=events_path,
        summary_path=summary_path,
        markdown_path=args.markdown,
        request_func=http_request,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
