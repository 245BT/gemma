import argparse
import glob
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import setup_local_codex


NUMERIC_FIELDS = [
    "first_byte_latency_ms",
    "first_text_latency_ms",
    "total_time_ms",
    "tokens_per_second",
    "tool_latency_ms",
    "model_calls",
    "prompt_tokens",
    "completion_tokens",
    "peak_ram_mb",
    "peak_vram_mb",
    "context_length_used",
    "cache_hit_rate",
    "test_pass_rate",
    "task_success_rate",
    "failed_json_tool_calls",
    "failed_tool_calls",
    "hallucinated_tool_calls",
]


def token_count(text):
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        words = re.findall(r"\S+", text)
        return math.ceil(len(words) * 1.25)


def text_metrics(text):
    return {
        "chars": len(text),
        "words": len(re.findall(r"\S+", text)),
        "tokens": token_count(text),
    }


def strip_repeat_suffix(label):
    return re.sub(r"-r\d+$", "", label)


def numeric(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def summarize_values(values):
    values = [value for value in values if numeric(value) is not None]
    if not values:
        return {}
    result = {
        "avg": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }
    result["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result


def summarize_runs(paths):
    groups = {}
    for path in sorted(paths):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        label = strip_repeat_suffix(payload["run_label"])
        groups.setdefault(label, []).append((str(path), payload))

    benchmarks = []
    for label, items in sorted(groups.items()):
        benchmark = {"label": label, "n": len(items)}
        for field in NUMERIC_FIELDS:
            stats = summarize_values([payload.get(field) for _, payload in items])
            for key, value in stats.items():
                benchmark[f"{field}_{key}"] = value
        benchmark["files"] = [path for path, _ in items]
        benchmarks.append(benchmark)
    return benchmarks


def find_base_instruction_texts(value):
    if isinstance(value, dict):
        base = value.get("base_instructions")
        if isinstance(base, str):
            yield base
        elif isinstance(base, dict) and isinstance(base.get("text"), str):
            yield base["text"]
        for item in value.values():
            yield from find_base_instruction_texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from find_base_instruction_texts(item)


def legacy_prompt_candidates(pattern, exclude_texts=None):
    if not pattern:
        return []
    exclude_texts = set(exclude_texts or [])
    candidates = {}
    for path in sorted(glob.glob(pattern, recursive=True)):
        session_path = Path(path)
        try:
            lines = session_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            for text in find_base_instruction_texts(payload):
                if "Codex running locally on Gemma" not in text:
                    continue
                if text in exclude_texts:
                    continue
                key = (str(session_path), text)
                candidates[key] = {
                    "path": str(session_path),
                    **text_metrics(text),
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
    return sorted(candidates.values(), key=lambda item: item["tokens"], reverse=True)


def add_comparison(output, before_label, after_label, name):
    by_label = {item["label"]: item for item in output["benchmarks"]}
    before = by_label.get(before_label)
    after = by_label.get(after_label)
    if not before or not after:
        return
    before_ms = before.get("total_time_ms_avg")
    after_ms = after.get("total_time_ms_avg")
    if numeric(before_ms) is None or numeric(after_ms) is None or before_ms == 0:
        return
    delta_ms = after_ms - before_ms
    output.setdefault("comparisons", {})[name] = {
        "before_label": before_label,
        "after_label": after_label,
        "before_total_ms_avg": before_ms,
        "after_total_ms_avg": after_ms,
        "delta_ms": delta_ms,
        "reduction_percent": ((before_ms - after_ms) / before_ms) * 100,
        "speedup": before_ms / after_ms if after_ms else None,
    }


def build_output(summary_glob, session_glob):
    paths = glob.glob(summary_glob)
    if not paths:
        raise SystemExit(f"No summary files matched {summary_glob}")

    current_prompt = setup_local_codex.build_base_instructions()
    output = {
        "benchmarks": summarize_runs(paths),
        "prompt_summary": {
            "current": text_metrics(current_prompt),
            "legacy_candidates": legacy_prompt_candidates(session_glob, exclude_texts={current_prompt}),
        },
    }
    add_comparison(
        output,
        "remeasure-reasoning-graph-forced-8082",
        "remeasure-reasoning-fastpath-8082",
        "current_code_forced_graph_to_fast_path",
    )
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate repeated Gemma benchmark summaries.")
    parser.add_argument(
        "--summary-glob",
        default="benchmarks/runs/remeasure-*-r*.summary.json",
        help="Glob for per-run summary JSON files.",
    )
    parser.add_argument(
        "--session-glob",
        default="",
        help="Optional glob for local session JSONL files used to find legacy prompt baselines.",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/runs/remeasure-2026-06-12.aggregate.json",
        help="Aggregate JSON output path.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = build_output(args.summary_glob, args.session_glob)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
