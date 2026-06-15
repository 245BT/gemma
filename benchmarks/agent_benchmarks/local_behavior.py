from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from gemma_agent import AgentSupervisor, ContextBuilder, MemoryStore, SafetyGuard, TaskEffortPolicy, ToolExecutor, ToolRegistry
from gemma_agent.terminal import TerminalCommandRunner
from .taxonomy import normalize_failure_mode


REQUIRED_RESEARCH_ARTIFACTS = [
    {
        "pdf": "composer2-user-provided.pdf",
        "text": "composer2-user-provided.txt",
        "markers": ["CursorBench", "Self-Summarization"],
    },
    {
        "pdf": "composer-2-technical-report.pdf",
        "text": "composer-2-technical-report.txt",
        "markers": ["CursorBench", "Self-Summarization"],
    },
    {
        "pdf": "swe-agent-agent-computer-interfaces.pdf",
        "text": "swe-agent-agent-computer-interfaces.txt",
        "markers": ["SWE-agent", "Agent-Computer Interface"],
    },
    {
        "pdf": "swe-bench-real-world-github-issues.pdf",
        "text": "swe-bench-real-world-github-issues.txt",
        "markers": ["SWE-bench"],
    },
    {
        "pdf": "terminal-bench-2.pdf",
        "text": "terminal-bench-2.txt",
        "markers": ["Terminal-Bench"],
    },
    {
        "pdf": "agentdojo-prompt-injection-agents.pdf",
        "text": "agentdojo-prompt-injection-agents.txt",
        "markers": ["AgentDojo"],
    },
    {
        "pdf": "swe-evo-long-horizon-software-evolution.pdf",
        "text": "swe-evo-long-horizon-software-evolution.txt",
        "markers": ["SWE-EVO"],
    },
]


STRICT_TEXT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


class StaticModelClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    def create_response(self, payload: dict[str, Any]) -> str:
        self.payloads.append(payload)
        if not self.responses:
            return json.dumps({"action": "final", "content": "out of scripted responses"})
        return self.responses.pop(0)


def run_local_behavior_benchmark(
    root: str | Path,
    *,
    run_id: str = "local-agent-behavior",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    output_root = Path(output_dir) if output_dir is not None else root / "benchmarks" / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    safe_run_id = _safe_run_id(run_id)
    results = [_run_case(case) for case in _cases(root)]
    passed = sum(1 for result in results if result["success"])
    category_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    for result in results:
        category_counts[result["category"]] = category_counts.get(result["category"], 0) + 1
        if not result["success"]:
            failure_mode = str(result.get("failure_mode") or "unknown")
            failure_counts[failure_mode] = failure_counts.get(failure_mode, 0) + 1

    summary_path = output_root / f"{safe_run_id}.local-agent-behavior.summary.json"
    summary = {
        "suite": "local-agent-behavior",
        "run_id": run_id,
        "total_count": len(results),
        "resolved_count": passed,
        "pass_rate": passed / len(results) if results else None,
        "category_counts": category_counts,
        "failure_counts": failure_counts,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "results": results,
        "summary_path": str(summary_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _safe_run_id(run_id: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "._-" else "-" for character in run_id)
    return cleaned.strip("._-") or "local-agent-behavior"


def _cases(root: Path) -> list[dict[str, Any]]:
    return [
        _case("terminal_idle_timeout", "stall_recovery", lambda: _terminal_idle_timeout()),
        _case("terminal_hard_timeout", "terminal_execution", lambda: _terminal_hard_timeout()),
        _case("effort_policy_budgeting", "efficiency_budgeting", lambda: _effort_policy_budgeting()),
        _case("repeated_tool_recovery", "tool_use_reliability", lambda: _repeated_tool_recovery()),
        _case("invalid_json_recovery", "planning_recovery", lambda: _invalid_json_recovery()),
        _case("large_file_context_budget", "large_file_handling", lambda: _large_file_context_budget()),
        _case("large_repo_retrieval", "large_repo_navigation", lambda: _large_repo_retrieval(root)),
        _case(
            "research_artifacts_available",
            "research_citation",
            lambda: _research_artifacts_available(root),
            failure_mode="research_artifact_missing",
        ),
        _case("schema_rejects_bad_tool_args", "tool_use_reliability", lambda: _schema_rejects_bad_tool_args()),
    ]


def _case(
    name: str,
    category: str,
    fn: Callable[[], bool],
    *,
    failure_mode: str = "assertion_failed",
) -> dict[str, Any]:
    return {"name": name, "category": category, "fn": fn, "failure_mode": failure_mode}


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        success = bool(case["fn"]())
        failure_mode = None if success else normalize_failure_mode(case.get("failure_mode"))
    except Exception as exc:
        success = False
        failure_mode = normalize_failure_mode(exc.__class__.__name__)
    return {
        "name": case["name"],
        "category": case["category"],
        "success": success,
        "failure_mode": failure_mode,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _terminal_idle_timeout() -> bool:
    runner = TerminalCommandRunner(
        default_timeout_sec=5,
        default_idle_timeout_sec=0.2,
        poll_interval_sec=0.02,
    )
    result = None
    try:
        result = runner.run(
            [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(5)"]
        )
        return (
            not result.ok
            and result.status == "stalled"
            and result.elapsed_ms < 2000
            and result.recovery.get("reason") == "idle_timeout"
        )
    finally:
        if result is not None and result.job_id:
            runner.kill_job(result.job_id, reason="benchmark idle-timeout cleanup")


def _terminal_hard_timeout() -> bool:
    runner = TerminalCommandRunner(
        default_timeout_sec=0.2,
        default_idle_timeout_sec=5,
        poll_interval_sec=0.02,
    )
    result = runner.run([sys.executable, "-c", "import time; time.sleep(5)"])
    return not result.ok and result.status == "timeout" and result.elapsed_ms < 2000


def _effort_policy_budgeting() -> bool:
    policy = TaskEffortPolicy()
    easy = policy.plan_for_task("Reply with only OK.")
    hard = policy.plan_for_task("Research, debug, benchmark, and verify a large repository fix.")
    return (
        easy.difficulty == "easy"
        and hard.difficulty == "hard"
        and easy.max_tool_calls < hard.max_tool_calls
        and policy.cost_score(turns=4, tool_calls=4, tool_output_tokens=400)
        > policy.cost_score(turns=1, tool_calls=1, tool_output_tokens=100)
    )


def _repeated_tool_recovery() -> bool:
    registry = ToolRegistry()
    registry.register("echo", lambda text: text, STRICT_TEXT_SCHEMA, execution_mode="thread")
    repeated_action = json.dumps({"action": "tool_call", "tool": "echo", "args": {"text": "same"}})
    model = StaticModelClient(
        [
            repeated_action,
            repeated_action,
            json.dumps({"action": "final", "content": "done"}),
        ]
    )
    result = AgentSupervisor(model, ToolExecutor(registry), max_iterations=3).run("avoid low progress")
    events = model.payloads[-1]["progress"]["recovery_events"]
    return result.ok and any(event["event"] == "repeated_tool_call" for event in events)


def _invalid_json_recovery() -> bool:
    model = StaticModelClient(
        [
            "not-json",
            "not-json",
            json.dumps({"action": "final", "content": "done"}),
        ]
    )
    result = AgentSupervisor(model, ToolExecutor(ToolRegistry()), max_iterations=3).run("recover schema")
    events = model.payloads[-1]["progress"]["recovery_events"]
    return result.ok and any(event["event"] == "repeated_invalid_action" for event in events)


def _large_file_context_budget() -> bool:
    memory = MemoryStore()
    memory.add("project_map", "irrelevant " * 1000, source="large")
    memory.add("project_map", "stall recovery lives in gemma_agent/progress.py", source="small")
    context = ContextBuilder(
        memory,
        max_entries_per_tier={"project_map": 1},
        max_chars_per_tier={"project_map": 80},
    ).build(task="find stall recovery implementation")
    return context["project_map"] == ["stall recovery lives in gemma_agent/progress.py"]


def _large_repo_retrieval(root: Path) -> bool:
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        repo = Path(tmp)
        for index in range(120):
            (repo / f"file_{index:03d}.py").write_text(f"# filler {index}\n", encoding="utf-8")
        for index in range(10):
            (repo / f"progress_decoy_{index:03d}.py").write_text(
                "progress unrelated launcher note\n" + ("noise " * 100),
                encoding="utf-8",
            )
        target = repo / "supervisor_progress.py"
        target.write_text(
            "class ProgressTracker:\n    recovery = 'terminal stall recovery'\n",
            encoding="utf-8",
        )
        memory = MemoryStore()
        for path in repo.rglob("*.py"):
            memory.add("project_map", f"{path.name}: {path.read_text(encoding='utf-8')}", source=str(path))
        context = ContextBuilder(
            memory,
            max_entries_per_tier={"project_map": 1},
            max_chars_per_tier={"project_map": 200},
        ).build(task="find ProgressTracker terminal stall recovery")
        return bool(context["project_map"]) and "supervisor_progress.py" in str(context["project_map"][0])


def _research_artifacts_available(root: Path) -> bool:
    research_dir = root / "docs" / "research"
    for artifact in REQUIRED_RESEARCH_ARTIFACTS:
        research_pdf = research_dir / artifact["pdf"]
        research_text = research_dir / artifact["text"]
        if not research_pdf.exists() or research_pdf.stat().st_size < 1000 or not research_text.exists():
            return False
        text = research_text.read_text(encoding="utf-8")
        if not all(marker in text for marker in artifact["markers"]):
            return False
    return True


def _schema_rejects_bad_tool_args() -> bool:
    calls = []
    registry = ToolRegistry()
    registry.register(
        "echo",
        lambda text: calls.append(text) or text,
        STRICT_TEXT_SCHEMA,
        execution_mode="thread",
    )
    result = ToolExecutor(registry, safety_guard=SafetyGuard([Path.cwd()])).execute(
        "echo",
        {"bad": "value"},
    )
    return not result.ok and not result.executed and calls == []


def parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Run deterministic local Gemma agent behavior checks.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="local-agent-behavior")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_local_behavior_benchmark(args.root, run_id=args.run_id, output_dir=args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("pass_rate") == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
