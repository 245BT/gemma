from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

from .subprocesses import output_metadata, run_bounded_subprocess, write_redacted_output
from .taxonomy import normalize_failure_mode


@dataclass(frozen=True)
class LocalRepoTask:
    name: str
    category: str
    instruction: str
    files: dict[str, str]
    oracle: Callable[[Path], None]


def run_local_repo_fix_benchmark(
    root: str | Path,
    *,
    run_id: str = "local-repo-fix",
    output_dir: str | Path | None = None,
    solver: str = "oracle",
    timeout: float = 20,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output_root = Path(output_dir) if output_dir is not None else root / "benchmarks" / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    safe_run_id = _safe_run_id(run_id)

    tasks = _tasks()
    if max_tasks is not None:
        tasks = tasks[: max(0, int(max_tasks))]
    artifact_root = output_root / f"{safe_run_id}.local-repo-fix-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    results = [
        _run_task(task, root=root, artifact_root=artifact_root, solver=solver, timeout=timeout)
        for task in tasks
    ]
    resolved = sum(1 for result in results if result["success"])
    fail_to_pass = sum(1 for result in results if result["fail_to_pass_passed"])
    pass_to_pass = sum(1 for result in results if result["pass_to_pass_passed"])
    regressions = sum(int(result["regression_count"]) for result in results)
    category_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    for result in results:
        category_counts[result["category"]] = category_counts.get(result["category"], 0) + 1
        if not result["success"]:
            failure_mode = str(result.get("failure_mode") or "unknown_failure")
            failure_counts[failure_mode] = failure_counts.get(failure_mode, 0) + 1

    summary_path = output_root / f"{safe_run_id}.local-repo-fix.summary.json"
    summary = {
        "suite": "local-repo-fix",
        "run_id": run_id,
        "solver": solver,
        "total_count": len(results),
        "resolved_count": resolved,
        "pass_rate": resolved / len(results) if results else None,
        "fail_to_pass_rate": fail_to_pass / len(results) if results else None,
        "pass_to_pass_rate": pass_to_pass / len(results) if results else None,
        "patch_apply_rate": sum(1 for result in results if result["patch_applied"]) / len(results) if results else None,
        "regression_count": regressions,
        "category_counts": category_counts,
        "failure_counts": dict(sorted(failure_counts.items())),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "results": results,
        "summary_path": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _safe_run_id(run_id: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "._-" else "-" for character in run_id)
    return cleaned.strip("._-") or "local-repo-fix"


def _run_task(
    task: LocalRepoTask,
    *,
    root: Path,
    artifact_root: Path,
    solver: str,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_root = Path(tempfile.mkdtemp(prefix=f"{task.name}-", dir=root))
    task_artifact_dir = artifact_root / task.name
    task_artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        _write_files(task_root, task.files)
        before = _snapshot(task_root)
        baseline = _run_tests(task_root, timeout=timeout)
        patch_applied = False
        failure_mode = None
        solver_result = {"status": "not_run", "return_code": None, "stdout": "", "stderr": ""}

        if solver == "oracle":
            task.oracle(task_root)
            solver_result = {"status": "completed", "return_code": 0, "stdout": "", "stderr": ""}
        elif solver == "none":
            solver_result = {"status": "not_run", "return_code": None, "stdout": "", "stderr": ""}
        elif solver == "gemma":
            solver_result = _run_gemma_solver(root=root, task_root=task_root, task=task, timeout=timeout)
        else:
            raise ValueError(f"unsupported solver {solver!r}; expected oracle, gemma, or none")

        after = _snapshot(task_root)
        test_result = _run_tests(task_root, timeout=timeout)
        diff_stats = _diff_stats(before, after)
        diff_path = task_artifact_dir / "patch.diff"
        diff_path.write_text(_diff_text(before, after), encoding="utf-8")
        write_redacted_output(task_artifact_dir / "solver.stdout.log", solver_result["stdout"])
        write_redacted_output(task_artifact_dir / "solver.stderr.log", solver_result["stderr"])
        solver_metadata = output_metadata(solver_result["stdout"], solver_result["stderr"])
        patch_applied = diff_stats["changed_files"] > 0
        fail_to_pass_passed = _unittest_label_passed(test_result, "FAIL_TO_PASS")
        pass_to_pass_passed = _unittest_label_passed(test_result, "PASS_TO_PASS")
        regression_count = 0 if pass_to_pass_passed else 1
        test_pass_rate = _unittest_pass_rate(test_result)
        success = (
            patch_applied
            and baseline["return_code"] != 0
            and fail_to_pass_passed
            and pass_to_pass_passed
            and test_result["return_code"] == 0
        )
        if not success:
            failure_mode = normalize_failure_mode(
                "patch_failed" if not patch_applied else "test_failed"
            )
        return {
            "name": task.name,
            "category": task.category,
            "success": success,
            "failure_mode": failure_mode,
            "patch_applied": patch_applied,
            "solver_status": solver_result["status"],
            "solver_return_code": solver_result["return_code"],
            "baseline_failed": baseline["return_code"] != 0,
            "fail_to_pass_passed": fail_to_pass_passed,
            "pass_to_pass_passed": pass_to_pass_passed,
            "regression_count": regression_count,
            "test_pass_rate": test_pass_rate,
            "changed_files": diff_stats["changed_files"],
            "changed_lines": diff_stats["changed_lines"],
            "diff_path": str(diff_path),
            "solver_stdout_path": str(task_artifact_dir / "solver.stdout.log"),
            "solver_stderr_path": str(task_artifact_dir / "solver.stderr.log"),
            "solver_stdout_tail": solver_metadata["stdout_tail"],
            "solver_stdout_length": solver_metadata["stdout_length"],
            "solver_stdout_sha256": solver_metadata["stdout_sha256"],
            "solver_stderr_tail": solver_metadata["stderr_tail"],
            "solver_stderr_length": solver_metadata["stderr_length"],
            "solver_stderr_sha256": solver_metadata["stderr_sha256"],
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "stdout_length": len(test_result["stdout"]),
            "stderr_length": len(test_result["stderr"]),
        }
    except Exception as exc:
        return {
            "name": task.name,
            "category": task.category,
            "success": False,
            "failure_mode": normalize_failure_mode(exc.__class__.__name__),
            "patch_applied": False,
            "solver_status": "exception",
            "solver_return_code": None,
            "baseline_failed": False,
            "fail_to_pass_passed": False,
            "pass_to_pass_passed": False,
            "regression_count": 1,
            "test_pass_rate": 0.0,
            "changed_files": 0,
            "changed_lines": 0,
            "diff_path": "",
            "solver_stdout_path": "",
            "solver_stderr_path": "",
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "stdout_length": 0,
            "stderr_length": 0,
        }
    finally:
        shutil.rmtree(task_root, ignore_errors=True)


def _run_gemma_solver(*, root: Path, task_root: Path, task: LocalRepoTask, timeout: float) -> dict[str, Any]:
    launcher = root / "gemma-codex.cmd"
    prompt = (
        f"{task.instruction}\n\n"
        "You are in a generated benchmark repository. Edit the implementation files only. "
        "Run `python -m unittest discover -v` before finishing. "
        "Return a concise summary of changed files and test results."
    )
    env = dict(os.environ)
    env["GEMMA_CODEX_TARGET_DIR"] = str(task_root)
    env["GEMMA_CODEX_EXEC_TIMEOUT"] = str(timeout)
    command = [
        str(launcher),
        "--reasoning",
        "--yolo",
        "exec",
        "--skip-git-repo-check",
        prompt,
    ]
    completed = run_bounded_subprocess(
        command,
        cwd=task_root,
        env=env,
        timeout=timeout + 30,
    )
    if completed.timed_out or completed.return_code == 124:
        status = "timeout"
    elif completed.return_code == 0:
        status = "completed"
    else:
        status = "failed"
    return {
        "status": status,
        "return_code": completed.return_code,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


def _tasks() -> list[LocalRepoTask]:
    return [
        LocalRepoTask(
            name="nested_response_output_text",
            category="real_world_coding",
            instruction="Normalize Responses API payloads that only contain nested output text.",
            files={
                "response_utils.py": (
                    "def output_text(payload):\n"
                    "    return payload.get('output_text', '')\n"
                ),
                "test_response_utils.py": (
                    "import unittest\n"
                    "from response_utils import output_text\n\n"
                    "class ResponseUtilsTests(unittest.TestCase):\n"
                    "    def test_FAIL_TO_PASS_nested_output_text(self):\n"
                    "        payload = {'output': [{'content': [{'type': 'output_text', 'text': 'OK'}]}]}\n"
                    "        self.assertEqual(output_text(payload), 'OK')\n\n"
                    "    def test_PASS_TO_PASS_top_level_output_text(self):\n"
                    "        self.assertEqual(output_text({'output_text': 'ready'}), 'ready')\n\n"
                    "if __name__ == '__main__':\n"
                    "    unittest.main()\n"
                ),
            },
            oracle=_oracle_nested_response_output_text,
        ),
        LocalRepoTask(
            name="terminal_tail_truncation",
            category="terminal_execution",
            instruction="Keep terminal output tail bounded to the newest text.",
            files={
                "terminal_utils.py": (
                    "def tail(text, max_chars):\n"
                    "    return text[:max_chars]\n"
                ),
                "test_terminal_utils.py": (
                    "import unittest\n"
                    "from terminal_utils import tail\n\n"
                    "class TerminalUtilsTests(unittest.TestCase):\n"
                    "    def test_FAIL_TO_PASS_keeps_newest_tail(self):\n"
                    "        self.assertEqual(tail('abcdef', 3), 'def')\n\n"
                    "    def test_PASS_TO_PASS_short_text_unchanged(self):\n"
                    "        self.assertEqual(tail('ab', 3), 'ab')\n\n"
                    "if __name__ == '__main__':\n"
                    "    unittest.main()\n"
                ),
            },
            oracle=_oracle_terminal_tail_truncation,
        ),
        LocalRepoTask(
            name="context_relevance_ranking",
            category="large_repo_navigation",
            instruction="Prefer relevant context entries over earlier irrelevant entries.",
            files={
                "ranker.py": (
                    "def choose(task, entries):\n"
                    "    return entries[0] if entries else ''\n"
                ),
                "test_ranker.py": (
                    "import unittest\n"
                    "from ranker import choose\n\n"
                    "class RankerTests(unittest.TestCase):\n"
                    "    def test_FAIL_TO_PASS_prefers_matching_entry(self):\n"
                    "        entries = ['unrelated launcher note', 'ProgressTracker terminal stall recovery']\n"
                    "        self.assertEqual(choose('terminal stall recovery', entries), entries[1])\n\n"
                    "    def test_PASS_TO_PASS_empty_entries(self):\n"
                    "        self.assertEqual(choose('anything', []), '')\n\n"
                    "if __name__ == '__main__':\n"
                    "    unittest.main()\n"
                ),
            },
            oracle=_oracle_context_relevance_ranking,
        ),
    ]


def _oracle_nested_response_output_text(repo: Path) -> None:
    (repo / "response_utils.py").write_text(
        "def output_text(payload):\n"
        "    if isinstance(payload.get('output_text'), str):\n"
        "        return payload['output_text']\n"
        "    parts = []\n"
        "    for item in payload.get('output', []):\n"
        "        for content in item.get('content', []):\n"
        "            text = content.get('text')\n"
        "            if isinstance(text, str):\n"
        "                parts.append(text)\n"
        "    return '\\n'.join(parts)\n",
        encoding="utf-8",
    )


def _oracle_terminal_tail_truncation(repo: Path) -> None:
    (repo / "terminal_utils.py").write_text(
        "def tail(text, max_chars):\n"
        "    if len(text) <= max_chars:\n"
        "        return text\n"
        "    return text[-max_chars:]\n",
        encoding="utf-8",
    )


def _oracle_context_relevance_ranking(repo: Path) -> None:
    (repo / "ranker.py").write_text(
        "def choose(task, entries):\n"
        "    if not entries:\n"
        "        return ''\n"
        "    tokens = {part.lower() for part in task.split() if len(part) > 2}\n"
        "    return max(entries, key=lambda entry: len(tokens.intersection(entry.lower().split())))\n",
        encoding="utf-8",
    )


def _write_files(repo: Path, files: dict[str, str]) -> None:
    for relative_path, content in files.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _snapshot(repo: Path) -> dict[str, str]:
    return {
        str(path.relative_to(repo)): path.read_text(encoding="utf-8")
        for path in sorted(repo.rglob("*.py"))
    }


def _run_tests(repo: Path, *, timeout: float) -> dict[str, Any]:
    completed = run_bounded_subprocess(
        [sys.executable, "-m", "unittest", "discover", "-v"],
        cwd=repo,
        timeout=timeout,
    )
    return {
        "return_code": completed.return_code,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _unittest_label_passed(result: dict[str, Any], label: str) -> bool:
    marker = f"test_{label}_"
    for line in str(result.get("stderr", "")).splitlines():
        if marker in line:
            return line.rstrip().endswith("ok")
    return False


def _unittest_pass_rate(result: dict[str, Any]) -> float:
    test_lines = [
        line
        for line in str(result.get("stderr", "")).splitlines()
        if line.startswith("test_") and " ... " in line
    ]
    if not test_lines:
        return 0.0
    passed = sum(1 for line in test_lines if line.rstrip().endswith("ok"))
    return passed / len(test_lines)


def _diff_stats(before: dict[str, str], after: dict[str, str]) -> dict[str, int]:
    changed_files = 0
    changed_lines = 0
    for path in sorted(set(before).union(after)):
        old = before.get(path, "").splitlines()
        new = after.get(path, "").splitlines()
        if old == new:
            continue
        changed_files += 1
        for line in difflib.unified_diff(old, new, lineterm=""):
            if line.startswith(("+++", "---", "@@")):
                continue
            if line.startswith(("+", "-")):
                changed_lines += 1
    return {"changed_files": changed_files, "changed_lines": changed_lines}


def _diff_text(before: dict[str, str], after: dict[str, str]) -> str:
    lines: list[str] = []
    for path in sorted(set(before).union(after)):
        old = before.get(path, "").splitlines()
        new = after.get(path, "").splitlines()
        if old == new:
            continue
        lines.extend(
            difflib.unified_diff(
                old,
                new,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run generated local repo-fix checks.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="local-repo-fix")
    parser.add_argument("--solver", choices=("oracle", "none", "gemma"), default="oracle")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_local_repo_fix_benchmark(
        args.root,
        run_id=args.run_id,
        output_dir=args.out_dir,
        solver=args.solver,
        timeout=args.timeout,
        max_tasks=args.max_tasks,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("pass_rate") == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
