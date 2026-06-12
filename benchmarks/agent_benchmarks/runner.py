from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .registry import get_suite, list_suites
from .results import normalize_result_file
from .suites import codex_home, mode_args, model_slug, yolo_args


DEFAULT_TIMEOUT_SECONDS = 3600


@dataclass(frozen=True)
class BenchmarkRequest:
    suite: str
    mode: str = "direct"
    yolo: bool = False
    run_id: str = "agent-benchmark"
    predictions_path: Path | None = None
    manifest_path: Path | None = None
    result_path: Path | None = None
    dataset: str | None = None
    n_concurrent: int = 1
    out_dir: Path = Path("benchmarks/runs")
    dry_run: bool = False
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    extra_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExecutionPlan:
    command: list[str]
    env: dict[str, str]
    summary_path: Path
    stdout_path: Path
    stderr_path: Path


def build_execution_plan(root: str | Path, request: BenchmarkRequest) -> ExecutionPlan:
    root = Path(root)
    suite = get_suite(request.suite)
    dataset = request.dataset or suite.default_dataset
    out_dir = Path(request.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_run_id = _safe_label(request.run_id)
    env = _benchmark_env(root, request)

    if request.suite == "swe-bench":
        command = _swe_bench_command(root, request, dataset)
    elif request.suite == "terminal-bench":
        command = _terminal_bench_command(request, dataset)
    elif request.suite == "terminal-bench-2":
        command = _terminal_bench_2_command(request, dataset)
    elif request.suite == "github-bugs":
        command = _github_bugs_command(root, request)
    else:  # pragma: no cover - guarded by get_suite
        raise ValueError(f"Unsupported benchmark suite {request.suite!r}")

    if request.extra_args:
        command.extend(request.extra_args)

    return ExecutionPlan(
        command=command,
        env=env,
        summary_path=out_dir / f"{safe_run_id}.agent-benchmark.summary.json",
        stdout_path=out_dir / f"{safe_run_id}.agent-benchmark.stdout.log",
        stderr_path=out_dir / f"{safe_run_id}.agent-benchmark.stderr.log",
    )


def run_benchmark_request(root: str | Path, request: BenchmarkRequest) -> dict[str, object]:
    plan = build_execution_plan(root, request)
    started = time.perf_counter()
    status = "dry_run" if request.dry_run else "not_run"
    return_code = None
    normalized = None

    if not request.dry_run:
        missing = missing_dependency(plan.command[0])
        if missing:
            status = "dependency_missing"
            plan.stderr_path.write_text(missing + "\n", encoding="utf-8")
        else:
            completed = subprocess.run(
                plan.command,
                cwd=Path(root),
                env={**os.environ, **plan.env},
                capture_output=True,
                text=True,
                timeout=request.timeout,
                check=False,
            )
            return_code = completed.returncode
            status = "passed" if completed.returncode == 0 else "failed"
            plan.stdout_path.write_text(completed.stdout, encoding="utf-8")
            plan.stderr_path.write_text(completed.stderr, encoding="utf-8")

    if request.result_path and Path(request.result_path).exists():
        normalized = normalize_result_file(request.suite, request.result_path)

    elapsed_ms = (time.perf_counter() - started) * 1000
    summary = {
        "suite": request.suite,
        "mode": request.mode,
        "yolo": request.yolo,
        "run_id": request.run_id,
        "status": status,
        "return_code": return_code,
        "elapsed_ms": elapsed_ms,
        "command": plan.command,
        "env": plan.env,
        "normalized_result": normalized,
        "summary_path": str(plan.summary_path),
        "stdout_path": str(plan.stdout_path),
        "stderr_path": str(plan.stderr_path),
    }
    plan.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def missing_dependency(executable: str) -> str:
    path = Path(executable)
    if path.is_absolute() or len(path.parts) > 1:
        return "" if path.exists() else f"Missing executable: {executable}"
    return "" if shutil.which(executable) else f"Missing executable on PATH: {executable}"


def _benchmark_env(root: Path, request: BenchmarkRequest) -> dict[str, str]:
    return {
        "CODEX_HOME": str(codex_home(root, request.mode)),
        "GEMMA_BENCH_MODE": request.mode,
        "GEMMA_BENCH_YOLO": "1" if request.yolo else "0",
        "GEMMA_BENCH_RUN_ID": request.run_id,
    }


def _swe_bench_command(root: Path, request: BenchmarkRequest, dataset: str) -> list[str]:
    if request.predictions_path is None:
        raise ValueError("swe-bench requires predictions_path")
    python = root / ".venv" / "Scripts" / "python.exe"
    return [
        str(python),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset,
        "--predictions_path",
        str(request.predictions_path),
        "--max_workers",
        str(request.n_concurrent),
        "--run_id",
        request.run_id,
    ]


def _terminal_bench_command(request: BenchmarkRequest, dataset: str) -> list[str]:
    dataset_name, dataset_version = _split_terminal_dataset(dataset)
    return [
        "tb",
        "run",
        "--agent",
        "codex-cli",
        "--model",
        model_slug(),
        "--dataset-name",
        dataset_name,
        "--dataset-version",
        dataset_version,
        "--n-concurrent",
        str(request.n_concurrent),
    ]


def _terminal_bench_2_command(request: BenchmarkRequest, dataset: str) -> list[str]:
    return [
        "harbor",
        "run",
        "--dataset",
        dataset,
        "--agent",
        "codex-cli",
        "--model",
        model_slug(),
        "--n-concurrent",
        str(request.n_concurrent),
    ]


def _github_bugs_command(root: Path, request: BenchmarkRequest) -> list[str]:
    if request.manifest_path is None:
        raise ValueError("github-bugs requires manifest_path")
    prompt = f"Fix the GitHub bug described by {request.manifest_path}. Run the manifest tests and leave a patch."
    return [
        str(root / "gemma-codex.cmd"),
        *mode_args(request.mode),
        *yolo_args(request.yolo),
        "exec",
        "--skip-git-repo-check",
        prompt,
    ]


def _split_terminal_dataset(dataset: str) -> tuple[str, str]:
    if "@" not in dataset:
        return dataset, "latest"
    name, version = dataset.rsplit("@", 1)
    return name, version


def _safe_label(label: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "._-" else "-" for character in label)
    return cleaned.strip("-") or "agent-benchmark"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run Gemma agent benchmark wrappers.")
    parser.add_argument("--suite", choices=[suite.name for suite in list_suites()], required=False)
    parser.add_argument("--list", action="store_true", help="List available benchmark suites.")
    parser.add_argument("--mode", choices=("direct", "reasoning"), default="direct")
    parser.add_argument("--yolo", action="store_true")
    parser.add_argument("--run-id", default="agent-benchmark")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--predictions-path", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/runs"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        print(json.dumps([suite.__dict__ for suite in list_suites()], indent=2, sort_keys=True))
        return 0
    if not args.suite:
        raise SystemExit("--suite is required unless --list is used")
    request = BenchmarkRequest(
        suite=args.suite,
        mode=args.mode,
        yolo=args.yolo,
        run_id=args.run_id,
        predictions_path=args.predictions_path,
        manifest_path=args.manifest_path,
        result_path=args.result_path,
        dataset=args.dataset,
        n_concurrent=args.n_concurrent,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
        timeout=args.timeout,
    )
    summary = run_benchmark_request(Path.cwd(), request)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] in {"dry_run", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
