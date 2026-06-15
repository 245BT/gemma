import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from benchmarks.agent_benchmarks.registry import get_suite, list_suites
from benchmarks.agent_benchmarks.results import normalize_result_file
from benchmarks.agent_benchmarks.runner import BenchmarkRequest, build_execution_plan, run_benchmark_request
from benchmarks.agent_benchmarks.subprocesses import BoundedCompletedProcess


def write_research_fixture(root: Path) -> None:
    from benchmarks.agent_benchmarks.local_behavior import REQUIRED_RESEARCH_ARTIFACTS

    research_dir = root / "docs" / "research"
    research_dir.mkdir(parents=True)
    for artifact in REQUIRED_RESEARCH_ARTIFACTS:
        (research_dir / artifact["pdf"]).write_bytes(b"%PDF-1.7\n" + b"x" * 1200)
        (research_dir / artifact["text"]).write_text(
            "\n".join(artifact["markers"]),
            encoding="utf-8",
        )


class AgentBenchmarkTests(unittest.TestCase):
    def test_registry_includes_required_agent_benchmark_suites(self):
        suite_names = [suite.name for suite in list_suites()]

        self.assertEqual(
            suite_names,
            [
                "swe-bench",
                "terminal-bench",
                "terminal-bench-2",
                "github-bugs",
                "local-agent-behavior",
                "local-repo-fix",
            ],
        )
        self.assertEqual(get_suite("terminal-bench-2").default_dataset, "terminal-bench@2.0")
        self.assertEqual(get_suite("local-agent-behavior").default_dataset, "deterministic-local")
        self.assertEqual(get_suite("local-repo-fix").default_dataset, "generated-local")

    def test_terminal_bench_2_reasoning_yolo_plan_uses_harbor_and_reasoning_home(self):
        root = Path("C:/Users/Agent-1/Desktop/gemma")
        request = BenchmarkRequest(
            suite="terminal-bench-2",
            mode="reasoning",
            yolo=True,
            run_id="tb2-smoke",
            n_concurrent=2,
        )

        plan = build_execution_plan(root, request)

        self.assertEqual(plan.env["CODEX_HOME"], str(root / ".codex-local-reasoning"))
        self.assertEqual(plan.env["GEMMA_BENCH_YOLO"], "1")
        self.assertEqual(plan.command[:4], ["harbor", "run", "--dataset", "terminal-bench@2.0"])
        self.assertIn("--agent", plan.command)
        self.assertIn("codex-cli", plan.command)
        self.assertIn("--n-concurrent", plan.command)
        self.assertIn("2", plan.command)

    def test_swe_bench_plan_requires_predictions_path(self):
        with self.assertRaisesRegex(ValueError, "predictions_path"):
            build_execution_plan(
                Path("C:/repo"),
                BenchmarkRequest(suite="swe-bench", mode="direct", run_id="missing"),
            )

    def test_swe_bench_plan_builds_official_harness_command(self):
        root = Path("C:/repo")
        request = BenchmarkRequest(
            suite="swe-bench",
            mode="direct",
            run_id="verified-001",
            predictions_path=Path("benchmarks/predictions/gemma.jsonl"),
            n_concurrent=3,
        )

        plan = build_execution_plan(root, request)

        self.assertEqual(plan.env["CODEX_HOME"], str(root / ".codex-local"))
        self.assertEqual(plan.command[:3], [str(root / ".venv" / "Scripts" / "python.exe"), "-m", "swebench.harness.run_evaluation"])
        self.assertIn("--dataset_name", plan.command)
        self.assertIn("SWE-bench/SWE-bench_Verified", plan.command)
        self.assertIn("--predictions_path", plan.command)
        self.assertIn("benchmarks\\predictions\\gemma.jsonl", plan.command)

    def test_github_bugs_plan_uses_local_launcher_with_reasoning_yolo_exec(self):
        root = Path("C:/repo")
        request = BenchmarkRequest(
            suite="github-bugs",
            mode="reasoning",
            yolo=True,
            run_id="bug-001",
            manifest_path=Path("benchmarks/manifests/github_bugs/smoke.json"),
        )

        plan = build_execution_plan(root, request)

        self.assertEqual(plan.command[0], str(root / "gemma-codex.cmd"))
        self.assertEqual(plan.command[1:5], ["--reasoning", "--yolo", "exec", "--skip-git-repo-check"])
        self.assertEqual(plan.env["GEMMA_BENCH_MODE"], "reasoning")

    def test_local_repo_fix_plan_uses_local_generated_harness(self):
        root = Path("C:/repo")
        request = BenchmarkRequest(
            suite="local-repo-fix",
            mode="direct",
            run_id="repo-fix-smoke",
            out_dir=Path("benchmarks/runs"),
        )

        plan = build_execution_plan(root, request)

        self.assertEqual(plan.command[:3], [str(root / ".venv" / "Scripts" / "python.exe"), "-m", "benchmarks.agent_benchmarks.local_repo_fix"])
        self.assertIn("--solver", plan.command)
        self.assertIn("oracle", plan.command)

    def test_dry_run_writes_summary_without_executing_external_harness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = BenchmarkRequest(
                suite="terminal-bench-2",
                mode="direct",
                yolo=True,
                run_id="dry",
                out_dir=root / "runs",
                dry_run=True,
            )

            summary = run_benchmark_request(root, request)

            self.assertEqual(summary["status"], "dry_run")
            self.assertEqual(summary["suite"], "terminal-bench-2")
            self.assertTrue(Path(summary["summary_path"]).exists())

    def test_local_agent_behavior_benchmark_runs_deterministic_tasks(self):
        from benchmarks.agent_benchmarks.local_behavior import run_local_behavior_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_research_fixture(root)
            out_dir = root / "runs"
            summary = run_local_behavior_benchmark(root, run_id="behavior-smoke", output_dir=out_dir)

            self.assertEqual(summary["suite"], "local-agent-behavior")
            self.assertEqual(Path(summary["summary_path"]).parent, out_dir)
            self.assertTrue(Path(summary["summary_path"]).exists())
            self.assertGreaterEqual(summary["total_count"], 9)
            self.assertIn("stall_recovery", summary["category_counts"])
            self.assertIn("efficiency_budgeting", summary["category_counts"])
            self.assertIn("large_repo_navigation", summary["category_counts"])
            self.assertIn("research_citation", summary["category_counts"])
            self.assertIn("tool_use_reliability", summary["category_counts"])
            self.assertIn("latency_ms", summary)
            self.assertIn("failure_counts", summary)

    def test_local_repo_fix_benchmark_runs_generated_oracle_tasks(self):
        from benchmarks.agent_benchmarks.local_repo_fix import run_local_repo_fix_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "runs"
            summary = run_local_repo_fix_benchmark(root, run_id="repo-fix-smoke", output_dir=out_dir)

            self.assertEqual(summary["suite"], "local-repo-fix")
            self.assertEqual(summary["total_count"], 3)
            self.assertEqual(summary["resolved_count"], 3)
            self.assertEqual(summary["pass_rate"], 1.0)
            self.assertEqual(summary["regression_count"], 0)
            self.assertEqual(summary["fail_to_pass_rate"], 1.0)
            self.assertEqual(summary["pass_to_pass_rate"], 1.0)
            self.assertTrue(Path(summary["summary_path"]).exists())
            self.assertGreater(summary["latency_ms"], 0)
            for result in summary["results"]:
                self.assertTrue(result["success"])
                self.assertGreaterEqual(result["changed_lines"], 1)
                self.assertGreaterEqual(result["test_pass_rate"], 1.0)

    def test_local_repo_fix_gemma_solver_invokes_launcher_with_target_dir(self):
        from benchmarks.agent_benchmarks import local_repo_fix

        completed = BoundedCompletedProcess(
            return_code=124,
            stdout="partial",
            stderr="timed out",
            timed_out=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gemma-codex.cmd").write_text("@echo off\n", encoding="utf-8")

            calls = []

            def bounded_run(command, **kwargs):
                calls.append((command, kwargs))
                if "gemma-codex.cmd" in str(command[0]):
                    return completed
                return BoundedCompletedProcess(return_code=0, stdout="", stderr="", timed_out=False)

            with mock.patch("benchmarks.agent_benchmarks.local_repo_fix.run_bounded_subprocess", side_effect=bounded_run):
                summary = local_repo_fix.run_local_repo_fix_benchmark(
                    root,
                    run_id="repo-fix-gemma",
                    output_dir=root / "runs",
                    solver="gemma",
                    timeout=7,
                    max_tasks=1,
                )

        self.assertEqual(summary["total_count"], 1)
        self.assertEqual(summary["resolved_count"], 0)
        result = summary["results"][0]
        self.assertEqual(result["solver_status"], "timeout")
        self.assertEqual(result["solver_return_code"], 124)
        self.assertFalse(result["patch_applied"])
        gemma_call = next(
            call
            for call in calls
            if "gemma-codex.cmd" in str(call[0][0])
        )
        command = gemma_call[0]
        self.assertIn("gemma-codex.cmd", command[0])
        self.assertIn("--yolo", command)
        self.assertIn("exec", command)
        env = gemma_call[1]["env"]
        self.assertIn("GEMMA_CODEX_TARGET_DIR", env)
        self.assertEqual(env["GEMMA_CODEX_EXEC_TIMEOUT"], "7")

    def test_local_repo_fix_gemma_solver_reports_outer_timeout(self):
        from benchmarks.agent_benchmarks import local_repo_fix

        task = local_repo_fix.LocalRepoTask(
            name="timeout_task",
            category="terminal_execution",
            instruction="Fix the bug.",
            files={},
            oracle=lambda _repo: None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_root = root / "task"
            task_root.mkdir()
            completed = BoundedCompletedProcess(
                return_code=124,
                stdout="partial stdout",
                stderr="partial stderr",
                timed_out=True,
            )
            with mock.patch("benchmarks.agent_benchmarks.local_repo_fix.run_bounded_subprocess", return_value=completed):
                result = local_repo_fix._run_gemma_solver(
                    root=root,
                    task_root=task_root,
                    task=task,
                    timeout=7,
                )

        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["return_code"], 124)
        self.assertIn("partial stdout", result["stdout"])
        self.assertIn("partial stderr", result["stderr"])

    def test_local_agent_behavior_requires_all_cited_research_artifacts(self):
        from benchmarks.agent_benchmarks.local_behavior import REQUIRED_RESEARCH_ARTIFACTS, run_local_behavior_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_research_fixture(root)
            missing = REQUIRED_RESEARCH_ARTIFACTS[-1]
            (root / "docs" / "research" / missing["text"]).unlink()
            summary = run_local_behavior_benchmark(root, run_id="missing-research")

        research_case = next(item for item in summary["results"] if item["name"] == "research_artifacts_available")
        self.assertFalse(research_case["success"])
        self.assertEqual(research_case["failure_mode"], "research_artifact_missing")

    def test_local_agent_behavior_default_output_dir_is_under_root_benchmarks_runs(self):
        from benchmarks.agent_benchmarks.local_behavior import run_local_behavior_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_research_fixture(root)
            summary = run_local_behavior_benchmark(root, run_id="default-out")

            self.assertEqual(
                Path(summary["summary_path"]).parent,
                root / "benchmarks" / "runs",
            )
            self.assertTrue(Path(summary["summary_path"]).exists())

    def test_local_agent_behavior_runner_keeps_artifacts_in_out_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path.cwd()
            out_dir = Path(tmp) / "runs"
            summary = run_benchmark_request(
                root,
                BenchmarkRequest(
                    suite="local-agent-behavior",
                    mode="direct",
                    run_id="local-artifacts",
                    out_dir=out_dir,
                    timeout=5,
                ),
            )

            self.assertEqual(summary["status"], "passed")
            self.assertEqual(Path(summary["summary_path"]).parent, out_dir)
            self.assertTrue((out_dir / "local-artifacts.local-agent-behavior.summary.json").exists())

    def test_local_agent_behavior_sanitizes_run_id_before_creating_summary_path(self):
        from benchmarks.agent_benchmarks import local_behavior

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            out_dir = Path(tmp) / "runs"
            with mock.patch.object(local_behavior, "_cases", return_value=[]):
                summary = local_behavior.run_local_behavior_benchmark(
                    root,
                    run_id="..\\outside",
                    output_dir=out_dir,
                )

            summary_path = Path(summary["summary_path"]).resolve()
            self.assertTrue(summary_path.is_relative_to(out_dir.resolve()))
            self.assertEqual(summary_path.name, "outside.local-agent-behavior.summary.json")
            self.assertFalse((Path(tmp) / "outside.local-agent-behavior.summary.json").exists())

    def test_benchmark_plan_resolves_relative_out_dir_under_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_execution_plan(
                root,
                BenchmarkRequest(
                    suite="local-agent-behavior",
                    run_id="relative-out",
                    out_dir=Path("relative-runs"),
                    dry_run=True,
                ),
            )

        self.assertEqual(plan.summary_path.parent, root / "relative-runs")

    def test_local_agent_behavior_runner_reports_subprocess_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path.cwd()
            out_dir = Path(tmp) / "runs"
            completed = BoundedCompletedProcess(
                return_code=124,
                stdout="partial stdout",
                stderr="partial stderr",
                timed_out=True,
            )
            with mock.patch("benchmarks.agent_benchmarks.runner.run_bounded_subprocess", return_value=completed):
                summary = run_benchmark_request(
                    root,
                    BenchmarkRequest(
                        suite="local-agent-behavior",
                        mode="direct",
                        run_id="local-timeout",
                        out_dir=out_dir,
                        timeout=0.01,
                    ),
                )

            self.assertEqual(summary["status"], "timeout")
            self.assertEqual(summary["return_code"], 124)
            self.assertIn("partial stdout", (out_dir / "local-timeout.agent-benchmark.stdout.log").read_text())
            self.assertIn("partial stderr", (out_dir / "local-timeout.agent-benchmark.stderr.log").read_text())

    def test_agent_benchmark_timeout_kills_child_process_tree(self):
        from benchmarks.agent_benchmarks.subprocesses import run_bounded_subprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "child-survived.txt"
            child_code = (
                "import pathlib,time; "
                "time.sleep(1); "
                f"pathlib.Path({str(marker)!r}).write_text('alive', encoding='utf-8')"
            )
            command = [
                sys.executable,
                "-c",
                (
                    "import subprocess, sys, time; "
                    f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                    "time.sleep(5)"
                ),
            ]

            result = run_bounded_subprocess(command, cwd=root, timeout=0.2)
            time.sleep(1.2)

            self.assertEqual(result.return_code, 124)
            self.assertTrue(result.timed_out)
            self.assertFalse(marker.exists())

    def test_agent_benchmark_logs_redacted_tails_and_hash_metadata(self):
        from benchmarks.agent_benchmarks.subprocesses import BoundedCompletedProcess

        secret_stdout = (
            "OPENAI_API_KEY=sk-test-secret-123456789\n"
            "password=hunter2\n"
            "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n"
        )
        secret_stderr = "token=secret-token-value\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path.cwd()
            out_dir = Path(tmp) / "runs"
            completed = BoundedCompletedProcess(
                return_code=1,
                stdout=secret_stdout,
                stderr=secret_stderr,
                timed_out=False,
            )

            with mock.patch("benchmarks.agent_benchmarks.runner.run_bounded_subprocess", return_value=completed):
                summary = run_benchmark_request(
                    root,
                    BenchmarkRequest(
                        suite="local-agent-behavior",
                        mode="direct",
                        run_id="redacted",
                        out_dir=out_dir,
                        timeout=5,
                    ),
                )

            summary_text = Path(summary["summary_path"]).read_text(encoding="utf-8")
            stdout_log = Path(summary["stdout_path"]).read_text(encoding="utf-8")
            stderr_log = Path(summary["stderr_path"]).read_text(encoding="utf-8")
            combined = "\n".join([summary_text, stdout_log, stderr_log])

            self.assertEqual(summary["status"], "failed")
            self.assertNotIn("sk-test-secret", combined)
            self.assertNotIn("hunter2", combined)
            self.assertNotIn("secret-token-value", combined)
            self.assertNotIn("BEGIN PRIVATE KEY", combined)
            self.assertIn("[REDACTED]", combined)
            self.assertEqual(summary["stdout_length"], len(secret_stdout))
            self.assertEqual(summary["stderr_length"], len(secret_stderr))
            self.assertIn("stdout_sha256", summary)
            self.assertIn("stderr_sha256", summary)
            self.assertIn("stdout_tail", summary)
            self.assertIn("stderr_tail", summary)

    def test_local_agent_behavior_timeout_ignores_stale_local_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path.cwd()
            out_dir = Path(tmp) / "runs"
            out_dir.mkdir()
            stale_summary = out_dir / "local-timeout.local-agent-behavior.summary.json"
            stale_summary.write_text(
                json.dumps(
                    {
                        "suite": "local-agent-behavior",
                        "results": [{"success": True, "category": "terminal_execution"}],
                    }
                ),
                encoding="utf-8",
            )
            completed = BoundedCompletedProcess(
                return_code=124,
                stdout="partial stdout",
                stderr="partial stderr",
                timed_out=True,
            )

            with mock.patch("benchmarks.agent_benchmarks.runner.run_bounded_subprocess", return_value=completed):
                summary = run_benchmark_request(
                    root,
                    BenchmarkRequest(
                        suite="local-agent-behavior",
                        mode="direct",
                        run_id="local-timeout",
                        out_dir=out_dir,
                        timeout=0.01,
                    ),
                )

            self.assertEqual(summary["status"], "timeout")
            self.assertIsNone(summary["normalized_result"])
            self.assertFalse(stale_summary.exists())

    def test_local_agent_behavior_runner_normalizes_failed_local_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path.cwd()
            out_dir = Path(tmp) / "runs"

            def failed_local_run(command, **kwargs):
                local_summary = out_dir / "local-failed.local-agent-behavior.summary.json"
                local_summary.parent.mkdir(parents=True, exist_ok=True)
                local_summary.write_text(
                    json.dumps(
                        {
                            "suite": "local-agent-behavior",
                            "results": [
                                {"success": True, "category": "terminal_execution"},
                                {
                                    "success": False,
                                    "category": "planning_recovery",
                                    "failure_mode": "assertion_failed",
                                },
                            ],
                            "category_counts": {
                                "terminal_execution": 1,
                                "planning_recovery": 1,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return BoundedCompletedProcess(
                    return_code=1,
                    stdout="summary written",
                    stderr="1 failed",
                    timed_out=False,
                )

            with mock.patch("benchmarks.agent_benchmarks.runner.run_bounded_subprocess", side_effect=failed_local_run):
                summary = run_benchmark_request(
                    root,
                    BenchmarkRequest(
                        suite="local-agent-behavior",
                        mode="direct",
                        run_id="local-failed",
                        out_dir=out_dir,
                        timeout=5,
                    ),
                )

            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["return_code"], 1)
            self.assertEqual(summary["normalized_result"]["total_count"], 2)
            self.assertEqual(summary["normalized_result"]["resolved_count"], 1)
            self.assertAlmostEqual(summary["normalized_result"]["pass_rate"], 0.5)
            self.assertEqual(summary["normalized_result"]["failure_counts"], {"assertion_failed": 1})

    def test_local_repo_fix_sanitizes_run_id_before_creating_artifact_paths(self):
        from benchmarks.agent_benchmarks.local_repo_fix import run_local_repo_fix_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            out_dir = Path(tmp) / "runs"
            summary = run_local_repo_fix_benchmark(
                root,
                run_id="..\\outside",
                output_dir=out_dir,
                max_tasks=0,
            )

            summary_path = Path(summary["summary_path"]).resolve()
            self.assertTrue(summary_path.is_relative_to(out_dir.resolve()))
            self.assertTrue(summary_path.exists())
            self.assertFalse((Path(tmp) / "outside.local-repo-fix.summary.json").exists())
            self.assertFalse((Path(tmp) / "outside.local-repo-fix-artifacts").exists())

    def test_local_repo_fix_sanitizes_absolute_run_id_before_creating_paths(self):
        from benchmarks.agent_benchmarks.local_repo_fix import run_local_repo_fix_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            out_dir = Path(tmp) / "runs"
            outside = Path(tmp) / "outside"
            run_id = str(outside)
            summary = run_local_repo_fix_benchmark(
                root,
                run_id=run_id,
                output_dir=out_dir,
                max_tasks=0,
            )

            summary_path = Path(summary["summary_path"]).resolve()
            self.assertEqual(summary["run_id"], run_id)
            self.assertTrue(summary_path.is_relative_to(out_dir.resolve()))
            self.assertTrue(summary_path.exists())
            self.assertFalse(Path(str(outside) + ".local-repo-fix.summary.json").exists())
            self.assertFalse(Path(str(outside) + ".local-repo-fix-artifacts").exists())

    def test_runner_sanitizes_local_summary_run_id_before_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path.cwd()
            out_dir = Path(tmp) / "runs"
            out_dir.mkdir()
            outside_summary = Path(tmp) / "outside.local-agent-behavior.summary.json"
            outside_summary.write_text('{"results":[]}', encoding="utf-8")
            completed = BoundedCompletedProcess(
                return_code=124,
                stdout="partial stdout",
                stderr="partial stderr",
                timed_out=True,
            )

            with mock.patch("benchmarks.agent_benchmarks.runner.run_bounded_subprocess", return_value=completed):
                summary = run_benchmark_request(
                    root,
                    BenchmarkRequest(
                        suite="local-agent-behavior",
                        mode="direct",
                        run_id="..\\outside",
                        out_dir=out_dir,
                        timeout=0.01,
                    ),
                )

            self.assertEqual(summary["status"], "timeout")
            self.assertTrue(outside_summary.exists())
            self.assertIsNone(summary["normalized_result"])

    def test_runner_reads_sanitized_local_repo_fix_summary_from_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path.cwd()
            out_dir = Path(tmp) / "runs"

            def local_repo_fix_run(command, **kwargs):
                local_summary = out_dir / "outside.local-repo-fix.summary.json"
                local_summary.parent.mkdir(parents=True, exist_ok=True)
                local_summary.write_text(
                    json.dumps(
                        {
                            "suite": "local-repo-fix",
                            "results": [{"success": True, "category": "terminal_execution"}],
                        }
                    ),
                    encoding="utf-8",
                )
                return BoundedCompletedProcess(
                    return_code=0,
                    stdout="summary written",
                    stderr="",
                    timed_out=False,
                )

            with mock.patch("benchmarks.agent_benchmarks.runner.run_bounded_subprocess", side_effect=local_repo_fix_run):
                summary = run_benchmark_request(
                    root,
                    BenchmarkRequest(
                        suite="local-repo-fix",
                        mode="direct",
                        run_id="..\\outside",
                        out_dir=out_dir,
                        timeout=5,
                    ),
                )

            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["normalized_result"]["total_count"], 1)
            self.assertEqual(summary["normalized_result"]["resolved_count"], 1)

    def test_local_behavior_idle_timeout_kills_stalled_job_after_check(self):
        from benchmarks.agent_benchmarks import local_behavior

        class FakeRunner:
            instances = []

            def __init__(self, **kwargs):
                self.killed = []
                FakeRunner.instances.append(self)

            def run(self, command):
                return SimpleNamespace(
                    ok=False,
                    status="stalled",
                    elapsed_ms=100,
                    recovery={"reason": "idle_timeout"},
                    job_id="job-1",
                )

            def kill_job(self, job_id, reason):
                self.killed.append((job_id, reason))
                return SimpleNamespace(status="killed")

        with mock.patch.object(local_behavior, "TerminalCommandRunner", FakeRunner):
            self.assertTrue(local_behavior._terminal_idle_timeout())

        self.assertEqual(FakeRunner.instances[0].killed, [("job-1", "benchmark idle-timeout cleanup")])

    def test_normalize_swe_result_file_counts_resolved_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(
                json.dumps(
                    {
                        "resolved": ["a", "b"],
                        "unresolved": ["c"],
                        "error": [],
                    }
                ),
                encoding="utf-8",
            )

            normalized = normalize_result_file("swe-bench", path)

            self.assertEqual(normalized["resolved_count"], 2)
            self.assertEqual(normalized["total_count"], 3)
            self.assertAlmostEqual(normalized["pass_rate"], 2 / 3)

    def test_normalize_result_file_counts_failure_taxonomy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local-results.json"
            path.write_text(
                json.dumps(
                    [
                        {"success": True, "category": "terminal_execution"},
                        {"success": False, "category": "terminal_execution", "failure_mode": "TimeoutError"},
                        {"success": False, "category": "research_citation", "failure_mode": "bad_search"},
                        {"success": False, "category": "research_citation", "failure_mode": "bad label"},
                        {"success": False, "category": "planning_recovery"},
                    ]
                ),
                encoding="utf-8",
            )

            normalized = normalize_result_file("local-agent-behavior", path)

            self.assertEqual(normalized["total_count"], 5)
            self.assertAlmostEqual(normalized["pass_rate"], 1 / 5)
            self.assertEqual(
                normalized["failure_counts"],
                {
                    "bad_search": 1,
                    "timeout": 1,
                    "unexpected_exception": 1,
                    "unknown_failure": 1,
                },
            )


if __name__ == "__main__":
    unittest.main()
