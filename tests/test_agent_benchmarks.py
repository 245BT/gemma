import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.agent_benchmarks.registry import get_suite, list_suites
from benchmarks.agent_benchmarks.results import normalize_result_file
from benchmarks.agent_benchmarks.runner import BenchmarkRequest, build_execution_plan, run_benchmark_request


class AgentBenchmarkTests(unittest.TestCase):
    def test_registry_includes_required_agent_benchmark_suites(self):
        suite_names = [suite.name for suite in list_suites()]

        self.assertEqual(
            suite_names,
            ["swe-bench", "terminal-bench", "terminal-bench-2", "github-bugs"],
        )
        self.assertEqual(get_suite("terminal-bench-2").default_dataset, "terminal-bench@2.0")

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


if __name__ == "__main__":
    unittest.main()
