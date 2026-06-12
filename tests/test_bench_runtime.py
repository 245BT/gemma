import json
import tempfile
import unittest
from pathlib import Path

from benchmarks import bench_runtime


class FakeClock:
    def __init__(self, start=100.0, step=0.1):
        self.value = start
        self.step = step

    def __call__(self):
        current = self.value
        self.value += self.step
        return current


class BenchmarkRuntimeTests(unittest.TestCase):
    def test_parse_nvidia_smi_memory_used_sums_visible_gpus(self):
        self.assertEqual(
            bench_runtime.parse_nvidia_smi_memory_used_mb("1024\n2048 MiB\n"),
            3072.0,
        )
        self.assertIsNone(bench_runtime.parse_nvidia_smi_memory_used_mb("N/A\n"))

    def test_sample_vram_returns_none_when_nvidia_smi_fails(self):
        class FailedRun:
            returncode = 1
            stdout = ""

        def failed_runner(*args, **kwargs):
            return FailedRun()

        self.assertIsNone(bench_runtime.sample_vram_mb(command_runner=failed_runner))

    def test_sample_vram_uses_nvidia_smi_memory_query(self):
        captured = {}

        class SuccessfulRun:
            returncode = 0
            stdout = "1234\n"

        def successful_runner(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return SuccessfulRun()

        self.assertEqual(bench_runtime.sample_vram_mb(command_runner=successful_runner), 1234.0)
        self.assertIn("--query-gpu=memory.used", captured["command"])
        self.assertEqual(captured["kwargs"]["timeout"], 1)

    def test_default_http_timeout_is_bounded(self):
        args = bench_runtime.parse_args(["--endpoint", "http://unit.test/v1/responses"])

        self.assertLessEqual(args.timeout, 120)

    def test_run_benchmark_writes_events_summary_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workload_path = root / "workload.jsonl"
            events_path = root / "events.jsonl"
            summary_path = root / "summary.json"
            markdown_path = root / "BENCHMARKS.md"
            workload_path.write_text(
                json.dumps(
                    {
                        "id": "math_smoke",
                        "request": {"prompt": "What is 2+2?"},
                        "expected_contains": "4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            samples = iter(
                [
                    {"ram_mb": 120.0, "vram_mb": 512.0},
                    {"ram_mb": 150.0, "vram_mb": 768.0},
                ]
            )

            def fake_request(endpoint, payload, emit_event):
                self.assertEqual(endpoint, "http://unit.test/v1/responses")
                self.assertEqual(payload["prompt"], "What is 2+2?")
                emit_event("first_byte")
                emit_event("first_text", text="4")
                emit_event("tool_call", latency_ms=12.5, tool_name="unit_tool")
                emit_event("model_call")
                return {
                    "text": "4",
                    "usage": {"prompt_tokens": 7, "completion_tokens": 1},
                    "model_calls": 3,
                    "context_length": 128,
                    "cache_hit": True,
                    "cache_hit_rate": 0.5,
                    "test_pass_rate": 1.0,
                    "hallucinated_tool_calls": 0,
                    "failed_json_tool_calls": 0,
                }

            summary = bench_runtime.run_benchmark(
                workload_path=workload_path,
                endpoint="http://unit.test/v1/responses",
                phase="before",
                run_label="unit-before",
                events_path=events_path,
                summary_path=summary_path,
                markdown_path=markdown_path,
                request_func=fake_request,
                resource_sampler=lambda: next(samples),
                clock=FakeClock(),
            )

            self.assertEqual(summary["run_label"], "unit-before")
            self.assertEqual(summary["phase"], "before")
            self.assertEqual(summary["workload_count"], 1)
            self.assertEqual(summary["model_calls"], 3)
            self.assertEqual(summary["prompt_tokens"], 7)
            self.assertEqual(summary["completion_tokens"], 1)
            self.assertGreater(summary["tokens_per_second"], 0)
            self.assertEqual(summary["peak_ram_mb"], 150.0)
            self.assertEqual(summary["peak_vram_mb"], 768.0)
            self.assertEqual(summary["context_length_used"], 128)
            self.assertTrue(summary["cache_hit"])
            self.assertEqual(summary["cache_hit_rate"], 0.5)
            self.assertEqual(summary["test_pass_rate"], 1.0)
            self.assertEqual(summary["task_success_rate"], 1.0)
            self.assertEqual(summary["hallucinated_tool_calls"], 0)
            self.assertEqual(summary["failed_json_tool_calls"], 0)
            self.assertEqual(summary["failed_tool_calls"], 0)
            self.assertIn("first_byte_latency_ms", summary)
            self.assertIn("first_text_latency_ms", summary)
            self.assertIn("tool_latency_ms", summary)

            raw_events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            event_types = {event["event"] for event in raw_events}
            self.assertIn("run_start", event_types)
            self.assertIn("first_byte", event_types)
            self.assertIn("first_text", event_types)
            self.assertIn("tool_call", event_types)
            self.assertIn("workload_result", event_types)
            self.assertIn("run_summary", event_types)
            first_text_event = next(event for event in raw_events if event["event"] == "first_text")
            self.assertNotIn("text", first_text_event)
            self.assertEqual(first_text_event["text_length"], 1)
            self.assertIn("text_sha256", first_text_event)

            saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["run_label"], "unit-before")

            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("| Run | Phase | Workloads | First byte ms |", markdown)
            self.assertIn("| unit-before | before | 1 |", markdown)

    def test_expected_exact_requires_stripped_exact_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workload_path = root / "workload.jsonl"
            events_path = root / "events.jsonl"
            summary_path = root / "summary.json"
            markdown_path = root / "BENCHMARKS.md"
            workload_path.write_text(
                json.dumps(
                    {
                        "id": "math_exact",
                        "request": {"prompt": "What is 2+2?"},
                        "expected_exact": "4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = bench_runtime.run_benchmark(
                workload_path=workload_path,
                endpoint="http://unit.test/v1/responses",
                phase="before",
                run_label="unit-before",
                events_path=events_path,
                summary_path=summary_path,
                markdown_path=markdown_path,
                request_func=lambda endpoint, payload, emit_event: {"text": "4, father"},
                resource_sampler=lambda: {},
                clock=FakeClock(),
            )

        self.assertEqual(summary["task_success_rate"], 0.0)

    def test_sample_process_ram_matches_target_command_lines(self):
        class Proc:
            def __init__(self, cmdline, rss):
                self.info = {"cmdline": cmdline}
                self._rss = rss

            def memory_info(self):
                return type("Mem", (), {"rss": self._rss})()

        processes = [
            Proc(["python.exe", "gemma_response_proxy.py"], 2 * 1024 * 1024),
            Proc(["python.exe", "other.py"], 99 * 1024 * 1024),
            Proc(["llama-server.exe", "--port", "8080"], 8 * 1024 * 1024),
        ]

        ram_mb = bench_runtime.sample_process_ram_mb(
            lambda attrs=None: processes,
            ["gemma_response_proxy.py", "llama-server.exe"],
        )

        self.assertEqual(ram_mb, 10.0)

    def test_markdown_table_appends_before_and_after_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            markdown_path = Path(tmp) / "BENCHMARKS.md"
            before = {
                "run_label": "baseline",
                "phase": "before",
                "workload_count": 1,
                "first_byte_latency_ms": 10.0,
                "first_text_latency_ms": 20.0,
                "total_time_ms": 50.0,
                "tokens_per_second": 12.0,
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "model_calls": 1,
                "peak_ram_mb": 100.0,
                "peak_vram_mb": None,
                "context_length_used": 256,
                "cache_hit": False,
                "cache_hit_rate": None,
                "test_pass_rate": None,
                "task_success_rate": None,
                "hallucinated_tool_calls": 0,
                "failed_json_tool_calls": 0,
                "failed_tool_calls": 0,
            }
            after = dict(before, run_label="optimized", phase="after", total_time_ms=25.0)

            bench_runtime.update_markdown_table(markdown_path, before)
            bench_runtime.update_markdown_table(markdown_path, after)

            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertEqual(markdown.count("| Run | Phase | Workloads |"), 1)
            self.assertIn("| baseline | before | 1 |", markdown)
            self.assertIn("| optimized | after | 1 |", markdown)

    def test_test_pass_rate_is_not_inferred_from_task_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workload_path = root / "workload.jsonl"
            events_path = root / "events.jsonl"
            summary_path = root / "summary.json"
            markdown_path = root / "BENCHMARKS.md"
            workload_path.write_text(
                json.dumps(
                    {
                        "id": "math_smoke",
                        "request": {"prompt": "What is 2+2?"},
                        "expected_contains": "4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = bench_runtime.run_benchmark(
                workload_path=workload_path,
                endpoint="http://unit.test/v1/responses",
                phase="before",
                run_label="unit-before",
                events_path=events_path,
                summary_path=summary_path,
                markdown_path=markdown_path,
                request_func=lambda endpoint, payload, emit_event: {"text": "4"},
                resource_sampler=lambda: {},
                clock=FakeClock(),
            )

        self.assertEqual(summary["task_success_rate"], 1.0)
        self.assertIsNone(summary["test_pass_rate"])


if __name__ == "__main__":
    unittest.main()
