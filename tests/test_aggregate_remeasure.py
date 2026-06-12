import json
import tempfile
import unittest
from pathlib import Path

from benchmarks import aggregate_remeasure


class AggregateRemeasureTests(unittest.TestCase):
    def test_strip_repeat_suffix_keeps_non_repeat_labels(self):
        self.assertEqual(
            aggregate_remeasure.strip_repeat_suffix("remeasure-reasoning-fastpath-8082-r5"),
            "remeasure-reasoning-fastpath-8082",
        )
        self.assertEqual(aggregate_remeasure.strip_repeat_suffix("live-before-8080"), "live-before-8080")

    def test_summarize_runs_groups_repeated_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "run-r1.summary.json"
            second = root / "run-r2.summary.json"
            first.write_text(
                json.dumps(
                    {
                        "run_label": "bench-r1",
                        "total_time_ms": 100,
                        "first_byte_latency_ms": 90,
                        "task_success_rate": 1.0,
                        "hallucinated_tool_calls": 0,
                    }
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "run_label": "bench-r2",
                        "total_time_ms": 120,
                        "first_byte_latency_ms": 100,
                        "task_success_rate": 1.0,
                        "hallucinated_tool_calls": 0,
                    }
                ),
                encoding="utf-8",
            )

            [summary] = aggregate_remeasure.summarize_runs([first, second])

        self.assertEqual(summary["label"], "bench")
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["total_time_ms_avg"], 110)
        self.assertEqual(summary["total_time_ms_min"], 100)
        self.assertEqual(summary["total_time_ms_max"], 120)
        self.assertEqual(summary["task_success_rate_avg"], 1.0)

    def test_numeric_fields_cover_required_runtime_metrics(self):
        for field in (
            "model_calls",
            "tool_latency_ms",
            "context_length_used",
            "cache_hit_rate",
            "test_pass_rate",
            "task_success_rate",
            "failed_json_tool_calls",
            "hallucinated_tool_calls",
        ):
            self.assertIn(field, aggregate_remeasure.NUMERIC_FIELDS)

    def test_default_args_do_not_scan_local_session_logs(self):
        args = aggregate_remeasure.parse_args([])

        self.assertEqual(args.session_glob, "")

    def test_prompt_candidate_extraction_reads_session_base_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "base_instructions": {
                                "text": "You are Codex running locally on Gemma 4. Use tools."
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            candidates = aggregate_remeasure.legacy_prompt_candidates(str(Path(tmp) / "*.jsonl"))

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["path"], str(path))
            self.assertGreater(candidates[0]["tokens"], 0)
            self.assertNotIn("text", candidates[0])
            self.assertIn("text_sha256", candidates[0])

    def test_prompt_candidate_extraction_can_exclude_current_prompt(self):
        prompt = "You are Codex running locally on Gemma 4. Use tools."
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {"base_instructions": {"text": prompt}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            candidates = aggregate_remeasure.legacy_prompt_candidates(
                str(Path(tmp) / "*.jsonl"),
                exclude_texts={prompt},
            )

        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
