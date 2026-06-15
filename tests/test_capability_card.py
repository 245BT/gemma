import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.agent_benchmarks import capability_card


class CapabilityCardTests(unittest.TestCase):
    def test_requirements_declare_pillow_for_png_rendering(self):
        requirements = Path("requirements-gemma.txt").read_text(encoding="utf-8")

        self.assertRegex(requirements, r"(?im)^pillow(?:[<=>!~]=?|$)")

    def test_parse_choice_accepts_exact_and_sentence_answers(self):
        self.assertEqual(capability_card.parse_choice("B"), "B")
        self.assertEqual(capability_card.parse_choice("The answer is C."), "C")
        self.assertIsNone(capability_card.parse_choice("I do not know."))

    def test_score_choice_results_groups_by_benchmark_row(self):
        tasks = [
            capability_card.ChoiceTask(
                row="Knowledge work",
                benchmark="LocalQA",
                prompt="q1",
                expected="A",
                maturity="live",
            ),
            capability_card.ChoiceTask(
                row="Knowledge work",
                benchmark="LocalQA",
                prompt="q2",
                expected="B",
                maturity="live",
            ),
            capability_card.ChoiceTask(
                row="Spatial reasoning",
                benchmark="LocalSpatial",
                prompt="q3",
                expected="C",
                maturity="live",
            ),
        ]
        results = [
            {"row": "Knowledge work", "benchmark": "LocalQA", "success": True, "latency_ms": 10},
            {"row": "Knowledge work", "benchmark": "LocalQA", "success": False, "latency_ms": 20},
            {"row": "Spatial reasoning", "benchmark": "LocalSpatial", "success": True, "latency_ms": 30},
        ]

        rows = capability_card.score_choice_results(tasks, results)

        self.assertEqual(rows[0]["row"], "Knowledge work")
        self.assertEqual(rows[0]["score"], 50.0)
        self.assertEqual(rows[0]["passed"], 1)
        self.assertEqual(rows[0]["total"], 2)
        self.assertEqual(rows[0]["latency_ms"], 30)
        self.assertEqual(rows[1]["score"], 100.0)

    def test_render_capability_card_writes_json_markdown_html_and_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            rows = [
                {
                    "row": "Agentic coding",
                    "benchmark": "Local Repo Fix",
                    "score": 66.7,
                    "passed": 2,
                    "total": 3,
                    "maturity": "live",
                    "latency_ms": 1234,
                    "notes": "model solver",
                },
                {
                    "row": "Tool use",
                    "benchmark": "Local Behavior",
                    "score": 100.0,
                    "passed": 2,
                    "total": 2,
                    "maturity": "deterministic",
                    "latency_ms": 10,
                    "notes": "tool/schema checks",
                },
            ]

            paths = capability_card.write_capability_card(
                rows,
                output_dir=out_dir,
                run_id="card-test",
                model_label="Gamma Local",
            )

            for key in ("json", "markdown", "html", "png"):
                self.assertTrue(paths[key].exists(), key)
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["model_label"], "Gamma Local")
            self.assertEqual(payload["rows"][0]["score"], 66.7)
            self.assertIn("Agentic coding", paths["markdown"].read_text(encoding="utf-8"))
            self.assertIn("Gamma Local", paths["html"].read_text(encoding="utf-8"))

    def test_write_capability_card_sanitizes_run_id_under_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "runs"
            outside = Path(tmp) / "outside"
            run_id = str(outside)

            paths = capability_card.write_capability_card(
                [],
                output_dir=out_dir,
                run_id=run_id,
                model_label="Gemma",
            )

            resolved_out_dir = out_dir.resolve()
            for path in paths.values():
                self.assertTrue(path.resolve().is_relative_to(resolved_out_dir))
            self.assertFalse(Path(str(outside) + ".capability-card.json").exists())
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], run_id)

    def test_write_comparison_card_outputs_multi_model_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            columns = ["Gamma Local", "Gamma Before", "Codex CLI"]
            rows = [
                {
                    "row": "Agentic coding",
                    "benchmark": "Local Repo Fix",
                    "scores": {
                        "Gamma Local": {"value": 100.0, "note": "3/3 live"},
                        "Gamma Before": {"value": 0.0, "note": "0/1 blocked"},
                        "Codex CLI": None,
                    },
                }
            ]

            paths = capability_card.write_comparison_card(
                columns,
                rows,
                output_dir=out_dir,
                run_id="comparison",
                primary_column="Gamma Local",
            )

            for key in ("json", "markdown", "html", "png"):
                self.assertTrue(paths[key].exists(), key)
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["columns"][1], "Gamma Before")
            self.assertIn("Gamma Before", paths["markdown"].read_text(encoding="utf-8"))
            self.assertGreater(paths["png"].stat().st_size, 1000)

    def test_write_comparison_card_sanitizes_run_id_under_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "runs"
            outside = Path(tmp) / "outside"
            run_id = str(outside)

            paths = capability_card.write_comparison_card(
                ["Gemma"],
                [],
                output_dir=out_dir,
                run_id=run_id,
                primary_column="Gemma",
            )

            resolved_out_dir = out_dir.resolve()
            for path in paths.values():
                self.assertTrue(path.resolve().is_relative_to(resolved_out_dir))
            self.assertFalse(Path(str(outside) + ".comparison-card.json").exists())
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], run_id)

    def test_run_choice_benchmark_sanitizes_run_id_under_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "runs"
            outside = Path(tmp) / "outside"
            run_id = str(outside)

            summary = capability_card.run_choice_benchmark(
                tasks=[],
                output_dir=out_dir,
                run_id=run_id,
            )

            summary_path = Path(summary["summary_path"])
            self.assertTrue(summary_path.resolve().is_relative_to(out_dir.resolve()))
            self.assertFalse(Path(str(outside) + ".choice.summary.json").exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], run_id)


if __name__ == "__main__":
    unittest.main()
