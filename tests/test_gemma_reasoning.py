import unittest
from unittest.mock import patch

from gemma_reasoning.graph import ReasoningConfig, run_reasoning_graph


class FakePrograms:
    def __init__(self, approvals=None):
        self.approvals = list(approvals or [True])
        self.plan_calls = []
        self.verify_calls = []

    def plan(self, task, constraints):
        self.plan_calls.append((task, constraints))
        return "Plan: answer directly."

    def verify(self, task, plan, draft, constraints=None):
        self.verify_calls.append((task, plan, draft, constraints or []))
        approved = self.approvals.pop(0) if self.approvals else True
        return {"approved": approved, "notes": "Checked draft."}


class FakeClient:
    def __init__(self, texts=None):
        self.texts = list(texts or ["final answer"])
        self.calls = []

    def create_response(self, payload):
        self.calls.append(payload)
        text = self.texts.pop(0) if self.texts else "final answer"
        return {"output": [{"content": [{"type": "output_text", "text": text}]}]}


class UsageClient:
    def __init__(self):
        self.calls = []

    def create_response(self, payload):
        self.calls.append(payload)
        return {
            "output": [{"content": [{"type": "output_text", "text": "final answer"}]}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            "context_length": 256,
            "cache_hit": True,
            "cache_hit_rate": 0.75,
        }


class ReasoningGraphTests(unittest.TestCase):
    def test_run_reasoning_graph_returns_approved_draft(self):
        client = FakeClient()
        programs = FakePrograms()
        result = run_reasoning_graph(
            {"input": [{"role": "user", "content": "say hi"}]},
            client=client,
            programs=programs,
            config=ReasoningConfig(),
        )
        self.assertEqual(result["final_text"], "final answer")
        self.assertEqual(result["plan"], "Plan: answer directly.")
        self.assertEqual(result["verifier"]["approved"], True)
        self.assertEqual(result["revisions"], 0)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(programs.plan_calls, [("say hi", [])])

    def test_run_reasoning_graph_returns_usage_metadata_from_model_calls(self):
        client = UsageClient()

        result = run_reasoning_graph(
            {"input": [{"role": "user", "content": "say hi"}]},
            client=client,
            programs=FakePrograms(),
            config=ReasoningConfig(),
        )

        self.assertEqual(result["usage"], {"prompt_tokens": 11, "completion_tokens": 3})
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["context_length"], 256)
        self.assertTrue(result["cache_hit"])
        self.assertEqual(result["cache_hit_rate"], 0.75)

    def test_run_reasoning_graph_revises_once_when_verifier_rejects(self):
        client = FakeClient(["draft answer", "revised answer"])
        result = run_reasoning_graph(
            {"input": [{"role": "user", "content": "say hi"}]},
            client=client,
            programs=FakePrograms([False, True]),
            config=ReasoningConfig(max_revisions=1),
        )
        self.assertEqual(result["final_text"], "revised answer")
        self.assertEqual(result["verifier"]["approved"], True)
        self.assertEqual(result["revisions"], 1)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("Missing or incomplete answer", client.calls[1]["input"][2]["content"])

    def test_draft_keeps_original_instructions_and_adds_plan_as_input_context(self):
        client = FakeClient()
        run_reasoning_graph(
            {
                "instructions": "System level instruction.",
                "input": [{"role": "user", "content": "say hi"}],
            },
            client=client,
            programs=FakePrograms(),
            config=ReasoningConfig(),
        )
        self.assertEqual(client.calls[0]["instructions"], "System level instruction.")
        self.assertEqual(client.calls[0]["input"][0], {"role": "user", "content": "say hi"})
        self.assertIn("Private non-authoritative reasoning plan", client.calls[0]["input"][1]["content"])

    def test_planner_receives_instructions_as_constraints(self):
        programs = FakePrograms()
        run_reasoning_graph(
            {
                "instructions": "Follow project rules.",
                "input": [{"role": "developer", "content": "Developer rule."}, {"role": "user", "content": "task"}],
            },
            client=FakeClient(),
            programs=programs,
            config=ReasoningConfig(constraints=["extra constraint"]),
        )
        constraints = programs.plan_calls[0][1]
        self.assertIn("instructions: Follow project rules.", constraints)
        self.assertIn("developer: Developer rule.", constraints)
        self.assertIn("extra constraint", constraints)
        verifier_constraints = programs.verify_calls[0][3]
        self.assertIn("instructions: Follow project rules.", verifier_constraints)
        self.assertIn("developer: Developer rule.", verifier_constraints)
        self.assertIn("extra constraint", verifier_constraints)

    def test_revision_preserves_original_input_context(self):
        client = FakeClient(["draft answer", "revised answer"])
        run_reasoning_graph(
            {"input": [{"role": "user", "content": "original task"}]},
            client=client,
            programs=FakePrograms([False, True]),
            config=ReasoningConfig(max_revisions=1),
        )
        revision_input = client.calls[1]["input"]
        self.assertEqual(revision_input[0], {"role": "user", "content": "original task"})
        self.assertIn("draft answer", revision_input[1]["content"])
        self.assertIn("Verifier notes", revision_input[2]["content"])

    def test_run_reasoning_graph_extracts_latest_user_message(self):
        programs = FakePrograms()
        result = run_reasoning_graph(
            {
                "input": [
                    {"role": "user", "content": "old task"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "new task"},
                ]
            },
            client=FakeClient(),
            programs=programs,
            config=ReasoningConfig(),
        )
        self.assertEqual(result["final_text"], "final answer")
        self.assertEqual(programs.plan_calls[0][0], "new task")

    def test_run_reasoning_graph_extracts_text_from_content_list(self):
        programs = FakePrograms()
        run_reasoning_graph(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "first"},
                            {"type": "input_text", "text": "second"},
                        ],
                    }
                ]
            },
            client=FakeClient(),
            programs=programs,
            config=ReasoningConfig(),
        )
        self.assertEqual(programs.plan_calls[0][0], "first\nsecond")

    def test_run_reasoning_graph_accepts_string_input(self):
        programs = FakePrograms()
        run_reasoning_graph(
            {"input": "plain task"},
            client=FakeClient(),
            programs=programs,
            config=ReasoningConfig(),
        )
        self.assertEqual(programs.plan_calls[0][0], "plain task")

    def test_run_reasoning_graph_requires_langgraph(self):
        with patch("gemma_reasoning.graph._load_langgraph", side_effect=ImportError("missing langgraph"), create=True):
            with self.assertRaisesRegex(RuntimeError, "LangGraph is required"):
                run_reasoning_graph(
                    {"input": "plain task"},
                    client=FakeClient(),
                    programs=FakePrograms(),
                    config=ReasoningConfig(),
                )

    def test_run_reasoning_graph_can_use_sequential_execution_without_langgraph(self):
        with patch("gemma_reasoning.graph._load_langgraph", side_effect=ImportError("missing langgraph"), create=True):
            result = run_reasoning_graph(
                {"input": "plain task"},
                client=FakeClient(),
                programs=FakePrograms(),
                config=ReasoningConfig(use_langgraph=False),
            )

        self.assertEqual(result["final_text"], "final answer")
        self.assertEqual(result["model_calls"], 1)


if __name__ == "__main__":
    unittest.main()
