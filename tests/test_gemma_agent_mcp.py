import json
import unittest

import gemma_agent_mcp


class FakeModelClient:
    def __init__(self):
        self.payloads = []

    def create_response(self, payload):
        self.payloads.append(payload)
        task = payload["input"][1]["content"]
        return json.dumps({"action": "final", "content": f"done:{task}"})


class RecursiveModelClient:
    def __init__(self):
        self.grandchild_calls = 0

    def create_response(self, payload):
        task = payload["input"][1]["content"]
        if task == "child" and not payload["subagent_results"]:
            return json.dumps(
                {
                    "action": "spawn_subagents",
                    "tasks": [{"id": "grandchild", "task": "grandchild"}],
                }
            )
        if task == "grandchild":
            self.grandchild_calls += 1
            return json.dumps({"action": "final", "content": "grandchild done"})
        return json.dumps({"action": "final", "content": "child done"})


class GemmaAgentMCPTests(unittest.TestCase):
    def test_gemma_run_subagents_response_runs_tasks_with_external_supervisor(self):
        model = FakeModelClient()

        result = gemma_agent_mcp.gemma_run_subagents_response(
            ["alpha", "beta"],
            model_client=model,
            concurrent=False,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["task_count"], 2)
        self.assertEqual([item["final"] for item in result["subagents"]], ["done:alpha", "done:beta"])
        self.assertTrue(all(item["ok"] for item in result["subagents"]))
        self.assertEqual(len(model.payloads), 2)

    def test_gemma_run_subagents_response_rejects_unsafe_fanout(self):
        result = gemma_agent_mcp.gemma_run_subagents_response([str(index) for index in range(9)])

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "validation_error")
        self.assertIn("between 1 and 8", result["message"])

    def test_gemma_run_subagents_response_rejects_non_string_tasks(self):
        result = gemma_agent_mcp.gemma_run_subagents_response([{"task": "alpha"}])

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "validation_error")
        self.assertIn("non-empty strings", result["message"])

    def test_gemma_run_subagents_disables_nested_subagents(self):
        model = RecursiveModelClient()

        result = gemma_agent_mcp.gemma_run_subagents_response(
            ["child"],
            model_client=model,
            concurrent=False,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["subagents"][0]["final"], "child done")
        self.assertEqual(model.grandchild_calls, 0)

    def test_gemma_run_subagents_default_client_includes_model_slug(self):
        fake_model = FakeModelClient()

        with unittest.mock.patch("gemma_agent_mcp.LocalResponsesClient", return_value=fake_model) as client_cls:
            gemma_agent_mcp.gemma_run_subagents_response(["alpha"], concurrent=False)

        self.assertEqual(client_cls.call_args.kwargs["model"], gemma_agent_mcp.DEFAULT_MODEL)

    def test_build_server_returns_fastmcp_server(self):
        server = gemma_agent_mcp.build_server()

        self.assertEqual(server.name, "gemma-agent")


if __name__ == "__main__":
    unittest.main()
