import json
import unittest
from io import BytesIO
from unittest.mock import patch

import gemma_reasoning_proxy


class ReasoningProxyTests(unittest.TestCase):
    def test_upstream_timeout_is_bounded(self):
        self.assertLessEqual(gemma_reasoning_proxy.UPSTREAM_TIMEOUT_SECONDS, 120)

    def _handler_for_payload(self, payload):
        body = json.dumps(payload).encode("utf-8")
        handler = object.__new__(gemma_reasoning_proxy.ReasoningProxyHandler)
        handler.upstream = "http://127.0.0.1:9000"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = BytesIO(body)
        sent = {}

        def send_json(status, response_payload):
            sent["status"] = status
            sent["payload"] = response_payload

        handler.send_json = send_json
        return handler, sent

    def test_build_response_payload_wraps_final_text(self):
        payload = gemma_reasoning_proxy.build_response_payload("hello")
        text = payload["output"][0]["content"][0]["text"]
        self.assertEqual(text, "hello")
        self.assertEqual(payload["object"], "response")

    def test_build_sse_payload_emits_done(self):
        body = gemma_reasoning_proxy.build_sse_payload("hello").decode("utf-8")
        self.assertIn("event: response.created", body)
        self.assertIn("event: response.output_text.delta", body)
        self.assertIn('"delta": "hello"', body)
        self.assertIn("event: response.output_text.done", body)
        self.assertIn("event: response.completed", body)

    def test_run_reasoning_request_uses_graph_runner(self):
        request_payload = {"input": [{"role": "user", "content": "say hi"}]}
        result = gemma_reasoning_proxy.run_reasoning_request(
            request_payload,
            graph_runner=lambda payload: {
                "final_text": "hello",
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                "model_calls": 1,
                "context_length": 128,
                "cache_hit": True,
                "cache_hit_rate": 0.5,
            },
        )
        self.assertEqual(result["output"][0]["content"][0]["text"], "hello")
        self.assertEqual(result["output_text"], "hello")
        self.assertEqual(result["usage"], {"prompt_tokens": 5, "completion_tokens": 1})
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["context_length"], 128)
        self.assertTrue(result["cache_hit"])
        self.assertEqual(result["cache_hit_rate"], 0.5)

    def test_build_error_payload_sanitizes_raw_exception_details(self):
        payload = gemma_reasoning_proxy.build_error_payload("boom password=secret-token")
        encoded = json.dumps(payload)
        self.assertEqual(payload["error"]["message"], "Reasoning proxy request failed.")
        self.assertNotIn("secret-token", encoded)

    def test_should_use_direct_fast_path_accepts_exact_short_prompt(self):
        payload = {"input": "Reply with only the digit 4.", "max_output_tokens": 8}
        self.assertTrue(gemma_reasoning_proxy.should_use_direct_fast_path(payload))

    def test_should_use_direct_fast_path_accepts_passive_tool_declarations_for_exact_output(self):
        payload = {
            "input": "Reply with only the digit 4.",
            "max_output_tokens": 8,
            "tools": [{"type": "function", "name": "noop"}],
        }

        self.assertTrue(gemma_reasoning_proxy.should_use_direct_fast_path(payload))

    def test_should_use_direct_fast_path_rejects_explicit_force_reasoning_flag(self):
        payload = {
            "input": "Reply with only the digit 4.",
            "max_output_tokens": 8,
            "x_gemma_force_reasoning": True,
        }

        self.assertFalse(gemma_reasoning_proxy.should_use_direct_fast_path(payload))
        self.assertNotIn("x_gemma_force_reasoning", gemma_reasoning_proxy.fast_path_payload(payload))

    def test_should_use_lightweight_reasoning_accepts_only_simple_forced_exact_prompts(self):
        simple = {
            "input": "Reply with only the digit 4.",
            "max_output_tokens": 8,
            "x_gemma_force_reasoning": True,
        }
        complex_prompt = {
            "input": "Research, cite sources, and compare the latest benchmarks.",
            "max_output_tokens": 8,
            "x_gemma_force_reasoning": True,
        }

        self.assertTrue(gemma_reasoning_proxy.should_use_lightweight_reasoning(simple))
        self.assertFalse(gemma_reasoning_proxy.should_use_lightweight_reasoning(complex_prompt))

    def test_should_use_direct_fast_path_rejects_bare_exactly_questions(self):
        cases = [
            {"input": "What exactly caused it?", "max_output_tokens": 8},
            {"input": "Explain exactly why 2+2 is 4.", "max_output_tokens": 8},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertFalse(gemma_reasoning_proxy.should_use_direct_fast_path(payload))

    def test_should_use_direct_fast_path_rejects_complex_or_uncapped_prompts(self):
        cases = [
            {"input": "Research the latest benchmark result.", "max_output_tokens": 8},
            {"input": "Edit gemma_reasoning_proxy.py to add a feature.", "max_output_tokens": 8},
            {"input": "Debug why the proxy test fails.", "max_output_tokens": 8},
            {"input": "Reply with only OK. " * 40, "max_output_tokens": 8},
            {"input": "Think deeply and plan the answer first.", "max_output_tokens": 8},
            {"input": "Reply with only OK.", "max_output_tokens": 64},
            {"input": "Reply with only OK.", "tool_choice": "required", "max_output_tokens": 8},
            {"input": "Reply with only OK.", "parallel_tool_calls": True, "max_output_tokens": 8},
            {"input": "Reply with only OK."},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertFalse(gemma_reasoning_proxy.should_use_direct_fast_path(payload))

    def test_reasoning_handler_fast_path_uses_upstream_without_graph_runner(self):
        request_payload = {"input": "Reply with only the digit 4.", "max_output_tokens": 8}
        upstream_payload = gemma_reasoning_proxy.build_response_payload("4")
        handler, sent = self._handler_for_payload(request_payload)

        with (
            patch("gemma_reasoning_proxy.forward_response_to_upstream", create=True, return_value=upstream_payload) as forward,
            patch("gemma_reasoning_proxy.run_reasoning_request") as run_reasoning_request,
        ):
            handler.handle_reasoning_response()

        run_reasoning_request.assert_not_called()
        forward.assert_called_once_with(request_payload, upstream="http://127.0.0.1:9000")
        self.assertEqual(sent["status"], 200)
        self.assertEqual(sent["payload"], upstream_payload)

    def test_reasoning_handler_strips_passive_tools_on_exact_output_fast_path(self):
        request_payload = {
            "input": "Reply with only the digit 4.",
            "max_output_tokens": 8,
            "tools": [{"type": "function", "name": "noop"}],
        }
        upstream_payload = gemma_reasoning_proxy.build_response_payload("4")
        handler, sent = self._handler_for_payload(request_payload)

        with patch("gemma_reasoning_proxy.forward_response_to_upstream", create=True, return_value=upstream_payload) as forward:
            handler.handle_reasoning_response()

        forwarded_payload = forward.call_args.kwargs["request_payload"] if "request_payload" in forward.call_args.kwargs else forward.call_args.args[0]
        self.assertNotIn("tools", forwarded_payload)
        self.assertEqual(forwarded_payload["input"], "Reply with only the digit 4.")
        self.assertEqual(sent["status"], 200)

    def test_run_reasoning_request_passes_custom_upstream_to_client(self):
        with (
            patch("gemma_reasoning.upstream.UpstreamResponsesClient") as client_cls,
            patch("gemma_reasoning.graph.run_reasoning_graph", return_value={"final_text": "hello"}),
        ):
            gemma_reasoning_proxy.run_reasoning_request(
                {"input": "say hi"},
                upstream="http://127.0.0.1:9000",
            )

        client_cls.assert_called_once_with("http://127.0.0.1:9000")

    def test_run_reasoning_request_uses_lightweight_graph_for_simple_forced_request(self):
        payload = {
            "input": "Reply with only the digit 4.",
            "max_output_tokens": 8,
            "x_gemma_force_reasoning": True,
        }

        with patch("gemma_reasoning.graph.run_reasoning_graph", return_value={"final_text": "4"}) as graph_runner:
            gemma_reasoning_proxy.run_reasoning_request(payload)

        _, kwargs = graph_runner.call_args
        self.assertFalse(kwargs["config"].use_langgraph)
        self.assertEqual(kwargs["programs"].plan("task", []), "Answer the task directly.")


if __name__ == "__main__":
    unittest.main()
