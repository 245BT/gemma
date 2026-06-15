import json
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import gemma_reasoning_proxy


class ReasoningProxyTests(unittest.TestCase):
    def test_upstream_timeout_is_unlimited_by_default(self):
        self.assertIsNone(gemma_reasoning_proxy.UPSTREAM_TIMEOUT_SECONDS)

    def test_reasoning_proxy_source_has_no_output_token_cap_controls(self):
        source = Path(gemma_reasoning_proxy.__file__).read_text(encoding="utf-8")

        for forbidden in (
            "MAX_OUTPUT",
            "_output_cap",
            "_small_output_cap",
            "_apply_output_token_budget",
            "max_output_tokens",
            "max_tokens",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

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
        payload = {"input": "Reply with only the digit 4."}
        self.assertTrue(gemma_reasoning_proxy.should_use_direct_fast_path(payload))

    def test_should_use_direct_fast_path_accepts_short_simple_uncapped_prompt_without_literal_matching(self):
        payload = {"input": "Give me a short status check.", "stream": True}

        self.assertTrue(gemma_reasoning_proxy.should_use_direct_fast_path(payload))

        forwarded = gemma_reasoning_proxy.fast_path_payload(payload)
        self.assertNotIn("max_output_tokens", forwarded)

    def test_should_use_direct_fast_path_accepts_passive_tool_declarations_for_exact_output(self):
        payload = {
            "input": "Reply with only the digit 4.",
            "tools": [{"type": "function", "name": "noop"}],
        }

        self.assertTrue(gemma_reasoning_proxy.should_use_direct_fast_path(payload))

    def test_should_use_direct_fast_path_accepts_passive_parallel_tool_flag_for_simple_prompt(self):
        payload = {
            "input": "hi",
            "tools": [{"type": "function", "name": "shell_command"}],
            "parallel_tool_calls": True,
        }

        self.assertTrue(gemma_reasoning_proxy.should_use_direct_fast_path(payload))

    def test_should_use_direct_fast_path_rejects_explicit_force_reasoning_flag(self):
        payload = {
            "input": "Reply with only the digit 4.",
            "x_gemma_force_reasoning": True,
        }

        self.assertFalse(gemma_reasoning_proxy.should_use_direct_fast_path(payload))
        self.assertNotIn("x_gemma_force_reasoning", gemma_reasoning_proxy.fast_path_payload(payload))

    def test_should_use_lightweight_reasoning_accepts_only_simple_forced_exact_prompts(self):
        simple = {
            "input": "Reply with only the digit 4.",
            "x_gemma_force_reasoning": True,
        }
        complex_prompt = {
            "input": "Research, cite sources, and compare the latest benchmarks.",
            "x_gemma_force_reasoning": True,
        }

        self.assertTrue(gemma_reasoning_proxy.should_use_lightweight_reasoning(simple))
        self.assertFalse(gemma_reasoning_proxy.should_use_lightweight_reasoning(complex_prompt))

    def test_should_use_direct_fast_path_rejects_bare_exactly_questions(self):
        cases = [
            {"input": "What exactly caused it?"},
            {"input": "Explain exactly why 2+2 is 4."},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertFalse(gemma_reasoning_proxy.should_use_direct_fast_path(payload))

    def test_should_use_direct_fast_path_rejects_complex_or_uncapped_prompts(self):
        cases = [
            {"input": "Research the latest benchmark result."},
            {"input": "Create probe.txt."},
            {"input": "Edit gemma_reasoning_proxy.py to add a feature."},
            {"input": "Debug why the proxy test fails."},
            {"input": "Reply with only OK. " * 40},
            {"input": "Think deeply and plan the answer first."},
            {"input": "Reply with only OK.", "tool_choice": "required"},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertFalse(gemma_reasoning_proxy.should_use_direct_fast_path(payload))

    def test_should_raw_forward_tool_request_ignores_no_tool_choices(self):
        cases = [
            {"input": "Reply with OK.", "tool_choice": "none"},
            {"input": "Reply with OK.", "tool_choice": "auto"},
            {"input": "Reply with OK.", "parallel_tool_calls": True},
            {"input": "Reply with OK.", "tools": []},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertFalse(gemma_reasoning_proxy.should_raw_forward_tool_request(payload))

    def test_should_raw_forward_tool_request_accepts_real_tool_requests(self):
        cases = [
            {"tools": [{"type": "function", "name": "noop"}]},
            {"tool_choice": "required"},
            {"tool_choice": {"type": "function", "name": "noop"}},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertTrue(gemma_reasoning_proxy.should_raw_forward_tool_request(payload))

    def test_should_raw_forward_tool_request_ignores_passive_tools_for_simple_greeting(self):
        payload = {
            "input": "hi",
            "tools": [{"type": "function", "name": "shell_command"}],
            "parallel_tool_calls": True,
        }

        self.assertFalse(gemma_reasoning_proxy.should_raw_forward_tool_request(payload))

    def test_prepare_reasoning_payload_strips_unsupported_tool_schemas_and_leaves_uncapped_requests_uncapped(self):
        payload = {
            "input": "Investigate why the local coding agent stalls and propose a fix.",
            "stream": True,
            "tools": [{"type": "custom", "name": "shell_command"}],
            "parallel_tool_calls": True,
            "x_gemma_force_reasoning": True,
        }

        prepared = gemma_reasoning_proxy.prepare_reasoning_payload(payload)

        self.assertNotIn("tools", prepared)
        self.assertNotIn("parallel_tool_calls", prepared)
        self.assertNotIn("x_gemma_force_reasoning", prepared)
        self.assertNotIn("max_output_tokens", prepared)
        self.assertTrue(prepared["stream"])

    def test_prepare_reasoning_payload_preserves_caller_fields_without_cap_rewrite(self):
        payload = {
            "input": "Investigate why the local coding agent stalls and propose a fix.",
            "max_tokens": 999999,
            "x_gemma_force_reasoning": True,
        }

        prepared = gemma_reasoning_proxy.prepare_reasoning_payload(payload)

        self.assertEqual(prepared["max_tokens"], 999999)
        self.assertNotIn("max_output_tokens", prepared)

    def test_fast_path_payload_preserves_caller_fields_without_cap_rewrite(self):
        payload = {
            "input": "Give me a short status check.",
            "max_tokens": 999999,
        }

        forwarded = gemma_reasoning_proxy.fast_path_payload(payload)

        self.assertEqual(forwarded["max_tokens"], 999999)
        self.assertNotIn("max_output_tokens", forwarded)

    def test_reasoning_handler_fast_path_uses_upstream_without_graph_runner(self):
        request_payload = {"input": "Reply with only the digit 4."}
        upstream_payload = gemma_reasoning_proxy.build_response_payload("4")
        handler, sent = self._handler_for_payload(request_payload)

        with (
            patch("gemma_reasoning_proxy.forward_response_to_upstream", create=True, return_value=upstream_payload) as forward,
            patch("gemma_reasoning_proxy.run_reasoning_request") as run_reasoning_request,
        ):
            handler.handle_reasoning_response()

        run_reasoning_request.assert_not_called()
        forwarded_payload = forward.call_args.args[0]
        self.assertEqual(
            forwarded_payload,
            {"input": "Reply with only the digit 4.", "stream": False},
        )
        self.assertEqual(forward.call_args.kwargs["upstream"], "http://127.0.0.1:9000")
        self.assertEqual(sent["status"], 200)
        self.assertEqual(sent["payload"], upstream_payload)

    def test_reasoning_handler_normalizes_fast_path_upstream_output_text(self):
        request_payload = {"input": "Reply with only OK."}
        upstream_payload = {
            "id": "resp_test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "OK",
                        }
                    ],
                }
            ],
        }
        handler, sent = self._handler_for_payload(request_payload)

        with patch("gemma_reasoning_proxy.forward_response_to_upstream", create=True, return_value=upstream_payload):
            handler.handle_reasoning_response()

        self.assertEqual(sent["status"], 200)
        self.assertEqual(sent["payload"]["output_text"], "OK")

    def test_reasoning_handler_cleans_fast_path_output_text_and_nested_output(self):
        request_payload = {"input": "Reply with only OK."}
        upstream_payload = {
            "id": "resp_test",
            "status": "completed",
            "output_text": "<|channel>final<channel|>OK",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "<|channel>final<channel|>OK",
                        }
                    ],
                }
            ],
        }
        handler, sent = self._handler_for_payload(request_payload)

        with patch("gemma_reasoning_proxy.forward_response_to_upstream", create=True, return_value=upstream_payload):
            handler.handle_reasoning_response()

        encoded = json.dumps(sent["payload"])
        self.assertEqual(sent["status"], 200)
        self.assertEqual(sent["payload"]["output_text"], "OK")
        self.assertEqual(sent["payload"]["output"][0]["content"][0]["text"], "OK")
        self.assertNotIn("<|channel>", encoded)
        self.assertNotIn("<channel|>", encoded)

    def test_reasoning_handler_streaming_fast_path_forwards_non_stream_and_wraps_sse(self):
        request_payload = {"input": "Give me a short status check.", "stream": True}
        upstream_payload = gemma_reasoning_proxy.build_response_payload("ok")
        handler, sent = self._handler_for_payload(request_payload)

        def send_bytes(status, body, content_type):
            sent["status"] = status
            sent["body"] = body
            sent["content_type"] = content_type

        handler.send_bytes = send_bytes

        with patch("gemma_reasoning_proxy.forward_response_to_upstream", create=True, return_value=upstream_payload) as forward:
            handler.handle_reasoning_response()

        forwarded_payload = forward.call_args.args[0]
        self.assertFalse(forwarded_payload["stream"])
        self.assertNotIn("max_output_tokens", forwarded_payload)
        self.assertEqual(sent["status"], 200)
        self.assertEqual(sent["content_type"], "text/event-stream")
        self.assertIn("event: response.completed", sent["body"].decode("utf-8"))

    def test_reasoning_handler_raw_forwards_tool_requests_to_preserve_codex_events(self):
        request_payload = {
            "input": "Create probe.txt.",
            "stream": True,
            "tools": [{"type": "function", "name": "shell_command"}],
        }
        handler, sent = self._handler_for_payload(request_payload)

        def send_bytes(status, body, content_type):
            sent["status"] = status
            sent["body"] = body
            sent["content_type"] = content_type

        handler.send_bytes = send_bytes

        with (
            patch("gemma_reasoning_proxy.forward_raw_response_to_upstream", return_value=(200, b"event: response.completed\n\n", "text/event-stream")) as forward,
            patch("gemma_reasoning_proxy.run_reasoning_request") as run_reasoning_request,
        ):
            handler.handle_reasoning_response()

        run_reasoning_request.assert_not_called()
        forward.assert_called_once_with(request_payload, upstream="http://127.0.0.1:9000")
        self.assertEqual(sent["status"], 200)
        self.assertEqual(sent["body"], b"event: response.completed\n\n")
        self.assertEqual(sent["content_type"], "text/event-stream")

    def test_reasoning_handler_cleans_raw_forwarded_stream_before_frontend(self):
        request_payload = {
            "input": "Create probe.txt.",
            "stream": True,
            "tools": [{"type": "function", "name": "noop"}],
        }
        raw_body = (
            b'data: {"type":"response.output_text.delta","delta":"<|channel>thought"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"<|channel>rium                         "}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"<channel|>hidden"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"<|channel>final<channel|>Hello father."}\n\n'
        )
        handler, sent = self._handler_for_payload(request_payload)

        def send_bytes(status, body, content_type):
            sent["status"] = status
            sent["body"] = body
            sent["content_type"] = content_type

        handler.send_bytes = send_bytes

        with patch(
            "gemma_reasoning_proxy.forward_raw_response_to_upstream",
            return_value=(200, raw_body, "text/event-stream"),
        ):
            handler.handle_reasoning_response()

        cleaned = sent["body"].decode("utf-8")
        self.assertEqual(sent["status"], 200)
        self.assertNotIn("<|channel>", cleaned)
        self.assertNotIn("<channel|>", cleaned)
        self.assertNotIn("hidden", cleaned)
        self.assertIn('"delta": "Hello father."', cleaned)

    def test_reasoning_handler_simple_greeting_strips_passive_tools_before_upstream(self):
        request_payload = {
            "input": "hi",
            "stream": True,
            "tools": [{"type": "function", "name": "shell_command"}],
            "parallel_tool_calls": True,
        }
        upstream_payload = gemma_reasoning_proxy.build_response_payload("Hi father.")
        handler, sent = self._handler_for_payload(request_payload)

        def send_bytes(status, body, content_type):
            sent["status"] = status
            sent["body"] = body
            sent["content_type"] = content_type

        handler.send_bytes = send_bytes

        with (
            patch("gemma_reasoning_proxy.forward_response_to_upstream", return_value=upstream_payload) as forward,
            patch("gemma_reasoning_proxy.forward_raw_response_to_upstream") as forward_raw,
            patch("gemma_reasoning_proxy.run_reasoning_request") as run_reasoning_request,
        ):
            handler.handle_reasoning_response()

        forward_raw.assert_not_called()
        run_reasoning_request.assert_not_called()
        forwarded_payload = forward.call_args.args[0]
        self.assertEqual(forwarded_payload["input"], "hi")
        self.assertFalse(forwarded_payload["stream"])
        self.assertNotIn("tools", forwarded_payload)
        self.assertNotIn("parallel_tool_calls", forwarded_payload)
        self.assertEqual(sent["status"], 200)
        self.assertEqual(sent["content_type"], "text/event-stream")
        self.assertIn("Hi father.", sent["body"].decode("utf-8"))

    def test_reasoning_handler_preserves_explicit_required_tools_by_raw_forwarding(self):
        request_payload = {
            "input": "Reply with only the digit 4.",
            "tools": [{"type": "function", "name": "noop"}],
            "tool_choice": "required",
        }
        handler, sent = self._handler_for_payload(request_payload)

        def send_bytes(status, body, content_type):
            sent["status"] = status
            sent["body"] = body
            sent["content_type"] = content_type

        handler.send_bytes = send_bytes

        with patch("gemma_reasoning_proxy.forward_raw_response_to_upstream", return_value=(200, b'{"output_text":"4"}', "application/json")) as forward:
            handler.handle_reasoning_response()

        forwarded_payload = forward.call_args.args[0]
        self.assertIn("tools", forwarded_payload)
        self.assertEqual(forwarded_payload["tool_choice"], "required")
        self.assertEqual(forwarded_payload["input"], "Reply with only the digit 4.")
        self.assertEqual(sent["status"], 200)
        self.assertEqual(sent["content_type"], "application/json")

    def test_run_reasoning_request_passes_custom_upstream_to_client(self):
        with (
            patch("gemma_reasoning.upstream.UpstreamResponsesClient") as client_cls,
            patch("gemma_reasoning.graph.run_reasoning_graph", return_value={"final_text": "hello"}),
        ):
            gemma_reasoning_proxy.run_reasoning_request(
                {"input": "say hi"},
                upstream="http://127.0.0.1:9000",
            )

        client_cls.assert_called_once_with(
            "http://127.0.0.1:9000",
            timeout=gemma_reasoning_proxy.UPSTREAM_TIMEOUT_SECONDS,
        )

    def test_run_reasoning_request_uses_lightweight_graph_for_simple_forced_request(self):
        payload = {
            "input": "Reply with only the digit 4.",
            "x_gemma_force_reasoning": True,
        }

        with patch("gemma_reasoning.graph.run_reasoning_graph", return_value={"final_text": "4"}) as graph_runner:
            gemma_reasoning_proxy.run_reasoning_request(payload)

        _, kwargs = graph_runner.call_args
        self.assertFalse(kwargs["config"].use_langgraph)
        self.assertEqual(kwargs["programs"].plan("task", []), "Answer the task directly.")

    def test_run_reasoning_request_uses_single_pass_local_reasoning_for_complex_codex_request(self):
        payload = {
            "input": "Investigate why the local coding agent stalls and propose a fix.",
            "stream": True,
            "tools": [{"type": "custom", "name": "shell_command"}],
        }

        with (
            patch("gemma_reasoning.dspy_programs.DspyPrograms") as dspy_programs,
            patch("gemma_reasoning.graph.run_reasoning_graph", return_value={"final_text": "bounded answer"}) as graph_runner,
        ):
            response = gemma_reasoning_proxy.run_reasoning_request(payload)

        dspy_programs.assert_not_called()
        graph_payload = graph_runner.call_args.args[0]
        _, kwargs = graph_runner.call_args
        self.assertFalse(kwargs["config"].use_langgraph)
        self.assertEqual(kwargs["config"].max_revisions, 0)
        self.assertNotIn("tools", graph_payload)
        self.assertNotIn("max_output_tokens", graph_payload)
        self.assertEqual(response["output_text"], "bounded answer")

    def test_forward_raw_response_to_upstream_returns_timeout_for_wrapped_urlerror(self):
        with patch("gemma_reasoning_proxy.urlopen", side_effect=URLError(TimeoutError("timed out"))):
            status, body, content_type = gemma_reasoning_proxy.forward_raw_response_to_upstream(
                {"input": "use a tool"},
                upstream="http://127.0.0.1:9000",
            )

        self.assertEqual(status, 504)
        self.assertEqual(content_type, "application/json")
        self.assertIn("Gemma upstream request timed out", body.decode("utf-8"))

    def test_reasoning_handler_fast_path_returns_timeout_for_wrapped_urlerror(self):
        request_payload = {"input": "Reply with only OK."}
        handler, sent = self._handler_for_payload(request_payload)

        with patch("gemma_reasoning_proxy.urlopen", side_effect=URLError(TimeoutError("timed out"))):
            handler.handle_reasoning_response()

        self.assertEqual(sent["status"], 504)
        self.assertIn("Gemma upstream request timed out", json.dumps(sent["payload"]))

    def test_reasoning_handler_forward_to_upstream_returns_bad_gateway_for_oserror(self):
        handler = object.__new__(gemma_reasoning_proxy.ReasoningProxyHandler)
        handler.upstream = "http://127.0.0.1:9000"
        handler.path = "/v1/models"
        handler.command = "GET"
        handler.headers = {"Content-Length": "0"}
        handler.rfile = BytesIO()
        sent = {}

        def send_bytes(status, body, content_type):
            sent["status"] = status
            sent["body"] = body
            sent["content_type"] = content_type

        handler.send_bytes = send_bytes

        with patch("gemma_reasoning_proxy.urlopen", side_effect=OSError("connection refused")):
            handler.forward_to_upstream()

        self.assertEqual(sent["status"], 502)
        self.assertEqual(sent["content_type"], "application/json")
        self.assertIn("Reasoning proxy request failed", sent["body"].decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
