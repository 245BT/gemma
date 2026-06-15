import json
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import gemma_response_proxy


class GemmaResponseProxyTests(unittest.TestCase):
    def test_upstream_timeout_is_unlimited_by_default(self):
        self.assertIsNone(gemma_response_proxy.UPSTREAM_TIMEOUT_SECONDS)

    def test_response_proxy_source_has_no_output_token_cap_controls(self):
        source = Path(gemma_response_proxy.__file__).read_text(encoding="utf-8")

        for forbidden in (
            "GEMMA_RESPONSE_MAX_OUTPUT_TOKENS",
            "MAX_OUTPUT_TOKENS",
            "response_max_output_tokens",
            "cap_response_request_payload",
            "max_output_tokens",
            "max_tokens",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_response_proxy_leaves_uncapped_responses_request_uncapped_before_forwarding(self):
        body = b'{"input":"write a bounded action"}'
        handler = object.__new__(gemma_response_proxy.ProxyHandler)
        handler.upstream = "http://127.0.0.1:9000"
        handler.path = "/v1/responses"
        handler.command = "POST"
        handler.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
        handler.rfile = BytesIO(body)
        handler.wfile = BytesIO()
        sent = {"headers": []}
        captured = {}

        class FakeResponse:
            status = 200
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"output_text":"ok"}'

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        handler.send_response = lambda status: sent.setdefault("status", status)
        handler.send_header = lambda key, value: sent["headers"].append((key, value))
        handler.end_headers = lambda: None

        with patch("gemma_response_proxy.urlopen", side_effect=fake_urlopen):
            handler.forward()

        self.assertNotIn("max_output_tokens", captured["payload"])
        self.assertIsNone(captured["timeout"])

    def test_response_proxy_forwards_legacy_token_fields_without_rewriting(self):
        body = b'{"input":"write a bounded action","max_tokens":999999}'
        handler = object.__new__(gemma_response_proxy.ProxyHandler)
        handler.upstream = "http://127.0.0.1:9000"
        handler.path = "/v1/responses"
        handler.command = "POST"
        handler.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
        handler.rfile = BytesIO(body)
        handler.wfile = BytesIO()
        sent = {"headers": []}
        captured = {}

        class FakeResponse:
            status = 200
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"output_text":"ok"}'

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        handler.send_response = lambda status: sent.setdefault("status", status)
        handler.send_header = lambda key, value: sent["headers"].append((key, value))
        handler.end_headers = lambda: None

        with patch("gemma_response_proxy.urlopen", side_effect=fake_urlopen):
            handler.forward()

        self.assertEqual(captured["payload"]["max_tokens"], 999999)
        self.assertNotIn("max_output_tokens", captured["payload"])

    def test_response_proxy_short_circuits_exact_greeting_without_upstream_call(self):
        body = json.dumps(
            {
                "model": "gemma",
                "input": "hi",
                "tools": [{"type": "function", "name": "shell_command"}],
                "parallel_tool_calls": True,
            }
        ).encode("utf-8")
        handler = object.__new__(gemma_response_proxy.ProxyHandler)
        handler.upstream = "http://127.0.0.1:9000"
        handler.path = "/v1/responses"
        handler.command = "POST"
        handler.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
        handler.rfile = BytesIO(body)
        handler.wfile = BytesIO()
        sent = {"headers": []}

        handler.send_response = lambda status: sent.setdefault("status", status)
        handler.send_header = lambda key, value: sent["headers"].append((key, value))
        handler.end_headers = lambda: None

        with patch("gemma_response_proxy.urlopen", side_effect=AssertionError("upstream should not be called")):
            handler.forward()

        payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(sent["status"], 200)
        self.assertIn(("Content-Type", "application/json"), sent["headers"])
        self.assertEqual(payload["output_text"], "Hi father.")
        self.assertEqual(payload["output"][0]["type"], "message")

    def test_response_proxy_short_circuits_streaming_exact_greeting(self):
        body = json.dumps(
            {
                "model": "gemma",
                "input": [{"role": "user", "content": "hello"}],
                "stream": True,
                "tools": [{"type": "function", "name": "shell_command"}],
            }
        ).encode("utf-8")
        handler = object.__new__(gemma_response_proxy.ProxyHandler)
        handler.upstream = "http://127.0.0.1:9000"
        handler.path = "/v1/responses"
        handler.command = "POST"
        handler.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
        handler.rfile = BytesIO(body)
        handler.wfile = BytesIO()
        sent = {"headers": []}

        handler.send_response = lambda status: sent.setdefault("status", status)
        handler.send_header = lambda key, value: sent["headers"].append((key, value))
        handler.end_headers = lambda: None

        with patch("gemma_response_proxy.urlopen", side_effect=AssertionError("upstream should not be called")):
            handler.forward()

        body_text = handler.wfile.getvalue().decode("utf-8")
        self.assertEqual(sent["status"], 200)
        self.assertIn(("Content-Type", "text/event-stream"), sent["headers"])
        self.assertIn("event: response.completed", body_text)
        self.assertIn("Hi father.", body_text)
        self.assertNotIn(gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE, body_text)

    def test_response_proxy_returns_gateway_timeout_for_urlerror_wrapped_timeout(self):
        body = b'{"input":"slow local turn"}'
        handler = object.__new__(gemma_response_proxy.ProxyHandler)
        handler.upstream = "http://127.0.0.1:9000"
        handler.path = "/v1/responses"
        handler.command = "POST"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = BytesIO(body)
        handler.wfile = BytesIO()
        sent = {"headers": []}

        handler.send_response = lambda status: sent.setdefault("status", status)
        handler.send_header = lambda key, value: sent["headers"].append((key, value))
        handler.end_headers = lambda: None

        with patch("gemma_response_proxy.urlopen", side_effect=URLError(TimeoutError("timed out"))):
            handler.forward()

        self.assertEqual(sent["status"], 504)
        response = handler.wfile.getvalue().decode("utf-8")
        self.assertIn("Gemma upstream request timed out", response)
        self.assertNotIn("URLError", response)

    def test_response_proxy_returns_gateway_timeout_when_upstream_times_out(self):
        body = b'{"input":"slow local turn"}'
        handler = object.__new__(gemma_response_proxy.ProxyHandler)
        handler.upstream = "http://127.0.0.1:9000"
        handler.path = "/v1/responses"
        handler.command = "POST"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = BytesIO(body)
        handler.wfile = BytesIO()
        sent = {"headers": []}

        def send_response(status):
            sent["status"] = status

        def send_header(key, value):
            sent["headers"].append((key, value))

        handler.send_response = send_response
        handler.send_header = send_header
        handler.end_headers = lambda: None

        with patch("gemma_response_proxy.urlopen", side_effect=TimeoutError("timed out")):
            handler.forward()

        self.assertEqual(sent["status"], 504)
        response = handler.wfile.getvalue().decode("utf-8")
        self.assertIn("Gemma upstream request timed out", response)
        self.assertNotIn("TimeoutError", response)

    def test_clean_channel_markers_suppresses_thought_channel_without_final(self):
        self.assertEqual(
            gemma_response_proxy.clean_channel_markers("<|channel>thought\n<channel|>4"),
            "",
        )

    def test_clean_channel_markers_suppresses_compact_thought_channel_without_final(self):
        self.assertEqual(
            gemma_response_proxy.clean_channel_markers("<|channel>thought<channel|>answer"),
            "",
        )

    def test_clean_channel_markers_suppresses_malformed_duplicate_thought_channel_without_final(self):
        self.assertEqual(
            gemma_response_proxy.clean_channel_markers(
                "<|channel>thought<|channel>rium                         <channel|>Hello father."
            ),
            "",
        )

    def test_clean_channel_markers_uses_final_after_malformed_thought_channel(self):
        self.assertEqual(
            gemma_response_proxy.clean_channel_markers(
                "<|channel>thought_thought\n  <channel|>hidden leak\n<|channel>final<channel|>visible"
            ),
            "visible",
        )

    def test_clean_channel_markers_uses_malformed_final_after_thought_channel(self):
        self.assertEqual(
            gemma_response_proxy.clean_channel_markers(
                "<|channel>thought<channel|>hidden\n<|channel>final\nvisible"
            ),
            "visible",
        )

    def test_clean_channel_markers_suppresses_unterminated_thought_channel(self):
        cleaned = gemma_response_proxy.clean_channel_markers(
            "<|channel>thought<|channel>979c4710136670e77b88916412901b7926e454626910433ba52db89e287f82\n"
            "</code>\n</code>"
        )

        self.assertEqual(cleaned, "")

    def test_clean_response_payload_updates_output_text(self):
        payload = {
            "output_text": "<|channel>thought\n<channel|>hidden\n<|channel>final<channel|>hello",
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": "<|channel>thought\n<channel|>hidden\n<|channel>final<channel|>hello",
                        }
                    ]
                }
            ]
        }
        cleaned = gemma_response_proxy.clean_response_payload(payload)
        self.assertEqual(cleaned["output_text"], "hello")
        self.assertEqual(cleaned["output"][0]["content"][0]["text"], "hello")

    def test_clean_response_payload_recovers_internal_only_tool_call_marker(self):
        marker = "<|tool_call>call:codex:using_superpowers{}<tool_call|>"
        payload = {
            "output_text": marker,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": marker,
                        }
                    ]
                }
            ],
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)

        self.assertEqual(cleaned["output_text"], gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE)
        self.assertEqual(
            cleaned["output"][0]["content"][0]["text"],
            gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE,
        )
        encoded = json.dumps(cleaned)
        self.assertNotIn("<|tool_call>", encoded)
        self.assertNotIn("call:codex", encoded)

    def test_clean_response_payload_recovers_empty_completed_response(self):
        payload = {
            "id": "resp_empty",
            "object": "response",
            "status": "completed",
            "output": [],
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)

        self.assertEqual(cleaned["output_text"], gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE)
        self.assertEqual(cleaned["output"][0]["type"], "message")
        self.assertEqual(
            cleaned["output"][0]["content"][0]["text"],
            gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE,
        )

    def test_clean_response_payload_recovers_gemma_thought_only_visible_answer(self):
        raw_text = "<|channel>thought\n<channel|>visible raw server check"
        payload = {
            "output_text": raw_text,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": raw_text}],
                }
            ],
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)

        self.assertEqual(cleaned["output_text"], "visible raw server check")
        self.assertEqual(cleaned["output"][0]["content"][0]["text"], "visible raw server check")
        self.assertNotIn("<|channel>", json.dumps(cleaned))

    def test_clean_response_payload_recovers_answer_that_mentions_hidden_channel_text(self):
        answer = "A regression test should assert hidden channel text is stripped while the visible answer remains."
        raw_text = f"<|channel>thought\n<channel|>{answer}"
        payload = {
            "output_text": raw_text,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": raw_text}],
                }
            ],
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)

        self.assertEqual(cleaned["output_text"], answer)
        self.assertEqual(cleaned["output"][0]["content"][0]["text"], answer)

    def test_clean_response_payload_recovers_when_sanitizer_removes_all_visible_text(self):
        payload = {
            "output_text": "<|channel>thought<channel|>hidden work without final",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "<|channel>thought<channel|>hidden work without final",
                        }
                    ],
                }
            ],
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)

        self.assertEqual(
            cleaned["output_text"],
            gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE,
        )
        self.assertEqual(
            cleaned["output"][0]["content"][0]["text"],
            gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE,
        )
        encoded = json.dumps(cleaned)
        self.assertNotIn("<|channel>", encoded)
        self.assertNotIn("hidden work", encoded)

    def test_clean_response_payload_does_not_insert_fallback_for_tool_call_turn(self):
        payload = {
            "output_text": "<|channel>thought<channel|>hidden work before tool call",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "<|channel>thought<channel|>hidden work before tool call",
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "name": "shell_command",
                    "arguments": "{\"command\":\"echo ok\"}",
                    "call_id": "call_1",
                },
            ],
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)
        encoded = json.dumps(cleaned)

        self.assertEqual(cleaned["output_text"], "")
        self.assertEqual(len(cleaned["output"]), 1)
        self.assertEqual(cleaned["output"][0]["type"], "function_call")
        self.assertNotIn(gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE, encoded)
        self.assertNotIn("<|channel>", encoded)
        self.assertNotIn("hidden work", encoded)

    def test_clean_response_payload_recovers_thought_only_visible_answer_with_tool_call(self):
        raw_text = "<|channel>thought\n<channel|>visible final after tool"
        payload = {
            "output_text": raw_text,
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": raw_text}],
                },
                {
                    "id": "call_1",
                    "type": "function_call",
                    "name": "shell_command",
                    "arguments": "{\"command\":\"echo ok\"}",
                    "call_id": "call_1",
                },
            ],
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)

        self.assertEqual(cleaned["output_text"], "visible final after tool")
        self.assertEqual(cleaned["output"][0]["type"], "message")
        self.assertEqual(cleaned["output"][0]["content"][0]["text"], "visible final after tool")
        self.assertEqual(cleaned["output"][1]["type"], "function_call")
        self.assertNotIn("<|channel>", json.dumps(cleaned))

    def test_clean_response_payload_prunes_internal_only_message_from_tool_call_turn(self):
        payload = {
            "output_text": "<|channel>thought<channel|>hidden work before tool call",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "<|channel>thought<channel|>hidden work before tool call",
                        }
                    ],
                },
                {
                    "id": "call_1",
                    "type": "function_call",
                    "name": "shell_command",
                    "arguments": "{\"command\":\"echo ok\"}",
                    "call_id": "call_1",
                },
            ],
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)

        self.assertEqual(cleaned["output_text"], "")
        self.assertEqual(len(cleaned["output"]), 1)
        self.assertEqual(cleaned["output"][0]["type"], "function_call")

    def test_clean_response_payload_preserves_non_text_message_content_on_tool_call_turn(self):
        payload = {
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "refusal",
                            "refusal": "Cannot provide that.",
                        }
                    ],
                },
                {
                    "id": "call_1",
                    "type": "function_call",
                    "name": "shell_command",
                    "arguments": "{\"command\":\"echo ok\"}",
                    "call_id": "call_1",
                },
            ],
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)

        self.assertEqual(len(cleaned["output"]), 2)
        self.assertEqual(cleaned["output"][0]["type"], "message")
        self.assertEqual(cleaned["output"][0]["content"][0]["refusal"], "Cannot provide that.")
        self.assertEqual(cleaned["output"][1]["type"], "function_call")

    def test_clean_response_payload_preserves_function_call_arguments_with_channel_like_text(self):
        arguments = json.dumps({"query": "<|channel>final<channel|>literal text"})
        payload = {
            "output_text": "<|channel>thought<channel|>hidden work before tool call",
            "output": [
                {
                    "type": "function_call",
                    "name": "search",
                    "arguments": arguments,
                    "call_id": "call_1",
                }
            ],
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)

        self.assertEqual(cleaned["output"][0]["arguments"], arguments)
        self.assertEqual(json.loads(cleaned["output"][0]["arguments"])["query"], "<|channel>final<channel|>literal text")

    def test_clean_response_payload_does_not_treat_nested_metadata_type_as_action(self):
        payload = {
            "output_text": "<|channel>thought<channel|>hidden work without final",
            "metadata": {"debug": {"type": "function_call"}},
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "<|channel>thought<channel|>hidden work without final",
                        }
                    ],
                }
            ],
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)

        self.assertEqual(cleaned["output_text"], gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE)
        self.assertEqual(
            cleaned["output"][0]["content"][0]["text"],
            gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE,
        )

    def test_clean_response_payload_removes_internal_tool_call_prefix_before_answer(self):
        marker = "<|tool_call>call:codex:using_superpowers{}<tool_call|>"
        payload = {"output_text": marker + "Continuing normally."}

        cleaned = gemma_response_proxy.clean_response_payload(payload)

        self.assertEqual(cleaned["output_text"], "Continuing normally.")

    def test_clean_response_payload_preserves_ordinary_model_text_without_markers(self):
        payload = {
            "output_text": (
                "Ship it \U0001f680 \u26a0\ufe0f It is 100% achieveable and production-ready. "
                "warning terms may appear as model content."
            ),
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": "This is 100% achievable and production-ready \u2705",
                        }
                    ]
                }
            ],
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)

        self.assertEqual(cleaned["output_text"], payload["output_text"])
        self.assertEqual(
            cleaned["output"][0]["content"][0]["text"],
            payload["output"][0]["content"][0]["text"],
        )

    def test_clean_response_payload_sanitizes_nested_error_and_tool_strings(self):
        payload = {
            "error": {
                "message": "Proxy says \u26d4 100% achievable and production-ready",
                "details": {"tool_output": "Tool result \u2757 WARNING"},
            }
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)
        combined = json.dumps(cleaned, ensure_ascii=False)

        self.assertNotIn("\u26d4", combined)
        self.assertNotIn("\u2757", combined)
        self.assertNotIn("100% achievable", combined)
        self.assertNotIn("production-ready", combined)
        self.assertNotIn("WARNING", combined)
        self.assertIn("likely achievable with verification", combined)
        self.assertIn("ready for review", combined)
        self.assertIn("warning", combined)

    def test_clean_sse_payload_preserves_ordinary_streamed_model_text_without_markers(self):
        raw = (
            b'data: {"type":"response.output_text.delta","delta":"Deploy \\ud83d\\udea8 100% achievable"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":" and production-ready warning"}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertIn("\\ud83d\\udea8", cleaned)
        self.assertIn("100% achievable", cleaned)
        self.assertIn("production-ready", cleaned)
        self.assertIn("warning", cleaned)

    def test_clean_sse_payload_updates_stream_text_fields(self):
        raw = (
            b'data: {"type":"response.output_text.delta","delta":"<|channel>final<channel|>OK"}\n\n'
            b"data: [DONE]\n\n"
        )
        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")
        self.assertIn('"delta": "OK"', cleaned)
        self.assertNotIn("<|channel>final", cleaned)

    def test_clean_sse_payload_parses_multiline_data_json_frame(self):
        raw = (
            b"event: response.output_text.delta\n"
            b'data: {"type":"response.output_text.delta",\n'
            b'data: "delta":"<|channel>final<channel|>OK"}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertIn("event: response.output_text.delta\ndata: ", cleaned)
        self.assertIn('"delta": "OK"', cleaned)
        self.assertNotIn("<|channel>", cleaned)

    def test_clean_sse_payload_removes_channel_marker_split_across_events(self):
        raw = (
            b'data: {"type":"response.output_text.delta","delta":"<|channel>"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"thought"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"<channel|>hidden"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"<|channel>final<channel|>Hello father"}\n\n'
            b"data: [DONE]\n\n"
        )
        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")
        self.assertNotIn("<|channel>", cleaned)
        self.assertNotIn("thought<channel|>", cleaned)
        self.assertNotIn("hidden", cleaned)
        self.assertIn('"delta": "Hello father"', cleaned)

    def test_clean_sse_payload_removes_malformed_duplicate_channel_marker_split_across_events(self):
        raw = (
            b'data: {"type":"response.output_text.delta","delta":"<|channel>thought"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"<|channel>rium                         "}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"<channel|>hidden"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"<|channel>final<channel|>Hello father."}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertNotIn("<|channel>", cleaned)
        self.assertNotIn("<channel|>", cleaned)
        self.assertNotIn("hidden", cleaned)
        self.assertIn('"delta": "Hello father."', cleaned)

    def test_clean_sse_payload_suppresses_unterminated_thought_channel_without_final(self):
        raw = (
            b'data: {"type":"response.output_text.delta","delta":"<|channel>thought"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"<|channel>979c4710136670e77b88916412901b7926e454626910433ba52db89e287f82"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"</code>\\n</code>"}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertNotIn("<|channel>", cleaned)
        self.assertNotIn("979c4710136670e77b88916412901b7926e454626910433ba52db89e287f82", cleaned)
        self.assertNotIn("</code>", cleaned)

    def test_clean_sse_payload_recovers_when_stream_sanitizes_all_visible_text(self):
        raw = (
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<|channel>thought"}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<channel|>hidden work"}\n\n'
            b'data: {"type":"response.output_text.done","item_id":"msg_1","text":"<|channel>thought<channel|>hidden work"}\n\n'
            b'data: {"type":"response.content_part.done","item_id":"msg_1","part":{"type":"output_text","text":"<|channel>thought<channel|>hidden work"}}\n\n'
            b'data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"<|channel>thought<channel|>hidden work"}]}}\n\n'
            b'data: {"type":"response.completed","response":{"id":"resp_1","output_text":"<|channel>thought<channel|>hidden work","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"<|channel>thought<channel|>hidden work"}]}]}}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertIn(gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE, cleaned)
        self.assertNotIn("<|channel>", cleaned)
        self.assertNotIn("hidden work", cleaned)

    def test_clean_sse_payload_emits_empty_response_fallback_events_at_completion(self):
        raw = (
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<|channel>thought"}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<channel|>hidden work"}\n\n'
            b'data: {"type":"response.output_text.done","item_id":"msg_1","text":"<|channel>thought<channel|>hidden work"}\n\n'
            b'data: {"type":"response.content_part.done","item_id":"msg_1","part":{"type":"output_text","text":"<|channel>thought<channel|>hidden work"}}\n\n'
            b'data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"<|channel>thought<channel|>hidden work"}]}}\n\n'
            b'data: {"type":"response.completed","response":{"id":"resp_1","output_text":"<|channel>thought<channel|>hidden work","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"<|channel>thought<channel|>hidden work"}]}]}}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")
        events_with_fallback = [
            json.loads(line[6:].strip())["type"]
            for line in cleaned.splitlines()
            if line.startswith("data: ")
            and line[6:].strip() != "[DONE]"
            and gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE in line
        ]

        self.assertEqual(
            events_with_fallback,
            [
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(
            cleaned.count(gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE),
            6,
        )

    def test_clean_sse_payload_does_not_insert_fallback_for_tool_call_turn(self):
        raw = (
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<|channel>thought"}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<channel|>hidden work before tool call"}\n\n'
            b'data: {"type":"response.output_item.added","item":{"id":"call_1","type":"function_call","name":"shell_command","arguments":"","call_id":"call_1"}}\n\n'
            b'data: {"type":"response.output_text.done","item_id":"msg_1","text":"<|channel>thought<channel|>hidden work before tool call"}\n\n'
            b'data: {"type":"response.output_item.done","item":{"id":"call_1","type":"function_call","name":"shell_command","arguments":"{\\"command\\":\\"echo ok\\"}","call_id":"call_1"}}\n\n'
            b'data: {"type":"response.completed","response":{"id":"resp_1","output_text":"<|channel>thought<channel|>hidden work before tool call","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"<|channel>thought<channel|>hidden work before tool call"}]},{"id":"call_1","type":"function_call","name":"shell_command","arguments":"{\\"command\\":\\"echo ok\\"}","call_id":"call_1"}]}}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertNotIn(gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE, cleaned)
        self.assertNotIn("<|channel>", cleaned)
        self.assertNotIn("hidden work", cleaned)
        self.assertIn('"type": "function_call"', cleaned)

    def test_clean_sse_payload_drops_internal_only_message_events_from_tool_call_turn(self):
        raw = (
            b'data: {"type":"response.output_item.added","item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<|channel>thought"}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<channel|>hidden work before tool call"}\n\n'
            b'data: {"type":"response.output_item.added","item":{"id":"call_1","type":"function_call","name":"shell_command","arguments":"","call_id":"call_1"}}\n\n'
            b'data: {"type":"response.output_text.done","item_id":"msg_1","text":"<|channel>thought<channel|>hidden work before tool call"}\n\n'
            b'data: {"type":"response.content_part.done","item_id":"msg_1","part":{"type":"output_text","text":"<|channel>thought<channel|>hidden work before tool call"}}\n\n'
            b'data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"<|channel>thought<channel|>hidden work before tool call"}]}}\n\n'
            b'data: {"type":"response.output_item.done","item":{"id":"call_1","type":"function_call","name":"shell_command","arguments":"{\\"command\\":\\"echo ok\\"}","call_id":"call_1"}}\n\n'
            b'data: {"type":"response.completed","response":{"id":"resp_1","output_text":"<|channel>thought<channel|>hidden work before tool call","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"<|channel>thought<channel|>hidden work before tool call"}]},{"id":"call_1","type":"function_call","name":"shell_command","arguments":"{\\"command\\":\\"echo ok\\"}","call_id":"call_1"}]}}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")
        event_types = [
            json.loads(line[6:].strip())["type"]
            for line in cleaned.splitlines()
            if line.startswith("data: ") and line[6:].strip() != "[DONE]"
        ]
        completed = [
            json.loads(line[6:].strip())["response"]
            for line in cleaned.splitlines()
            if line.startswith("data: ")
            and line[6:].strip() != "[DONE]"
            and json.loads(line[6:].strip())["type"] == "response.completed"
        ][0]

        self.assertNotIn("response.output_text.delta", event_types)
        self.assertNotIn("response.output_text.done", event_types)
        self.assertNotIn("response.content_part.done", event_types)
        self.assertFalse(
            any(
                event.get("type") == "response.output_item.added"
                and event.get("item", {}).get("id") == "msg_1"
                for event in (
                    json.loads(line[6:].strip())
                    for line in cleaned.splitlines()
                    if line.startswith("data: ") and line[6:].strip() != "[DONE]"
                )
            )
        )
        self.assertEqual(completed["output"], [{"id": "call_1", "type": "function_call", "name": "shell_command", "arguments": "{\"command\":\"echo ok\"}", "call_id": "call_1"}])

    def test_clean_sse_payload_keeps_separate_text_state_per_item(self):
        raw = (
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<|channel>thought<channel|>hidden"}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_2","delta":"visible second"}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<|channel>final<channel|>visible first"}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertIn('"delta": "visible second"', cleaned)
        self.assertIn('"delta": "visible first"', cleaned)
        self.assertNotIn("hidden", cleaned)

    def test_clean_sse_payload_preserves_empty_message_added_event_for_visible_item(self):
        raw = (
            b'data: {"type":"response.output_item.added","item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"visible"}\n\n'
            b'data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"visible"}]}}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")
        events = [
            json.loads(line[6:].strip())
            for line in cleaned.splitlines()
            if line.startswith("data: ") and line[6:].strip() != "[DONE]"
        ]

        self.assertEqual(events[0]["type"], "response.output_item.added")
        self.assertEqual(events[0]["item"]["id"], "msg_1")
        self.assertEqual(events[1]["type"], "response.output_text.delta")
        self.assertEqual(events[1]["delta"], "visible")

    def test_clean_sse_payload_preserves_message_added_when_hidden_text_precedes_visible_text(self):
        raw = (
            b'data: {"type":"response.output_item.added","item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<|channel>thought<channel|>hidden"}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<|channel>final<channel|>visible"}\n\n'
            b'data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"visible"}]}}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")
        events = [
            json.loads(line[6:].strip())
            for line in cleaned.splitlines()
            if line.startswith("data: ") and line[6:].strip() != "[DONE]"
        ]

        self.assertEqual(events[0]["type"], "response.output_item.added")
        self.assertEqual(events[0]["item"]["id"], "msg_1")
        self.assertEqual(events[1]["type"], "response.output_text.delta")
        self.assertEqual(events[1]["delta"], "visible")
        self.assertNotIn("hidden", cleaned)

    def test_clean_sse_payload_preserves_function_call_arguments_with_channel_like_text(self):
        arguments = json.dumps({"query": "<|channel>final<channel|>literal text"})
        raw = (
            "data: "
            + json.dumps(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "id": "call_1",
                        "type": "function_call",
                        "name": "search",
                        "arguments": arguments,
                        "call_id": "call_1",
                    },
                }
            )
            + "\n\n"
            + "data: "
            + json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "output": [
                            {
                                "id": "call_1",
                                "type": "function_call",
                                "name": "search",
                                "arguments": arguments,
                                "call_id": "call_1",
                            }
                        ],
                    },
                }
            )
            + "\n\n"
            + "data: [DONE]\n\n"
        ).encode("utf-8")

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")
        events = [
            json.loads(line[6:].strip())
            for line in cleaned.splitlines()
            if line.startswith("data: ") and line[6:].strip() != "[DONE]"
        ]

        self.assertEqual(events[0]["item"]["arguments"], arguments)
        self.assertEqual(events[1]["response"]["output"][0]["arguments"], arguments)

    def test_clean_sse_payload_recovers_when_split_channel_stream_sanitizes_all_visible_text(self):
        raw = (
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<|channel>"}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"thought"}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<channel|>hidden work"}\n\n'
            b'data: {"type":"response.output_text.done","item_id":"msg_1","text":""}\n\n'
            b'data: {"type":"response.content_part.done","item_id":"msg_1","part":{"type":"output_text","text":""}}\n\n'
            b'data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":""}]}}\n\n'
            b'data: {"type":"response.completed","response":{"id":"resp_1","output_text":"","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":""}]}]}}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertIn(gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE, cleaned)
        self.assertNotIn("<|channel>", cleaned)
        self.assertNotIn("hidden work", cleaned)

    def test_clean_sse_payload_suppresses_thought_until_split_final_marker(self):
        raw = (
            b'data: {"type":"response.output_text.delta","delta":"<|channel>thought_thought\\n  <channel|>secret"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":" more secret <|channel>"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"final<channel|>visible"}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertNotIn("secret", cleaned)
        self.assertNotIn("<|channel>", cleaned)
        self.assertIn('"delta": "visible"', cleaned)

    def test_clean_sse_payload_preserves_streamed_final_model_text_after_marker_strip(self):
        raw = (
            b'data: {"type":"response.output_text.delta","delta":"<|channel>final<channel|>warning \\u2705 100% achievable production-ready"}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertNotIn("<|channel>final", cleaned)
        self.assertIn("warning", cleaned)
        self.assertIn("\\u2705", cleaned)
        self.assertIn("100% achievable", cleaned)
        self.assertIn("production-ready", cleaned)

    def test_clean_sse_payload_removes_internal_tool_call_marker_split_across_events(self):
        raw = (
            b'data: {"type":"response.output_text.delta","delta":"<|tool_call>"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"call:codex:using_superpowers{}"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"<tool_call|>Hello father"}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertNotIn("<|tool_call>", cleaned)
        self.assertNotIn("call:codex:using_superpowers", cleaned)
        self.assertIn('"delta": "Hello father"', cleaned)

    def test_clean_sse_payload_recovers_internal_only_tool_call_marker(self):
        raw = (
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<|tool_call>"}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"call:codex:using_superpowers{}"}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<tool_call|>"}\n\n'
            b'data: {"type":"response.output_text.done","item_id":"msg_1","text":"<|tool_call>call:codex:using_superpowers{}<tool_call|>"}\n\n'
            b'data: {"type":"response.content_part.done","item_id":"msg_1","part":{"type":"output_text","text":"<|tool_call>call:codex:using_superpowers{}<tool_call|>"}}\n\n'
            b'data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"<|tool_call>call:codex:using_superpowers{}<tool_call|>"}]}}\n\n'
            b'data: {"type":"response.completed","response":{"id":"resp_1","output_text":"<|tool_call>call:codex:using_superpowers{}<tool_call|>","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"<|tool_call>call:codex:using_superpowers{}<tool_call|>"}]}]}}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertIn(gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE, cleaned)
        self.assertNotIn("<|tool_call>", cleaned)
        self.assertNotIn("call:codex", cleaned)

    def test_clean_sse_payload_recovers_empty_completed_response(self):
        raw = (
            b'data: {"type":"response.completed","response":{"id":"resp_empty","object":"response","status":"completed","output":[]}}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertIn(gemma_response_proxy.EMPTY_VISIBLE_RESPONSE_MESSAGE, cleaned)
        self.assertIn('"type": "message"', cleaned)

    def test_clean_sse_payload_recovers_gemma_thought_only_visible_answer_on_completed_event(self):
        raw = (
            b'data: {"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed","output_text":"<|channel>thought\\n<channel|>visible raw server check","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"<|channel>thought\\n<channel|>visible raw server check"}]}]}}\n\n'
            b"data: [DONE]\n\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertIn('"output_text": "visible raw server check"', cleaned)
        self.assertIn('"text": "visible raw server check"', cleaned)
        self.assertNotIn("<|channel>", cleaned)

    def test_clean_sse_payload_synthesizes_visible_events_from_gemma_thought_only_completed_response(self):
        raw = (
            b'data: {"type":"response.output_item.added","item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}\n\n'
            b'data: {"type":"response.content_part.added","item_id":"msg_1","part":{"type":"output_text","text":""}}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<|channel>thought"}\n\n'
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"\\n<channel|>hi"}\n\n'
            b'data: {"type":"response.output_text.done","item_id":"msg_1","text":"<|channel>thought\\n<channel|>hi"}\n\n'
            b'data: {"type":"response.content_part.done","item_id":"msg_1","part":{"type":"output_text","text":"<|channel>thought\\n<channel|>hi"}}\n\n'
            b'data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"<|channel>thought\\n<channel|>hi"}]}}\n\n'
            b'data: {"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed","output_text":"<|channel>thought\\n<channel|>hi","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"<|channel>thought\\n<channel|>hi"}]}]}}\n\n'
            b"data: [DONE]\n\n"
        )

        events = [
            json.loads(line[6:].strip())
            for line in gemma_response_proxy.clean_sse_payload(raw).decode("utf-8").splitlines()
            if line.startswith("data: ") and line.strip() != "data: [DONE]"
        ]
        event_types = [event["type"] for event in events]

        self.assertIn("response.output_text.delta", event_types)
        self.assertIn("response.output_text.done", event_types)
        self.assertIn("response.content_part.done", event_types)
        self.assertIn("response.output_item.done", event_types)
        self.assertLess(event_types.index("response.output_text.delta"), event_types.index("response.completed"))
        self.assertIn('"delta": "hi"', json.dumps(events))

    def test_clean_sse_payload_keeps_synthetic_event_and_data_lines_adjacent(self):
        raw = (
            b"event: response.output_item.added\n"
            b'data: {"type":"response.output_item.added","item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}\n\n'
            b"event: response.content_part.added\n"
            b'data: {"type":"response.content_part.added","item_id":"msg_1","part":{"type":"output_text","text":""}}\n\n'
            b"event: response.output_text.delta\n"
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"<|channel>thought"}\n\n'
            b"event: response.output_text.delta\n"
            b'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"\\n<channel|>hi"}\n\n'
            b"event: response.output_text.done\n"
            b'data: {"type":"response.output_text.done","item_id":"msg_1","text":"<|channel>thought\\n<channel|>hi"}\n\n'
            b"event: response.content_part.done\n"
            b'data: {"type":"response.content_part.done","item_id":"msg_1","part":{"type":"output_text","text":"<|channel>thought\\n<channel|>hi"}}\n\n'
            b"event: response.output_item.done\n"
            b'data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"<|channel>thought\\n<channel|>hi"}]}}\n\n'
            b"event: response.completed\n"
            b'data: {"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed","output_text":"<|channel>thought\\n<channel|>hi","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"<|channel>thought\\n<channel|>hi"}]}]}}\n\n'
        )

        lines = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8").splitlines()

        for index, line in enumerate(lines):
            if line.startswith("event: "):
                self.assertLess(index + 1, len(lines))
                self.assertTrue(lines[index + 1].startswith("data: "), lines[index : index + 3])


if __name__ == "__main__":
    unittest.main()
