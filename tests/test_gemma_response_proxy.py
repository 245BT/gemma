import unittest

import gemma_response_proxy


class GemmaResponseProxyTests(unittest.TestCase):
    def test_upstream_timeout_is_bounded(self):
        self.assertLessEqual(gemma_response_proxy.UPSTREAM_TIMEOUT_SECONDS, 120)

    def test_clean_channel_markers_removes_thought_prefix(self):
        self.assertEqual(
            gemma_response_proxy.clean_channel_markers("<|channel>thought\n<channel|>4"),
            "4",
        )

    def test_clean_channel_markers_removes_compact_thought_prefix(self):
        self.assertEqual(
            gemma_response_proxy.clean_channel_markers("<|channel>thought<channel|>answer"),
            "answer",
        )

    def test_clean_response_payload_updates_output_text(self):
        payload = {
            "output_text": "<|channel>thought\n<channel|>hello",
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": "<|channel>thought\n<channel|>hello",
                        }
                    ]
                }
            ]
        }
        cleaned = gemma_response_proxy.clean_response_payload(payload)
        self.assertEqual(cleaned["output_text"], "hello")
        self.assertEqual(cleaned["output"][0]["content"][0]["text"], "hello")

    def test_clean_response_payload_removes_internal_codex_tool_call_marker(self):
        marker = "<|tool_call>call:codex:using_superpowers{}<tool_call|>"
        payload = {
            "output_text": marker,
            "output": [
                {
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

        self.assertEqual(cleaned["output_text"], "")
        self.assertEqual(cleaned["output"][0]["content"][0]["text"], "")

    def test_clean_response_payload_removes_internal_tool_call_prefix_before_answer(self):
        marker = "<|tool_call>call:codex:using_superpowers{}<tool_call|>"
        payload = {"output_text": marker + "Continuing normally."}

        cleaned = gemma_response_proxy.clean_response_payload(payload)

        self.assertEqual(cleaned["output_text"], "Continuing normally.")

    def test_clean_sse_payload_updates_stream_text_fields(self):
        raw = (
            b'data: {"type":"response.output_text.delta","delta":"<|channel>thought<channel|>OK"}\n\n'
            b"data: [DONE]\n\n"
        )
        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")
        self.assertIn('"delta": "OK"', cleaned)
        self.assertNotIn("<|channel>thought", cleaned)

    def test_clean_sse_payload_removes_channel_marker_split_across_events(self):
        raw = (
            b'data: {"type":"response.output_text.delta","delta":"<|channel>"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"thought"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"<channel|>Hello father"}\n\n'
            b"data: [DONE]\n\n"
        )
        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")
        self.assertNotIn("<|channel>", cleaned)
        self.assertNotIn("thought<channel|>", cleaned)
        self.assertIn('"delta": "Hello father"', cleaned)

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


if __name__ == "__main__":
    unittest.main()
