import json
import unittest
from unittest.mock import patch

from gemma_reasoning.upstream import UpstreamResponsesClient


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class UpstreamResponsesClientTests(unittest.TestCase):
    def test_default_timeout_is_bounded(self):
        client = UpstreamResponsesClient("http://127.0.0.1:8081")

        self.assertLessEqual(client.timeout, 120)

    def test_create_response_posts_to_responses_endpoint(self):
        response_payload = {"output": [{"content": [{"type": "output_text", "text": "ok"}]}]}
        with patch("gemma_reasoning.upstream.urlopen", return_value=FakeResponse(response_payload)) as urlopen:
            client = UpstreamResponsesClient("http://127.0.0.1:8081")
            result = client.create_response({"input": "hello", "stream": True})

        self.assertEqual(result, response_payload)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8081/v1/responses")
        self.assertEqual(json.loads(request.data.decode("utf-8")), {"input": "hello", "stream": False})


if __name__ == "__main__":
    unittest.main()
