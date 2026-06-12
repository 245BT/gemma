import copy
import json
from urllib.request import Request, urlopen


class UpstreamResponsesClient:
    def __init__(self, upstream="http://127.0.0.1:8081", timeout=120):
        self.upstream = upstream.rstrip("/")
        self.timeout = timeout

    def create_response(self, payload):
        request_payload = copy.deepcopy(payload)
        request_payload["stream"] = False
        body = json.dumps(request_payload).encode("utf-8")
        request = Request(
            self.upstream + "/v1/responses",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
