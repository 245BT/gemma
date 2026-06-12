from __future__ import annotations

import copy
import json
from typing import Any, Callable, Protocol
from urllib.request import Request, urlopen as urllib_urlopen


class ModelClient(Protocol):
    def create_response(self, payload: dict[str, Any]) -> Any:
        """Send a Responses-style payload to a model runtime."""


class LocalResponsesClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8081/v1",
        model: str | None = None,
        timeout: float = 120,
        urlopen: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._urlopen = urllib_urlopen if urlopen is None else urlopen

    def create_response(self, payload: dict[str, Any]) -> Any:
        request_payload = copy.deepcopy(payload)
        if self.model and "model" not in request_payload:
            request_payload["model"] = self.model
        body = json.dumps(request_payload).encode("utf-8")
        request = Request(
            self._responses_url(),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _responses_url(self) -> str:
        if self.base_url.endswith("/responses"):
            return self.base_url
        return self.base_url + "/responses"
