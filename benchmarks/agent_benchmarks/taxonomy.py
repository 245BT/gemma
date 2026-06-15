from __future__ import annotations

import re
from typing import Any


KNOWN_FAILURE_MODES = {
    "assertion_failed",
    "bad_search",
    "invalid_json",
    "large_file_context",
    "large_repo_navigation",
    "patch_failed",
    "research_artifact_missing",
    "regression",
    "schema_rejected",
    "stall",
    "test_failed",
    "timeout",
    "unexpected_exception",
    "unknown_failure",
}

_EXCEPTION_MAP = {
    "TimeoutError": "timeout",
    "TimeoutExpired": "timeout",
    "AssertionError": "assertion_failed",
    "JSONDecodeError": "invalid_json",
}


def normalize_failure_mode(value: Any) -> str:
    if value is None or value == "":
        return "unknown_failure"
    text = str(value)
    if text in _EXCEPTION_MAP:
        return _EXCEPTION_MAP[text]
    normalized = text.strip().lower().replace("-", "_")
    if normalized in KNOWN_FAILURE_MODES:
        return normalized
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,60}", normalized):
        return "unexpected_exception"
    return "unexpected_exception"
