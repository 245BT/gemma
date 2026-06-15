from __future__ import annotations

import json
from pathlib import Path

from .taxonomy import normalize_failure_mode


def normalize_result_file(suite: str, path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if suite == "swe-bench":
        return _normalize_swe_bench(payload)
    if suite == "local-agent-behavior":
        return _normalize_local_agent_behavior(payload)
    if suite == "local-repo-fix":
        return _normalize_local_repo_fix(payload)
    if isinstance(payload, dict):
        return _normalize_mapping(payload)
    if isinstance(payload, list):
        return _normalize_rows(payload)
    return {"suite": suite, "pass_rate": None, "total_count": None}


def _normalize_swe_bench(payload: dict[str, object]) -> dict[str, object]:
    resolved = _list_count(payload.get("resolved"))
    unresolved = _list_count(payload.get("unresolved"))
    errors = _list_count(payload.get("error")) + _list_count(payload.get("errors"))
    total = resolved + unresolved + errors
    return {
        "resolved_count": resolved,
        "unresolved_count": unresolved,
        "error_count": errors,
        "total_count": total,
        "pass_rate": resolved / total if total else None,
    }


def _normalize_mapping(payload: dict[str, object]) -> dict[str, object]:
    for key in ("pass_rate", "score", "accuracy", "success_rate"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return {"pass_rate": float(value), "total_count": payload.get("total_count")}
    return {"pass_rate": None, "total_count": payload.get("total_count")}


def _normalize_rows(rows: list[object]) -> dict[str, object]:
    total = 0
    passed = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        total += 1
        if row.get("resolved") is True or row.get("passed") is True or row.get("success") is True:
            passed += 1
    return {"resolved_count": passed, "total_count": total, "pass_rate": passed / total if total else None}


def _normalize_local_agent_behavior(payload: object) -> dict[str, object]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        rows = payload["results"]
        normalized = _normalize_rows(rows)
        normalized["failure_counts"] = _failure_counts(rows)
        if isinstance(payload.get("category_counts"), dict):
            normalized["category_counts"] = payload["category_counts"]
        return normalized
    if isinstance(payload, list):
        normalized = _normalize_rows(payload)
        normalized["failure_counts"] = _failure_counts(payload)
        return normalized
    return {"pass_rate": None, "total_count": None, "failure_counts": {}}


def _normalize_local_repo_fix(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {"pass_rate": None, "total_count": None, "failure_counts": {}}
    rows = payload.get("results")
    normalized = _normalize_rows(rows if isinstance(rows, list) else [])
    for key in (
        "fail_to_pass_rate",
        "pass_to_pass_rate",
        "patch_apply_rate",
        "regression_count",
    ):
        if key in payload:
            normalized[key] = payload[key]
    if isinstance(payload.get("failure_counts"), dict):
        normalized["failure_counts"] = payload["failure_counts"]
    elif isinstance(rows, list):
        normalized["failure_counts"] = _failure_counts(rows)
    if isinstance(payload.get("category_counts"), dict):
        normalized["category_counts"] = payload["category_counts"]
    return normalized


def _failure_counts(rows: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("success") is True:
            continue
        failure_mode = normalize_failure_mode(row.get("failure_mode"))
        counts[failure_mode] = counts.get(failure_mode, 0) + 1
    return dict(sorted(counts.items()))


def _list_count(value: object) -> int:
    return len(value) if isinstance(value, list) else 0
