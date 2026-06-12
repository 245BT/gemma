from __future__ import annotations

import json
from pathlib import Path


def normalize_result_file(suite: str, path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if suite == "swe-bench":
        return _normalize_swe_bench(payload)
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


def _list_count(value: object) -> int:
    return len(value) if isinstance(value, list) else 0
