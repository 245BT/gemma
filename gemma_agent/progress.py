from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Any

from .safety import neutralize_model_facing_metadata, neutralize_runtime_markers
from .schemas import InvalidAction, ToolResult


@dataclass(frozen=True)
class RecoveryEvent:
    event: str
    reason: str
    next_action: str
    evidence: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "reason": self.reason,
            "next_action": self.next_action,
            "evidence": self.evidence,
        }


class ProgressTracker:
    def __init__(
        self,
        *,
        repeated_tool_threshold: int = 2,
        repeated_invalid_threshold: int = 2,
        safety_guard: Any | None = None,
    ) -> None:
        self.repeated_tool_threshold = max(2, int(repeated_tool_threshold))
        self.repeated_invalid_threshold = max(2, int(repeated_invalid_threshold))
        self._safety_guard = safety_guard
        self.iteration = 0
        self._last_tool_signature: str | None = None
        self._last_invalid_signature: str | None = None
        self._tool_repeat_count = 0
        self._invalid_repeat_count = 0
        self._events: list[RecoveryEvent] = []

    def start_iteration(self) -> None:
        self.iteration += 1

    def record_tool_result(self, result: ToolResult) -> None:
        signature = _tool_signature(result)
        if signature == self._last_tool_signature:
            self._tool_repeat_count += 1
        else:
            self._last_tool_signature = signature
            self._tool_repeat_count = 1

        if self._tool_repeat_count == self.repeated_tool_threshold:
            self._events.append(
                RecoveryEvent(
                    event="repeated_tool_call",
                    reason="same tool and arguments repeated without new strategy",
                    next_action=(
                        "Stop repeating the same tool call; inspect prior evidence, "
                        "summarize what changed, use Context7 or official docs if command "
                        "syntax is uncertain, and change strategy before another command."
                    ),
                    evidence={
                        "tool_name": self._tool_name_evidence(result),
                        "args_sha256": _stable_hash(result.args),
                        "repeat_count": self._tool_repeat_count,
                    },
                )
            )

        if result.error_code in {"timeout", "stalled"}:
            evidence = {
                "tool_name": self._tool_name_evidence(result),
                "error_code": result.error_code,
                "elapsed_ms": result.elapsed_ms,
            }
            evidence.update(_stall_metadata(result.output, safety_guard=self._safety_guard))
            self._events.append(
                RecoveryEvent(
                    event="tool_stall",
                    reason=f"tool returned {result.error_code}",
                    next_action=(
                        "Inspect partial output and current state; do not repeat the same action "
                        "or only shrink a timed-out range. Use an explicit timeout for expected-long "
                        "commands, and for subnet probes use parallel or vectorized checks with "
                        "per-probe timeouts. Check active install/package-manager processes and use "
                        "Context7 or official docs if command syntax is uncertain."
                    ),
                    evidence=evidence,
                )
            )

    def record_invalid_action(self, invalid: InvalidAction) -> None:
        signature = invalid.error
        if signature == self._last_invalid_signature:
            self._invalid_repeat_count += 1
        else:
            self._last_invalid_signature = signature
            self._invalid_repeat_count = 1
        if self._invalid_repeat_count == self.repeated_invalid_threshold:
            self._events.append(
                RecoveryEvent(
                    event="repeated_invalid_action",
                    reason="same invalid model action repeated",
                    next_action=(
                        "Stop the loop; emit a concise thinking_summary with the schema error "
                        "and then produce a valid JSON action."
                    ),
                    evidence={
                        "error": self._invalid_action_error_evidence(invalid.error),
                        "repeat_count": self._invalid_repeat_count,
                    },
                )
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "recovery_events": [event.to_payload() for event in self._events],
            "policy": {
                "easy_tasks": "keep turns and tool calls short",
                "hard_tasks": "summarize state, decompose, and use targeted tools",
                "stall_rule": "if progress is low, inspect, summarize, and change strategy",
                "search_rule": "if command syntax or installation steps are uncertain, use Context7 before retrying",
            },
        }

    def _tool_name_evidence(self, result: ToolResult) -> Any:
        if self._safety_guard is None:
            return neutralize_runtime_markers(result.tool_name)
        if result.error_code == "tool_not_allowed" and not result.executed:
            return self._safety_guard.wrap_untrusted_model_identifier(
                result.tool_name,
                field="tool_name",
            )
        return self._safety_guard.wrap_model_identifier(result.tool_name, field="tool_name")

    def _invalid_action_error_evidence(self, error: str) -> Any:
        if self._safety_guard is None:
            return neutralize_runtime_markers(error)
        return self._safety_guard.wrap_untrusted_evidence(
            "invalid_model_action",
            error,
            field="error",
        )


def _tool_signature(result: ToolResult) -> str:
    return json.dumps(
        {"tool": result.tool_name, "args": result.args},
        sort_keys=True,
        ensure_ascii=True,
        default=repr,
    )


def _stable_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=True, default=repr)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stall_metadata(output: Any, *, safety_guard: Any | None = None) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {}
    content = output.get("content")
    if not isinstance(content, dict):
        return {}
    metadata: dict[str, Any] = {}
    recovery = content.get("recovery")
    if isinstance(recovery, dict):
        if recovery.get("reason") is not None:
            metadata["recovery_reason"] = _stall_metadata_value(
                recovery["reason"],
                field="recovery_reason",
                safety_guard=safety_guard,
            )
        if recovery.get("threshold_sec") is not None:
            metadata["recovery_threshold_sec"] = _stall_metadata_value(
                recovery["threshold_sec"],
                field="recovery_threshold_sec",
                safety_guard=safety_guard,
            )
    for key in ("stdout_length", "stderr_length", "stdout_sha256", "stderr_sha256"):
        if key in content:
            metadata[key] = _stall_metadata_value(
                content[key],
                field=key,
                safety_guard=safety_guard,
            )
    return metadata


def _stall_metadata_value(value: Any, *, field: str, safety_guard: Any | None) -> Any:
    if safety_guard is None:
        return neutralize_model_facing_metadata(value)
    return safety_guard.wrap_untrusted_evidence(
        "tool_stall_metadata",
        value,
        field=field,
    )
