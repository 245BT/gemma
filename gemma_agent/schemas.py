from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

from .safety import neutralize_runtime_markers


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    args: dict[str, Any]
    ok: bool
    output: Any = None
    error: str | None = None
    error_code: str | None = None
    elapsed_ms: float = 0
    executed: bool = False

    def to_evidence(self, safety_guard: Any | None = None) -> dict[str, Any]:
        tool_name: Any = self.tool_name
        args: Any = self.args
        output: Any = self.output
        error: Any = self.error
        error_code: Any = self.error_code
        rejected_tool_identifier = self.error_code == "tool_not_allowed" and not self.executed
        if safety_guard is not None:
            tool_name = (
                safety_guard.wrap_untrusted_model_identifier(self.tool_name, field="tool_name")
                if rejected_tool_identifier
                else safety_guard.wrap_model_identifier(self.tool_name, field="tool_name")
            )
            args = safety_guard.wrap_untrusted_evidence(
                safety_guard.safe_untrusted_metadata_source("tool_args", self.tool_name)
                if rejected_tool_identifier
                else safety_guard.safe_metadata_source("tool_args", self.tool_name),
                self.args,
                field="args",
            )
            if output is not None:
                output = safety_guard.wrap_untrusted_output(
                    self.tool_name,
                    output,
                    untrusted_tool_identifier=rejected_tool_identifier,
                )
            if error is not None:
                error = safety_guard.wrap_untrusted_tool_error(
                    self.tool_name,
                    error,
                    untrusted_tool_identifier=rejected_tool_identifier,
                )
            if error_code is not None:
                error_code = _model_facing_error_code(error_code, safety_guard)
        else:
            error_code = neutralize_runtime_markers(error_code)
        return {
            "tool_name": tool_name,
            "args": args,
            "ok": self.ok,
            "output": output,
            "error": error,
            "error_code": error_code,
            "elapsed_ms": self.elapsed_ms,
            "executed": self.executed,
        }


@dataclass(frozen=True)
class InvalidAction:
    raw: str
    error: str


@dataclass(frozen=True)
class ThinkingSummary:
    goal: str
    plan: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    current_finding: str = ""
    confidence: str = "low"
    next_action: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ThinkingSummary":
        return cls(
            goal=str(value.get("goal", "")),
            plan=_string_list(value.get("plan", [])),
            evidence=_string_list(value.get("evidence", [])),
            current_finding=str(value.get("current_finding", value.get("current finding", ""))),
            confidence=str(value.get("confidence", "low")),
            next_action=str(value.get("next_action", value.get("next action", ""))),
        )

    @classmethod
    def validate_mapping(cls, value: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in value:
            if _is_forbidden_summary_key(key):
                errors.append("forbidden thinking_summary key is not allowed")
        return errors

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.goal:
            errors.append("goal is required")
        if not self.plan:
            errors.append("plan is required")
        if not self.evidence:
            errors.append("evidence is required")
        if not self.current_finding:
            errors.append("current_finding is required")
        if self.confidence not in {"low", "medium", "high"}:
            errors.append("confidence must be low, medium, or high")
        if not self.next_action:
            errors.append("next_action is required")
        if _contains_sensitive_summary_content(self.format_lines()):
            errors.append("sensitive raw prompt or chain-of-thought content is not allowed")
        if any(len(line) > 1000 for line in self.format_lines()):
            errors.append("thinking_summary fields must be concise")
        return errors

    def format_lines(self) -> list[str]:
        return [
            f"- Goal: {self.goal}",
            f"- Plan: {'; '.join(self.plan)}",
            f"- Evidence: {'; '.join(self.evidence)}",
            f"- Current finding: {self.current_finding}",
            f"- Confidence: {self.confidence}",
            f"- Next action: {self.next_action}",
        ]


@dataclass(frozen=True)
class SubAgentResult:
    agent_id: str
    task: str
    final: str = ""
    ok: bool = False
    error: str | None = None
    tool_results: list[ToolResult] = field(default_factory=list)

    def to_evidence(self, safety_guard: Any | None = None) -> dict[str, Any]:
        agent_id: Any = self.agent_id
        task: Any = self.task
        final: Any = self.final
        error: Any = self.error
        if safety_guard is not None:
            source = safety_guard.safe_metadata_source("subagent", self.agent_id)
            agent_id = safety_guard.wrap_model_identifier(self.agent_id, field="agent_id")
            task = safety_guard.wrap_untrusted_evidence(
                source,
                self.task,
                field="task",
            )
            final = safety_guard.wrap_untrusted_evidence(
                source,
                self.final,
                field="final",
            )
            if error is not None:
                error = safety_guard.wrap_untrusted_evidence(
                    source,
                    error,
                    field="error",
                )
        return {
            "agent_id": agent_id,
            "task": task,
            "final": final,
            "ok": self.ok,
            "error": error,
            "tool_results": [item.to_evidence(safety_guard) for item in self.tool_results],
        }


@dataclass(frozen=True)
class AgentRunResult:
    final: str | None
    ok: bool
    tool_results: list[ToolResult] = field(default_factory=list)
    subagent_results: list[SubAgentResult] = field(default_factory=list)
    thinking_summaries: list[ThinkingSummary] = field(default_factory=list)
    invalid_actions: list[InvalidAction] = field(default_factory=list)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


_TRUSTED_MODEL_FACING_ERROR_CODES = {
    "invalid_json",
    "path_validation_error",
    "stalled",
    "timeout",
    "tool_exception",
    "tool_failed",
    "tool_not_allowed",
    "tool_spawn_error",
    "validation_error",
}


def _model_facing_error_code(error_code: Any, safety_guard: Any) -> Any:
    if isinstance(error_code, str):
        normalized = error_code.strip().lower().replace("-", "_")
        if normalized in _TRUSTED_MODEL_FACING_ERROR_CODES:
            return normalized
    return safety_guard.wrap_untrusted_evidence(
        "tool_error_code",
        error_code,
        field="error_code",
    )


_SENSITIVE_SUMMARY_RE = re.compile(
    r"\b(raw[\s_-]+chain[\s_-]+of[\s_-]+thought|hidden[\s_-]+reasoning|"
    r"system[\s_-]+prompt|developer[\s_-]+message|"
    r"session\s+log|history\.jsonl|prompt\s+text)\b",
    re.IGNORECASE,
)
_FORBIDDEN_SUMMARY_KEYS = {"raw_chain_of_thought", "system_prompt"}


def _contains_sensitive_summary_content(lines: list[str]) -> bool:
    return any(_SENSITIVE_SUMMARY_RE.search(line) for line in lines)


def _is_forbidden_summary_key(key: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
    return normalized in _FORBIDDEN_SUMMARY_KEYS
