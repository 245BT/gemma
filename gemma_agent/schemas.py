from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    def to_evidence(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "args": self.args,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "error_code": self.error_code,
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

    def to_evidence(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "task": self.task,
            "final": self.final,
            "ok": self.ok,
            "error": self.error,
            "tool_results": [item.to_evidence() for item in self.tool_results],
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
