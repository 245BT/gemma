from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


_EASY_EXACT_RE = re.compile(
    r"\b(?:reply|respond|answer|output|return|print|say)\s+"
    r"(?:with\s+)?(?:only|exactly)\b",
    re.IGNORECASE,
)
_HARD_TASK_RE = re.compile(
    r"\b("
    r"ambiguous|benchmark|citations?|compare|debug|diagnose|evaluate|"
    r"fix|implement|investigate|large[-\s]?(?:file|repo|repository|branch)|"
    r"latency|long[-\s]?horizon|measure|multi[-\s]?step|optimi[sz]e|"
    r"recover|refactor|research|stall|sub[-\s]?agent|terminal|test|verify"
    r")\b",
    re.IGNORECASE,
)
_MEDIUM_TASK_RE = re.compile(
    r"\b(code|edit|explain|inspect|plan|review|search|tool|trace)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TaskEffortBudget:
    difficulty: str
    max_iterations: int
    max_tool_calls: int
    max_subagents: int
    max_context_chars: int
    terminal_timeout_sec: float
    terminal_idle_timeout_sec: float
    summarization_interval: int
    repeated_tool_threshold: int
    repeated_invalid_threshold: int

    def to_payload(self, *, nonlinear_cost_score: float) -> dict[str, Any]:
        return {
            "difficulty": self.difficulty,
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "max_subagents": self.max_subagents,
            "max_context_chars": self.max_context_chars,
            "terminal_timeout_sec": self.terminal_timeout_sec,
            "terminal_idle_timeout_sec": self.terminal_idle_timeout_sec,
            "summarization_interval": self.summarization_interval,
            "repeated_tool_threshold": self.repeated_tool_threshold,
            "repeated_invalid_threshold": self.repeated_invalid_threshold,
            "nonlinear_cost_score": nonlinear_cost_score,
            "policy": {
                "easy": "answer directly with minimal turns and no speculative tools",
                "medium": "inspect targeted evidence before editing or testing",
                "hard": (
                    "decompose, summarize state every few turns, verify assumptions, "
                    "and change strategy after stalls or repeated failures"
                ),
                "cost": (
                    "cost grows with turns, tool calls, tool output, and final tokens, "
                    "using a concave score so hard tasks can spend more effort"
                ),
            },
        }


class TaskEffortPolicy:
    def plan_for_task(self, task: str, *, context: dict[str, Any] | None = None) -> TaskEffortBudget:
        difficulty = self.classify(task, context=context)
        if difficulty == "easy":
            return TaskEffortBudget(
                difficulty="easy",
                max_iterations=2,
                max_tool_calls=1,
                max_subagents=0,
                max_context_chars=6000,
                terminal_timeout_sec=30,
                terminal_idle_timeout_sec=8,
                summarization_interval=2,
                repeated_tool_threshold=2,
                repeated_invalid_threshold=2,
            )
        if difficulty == "hard":
            return TaskEffortBudget(
                difficulty="hard",
                max_iterations=10,
                max_tool_calls=12,
                max_subagents=4,
                max_context_chars=32000,
                terminal_timeout_sec=1800,
                terminal_idle_timeout_sec=120,
                summarization_interval=2,
                repeated_tool_threshold=2,
                repeated_invalid_threshold=2,
            )
        return TaskEffortBudget(
            difficulty="medium",
            max_iterations=6,
            max_tool_calls=6,
            max_subagents=2,
            max_context_chars=16000,
            terminal_timeout_sec=900,
            terminal_idle_timeout_sec=90,
            summarization_interval=3,
            repeated_tool_threshold=2,
            repeated_invalid_threshold=2,
        )

    def classify(self, task: str, *, context: dict[str, Any] | None = None) -> str:
        text = _task_text(task, context)
        stripped = text.strip()
        if not stripped:
            return "medium"
        if _looks_easy(stripped):
            return "easy"
        if len(stripped) > 900 or _HARD_TASK_RE.search(stripped):
            return "hard"
        if _MEDIUM_TASK_RE.search(stripped):
            return "medium"
        return "easy" if len(stripped) <= 160 else "medium"

    def cost_score(
        self,
        *,
        turns: int = 0,
        tool_calls: int = 0,
        model_calls: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        tool_output_tokens: int = 0,
        final_tokens: int = 0,
    ) -> float:
        weighted = (
            max(0, turns) * 1.0
            + max(0, tool_calls) * 2.0
            + max(0, model_calls) * 1.5
            + max(0, prompt_tokens) / 800.0
            + max(0, completion_tokens) / 500.0
            + max(0, tool_output_tokens) / 200.0
            + max(0, final_tokens) / 150.0
        )
        return round(math.sqrt(weighted), 6) if weighted > 0 else 0.0


def estimate_text_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else repr(value)
    return max(0, math.ceil(len(text) / 4))


def _task_text(task: str, context: dict[str, Any] | None) -> str:
    if not context:
        return str(task)
    parts = [str(task)]
    for key in ("task", "parent_task", "delegated_task"):
        value = context.get(key)
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


def _looks_easy(task: str) -> bool:
    if len(task) > 240 or "\n" in task:
        return False
    if _HARD_TASK_RE.search(task) or _MEDIUM_TASK_RE.search(task):
        return False
    return bool(_EASY_EXACT_RE.search(task)) or len(task) <= 80
