from __future__ import annotations

import copy
import json
import threading
from typing import Any

from .citations import CitationManager
from .context import ContextBuilder
from .memory import MemoryStore
from .schemas import AgentRunResult, InvalidAction, SubAgentResult, ThinkingSummary, ToolResult
from .subagent import SubAgent
from .tool_executor import ToolExecutor

DEFAULT_MAX_SUBAGENTS = 8


ACTION_INSTRUCTIONS = (
    "Return exactly one strict JSON object per turn. "
    "Allowed actions: final, tool_call, spawn_subagents, thinking_summary. "
    "Tool execution is only real after the external runtime returns ToolResult evidence."
)


class AgentSupervisor:
    def __init__(
        self,
        model_client: Any,
        tool_executor: ToolExecutor,
        *,
        memory: MemoryStore | None = None,
        citations: CitationManager | None = None,
        context_builder: ContextBuilder | None = None,
        max_iterations: int = 8,
        max_subagents: int = DEFAULT_MAX_SUBAGENTS,
        subagent_budget: "SubAgentBudget | None" = None,
        subagent_factory: Any | None = None,
    ) -> None:
        self.model_client = model_client
        self.tool_executor = tool_executor
        self.memory = MemoryStore() if memory is None else memory
        self.citations = CitationManager() if citations is None else citations
        self.context_builder = (
            ContextBuilder(self.memory, self.citations) if context_builder is None else context_builder
        )
        self.max_iterations = max_iterations
        self.max_subagents = _non_negative_int(max_subagents, DEFAULT_MAX_SUBAGENTS)
        self.subagent_budget = (
            SubAgentBudget(self.max_subagents) if subagent_budget is None else subagent_budget
        )
        self.subagent_factory = subagent_factory

    def run(self, task: str, *, context: dict[str, Any] | None = None) -> AgentRunResult:
        run_context = copy.deepcopy(context or {})
        tool_results = []
        subagent_results: list[SubAgentResult] = []
        thinking_summaries: list[ThinkingSummary] = []
        invalid_actions: list[InvalidAction] = []

        for _ in range(self.max_iterations):
            payload = self._build_model_payload(
                task,
                run_context,
                tool_results=tool_results,
                subagent_results=subagent_results,
                thinking_summaries=thinking_summaries,
                invalid_actions=invalid_actions,
            )
            raw_response = self.model_client.create_response(payload)
            action, invalid = parse_model_action(raw_response)
            if invalid is not None:
                invalid_actions.append(invalid)
                self.memory.add("evidence", invalid.error, source="invalid_model_action")
                continue

            action_name = action.get("action")
            if action_name == "final":
                return AgentRunResult(
                    final=str(action.get("content", "")),
                    ok=True,
                    tool_results=tool_results,
                    subagent_results=subagent_results,
                    thinking_summaries=thinking_summaries,
                    invalid_actions=invalid_actions,
                )
            if action_name == "tool_call":
                tool_results.append(self._handle_tool_call(action))
                self.memory.add("evidence", tool_results[-1].to_evidence(), source=f"tool:{tool_results[-1].tool_name}")
                continue
            if action_name == "spawn_subagents":
                new_results = self._handle_spawn_subagents(action, task, run_context)
                subagent_results.extend(new_results)
                for result in new_results:
                    self.memory.add("evidence", result.to_evidence(), source=f"subagent:{result.agent_id}")
                continue
            if action_name == "thinking_summary":
                summary, summary_invalid = self._handle_thinking_summary(raw_response, action)
                if summary_invalid is not None:
                    invalid_actions.append(summary_invalid)
                    continue
                thinking_summaries.append(summary)
                self.memory.add("task_scratchpad", "\n".join(summary.format_lines()), source="thinking_summary")
                continue

            invalid_actions.append(
                InvalidAction(
                    raw=_extract_response_text(raw_response),
                    error=f"unknown action {action_name!r}; expected strict JSON action",
                )
            )

        return AgentRunResult(
            final=None,
            ok=False,
            tool_results=tool_results,
            subagent_results=subagent_results,
            thinking_summaries=thinking_summaries,
            invalid_actions=invalid_actions,
        )

    def _build_model_payload(
        self,
        task: str,
        run_context: dict[str, Any],
        *,
        tool_results: list[Any],
        subagent_results: list[SubAgentResult],
        thinking_summaries: list[ThinkingSummary],
        invalid_actions: list[InvalidAction],
    ) -> dict[str, Any]:
        return {
            "input": [
                {"role": "system", "content": ACTION_INSTRUCTIONS},
                {"role": "user", "content": task},
            ],
            "context": self.context_builder.build(task=task, extra=run_context),
            "tools": self.tool_executor.registry.list_tool_schemas(),
            "tool_results": [result.to_evidence() for result in tool_results],
            "subagent_results": [result.to_evidence() for result in subagent_results],
            "thinking_summaries": [summary.format_lines() for summary in thinking_summaries],
            "invalid_actions": [{"raw": item.raw, "error": item.error} for item in invalid_actions],
        }

    def _handle_tool_call(self, action: dict[str, Any]):
        tool_name = action.get("tool", action.get("name"))
        if not isinstance(tool_name, str) or not tool_name:
            return ToolResult(
                tool_name="",
                args={},
                ok=False,
                error="tool_call action must include a non-empty tool field",
                error_code="validation_error",
                executed=False,
            )
        return self.tool_executor.execute(tool_name, action.get("args", {}))

    def _handle_spawn_subagents(
        self,
        action: dict[str, Any],
        parent_task: str,
        run_context: dict[str, Any],
    ) -> list[SubAgentResult]:
        from .subagent import run_subagents

        raw_tasks = action.get("tasks", [])
        if not isinstance(raw_tasks, list):
            return [
                SubAgentResult(
                    agent_id="invalid",
                    task=parent_task,
                    ok=False,
                    error="spawn_subagents tasks must be a list",
                )
            ]
        if len(raw_tasks) > self.max_subagents:
            return [
                SubAgentResult(
                    agent_id="invalid",
                    task=parent_task,
                    ok=False,
                    error=(
                        f"spawn_subagents requested {len(raw_tasks)} tasks, "
                        f"which exceeds max_subagents={self.max_subagents}"
                    ),
                )
            ]
        if not self.subagent_budget.reserve(len(raw_tasks)):
            return [
                SubAgentResult(
                    agent_id="invalid",
                    task=parent_task,
                    ok=False,
                    error=(
                        f"spawn_subagents requested {len(raw_tasks)} tasks, "
                        f"which exceeds remaining subagent budget={self.subagent_budget.remaining}"
                    ),
                )
            ]

        agents = []
        for index, item in enumerate(raw_tasks):
            if isinstance(item, str):
                spec = {"id": f"subagent-{index + 1}", "task": item}
            elif isinstance(item, dict):
                spec = {
                    "id": str(item.get("id", f"subagent-{index + 1}")),
                    "task": str(item.get("task", "")),
                }
            else:
                spec = {"id": f"subagent-{index + 1}", "task": ""}
            child_context = copy.deepcopy(run_context)
            child_context["parent_task"] = parent_task
            child_context["task"] = spec["task"]
            agents.append(self._make_subagent(spec, child_context))

        return run_subagents(
            agents,
            concurrent=bool(action.get("concurrent", False)),
            max_workers=_bounded_requested_workers(action.get("max_workers"), self.max_subagents),
        )

    def _make_subagent(self, spec: dict[str, str], context: dict[str, Any]):
        if self.subagent_factory is not None:
            return self.subagent_factory(copy.deepcopy(spec), copy.deepcopy(context))
        return SubAgent(
            agent_id=spec["id"],
            task=spec["task"],
            model_client=self.model_client,
            tool_executor=self.tool_executor,
            context=context,
            max_subagents=self.max_subagents,
            subagent_budget=self.subagent_budget,
        )

    def _handle_thinking_summary(
        self,
        raw_response: Any,
        action: dict[str, Any],
    ) -> tuple[ThinkingSummary, InvalidAction | None]:
        summary_payload = action.get("summary")
        if not isinstance(summary_payload, dict):
            return (
                ThinkingSummary(goal=""),
                InvalidAction(
                    raw=_extract_response_text(raw_response),
                    error="thinking_summary action must include a summary object",
                ),
            )
        summary = ThinkingSummary.from_mapping(summary_payload)
        errors = summary.validate()
        if errors:
            return (
                summary,
                InvalidAction(
                    raw=_extract_response_text(raw_response),
                    error="invalid thinking_summary: " + "; ".join(errors),
                ),
            )
        return summary, None


def parse_model_action(raw_response: Any) -> tuple[dict[str, Any], InvalidAction | None]:
    text = _extract_response_text(raw_response)
    try:
        action = json.loads(text)
    except json.JSONDecodeError as exc:
        return (
            {},
            InvalidAction(
                raw=text,
                error=f"model output must be strict JSON action; JSON parse failed: {exc.msg}",
            ),
        )
    if not isinstance(action, dict):
        return {}, InvalidAction(raw=text, error="model output must be a strict JSON object")
    if "action" not in action:
        return {}, InvalidAction(raw=text, error="strict JSON action requires an action field")
    schema_error = _validate_action_schema(action)
    if schema_error:
        return {}, InvalidAction(raw=text, error=schema_error)
    return action, None


def _validate_action_schema(action: dict[str, Any]) -> str | None:
    action_name = action.get("action")
    if not isinstance(action_name, str):
        return "strict JSON action field must be a string"
    if action_name == "final":
        if not isinstance(action.get("content"), str):
            return "final action content must be a string"
        return None
    if action_name == "tool_call":
        tool_name = action.get("tool", action.get("name"))
        if not isinstance(tool_name, str) or not tool_name.strip():
            return "tool_call action must include a non-empty tool field"
        if not isinstance(action.get("args"), dict):
            return "tool_call args must be an object"
        return None
    if action_name == "spawn_subagents":
        tasks = action.get("tasks")
        if not isinstance(tasks, list):
            return "spawn_subagents tasks must be a list"
        for index, item in enumerate(tasks):
            if isinstance(item, str):
                if not item.strip():
                    return f"spawn_subagents tasks[{index}] must be a non-empty string"
                continue
            if isinstance(item, dict):
                task = item.get("task")
                if not isinstance(task, str) or not task.strip():
                    return f"spawn_subagents tasks[{index}].task must be a non-empty string"
                item_id = item.get("id")
                if item_id is not None and (not isinstance(item_id, str) or not item_id.strip()):
                    return f"spawn_subagents tasks[{index}].id must be a non-empty string when provided"
                continue
            return f"spawn_subagents tasks[{index}] must be a string or object"
        if "concurrent" in action and not isinstance(action["concurrent"], bool):
            return "spawn_subagents concurrent must be a boolean when provided"
        if "max_workers" in action and (
            isinstance(action["max_workers"], bool)
            or not isinstance(action["max_workers"], int)
            or action["max_workers"] < 1
        ):
            return "spawn_subagents max_workers must be a positive integer when provided"
        return None
    if action_name == "thinking_summary":
        if not isinstance(action.get("summary"), dict):
            return "thinking_summary action must include a summary object"
        return None
    return f"unknown action {action_name!r}; expected strict JSON action"


def _extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        return str(response)
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = response.get("output")
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            content = item.get("content") if isinstance(item, dict) else item
            text = _content_to_text(content)
            if text:
                texts.append(text)
        if texts:
            return "\n".join(texts)
    return _content_to_text(response.get("content", ""))


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if "content" in value:
            return _content_to_text(value["content"])
        return ""
    if isinstance(value, list):
        return "\n".join(part for part in (_content_to_text(item) for item in value) if part)
    return str(value)


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return value


def _non_negative_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _bounded_requested_workers(value: Any, max_workers: int) -> int:
    requested = _positive_int(value, max_workers)
    return max(1, min(requested, max_workers))


class SubAgentBudget:
    def __init__(self, total: int) -> None:
        self.total = max(0, int(total))
        self._remaining = self.total
        self._lock = threading.Lock()

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._remaining

    def reserve(self, count: int) -> bool:
        if count <= 0:
            return True
        with self._lock:
            if count > self._remaining:
                return False
            self._remaining -= count
            return True
