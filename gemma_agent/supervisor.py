from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import re
import shlex
import threading
from typing import Any

from .citations import CitationManager
from .context import ContextBuilder
from .effort import TaskEffortBudget, TaskEffortPolicy, estimate_text_tokens
from .memory import MemoryStore
from .progress import ProgressTracker
from .safety import neutralize_model_facing_metadata
from .schemas import AgentRunResult, InvalidAction, SubAgentResult, ThinkingSummary, ToolResult
from .subagent import SubAgent
from .tool_executor import ToolExecutor

DEFAULT_MAX_SUBAGENTS = 8
MIN_TOOL_EVIDENCE_BUDGET_CHARS = 2000


ACTION_INSTRUCTIONS = (
    "Return exactly one strict JSON object per turn. "
    "Allowed actions: final, tool_call, spawn_subagents, thinking_summary. "
    "Tool execution is only real after the external runtime returns ToolResult evidence. "
    "Use context7_search as the primary tool for software, code, package, install, "
    "command, API, SDK, and framework documentation. Do not use duckduckgo_search for "
    "software, code, or coding tasks; DuckDuckGo is only for public news, people, and "
    "non-code public information."
)

_FILE_EDIT_CLAIM_RE = re.compile(
    r"\b(?:edited|modified|wrote|updated|changed|created|patched|saved|"
    r"deleted|removed|added|moved|copied|replaced|renamed)\b"
    r".{0,120}\b(?:file|files|[\w./\\-]+\.[A-Za-z0-9]{1,12})\b"
    r"|\bmade\s+changes\s+to\b"
    r".{0,120}\b(?:file|files|[\w./\\-]+\.[A-Za-z0-9]{1,12})\b",
    re.IGNORECASE,
)
_PASSIVE_FILE_EDIT_CLAIM_RE = re.compile(
    r"\b(?:file|files|[\w./\\-]+\.[A-Za-z0-9]{1,12})\b"
    r".{0,80}\b(was|were|has been|have been|is|are)?\s*"
    r"(edited|modified|written|wrote|updated|changed|created|patched|saved|"
    r"deleted|removed|added|moved|copied|replaced|renamed)\b",
    re.IGNORECASE,
)
_EXECUTION_CLAIM_RE = re.compile(
    r"\b(ran|run|executed|called|used|invoked)\b"
    r".{0,80}\b(command|commands|test|tests|tool|tools|unittest|pytest|suite|script|shell)\b",
    re.IGNORECASE,
)
_DIRECT_EXECUTION_CLAIM_RE = re.compile(
    r"\b(?:ran|run|executed)\s+[\w./\\-]{2,}\b"
    r"|\b(?:called|used|invoked)\s+[\w./\\-]{2,}\b",
    re.IGNORECASE,
)
_PASSING_TEST_CLAIM_RE = re.compile(
    r"\b(tests?|unittest|pytest|suite)\b.{0,40}\b(pass|passed|passing|succeeded|green)\b",
    re.IGNORECASE,
)
_FAILED_TEST_REPORT_RE = re.compile(
    r"\b(tests?|unittest|pytest|suite)\b.{0,40}\b"
    r"((?:did\s+not|didn['’]?t)\s+pass|failed\s+to\s+pass|failed|failing)\b",
    re.IGNORECASE,
)
_NEGATED_TEST_PASS_RE = re.compile(
    r"\b(tests?|unittest|pytest|suite)\b.{0,40}\b"
    r"((?:did\s+not|didn['’]?t)\s+pass|failed\s+to\s+pass)\b",
    re.IGNORECASE,
)
_TEST_VERIFICATION_CLAIM_RE = re.compile(
    r"\b(verified|checked|validated)\s+with\s+(pytest|unittest|tests?|suite)\b"
    r"|\b(?:built\s+and\s+)?tested\s+locally\b",
    re.IGNORECASE,
)
_SPECIFIC_TOOL_CLAIM_RE = re.compile(
    r"\b(called|used|invoked)\s+([A-Za-z_][\w.-]{1,80})\b",
    re.IGNORECASE,
)
_DIRECT_RAN_CLAIM_RE = re.compile(
    r"\b(?:ran|run|executed)\s+([A-Za-z0-9_.\\/:-]{2,120})\b",
    re.IGNORECASE,
)
_TEST_EVIDENCE_RE = re.compile(r"\b(pytest|unittest|test|tests|suite)\b", re.IGNORECASE)
_TEST_COMMAND_RE = re.compile(
    r"\b("
    r"pytest|"
    r"unittest|"
    r"python(?:\.exe)?\s+-m\s+(?:pytest|unittest)|"
    r"npm\s+test|"
    r"yarn\s+test|"
    r"pnpm\s+test|"
    r"go\s+test|"
    r"cargo\s+test|"
    r"dotnet\s+test"
    r")\b",
    re.IGNORECASE,
)
_TEST_TOOL_TOKENS = {"pytest", "unittest", "test", "tests", "suite"}
_TEST_SUCCESS_STATUSES = {"completed", "ok", "passed", "success", "succeeded"}
_TEST_INCOMPLETE_OR_FAILED_STATUSES = {
    "failed",
    "killed",
    "running",
    "stalled",
    "timeout",
}
_TEST_EXECUTION_TOOL_TOKENS = {
    "exec",
    "execute",
    "pytest",
    "run",
    "runner",
    "unittest",
}
_READ_ONLY_TOOL_TOKENS = {
    "cat",
    "fetch",
    "get",
    "list",
    "open",
    "read",
    "show",
    "view",
}
_COMMAND_TOOL_TOKENS = {
    "shell",
    "command",
    "cmd",
    "powershell",
    "bash",
    "sh",
    "exec",
    "terminal",
    "subprocess",
}
_FILE_EDIT_TOOL_TOKENS = {
    "write",
    "edit",
    "patch",
    "save",
    "create",
    "add",
    "delete",
    "remove",
    "move",
    "copy",
    "replace",
    "rename",
}
_GENERIC_FILE_EDIT_TOOL_TOKENS = {"apply"}
_FILE_TARGET_TOOL_TOKENS = {
    "dir",
    "directory",
    "file",
    "files",
    "filesystem",
    "fs",
    "path",
    "paths",
}
_PATH_ARG_KEYS = {"path", "file", "file_path", "filepath", "target", "destination", "source"}
_COMMAND_ARG_KEYS = {"command", "cmd", "script", "shell", "argv", "args"}
_DIRECT_COMMAND_CLAIM_NAMES = {
    "bash",
    "bun",
    "cargo",
    "cmd",
    "dir",
    "dotnet",
    "echo",
    "gemma-codex.cmd",
    "git",
    "go",
    "grep",
    "ls",
    "make",
    "node",
    "npm",
    "npx",
    "pnpm",
    "powershell",
    "pip",
    "py",
    "pytest",
    "python",
    "python.exe",
    "pwd",
    "sh",
    "unittest",
    "yarn",
}
_NATURAL_LANGUAGE_CLAIM_OBJECTS = {
    "a",
    "an",
    "context",
    "provided",
    "that",
    "the",
    "this",
}
_TERMINAL_FILE_EDIT_COMMAND_RE = re.compile(
    r"\b("
    r"set-content|add-content|out-file|new-item|remove-item|move-item|copy-item|"
    r"rename-item|copy|move|del|erase|rm|mv|cp|touch|mkdir|rmdir|tee"
    r")\b"
    r"|(?:^|[\s;&|])(?:echo|printf)\b.{0,240}(?:>>?|/Y\s+)\s*",
    re.IGNORECASE,
)
_SPECIFIC_TOOL_ALIASES = {
    "context7": {"context7_search"},
    "ctx7": {"context7_search"},
}


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
        effort_policy: TaskEffortPolicy | None = None,
    ) -> None:
        self.model_client = model_client
        self.tool_executor = tool_executor
        self.memory = MemoryStore() if memory is None else memory
        self.citations = CitationManager() if citations is None else citations
        self.context_builder = (
            ContextBuilder(
                self.memory,
                self.citations,
                safety_guard=self.tool_executor.safety_guard,
            )
            if context_builder is None
            else context_builder
        )
        self.max_iterations = max_iterations
        self.max_subagents = _non_negative_int(max_subagents, DEFAULT_MAX_SUBAGENTS)
        self.subagent_budget = (
            SubAgentBudget(self.max_subagents) if subagent_budget is None else subagent_budget
        )
        self.subagent_factory = subagent_factory
        self.effort_policy = TaskEffortPolicy() if effort_policy is None else effort_policy

    def run(self, task: str, *, context: dict[str, Any] | None = None) -> AgentRunResult:
        run_context = copy.deepcopy(context or {})
        tool_results = []
        subagent_results: list[SubAgentResult] = []
        thinking_summaries: list[ThinkingSummary] = []
        invalid_actions: list[InvalidAction] = []
        effort_budget = _clamp_effort_budget_to_runtime_limit(
            self.effort_policy.plan_for_task(task, context=run_context),
            self.max_iterations,
        )
        progress = ProgressTracker(
            repeated_tool_threshold=effort_budget.repeated_tool_threshold,
            repeated_invalid_threshold=effort_budget.repeated_invalid_threshold,
            safety_guard=self.tool_executor.safety_guard,
        )

        for _ in range(self.max_iterations):
            progress.start_iteration()
            payload = self._build_model_payload(
                task,
                run_context,
                tool_results=tool_results,
                subagent_results=subagent_results,
                thinking_summaries=thinking_summaries,
                invalid_actions=invalid_actions,
                progress=progress,
                effort_budget=effort_budget,
            )
            raw_response = self.model_client.create_response(payload)
            action, invalid = parse_model_action(raw_response)
            if invalid is not None:
                invalid_actions.append(invalid)
                progress.record_invalid_action(invalid)
                self.memory.add(
                    "evidence",
                    self._invalid_action_error_to_evidence(invalid, source="invalid_model_action"),
                    source="invalid_model_action",
                    metadata={"runtime_wrapped": True},
                )
                continue

            action_name = action.get("action")
            if action_name == "final":
                final_content = str(action.get("content", ""))
                claim_evidence_error = _final_claim_evidence_error(
                    final_content,
                    tool_results,
                    subagent_results,
                )
                if claim_evidence_error:
                    invalid = InvalidAction(
                        raw=_extract_response_text(raw_response),
                        error=(
                            "final action claims file edits or command/test/tool execution, "
                            f"but {claim_evidence_error}"
                        ),
                    )
                    invalid_actions.append(invalid)
                    progress.record_invalid_action(invalid)
                    self.memory.add(
                        "evidence",
                        self._invalid_action_error_to_evidence(invalid, source="invalid_final_claim"),
                        source="invalid_final_claim",
                        metadata={"runtime_wrapped": True},
                    )
                    continue
                return AgentRunResult(
                    final=final_content,
                    ok=True,
                    tool_results=tool_results,
                    subagent_results=subagent_results,
                    thinking_summaries=thinking_summaries,
                    invalid_actions=invalid_actions,
                )
            if action_name == "tool_call":
                tool_result = self._handle_tool_call(action)
                tool_results.append(tool_result)
                progress.record_tool_result(tool_result)
                safety_guard = self.tool_executor.safety_guard
                if tool_results[-1].error_code == "tool_not_allowed" and not tool_results[-1].executed:
                    tool_source = safety_guard.safe_untrusted_metadata_source("tool", tool_results[-1].tool_name)
                else:
                    tool_source = safety_guard.safe_metadata_source("tool", tool_results[-1].tool_name)
                self.memory.add(
                    "evidence",
                    safety_guard.wrap_untrusted_evidence(
                        tool_source,
                        tool_results[-1].to_evidence(safety_guard),
                        field="tool_result",
                    ),
                    source=tool_source,
                    metadata={"runtime_wrapped": True},
                )
                continue
            if action_name == "spawn_subagents":
                new_results = self._handle_spawn_subagents(action, task, run_context)
                subagent_results.extend(new_results)
                for result in new_results:
                    safety_guard = self.tool_executor.safety_guard
                    subagent_source = safety_guard.safe_metadata_source("subagent", result.agent_id)
                    self.memory.add(
                        "evidence",
                        safety_guard.wrap_untrusted_evidence(
                            subagent_source,
                            result.to_evidence(safety_guard),
                            field="subagent_result",
                        ),
                        source=subagent_source,
                        metadata={"runtime_wrapped": True},
                    )
                continue
            if action_name == "thinking_summary":
                summary, summary_invalid = self._handle_thinking_summary(raw_response, action)
                if summary_invalid is not None:
                    invalid_actions.append(summary_invalid)
                    progress.record_invalid_action(summary_invalid)
                    self.memory.add(
                        "evidence",
                        self._invalid_action_error_to_evidence(summary_invalid, source="invalid_thinking_summary"),
                        source="invalid_thinking_summary",
                        metadata={"runtime_wrapped": True},
                    )
                    continue
                thinking_summaries.append(summary)
                self.memory.add(
                    "task_scratchpad",
                    self._thinking_summary_to_evidence(summary),
                    source="thinking_summary",
                    metadata={"runtime_wrapped": True},
                )
                continue

            invalid_actions.append(
                invalid_action := InvalidAction(
                    raw=_extract_response_text(raw_response),
                    error=f"unknown action {action_name!r}; expected strict JSON action",
                )
            )
            progress.record_invalid_action(invalid_action)

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
        progress: ProgressTracker,
        effort_budget: TaskEffortBudget,
    ) -> dict[str, Any]:
        safety_guard = self.tool_executor.safety_guard
        tool_output_tokens = sum(estimate_text_tokens(result.output) for result in tool_results)
        final_tokens = sum(estimate_text_tokens(result.final) for result in subagent_results)
        nonlinear_cost_score = self.effort_policy.cost_score(
            turns=progress.iteration,
            tool_calls=len(tool_results),
            model_calls=progress.iteration,
            tool_output_tokens=tool_output_tokens,
            final_tokens=final_tokens,
        )
        raw_tool_evidence = [result.to_evidence(safety_guard) for result in tool_results]
        tool_evidence = _fit_tool_evidence_to_budget(
            raw_tool_evidence,
            max(MIN_TOOL_EVIDENCE_BUDGET_CHARS, effort_budget.max_context_chars),
        )
        return {
            "input": [
                {"role": "system", "content": ACTION_INSTRUCTIONS},
                {"role": "user", "content": task},
            ],
            "context": self.context_builder.build(task=task, extra=run_context),
            "tools": self.tool_executor.registry.list_tool_schemas(),
            "tool_results": tool_evidence,
            "subagent_results": [result.to_evidence(safety_guard) for result in subagent_results],
            "thinking_summaries": [
                self._thinking_summary_to_evidence(summary) for summary in thinking_summaries
            ],
            "invalid_actions": [self._invalid_action_to_evidence(item) for item in invalid_actions],
            "progress": progress.to_payload(),
            "effort": effort_budget.to_payload(nonlinear_cost_score=nonlinear_cost_score),
        }

    def _invalid_action_to_evidence(self, invalid: InvalidAction) -> dict[str, Any]:
        safety_guard = self.tool_executor.safety_guard
        raw_text = str(invalid.raw or "")
        return {
            "raw_preview": safety_guard.wrap_untrusted_evidence(
                "invalid_model_action",
                _redacted_invalid_raw_preview(raw_text),
                field="raw_preview",
            ),
            "raw_length": len(raw_text),
            "raw_sha256": _sha256_text(raw_text),
            "error": safety_guard.wrap_untrusted_evidence(
                "invalid_model_action",
                invalid.error,
                field="error",
            ),
        }

    def _invalid_action_error_to_evidence(self, invalid: InvalidAction, *, source: str) -> dict[str, Any]:
        return self.tool_executor.safety_guard.wrap_untrusted_evidence(
            source,
            invalid.error,
            field="error",
        )

    def _thinking_summary_to_evidence(self, summary: ThinkingSummary) -> dict[str, Any]:
        return self.tool_executor.safety_guard.wrap_untrusted_evidence(
            "thinking_summary",
            summary.format_lines(),
            field="thinking_summary",
        )

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
        delegated_tasks: list[str] = []
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
            delegated_tasks.append(spec["task"])
            spec, child_context = self._prepare_subagent_spec(spec, child_context)
            agents.append(self._make_subagent(spec, child_context))

        concurrent = action.get("concurrent")
        if concurrent is None:
            concurrent = len(agents) > 1

        results = run_subagents(
            agents,
            concurrent=bool(concurrent),
            max_workers=_bounded_requested_workers(action.get("max_workers"), self.max_subagents),
        )
        return [
            _with_delegated_task(result, delegated_tasks[index])
            for index, result in enumerate(results)
        ]

    def _prepare_subagent_spec(
        self,
        spec: dict[str, str],
        context: dict[str, Any],
    ) -> tuple[dict[str, str], dict[str, Any]]:
        prepared_spec = copy.deepcopy(spec)
        task = spec["task"]
        prepared_context = copy.deepcopy(context)
        delegated_task = self.tool_executor.safety_guard.wrap_untrusted_evidence(
            "model_subagent_task",
            task,
            field="subagent_task",
        )
        prepared_context["task"] = delegated_task
        prepared_context["delegated_task"] = delegated_task
        prepared_spec["task"] = str(neutralize_model_facing_metadata(task))
        return prepared_spec, prepared_context

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
        errors = [*ThinkingSummary.validate_mapping(summary_payload), *summary.validate()]
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


def _with_delegated_task(result: SubAgentResult, delegated_task: str) -> SubAgentResult:
    return SubAgentResult(
        agent_id=result.agent_id,
        task=delegated_task,
        final=result.final,
        ok=result.ok,
        error=result.error,
        tool_results=list(result.tool_results),
    )


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


def _clamp_effort_budget_to_runtime_limit(
    budget: TaskEffortBudget,
    max_iterations: int,
) -> TaskEffortBudget:
    effective_iterations = max(0, int(max_iterations))
    if budget.max_iterations <= effective_iterations:
        return budget
    return replace(budget, max_iterations=effective_iterations)


def _claims_execution_without_evidence(
    final_content: str,
    tool_results: list[ToolResult],
    subagent_results: list[SubAgentResult],
) -> bool:
    return _final_claim_evidence_error(final_content, tool_results, subagent_results) is not None


def _final_claim_evidence_error(
    final_content: str,
    tool_results: list[ToolResult],
    subagent_results: list[SubAgentResult],
) -> str | None:
    requirements, specific_tools = _claim_requirements(final_content)
    if not requirements and not specific_tools:
        return None
    executed_results = _executed_tool_results(tool_results, subagent_results)
    if not executed_results:
        return "missing execution evidence: no executed ToolResult exists"
    if _requirements_have_matching_evidence(requirements, specific_tools, executed_results):
        return None
    return "missing matching execution evidence for the claim category"


def _claim_requirements(final_content: str) -> tuple[set[str], set[str]]:
    final_content = _strip_negated_execution_caveats(final_content)
    final_content = _strip_negated_file_edit_caveats(final_content)
    requirements: set[str] = set()
    specific_tools: set[str] = set()
    if _FILE_EDIT_CLAIM_RE.search(final_content) or _PASSIVE_FILE_EDIT_CLAIM_RE.search(final_content):
        requirements.add("file_edit")
    failed_test_report = _FAILED_TEST_REPORT_RE.search(final_content)
    if failed_test_report:
        requirements.add("test_failure")
    if (
        _PASSING_TEST_CLAIM_RE.search(final_content)
        and not _NEGATED_TEST_PASS_RE.search(final_content)
    ) or _TEST_VERIFICATION_CLAIM_RE.search(final_content):
        requirements.add("test_success")
    if _EXECUTION_CLAIM_RE.search(final_content):
        lowered = final_content.lower()
        if any(word in lowered for word in ("test", "tests", "pytest", "unittest", "suite")):
            requirements.add("test_execution")
        elif "tool" in lowered:
            requirements.add("tool_call")
        else:
            requirements.add("command")
    for match in _SPECIFIC_TOOL_CLAIM_RE.finditer(final_content):
        verb = match.group(1).lower()
        tool_name = _normalize_tool_name(match.group(2))
        if _is_specific_tool_claim_name(tool_name, verb=verb):
            specific_tools.add(tool_name)
    for match in _DIRECT_RAN_CLAIM_RE.finditer(final_content):
        command_name = match.group(1)
        if not _looks_like_explicit_command_claim(command_name):
            continue
        if _TEST_EVIDENCE_RE.search(command_name):
            requirements.add("test_execution")
        else:
            requirements.add("command")
    return requirements, specific_tools


def _strip_negated_execution_caveats(final_content: str) -> str:
    patterns = [
        r"\btests?\s+(?:were\s+|was\s+)?not\s+(?:run|executed)\b\.?",
        (
            r"\bI\s+(?:did\s+not|didn['’]?t|have\s+not|haven['’]?t)\s+"
            r"(?:run|execute|executed?)\s+"
            r"(?:pytest|unittest|tests?|suite|verification|checks?)\b\.?"
        ),
        r"\bno\s+(?:verification|tests?|checks?)\s+(?:was\s+|were\s+)?(?:run|executed)\b\.?",
        (
            r"\b(?:I\s+)?(?:couldn['’]?t|could\s+not|can['’]?t|cannot|"
            r"wasn['’]?t\s+able\s+to|was\s+not\s+able\s+to|am\s+unable\s+to|"
            r"unable\s+to)\s+(?:run|execute)\s+"
            r"(?:pytest|unittest|tests?|suite|verification|checks?)\b\.?"
        ),
        (
            r"\b(?:I\s+)?haven['’]?t\s+(?:run|executed?)\s+"
            r"(?:pytest|unittest|tests?|suite|verification|checks?)\b\.?"
        ),
    ]
    stripped = final_content
    for pattern in patterns:
        stripped = re.sub(pattern, " ", stripped, flags=re.IGNORECASE)
    return stripped


def _strip_negated_file_edit_caveats(final_content: str) -> str:
    edit_verbs = (
        r"edit|edited|modify|modified|write|written|wrote|update|updated|"
        r"change|changed|create|created|patch|patched|save|saved|"
        r"delete|deleted|remove|removed|add|added|move|moved|copy|copied|"
        r"replace|replaced|rename|renamed"
    )
    patterns = [
        (
            r"\bno\s+files?\s+(?:was\s+|were\s+|has\s+been\s+|have\s+been\s+)?"
            rf"(?:{edit_verbs})\b\.?"
        ),
        (
            rf"\bI\s+(?:did\s+not|didn['’]?t|have\s+not|haven['’]?t)\s+"
            rf"(?:{edit_verbs})\s+"
            r"(?:any\s+)?files?\b\.?"
        ),
        (
            r"\bfiles?\s+"
            r"(?:wasn['’]?t|was\s+not|weren['’]?t|were\s+not|"
            r"hasn['’]?t\s+been|has\s+not\s+been|haven['’]?t\s+been|"
            r"have\s+not\s+been|isn['’]?t|is\s+not|aren['’]?t|are\s+not)\s+"
            rf"(?:{edit_verbs})\b\.?"
        ),
    ]
    stripped = final_content
    for pattern in patterns:
        stripped = re.sub(pattern, " ", stripped, flags=re.IGNORECASE)
    return stripped


def _is_specific_tool_claim_name(tool_name: str, *, verb: str) -> bool:
    if tool_name in {
        "tool",
        "tools",
        "command",
        "commands",
        "test",
        "tests",
        *_NATURAL_LANGUAGE_CLAIM_OBJECTS,
    }:
        return False
    if verb == "used" and not _looks_like_explicit_used_tool_claim(tool_name):
        return False
    if verb in {"called", "invoked"} and not _looks_like_explicit_tool_claim(tool_name):
        return False
    return True


def _looks_like_explicit_used_tool_claim(tool_name: str) -> bool:
    return tool_name in _SPECIFIC_TOOL_ALIASES or bool(re.search(r"[_./\\:-]", tool_name))


def _looks_like_explicit_tool_claim(tool_name: str) -> bool:
    return (
        tool_name in _SPECIFIC_TOOL_ALIASES
        or bool(re.search(r"[_./\\:-]", tool_name))
        or tool_name in _DIRECT_COMMAND_CLAIM_NAMES
    )


def _looks_like_explicit_command_claim(command_name: str) -> bool:
    command_name = command_name.strip().strip(".,;:!?)]}'\"")
    if not command_name:
        return False
    command_base = _command_basename(command_name)
    return (
        command_base in _DIRECT_COMMAND_CLAIM_NAMES
        or bool(re.search(r"[./\\:-]", command_name))
    )


def _specific_tool_claim_matches_result(tool_name: str, result: ToolResult) -> bool:
    actual = _normalize_tool_name(result.tool_name)
    if actual == tool_name:
        return True
    return actual in _SPECIFIC_TOOL_ALIASES.get(tool_name, set())


def _requirements_have_matching_evidence(
    requirements: set[str],
    specific_tools: set[str],
    executed_results: list[ToolResult],
) -> bool:
    for tool_name in specific_tools:
        if not any(_specific_tool_claim_matches_result(tool_name, result) for result in executed_results):
            return False
    return all(
        any(_result_matches_claim_category(result, requirement) for result in executed_results)
        for requirement in requirements
    )


def _result_matches_claim_category(result: ToolResult, category: str) -> bool:
    if category == "tool_call":
        return True
    if category == "command":
        return _tool_result_looks_like_command(result)
    if category == "test_execution":
        return _tool_result_looks_like_test(result)
    if category == "test_success":
        return result.ok and _tool_result_looks_like_test(result) and _tool_result_completed_successfully(result)
    if category == "test_failure":
        return _tool_result_looks_like_test(result) and not _tool_result_completed_successfully(result)
    if category == "file_edit":
        return result.ok and _tool_result_looks_like_file_edit(result)
    return False


def _executed_tool_results(
    tool_results: list[ToolResult],
    subagent_results: list[SubAgentResult],
) -> list[ToolResult]:
    results = [result for result in tool_results if result.executed]
    for subagent in subagent_results:
        results.extend(result for result in subagent.tool_results if result.executed)
    return results


def _tool_result_looks_like_command(result: ToolResult) -> bool:
    return _tool_name_has_any_token(result.tool_name, _COMMAND_TOOL_TOKENS)


def _tool_result_looks_like_test(result: ToolResult) -> bool:
    if _tool_name_indicates_test_execution(result.tool_name):
        return True
    if not _tool_result_looks_like_command(result):
        return False
    return _command_args_start_with_test_runner(result)


def _tool_result_completed_successfully(result: ToolResult) -> bool:
    status = _tool_result_output_status(result)
    if status in _TEST_INCOMPLETE_OR_FAILED_STATUSES:
        return False
    if status in _TEST_SUCCESS_STATUSES:
        return True
    return result.ok


def _tool_result_output_status(result: ToolResult) -> str:
    output = result.output
    if isinstance(output, dict) and "content" in output:
        output = output["content"]
    if isinstance(output, dict):
        status = output.get("status")
        if isinstance(status, str):
            return status.strip().lower()
    return ""


def _tool_result_looks_like_file_edit(result: ToolResult) -> bool:
    name_tokens = _tool_name_tokens(result.tool_name)
    if name_tokens & {"write", "edit", "patch", "save"}:
        if name_tokens & _FILE_TARGET_TOOL_TOKENS:
            if _tool_args_include_explicit_path_arg(result.args):
                return True
            return _tool_args_include_path_like_value(result.args)
        non_target_tokens = name_tokens - _FILE_EDIT_TOOL_TOKENS - _GENERIC_FILE_EDIT_TOOL_TOKENS
        if non_target_tokens:
            return False
        return _tool_args_include_explicit_path_arg(result.args) or _tool_args_include_patch_path_arg(result.args)
    if name_tokens & {"create", "add", "delete", "remove", "move", "copy", "replace", "rename"}:
        if not name_tokens & _FILE_TARGET_TOOL_TOKENS:
            return False
        if _tool_args_include_explicit_path_arg(result.args):
            return True
        return _tool_args_include_path_like_value(result.args)
    if _tool_result_looks_like_command(result):
        return _command_args_include_file_edit(result)
    return False


def _tool_invocation_text(result: ToolResult) -> str:
    parts = [result.tool_name]
    for key, value in result.args.items():
        normalized_key = str(key).lower().replace("-", "_")
        if normalized_key in _COMMAND_ARG_KEYS:
            parts.append(str(value))
    return " ".join(parts)


def _command_args_start_with_test_runner(result: ToolResult) -> bool:
    for value in _command_arg_values(result.args):
        argv = _argv_from_command_value(value)
        if _argv_starts_with_test_runner(argv):
            return True
    return False


def _command_args_include_file_edit(result: ToolResult) -> bool:
    for value in _command_arg_values(result.args):
        argv = _argv_from_command_value(value)
        text = " ".join(argv) if argv else str(value)
        if _TERMINAL_FILE_EDIT_COMMAND_RE.search(text) and _tool_args_include_path_like_value(text):
            return True
    return False


def _command_arg_values(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            if normalized_key in _COMMAND_ARG_KEYS:
                yield item
            else:
                yield from _command_arg_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _command_arg_values(item)


def _argv_from_command_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            return shlex.split(value, posix=False)
        except ValueError:
            return value.split()
    return []


def _argv_starts_with_test_runner(argv: list[str]) -> bool:
    if not argv:
        return False
    executable = _command_basename(argv[0])
    if executable in {"pytest", "unittest"}:
        return True
    if executable in {"npm", "yarn", "pnpm", "go", "cargo", "dotnet"}:
        return len(argv) > 1 and argv[1] == "test"
    if executable in {"python", "python.exe", "py"}:
        return len(argv) > 2 and argv[1] == "-m" and argv[2] in {"pytest", "unittest"}
    if executable in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        return _shell_wrapped_command_starts_with_test_runner(
            argv,
            command_flags={"-command", "-c"},
        )
    if executable in {"cmd", "cmd.exe"}:
        return _shell_wrapped_command_starts_with_test_runner(
            argv,
            command_flags={"/c", "/k"},
        )
    return False


def _shell_wrapped_command_starts_with_test_runner(
    argv: list[str],
    *,
    command_flags: set[str],
) -> bool:
    for index, arg in enumerate(argv[1:], start=1):
        if arg.lower() not in command_flags or index + 1 >= len(argv):
            continue
        command_argv = _argv_from_command_value(argv[index + 1])
        return _argv_starts_with_test_runner(command_argv)
    return False


def _command_basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _tool_args_include_path_like_value(value: Any, property_name: str = "") -> bool:
    if isinstance(value, dict):
        return any(_tool_args_include_path_like_value(item, str(key)) for key, item in value.items())
    if isinstance(value, list):
        return any(_tool_args_include_path_like_value(item, property_name) for item in value)
    if not isinstance(value, str):
        return False
    normalized = property_name.lower().replace("-", "_")
    if normalized in _PATH_ARG_KEYS:
        return True
    return bool(re.search(r"[\\/]|[A-Za-z0-9_-]+\.[A-Za-z0-9]{1,12}\b", value))


def _tool_args_include_explicit_path_arg(value: Any, property_name: str = "") -> bool:
    if isinstance(value, dict):
        return any(
            _tool_args_include_explicit_path_arg(item, str(key))
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_tool_args_include_explicit_path_arg(item, property_name) for item in value)
    if not isinstance(value, str):
        return False
    normalized = property_name.lower().replace("-", "_")
    return normalized in _PATH_ARG_KEYS


def _tool_args_include_patch_path_arg(value: Any, property_name: str = "") -> bool:
    if isinstance(value, dict):
        return any(
            _tool_args_include_patch_path_arg(item, str(key))
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_tool_args_include_patch_path_arg(item, property_name) for item in value)
    if not isinstance(value, str):
        return False
    normalized = property_name.lower().replace("-", "_")
    if normalized not in {"patch", "diff"}:
        return False
    return bool(re.search(r"[\\/]|[A-Za-z0-9_-]+\.[A-Za-z0-9]{1,12}\b", value))


def _tool_name_has_any_token(tool_name: str, tokens: set[str]) -> bool:
    return bool(_tool_name_tokens(tool_name) & tokens)


def _tool_name_indicates_test_execution(tool_name: str) -> bool:
    name_tokens = _tool_name_tokens(tool_name)
    if not name_tokens & _TEST_TOOL_TOKENS:
        return False
    if name_tokens & _READ_ONLY_TOOL_TOKENS:
        return False
    return bool(name_tokens & _TEST_EXECUTION_TOOL_TOKENS)


def _tool_name_tokens(tool_name: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", _normalize_tool_name(tool_name))
        if token
    }


def _normalize_tool_name(value: str) -> str:
    return value.strip().strip(".,;:!?)]}'\"").lower().replace("-", "_")


def _fit_tool_evidence_to_budget(items: list[dict[str, Any]], budget_chars: int) -> list[dict[str, Any]]:
    budget = max(0, int(budget_chars))
    fitted: list[dict[str, Any]] = []
    used = 2
    for item in items:
        item_size = _json_size(item)
        if used + item_size <= budget:
            fitted.append(item)
            used += item_size
            continue
        summary = _summarize_tool_evidence(item)
        summary_size = _json_size(summary)
        if used + summary_size <= budget:
            fitted.append(summary)
            used += summary_size
            continue
        minimal = _minimal_tool_evidence(item)
        minimal_size = _json_size(minimal)
        if used + minimal_size <= budget or not fitted:
            fitted.append(minimal)
            used += minimal_size
    return fitted


def _summarize_tool_evidence(item: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "tool_name": item.get("tool_name"),
        "args": item.get("args"),
        "ok": item.get("ok"),
        "error": item.get("error"),
        "error_code": item.get("error_code"),
        "elapsed_ms": item.get("elapsed_ms"),
        "executed": item.get("executed"),
        "omitted_due_to_context_budget": True,
    }
    output = item.get("output")
    if output is not None:
        summary["output_summary"] = _summarize_value(output)
    return summary


def _minimal_tool_evidence(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_name": item.get("tool_name"),
        "ok": item.get("ok"),
        "error_code": item.get("error_code"),
        "elapsed_ms": item.get("elapsed_ms"),
        "executed": item.get("executed"),
        "omitted_due_to_context_budget": True,
    }


def _summarize_value(value: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "omitted_due_to_context_budget": True,
        "serialized_length": len(json.dumps(value, sort_keys=True, ensure_ascii=True, default=repr)),
    }
    _collect_summary_fields(value, fields)
    return fields


def _collect_summary_fields(value: Any, fields: dict[str, Any]) -> None:
    interesting = {
        "status",
        "job_id",
        "process_id",
        "exit_code",
        "stdout_length",
        "stderr_length",
        "stdout_sha256",
        "stderr_sha256",
        "elapsed_ms",
        "recovery",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key in interesting and key not in fields:
                fields[key] = child
            _collect_summary_fields(child, fields)
    elif isinstance(value, list):
        for child in value[:20]:
            _collect_summary_fields(child, fields)


def _json_size(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, ensure_ascii=True, default=repr))


def _redacted_invalid_raw_preview(text: str, *, max_chars: int = 500) -> str:
    preview = str(text or "")[:max(0, max_chars)]
    preview = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY]",
        preview,
        flags=re.DOTALL,
    )
    preview = re.sub(
        r"(?i)\b(password|passwd|token|api[_-]?key|bearer|openai_api_key)\s*[:=]\s*\S+",
        lambda match: f"{match.group(1)}=[REDACTED]",
        preview,
    )
    preview = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "sk-[REDACTED]", preview)
    return str(neutralize_model_facing_metadata(preview))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="replace")).hexdigest()
