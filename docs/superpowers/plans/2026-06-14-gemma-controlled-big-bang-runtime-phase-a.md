# Gemma Controlled Big-Bang Runtime Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MVP-shaped agent runtime core with focused, testable runtime modules while preserving the existing `AgentSupervisor` public API.

**Architecture:** Phase A introduces `gemma_agent.runtime` as the new runtime core and migrates behavior behind compatibility adapters. `AgentSupervisor` remains the external entry point during the migration, but delegates parsing, budgeting, evidence, claim verification, context assembly, tool execution, and subagent scheduling to focused modules.

**Tech Stack:** Python 3, dataclasses, standard-library `unittest`, existing `gemma_agent` schemas, existing `ToolExecutor`, existing `TaskEffortPolicy`, existing `SafetyGuard`.

---

## File Structure

- Create: `gemma_agent/runtime/__init__.py`
  - Exports the new runtime primitives.
- Create: `gemma_agent/runtime/budget.py`
  - Owns enforced per-run budgets and stop decisions.
- Create: `gemma_agent/runtime/actions.py`
  - Owns model action parsing and validation.
- Create: `gemma_agent/runtime/evidence.py`
  - Owns evidence recording and final-claim verification.
- Create: `gemma_agent/runtime/context.py`
  - Owns payload assembly and context fitting.
- Create: `gemma_agent/runtime/tools.py`
  - Wraps `ToolExecutor` with run-aware accounting and evidence recording.
- Create: `gemma_agent/runtime/subagents.py`
  - Owns subagent request validation, budget reservation, execution, and evidence fan-in.
- Create: `gemma_agent/runtime/run_loop.py`
  - Owns the runtime loop currently embedded in `AgentSupervisor.run`.
- Modify: `gemma_agent/supervisor.py`
  - Replace orchestration internals with delegation to `AgentRunLoop`; keep compatibility helpers until tests prove they can move.
- Modify: `gemma_agent/__init__.py`
  - Export new runtime primitives that should be public.
- Create: `tests/test_agent_runtime_phase_a_budget.py`
- Create: `tests/test_agent_runtime_phase_a_actions.py`
- Create: `tests/test_agent_runtime_phase_a_evidence.py`
- Create: `tests/test_agent_runtime_phase_a_context.py`
- Create: `tests/test_agent_runtime_phase_a_run_loop.py`
- Modify: `tests/test_gemma_agent_runtime.py`
  - Keep existing behavior tests as compatibility coverage.

## Execution Constraints

- Before implementation, create an isolated worktree from the current branch.
- If `.worktrees/` is not ignored in the committed tree, commit the ignore rule before creating the worktree.
- Do not delete old runtime code until the new module has equivalent tests and `AgentSupervisor` compatibility passes.
- Do not claim Phase A complete until the full unit suite and the local behavior benchmark pass on fresh output.

---

### Task 0: Worktree And Baseline Gate

**Files:**
- Modify if needed: `.gitignore`
- No production source changes in this task.

- [ ] **Step 1: Verify `.worktrees/` is ignored by the committed tree**

Run:

```powershell
git check-ignore -v .worktrees/
```

Expected if safe:

```text
.gitignore:<line>:.worktrees/	.worktrees/
```

- [ ] **Step 2: If the ignore rule is missing, add this exact entry**

Patch `.gitignore` under the local-runtime artifact section:

```gitignore
.superpowers/
.worktrees/
```

- [ ] **Step 3: Commit only the ignore safety change if it was needed**

Run:

```powershell
git add -- .gitignore
git commit -m "chore: ignore local superpowers worktrees"
```

Expected:

```text
[branch <sha>] chore: ignore local superpowers worktrees
```

- [ ] **Step 4: Create the isolated runtime rewrite worktree**

Run:

```powershell
New-Item -ItemType Directory -Force .worktrees | Out-Null
git worktree add .worktrees/gemma-runtime-phase-a -b codex/gemma-runtime-phase-a
```

Expected:

```text
Preparing worktree (new branch 'codex/gemma-runtime-phase-a')
HEAD is now at <sha> <message>
```

- [ ] **Step 5: Verify baseline in the worktree**

Run:

```powershell
Set-Location .worktrees/gemma-runtime-phase-a
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected:

```text
Ran <count> tests in <seconds>s
OK
```

- [ ] **Step 6: Commit nothing in this task unless `.gitignore` changed**

Run:

```powershell
git status --short
```

Expected:

```text
```

---

### Task 1: Enforced Budget Manager

**Files:**
- Create: `gemma_agent/runtime/__init__.py`
- Create: `gemma_agent/runtime/budget.py`
- Create: `tests/test_agent_runtime_phase_a_budget.py`
- Modify: `gemma_agent/__init__.py`

- [ ] **Step 1: Write the failing budget tests**

Create `tests/test_agent_runtime_phase_a_budget.py`:

```python
import unittest

from gemma_agent.effort import TaskEffortBudget
from gemma_agent.runtime.budget import BudgetDecision, BudgetManager
from gemma_agent.schemas import ToolResult


def hard_budget(**overrides):
    values = {
        "difficulty": "hard",
        "max_iterations": 3,
        "max_tool_calls": 2,
        "max_subagents": 2,
        "max_context_chars": 32000,
        "terminal_timeout_sec": 1800,
        "terminal_idle_timeout_sec": 120,
        "summarization_interval": 2,
        "repeated_tool_threshold": 2,
        "repeated_invalid_threshold": 2,
    }
    values.update(overrides)
    return TaskEffortBudget(**values)


class BudgetManagerTests(unittest.TestCase):
    def test_allows_work_until_limits_are_consumed(self):
        manager = BudgetManager(hard_budget())

        self.assertEqual(manager.before_iteration(), BudgetDecision.continue_run())
        manager.record_model_call(prompt_tokens=100, completion_tokens=20)
        manager.record_tool_result(ToolResult("shell_command", {}, ok=True, executed=True))
        manager.record_subagents(1)

        snapshot = manager.snapshot()
        self.assertEqual(snapshot.iterations_started, 1)
        self.assertEqual(snapshot.model_calls, 1)
        self.assertEqual(snapshot.tool_calls, 1)
        self.assertEqual(snapshot.subagents_started, 1)
        self.assertEqual(snapshot.remaining_tool_calls, 1)

    def test_stops_before_iteration_when_iteration_budget_is_exhausted(self):
        manager = BudgetManager(hard_budget(max_iterations=1))

        self.assertEqual(manager.before_iteration(), BudgetDecision.continue_run())
        decision = manager.before_iteration()

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "max_iterations_exhausted")

    def test_blocks_tool_calls_when_tool_budget_is_exhausted(self):
        manager = BudgetManager(hard_budget(max_tool_calls=1))
        manager.record_tool_result(ToolResult("shell_command", {}, ok=True, executed=True))

        decision = manager.before_tool_call("shell_command")

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "max_tool_calls_exhausted")

    def test_blocks_subagents_when_requested_count_exceeds_remaining_budget(self):
        manager = BudgetManager(hard_budget(max_subagents=2))
        manager.record_subagents(1)

        decision = manager.before_subagents(2)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "max_subagents_exhausted")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_runtime_phase_a_budget.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'gemma_agent.runtime'
```

- [ ] **Step 3: Create the runtime package exports**

Create `gemma_agent/runtime/__init__.py`:

```python
from .budget import BudgetDecision, BudgetManager, BudgetSnapshot

__all__ = [
    "BudgetDecision",
    "BudgetManager",
    "BudgetSnapshot",
]
```

- [ ] **Step 4: Implement the budget manager**

Create `gemma_agent/runtime/budget.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from gemma_agent.effort import TaskEffortBudget
from gemma_agent.schemas import ToolResult


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str = ""
    message: str = ""

    @classmethod
    def continue_run(cls) -> "BudgetDecision":
        return cls(True)

    @classmethod
    def stop(cls, reason: str, message: str) -> "BudgetDecision":
        return cls(False, reason, message)


@dataclass(frozen=True)
class BudgetSnapshot:
    difficulty: str
    iterations_started: int
    model_calls: int
    tool_calls: int
    subagents_started: int
    prompt_tokens: int
    completion_tokens: int
    max_iterations: int
    max_tool_calls: int
    max_subagents: int
    max_context_chars: int
    terminal_timeout_sec: float
    terminal_idle_timeout_sec: float

    @property
    def remaining_iterations(self) -> int:
        return max(0, self.max_iterations - self.iterations_started)

    @property
    def remaining_tool_calls(self) -> int:
        return max(0, self.max_tool_calls - self.tool_calls)

    @property
    def remaining_subagents(self) -> int:
        return max(0, self.max_subagents - self.subagents_started)

    def to_payload(self) -> dict[str, object]:
        return {
            "difficulty": self.difficulty,
            "iterations_started": self.iterations_started,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "subagents_started": self.subagents_started,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "max_subagents": self.max_subagents,
            "max_context_chars": self.max_context_chars,
            "terminal_timeout_sec": self.terminal_timeout_sec,
            "terminal_idle_timeout_sec": self.terminal_idle_timeout_sec,
            "remaining_iterations": self.remaining_iterations,
            "remaining_tool_calls": self.remaining_tool_calls,
            "remaining_subagents": self.remaining_subagents,
        }


class BudgetManager:
    def __init__(self, budget: TaskEffortBudget) -> None:
        self.budget = budget
        self._iterations_started = 0
        self._model_calls = 0
        self._tool_calls = 0
        self._subagents_started = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def before_iteration(self) -> BudgetDecision:
        if self._iterations_started >= max(0, int(self.budget.max_iterations)):
            return BudgetDecision.stop(
                "max_iterations_exhausted",
                f"iteration budget exhausted at {self._iterations_started}",
            )
        self._iterations_started += 1
        return BudgetDecision.continue_run()

    def record_model_call(self, *, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self._model_calls += 1
        self._prompt_tokens += max(0, int(prompt_tokens))
        self._completion_tokens += max(0, int(completion_tokens))

    def before_tool_call(self, tool_name: str) -> BudgetDecision:
        if self._tool_calls >= max(0, int(self.budget.max_tool_calls)):
            return BudgetDecision.stop(
                "max_tool_calls_exhausted",
                f"tool budget exhausted before {tool_name}",
            )
        return BudgetDecision.continue_run()

    def record_tool_result(self, result: ToolResult) -> None:
        if result.executed:
            self._tool_calls += 1

    def before_subagents(self, requested_count: int) -> BudgetDecision:
        requested = max(0, int(requested_count))
        if self._subagents_started + requested > max(0, int(self.budget.max_subagents)):
            return BudgetDecision.stop(
                "max_subagents_exhausted",
                f"subagent budget remaining {self.snapshot().remaining_subagents}, requested {requested}",
            )
        return BudgetDecision.continue_run()

    def record_subagents(self, count: int) -> None:
        self._subagents_started += max(0, int(count))

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            difficulty=self.budget.difficulty,
            iterations_started=self._iterations_started,
            model_calls=self._model_calls,
            tool_calls=self._tool_calls,
            subagents_started=self._subagents_started,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            max_iterations=max(0, int(self.budget.max_iterations)),
            max_tool_calls=max(0, int(self.budget.max_tool_calls)),
            max_subagents=max(0, int(self.budget.max_subagents)),
            max_context_chars=max(0, int(self.budget.max_context_chars)),
            terminal_timeout_sec=float(self.budget.terminal_timeout_sec),
            terminal_idle_timeout_sec=float(self.budget.terminal_idle_timeout_sec),
        )
```

- [ ] **Step 5: Export budget primitives from `gemma_agent/__init__.py`**

Add imports:

```python
from .runtime import BudgetDecision, BudgetManager, BudgetSnapshot
```

Add names to `__all__`:

```python
"BudgetDecision",
"BudgetManager",
"BudgetSnapshot",
```

- [ ] **Step 6: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_runtime_phase_a_budget.py -v
```

Expected:

```text
Ran 4 tests in <seconds>s
OK
```

- [ ] **Step 7: Commit**

Run:

```powershell
git add gemma_agent/runtime/__init__.py gemma_agent/runtime/budget.py gemma_agent/__init__.py tests/test_agent_runtime_phase_a_budget.py
git commit -m "feat: add enforced runtime budget manager"
```

---

### Task 2: Model Action Codec

**Files:**
- Create: `gemma_agent/runtime/actions.py`
- Create: `tests/test_agent_runtime_phase_a_actions.py`
- Modify: `gemma_agent/runtime/__init__.py`

- [ ] **Step 1: Write the failing action codec tests**

Create `tests/test_agent_runtime_phase_a_actions.py`:

```python
import json
import unittest

from gemma_agent.runtime.actions import ActionCodec


class ActionCodecTests(unittest.TestCase):
    def test_parses_strict_final_action(self):
        action, invalid = ActionCodec().parse(json.dumps({"action": "final", "content": "done"}))

        self.assertIsNone(invalid)
        self.assertEqual(action, {"action": "final", "content": "done"})

    def test_rejects_unknown_action(self):
        action, invalid = ActionCodec().parse(json.dumps({"action": "dance"}))

        self.assertIsNone(action)
        self.assertIsNotNone(invalid)
        self.assertIn("unknown action", invalid.error)

    def test_rejects_tool_call_without_tool_name(self):
        action, invalid = ActionCodec().parse(json.dumps({"action": "tool_call", "args": {}}))

        self.assertIsNone(action)
        self.assertIn("tool_call action must include", invalid.error)

    def test_extracts_text_from_response_shape(self):
        response = {"output": [{"content": [{"type": "output_text", "text": "{\"action\":\"final\",\"content\":\"ok\"}"}]}]}

        action, invalid = ActionCodec().parse(response)

        self.assertIsNone(invalid)
        self.assertEqual(action["content"], "ok")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run focused test and confirm it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_runtime_phase_a_actions.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'gemma_agent.runtime.actions'
```

- [ ] **Step 3: Implement the action codec**

Create `gemma_agent/runtime/actions.py`:

```python
from __future__ import annotations

import json
from typing import Any

from gemma_agent.schemas import InvalidAction

ALLOWED_ACTIONS = {"final", "tool_call", "spawn_subagents", "thinking_summary"}


class ActionCodec:
    def parse(self, raw_response: Any) -> tuple[dict[str, Any] | None, InvalidAction | None]:
        raw_text = extract_response_text(raw_response)
        try:
            action = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            return None, InvalidAction(raw=raw_text, error=f"response is not strict JSON: {exc.msg}")
        if not isinstance(action, dict):
            return None, InvalidAction(raw=raw_text, error="response JSON must be an object")

        action_name = action.get("action")
        if action_name not in ALLOWED_ACTIONS:
            return None, InvalidAction(
                raw=raw_text,
                error=f"unknown action {action_name!r}; expected one of {sorted(ALLOWED_ACTIONS)}",
            )
        if action_name == "tool_call":
            tool_name = action.get("tool", action.get("name"))
            if not isinstance(tool_name, str) or not tool_name:
                return None, InvalidAction(
                    raw=raw_text,
                    error="tool_call action must include a non-empty tool field",
                )
        if action_name == "spawn_subagents" and not isinstance(action.get("tasks", []), list):
            return None, InvalidAction(raw=raw_text, error="spawn_subagents tasks must be a list")
        if action_name == "thinking_summary":
            summary = action.get("summary", action)
            if not isinstance(summary, dict):
                return None, InvalidAction(raw=raw_text, error="thinking_summary must be an object")
        return action, None


def extract_response_text(raw_response: Any) -> str:
    if isinstance(raw_response, str):
        return raw_response
    if isinstance(raw_response, dict):
        if isinstance(raw_response.get("output_text"), str):
            return raw_response["output_text"]
        output = raw_response.get("output")
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                content = item.get("content") if isinstance(item, dict) else None
                if isinstance(content, list):
                    for child in content:
                        if isinstance(child, dict) and isinstance(child.get("text"), str):
                            chunks.append(child["text"])
            if chunks:
                return "".join(chunks)
    return str(raw_response)
```

- [ ] **Step 4: Export action codec**

Modify `gemma_agent/runtime/__init__.py`:

```python
from .actions import ActionCodec, extract_response_text
from .budget import BudgetDecision, BudgetManager, BudgetSnapshot

__all__ = [
    "ActionCodec",
    "BudgetDecision",
    "BudgetManager",
    "BudgetSnapshot",
    "extract_response_text",
]
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_runtime_phase_a_actions.py -v
```

Expected:

```text
Ran 4 tests in <seconds>s
OK
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add gemma_agent/runtime/actions.py gemma_agent/runtime/__init__.py tests/test_agent_runtime_phase_a_actions.py
git commit -m "feat: add model action codec"
```

---

### Task 3: Evidence Ledger And Claim Verifier

**Files:**
- Create: `gemma_agent/runtime/evidence.py`
- Create: `tests/test_agent_runtime_phase_a_evidence.py`
- Modify: `gemma_agent/runtime/__init__.py`
- Modify: `gemma_agent/supervisor.py`

- [ ] **Step 1: Write the failing evidence tests**

Create `tests/test_agent_runtime_phase_a_evidence.py`:

```python
import unittest

from gemma_agent.runtime.evidence import ClaimVerifier, EvidenceLedger
from gemma_agent.schemas import SubAgentResult, ToolResult


class EvidenceLedgerTests(unittest.TestCase):
    def test_records_executed_tool_results(self):
        ledger = EvidenceLedger()
        result = ToolResult("shell_command", {"command": "python -m unittest"}, ok=True, executed=True)

        ledger.record_tool_result(result)

        self.assertEqual(ledger.executed_tool_results(), [result])

    def test_includes_subagent_tool_results_as_execution_evidence(self):
        tool = ToolResult("shell_command", {"command": "python -m unittest"}, ok=True, executed=True)
        ledger = EvidenceLedger()
        ledger.record_subagent_result(SubAgentResult("a", "task", ok=True, tool_results=[tool]))

        self.assertEqual(ledger.executed_tool_results(), [tool])


class ClaimVerifierTests(unittest.TestCase):
    def test_rejects_test_pass_claim_without_evidence(self):
        error = ClaimVerifier().final_claim_error("Tests passed.", EvidenceLedger())

        self.assertEqual(error, "missing execution evidence: no executed ToolResult exists")

    def test_accepts_test_pass_claim_with_test_command_evidence(self):
        ledger = EvidenceLedger()
        ledger.record_tool_result(
            ToolResult(
                "shell_command",
                {"command": "python -m unittest discover -s tests -v"},
                ok=True,
                output={"status": "completed"},
                executed=True,
            )
        )

        error = ClaimVerifier().final_claim_error("Tests passed.", ledger)

        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run focused test and confirm it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_runtime_phase_a_evidence.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'gemma_agent.runtime.evidence'
```

- [ ] **Step 3: Implement ledger and claim verifier by moving existing proven logic**

Create `gemma_agent/runtime/evidence.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

from gemma_agent.schemas import SubAgentResult, ToolResult
from gemma_agent.supervisor import _final_claim_evidence_error


@dataclass
class EvidenceLedger:
    tool_results: list[ToolResult] = field(default_factory=list)
    subagent_results: list[SubAgentResult] = field(default_factory=list)

    def record_tool_result(self, result: ToolResult) -> None:
        self.tool_results.append(result)

    def record_subagent_result(self, result: SubAgentResult) -> None:
        self.subagent_results.append(result)

    def record_subagent_results(self, results: list[SubAgentResult]) -> None:
        for result in results:
            self.record_subagent_result(result)

    def executed_tool_results(self) -> list[ToolResult]:
        results = [result for result in self.tool_results if result.executed]
        for subagent in self.subagent_results:
            results.extend(result for result in subagent.tool_results if result.executed)
        return results


class ClaimVerifier:
    def final_claim_error(self, final_content: str, ledger: EvidenceLedger) -> str | None:
        return _final_claim_evidence_error(final_content, ledger.tool_results, ledger.subagent_results)
```

This imports the existing claim classifier as a temporary compatibility bridge. A later task moves the helper functions out of `supervisor.py` after the run loop no longer depends on supervisor internals.

- [ ] **Step 4: Export evidence primitives**

Modify `gemma_agent/runtime/__init__.py`:

```python
from .actions import ActionCodec, extract_response_text
from .budget import BudgetDecision, BudgetManager, BudgetSnapshot
from .evidence import ClaimVerifier, EvidenceLedger

__all__ = [
    "ActionCodec",
    "BudgetDecision",
    "BudgetManager",
    "BudgetSnapshot",
    "ClaimVerifier",
    "EvidenceLedger",
    "extract_response_text",
]
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_runtime_phase_a_evidence.py -v
```

Expected:

```text
Ran 4 tests in <seconds>s
OK
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add gemma_agent/runtime/evidence.py gemma_agent/runtime/__init__.py tests/test_agent_runtime_phase_a_evidence.py
git commit -m "feat: add runtime evidence ledger"
```

---

### Task 4: Prompt Assembler And Context Budgeter

**Files:**
- Create: `gemma_agent/runtime/context.py`
- Create: `tests/test_agent_runtime_phase_a_context.py`
- Modify: `gemma_agent/runtime/__init__.py`

- [ ] **Step 1: Write the failing context tests**

Create `tests/test_agent_runtime_phase_a_context.py`:

```python
import unittest

from gemma_agent.effort import TaskEffortPolicy
from gemma_agent.runtime.budget import BudgetManager
from gemma_agent.runtime.context import PromptAssembler
from gemma_agent.schemas import ToolResult


class FakeContextBuilder:
    def build(self, *, task, extra):
        return {"task": task, "extra": extra, "large": "x" * 1000}


class FakeRegistry:
    def list_tool_schemas(self):
        return [{"name": "shell_command", "schema": {"type": "object"}}]


class FakeToolExecutor:
    registry = FakeRegistry()


class PromptAssemblerTests(unittest.TestCase):
    def test_builds_payload_with_runtime_budget_snapshot(self):
        effort = TaskEffortPolicy().plan_for_task("fix and verify a bug")
        budget = BudgetManager(effort)
        budget.before_iteration()
        assembler = PromptAssembler(context_builder=FakeContextBuilder(), tool_executor=FakeToolExecutor())

        payload = assembler.build(
            task="fix and verify a bug",
            run_context={"cwd": "C:/repo"},
            tool_results=[],
            subagent_results=[],
            thinking_summaries=[],
            invalid_actions=[],
            progress_payload={"iteration": 1, "recovery_events": []},
            budget_manager=budget,
        )

        self.assertEqual(payload["input"][1]["content"], "fix and verify a bug")
        self.assertEqual(payload["budget"]["iterations_started"], 1)
        self.assertEqual(payload["tools"][0]["name"], "shell_command")

    def test_summarizes_tool_results_to_context_budget(self):
        effort = TaskEffortPolicy().plan_for_task("fix and verify a bug")
        budget = BudgetManager(effort)
        assembler = PromptAssembler(context_builder=FakeContextBuilder(), tool_executor=FakeToolExecutor())
        result = ToolResult("shell_command", {}, ok=True, output="x" * 10000, executed=True)

        payload = assembler.build(
            task="task",
            run_context={},
            tool_results=[result],
            subagent_results=[],
            thinking_summaries=[],
            invalid_actions=[],
            progress_payload={},
            budget_manager=budget,
            tool_evidence_budget_chars=500,
        )

        serialized = str(payload["tool_results"])
        self.assertLess(len(serialized), 1000)
        self.assertIn("omitted_due_to_context_budget", serialized)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run focused test and confirm it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_runtime_phase_a_context.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'gemma_agent.runtime.context'
```

- [ ] **Step 3: Implement prompt assembly using the existing payload shape**

Create `gemma_agent/runtime/context.py`:

```python
from __future__ import annotations

import json
from typing import Any

from gemma_agent.effort import estimate_text_tokens
from gemma_agent.schemas import InvalidAction, SubAgentResult, ThinkingSummary, ToolResult

ACTION_INSTRUCTIONS = (
    "Return exactly one strict JSON object per turn. "
    "Allowed actions: final, tool_call, spawn_subagents, thinking_summary. "
    "Tool execution is only real after the external runtime returns ToolResult evidence. "
    "Use context7_search as the primary tool for software, code, package, install, "
    "command, API, SDK, and framework documentation. Do not use duckduckgo_search for "
    "software, code, or coding tasks; DuckDuckGo is only for public news, people, and "
    "non-code public information."
)

MIN_TOOL_EVIDENCE_BUDGET_CHARS = 2000


class PromptAssembler:
    def __init__(self, *, context_builder: Any, tool_executor: Any) -> None:
        self.context_builder = context_builder
        self.tool_executor = tool_executor

    def build(
        self,
        *,
        task: str,
        run_context: dict[str, Any],
        tool_results: list[ToolResult],
        subagent_results: list[SubAgentResult],
        thinking_summaries: list[ThinkingSummary],
        invalid_actions: list[InvalidAction],
        progress_payload: dict[str, Any],
        budget_manager: Any,
        tool_evidence_budget_chars: int | None = None,
    ) -> dict[str, Any]:
        safety_guard = self.tool_executor.safety_guard if hasattr(self.tool_executor, "safety_guard") else None
        raw_tool_evidence = [result.to_evidence(safety_guard) for result in tool_results]
        budget_chars = (
            max(MIN_TOOL_EVIDENCE_BUDGET_CHARS, budget_manager.snapshot().max_context_chars)
            if tool_evidence_budget_chars is None
            else max(0, int(tool_evidence_budget_chars))
        )
        tool_evidence = fit_tool_evidence_to_budget(raw_tool_evidence, budget_chars)
        return {
            "input": [
                {"role": "system", "content": ACTION_INSTRUCTIONS},
                {"role": "user", "content": task},
            ],
            "context": self.context_builder.build(task=task, extra=run_context),
            "tools": self.tool_executor.registry.list_tool_schemas(),
            "tool_results": tool_evidence,
            "subagent_results": [result.to_evidence(safety_guard) for result in subagent_results],
            "thinking_summaries": [summary.format_lines() for summary in thinking_summaries],
            "invalid_actions": [{"raw_length": len(item.raw), "error": item.error} for item in invalid_actions],
            "progress": progress_payload,
            "budget": budget_manager.snapshot().to_payload(),
        }


def fit_tool_evidence_to_budget(items: list[dict[str, Any]], budget_chars: int) -> list[dict[str, Any]]:
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
        if used + _json_size(minimal) <= budget or not fitted:
            fitted.append(minimal)
            used += _json_size(minimal)
    return fitted


def estimate_payload_tokens(payload: dict[str, Any]) -> int:
    return estimate_text_tokens(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=repr))


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
        summary["output_summary"] = {
            "omitted_due_to_context_budget": True,
            "serialized_length": _json_size(output),
        }
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


def _json_size(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, ensure_ascii=True, default=repr))
```

- [ ] **Step 4: Export prompt assembler**

Modify `gemma_agent/runtime/__init__.py`:

```python
from .context import PromptAssembler, estimate_payload_tokens, fit_tool_evidence_to_budget
```

Add to `__all__`:

```python
"PromptAssembler",
"estimate_payload_tokens",
"fit_tool_evidence_to_budget",
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_runtime_phase_a_context.py -v
```

Expected:

```text
Ran 2 tests in <seconds>s
OK
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add gemma_agent/runtime/context.py gemma_agent/runtime/__init__.py tests/test_agent_runtime_phase_a_context.py
git commit -m "feat: add runtime prompt assembler"
```

---

### Task 5: Run-Aware Tool Runner

**Files:**
- Create: `gemma_agent/runtime/tools.py`
- Create: `tests/test_agent_runtime_phase_a_tools.py`
- Modify: `gemma_agent/runtime/__init__.py`

- [ ] **Step 1: Write the failing tool runner tests**

Create `tests/test_agent_runtime_phase_a_tools.py`:

```python
import unittest

from gemma_agent.runtime.budget import BudgetManager
from gemma_agent.runtime.evidence import EvidenceLedger
from gemma_agent.runtime.tools import RuntimeToolRunner
from gemma_agent.effort import TaskEffortBudget
from gemma_agent.schemas import ToolResult


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, tool_name, args):
        self.calls.append((tool_name, args))
        return ToolResult(tool_name, args, ok=True, output="ok", executed=True)


def one_tool_budget():
    return TaskEffortBudget(
        difficulty="hard",
        max_iterations=4,
        max_tool_calls=1,
        max_subagents=0,
        max_context_chars=32000,
        terminal_timeout_sec=1800,
        terminal_idle_timeout_sec=120,
        summarization_interval=2,
        repeated_tool_threshold=2,
        repeated_invalid_threshold=2,
    )


class RuntimeToolRunnerTests(unittest.TestCase):
    def test_executes_and_records_tool_result(self):
        executor = FakeExecutor()
        ledger = EvidenceLedger()
        runner = RuntimeToolRunner(executor, budget_manager=BudgetManager(one_tool_budget()), evidence=ledger)

        result = runner.execute("echo", {"text": "hello"})

        self.assertTrue(result.ok)
        self.assertEqual(executor.calls, [("echo", {"text": "hello"})])
        self.assertEqual(ledger.tool_results, [result])

    def test_returns_budget_error_without_calling_executor(self):
        executor = FakeExecutor()
        budget = BudgetManager(one_tool_budget())
        ledger = EvidenceLedger()
        runner = RuntimeToolRunner(executor, budget_manager=budget, evidence=ledger)
        runner.execute("echo", {"text": "first"})

        result = runner.execute("echo", {"text": "second"})

        self.assertFalse(result.ok)
        self.assertFalse(result.executed)
        self.assertEqual(result.error_code, "budget_exhausted")
        self.assertEqual(len(executor.calls), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run focused test and confirm it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_runtime_phase_a_tools.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'gemma_agent.runtime.tools'
```

- [ ] **Step 3: Implement run-aware tool execution**

Create `gemma_agent/runtime/tools.py`:

```python
from __future__ import annotations

from typing import Any

from gemma_agent.schemas import ToolResult


class RuntimeToolRunner:
    def __init__(self, tool_executor: Any, *, budget_manager: Any, evidence: Any) -> None:
        self.tool_executor = tool_executor
        self.budget_manager = budget_manager
        self.evidence = evidence

    def execute(self, tool_name: str, args: Any) -> ToolResult:
        decision = self.budget_manager.before_tool_call(tool_name)
        if not decision.allowed:
            result = ToolResult(
                tool_name=tool_name,
                args=args if isinstance(args, dict) else {},
                ok=False,
                error=decision.message,
                error_code="budget_exhausted",
                executed=False,
            )
            self.evidence.record_tool_result(result)
            return result
        result = self.tool_executor.execute(tool_name, args)
        self.budget_manager.record_tool_result(result)
        self.evidence.record_tool_result(result)
        return result
```

- [ ] **Step 4: Export tool runner**

Modify `gemma_agent/runtime/__init__.py`:

```python
from .tools import RuntimeToolRunner
```

Add to `__all__`:

```python
"RuntimeToolRunner",
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_runtime_phase_a_tools.py -v
```

Expected:

```text
Ran 2 tests in <seconds>s
OK
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add gemma_agent/runtime/tools.py gemma_agent/runtime/__init__.py tests/test_agent_runtime_phase_a_tools.py
git commit -m "feat: add run-aware tool runner"
```

---

### Task 6: Subagent Scheduler

**Files:**
- Create: `gemma_agent/runtime/subagents.py`
- Create: `tests/test_agent_runtime_phase_a_subagents.py`
- Modify: `gemma_agent/runtime/__init__.py`

- [ ] **Step 1: Write the failing subagent scheduler tests**

Create `tests/test_agent_runtime_phase_a_subagents.py`:

```python
import unittest

from gemma_agent.effort import TaskEffortBudget
from gemma_agent.runtime.budget import BudgetManager
from gemma_agent.runtime.evidence import EvidenceLedger
from gemma_agent.runtime.subagents import SubagentScheduler
from gemma_agent.schemas import SubAgentResult


def subagent_budget(max_subagents=2):
    return TaskEffortBudget(
        difficulty="hard",
        max_iterations=4,
        max_tool_calls=4,
        max_subagents=max_subagents,
        max_context_chars=32000,
        terminal_timeout_sec=1800,
        terminal_idle_timeout_sec=120,
        summarization_interval=2,
        repeated_tool_threshold=2,
        repeated_invalid_threshold=2,
    )


class StaticAgent:
    def __init__(self, agent_id, task):
        self.agent_id = agent_id
        self.task = task

    def run(self):
        return SubAgentResult(self.agent_id, self.task, final="done", ok=True)


class SubagentSchedulerTests(unittest.TestCase):
    def test_runs_and_records_subagents(self):
        ledger = EvidenceLedger()
        scheduler = SubagentScheduler(
            budget_manager=BudgetManager(subagent_budget()),
            evidence=ledger,
            agent_factory=lambda spec, context: StaticAgent(spec["id"], spec["task"]),
        )

        results = scheduler.spawn(
            [{"id": "a", "task": "inspect A"}, {"id": "b", "task": "inspect B"}],
            parent_task="main",
            run_context={"cwd": "C:/repo"},
        )

        self.assertEqual([result.agent_id for result in results], ["a", "b"])
        self.assertEqual(ledger.subagent_results, results)

    def test_rejects_requests_over_budget(self):
        scheduler = SubagentScheduler(
            budget_manager=BudgetManager(subagent_budget(max_subagents=1)),
            evidence=EvidenceLedger(),
            agent_factory=lambda spec, context: StaticAgent(spec["id"], spec["task"]),
        )

        results = scheduler.spawn(
            [{"id": "a", "task": "inspect A"}, {"id": "b", "task": "inspect B"}],
            parent_task="main",
            run_context={},
        )

        self.assertFalse(results[0].ok)
        self.assertEqual(results[0].agent_id, "invalid")
        self.assertIn("max_subagents_exhausted", results[0].error)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run focused test and confirm it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_runtime_phase_a_subagents.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'gemma_agent.runtime.subagents'
```

- [ ] **Step 3: Implement the scheduler**

Create `gemma_agent/runtime/subagents.py`:

```python
from __future__ import annotations

import copy
from typing import Any

from gemma_agent.schemas import SubAgentResult
from gemma_agent.subagent import run_subagents


class SubagentScheduler:
    def __init__(
        self,
        *,
        budget_manager: Any,
        evidence: Any,
        agent_factory: Any,
        concurrent: bool = True,
        max_workers: int | None = None,
    ) -> None:
        self.budget_manager = budget_manager
        self.evidence = evidence
        self.agent_factory = agent_factory
        self.concurrent = concurrent
        self.max_workers = max_workers

    def spawn(
        self,
        raw_tasks: list[Any],
        *,
        parent_task: str,
        run_context: dict[str, Any],
    ) -> list[SubAgentResult]:
        specs = normalize_subagent_specs(raw_tasks)
        decision = self.budget_manager.before_subagents(len(specs))
        if not decision.allowed:
            return [SubAgentResult("invalid", parent_task, ok=False, error=f"{decision.reason}: {decision.message}")]
        agents = [self.agent_factory(spec, copy.deepcopy(run_context)) for spec in specs]
        results = run_subagents(agents, concurrent=self.concurrent, max_workers=self.max_workers)
        self.budget_manager.record_subagents(len(specs))
        self.evidence.record_subagent_results(results)
        return results


def normalize_subagent_specs(raw_tasks: list[Any]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for index, item in enumerate(raw_tasks):
        if isinstance(item, str):
            specs.append({"id": f"subagent-{index + 1}", "task": item})
        elif isinstance(item, dict):
            specs.append({
                "id": str(item.get("id", f"subagent-{index + 1}")),
                "task": str(item.get("task", "")),
            })
        else:
            specs.append({"id": f"subagent-{index + 1}", "task": ""})
    return specs
```

- [ ] **Step 4: Export scheduler**

Modify `gemma_agent/runtime/__init__.py`:

```python
from .subagents import SubagentScheduler, normalize_subagent_specs
```

Add to `__all__`:

```python
"SubagentScheduler",
"normalize_subagent_specs",
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_runtime_phase_a_subagents.py -v
```

Expected:

```text
Ran 2 tests in <seconds>s
OK
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add gemma_agent/runtime/subagents.py gemma_agent/runtime/__init__.py tests/test_agent_runtime_phase_a_subagents.py
git commit -m "feat: add runtime subagent scheduler"
```

---

### Task 7: Agent Run Loop Adapter

**Files:**
- Create: `gemma_agent/runtime/run_loop.py`
- Create: `tests/test_agent_runtime_phase_a_run_loop.py`
- Modify: `gemma_agent/runtime/__init__.py`
- Modify: `gemma_agent/supervisor.py`

- [ ] **Step 1: Write the failing run loop tests**

Create `tests/test_agent_runtime_phase_a_run_loop.py`:

```python
import json
import unittest

from gemma_agent import AgentSupervisor, ToolExecutor, ToolRegistry
from gemma_agent.runtime.run_loop import AgentRunLoop


class FakeModelClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def create_response(self, payload):
        self.payloads.append(payload)
        if not self.responses:
            raise AssertionError("model called more times than expected")
        return self.responses.pop(0)


class AgentRunLoopTests(unittest.TestCase):
    def test_run_loop_returns_final_action(self):
        registry = ToolRegistry()
        model = FakeModelClient([json.dumps({"action": "final", "content": "done"})])
        loop = AgentRunLoop(model, ToolExecutor(registry), max_iterations=2)

        result = loop.run("say done")

        self.assertTrue(result.ok)
        self.assertEqual(result.final, "done")

    def test_supervisor_delegates_to_run_loop_compatibility_path(self):
        registry = ToolRegistry()
        model = FakeModelClient([json.dumps({"action": "final", "content": "done"})])

        result = AgentSupervisor(model, ToolExecutor(registry), max_iterations=2).run("say done")

        self.assertTrue(result.ok)
        self.assertEqual(result.final, "done")
        self.assertIn("budget", model.payloads[0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run focused test and confirm it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_runtime_phase_a_run_loop.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'gemma_agent.runtime.run_loop'
```

- [ ] **Step 3: Implement the new run loop**

Create `gemma_agent/runtime/run_loop.py`:

```python
from __future__ import annotations

import copy
from typing import Any

from gemma_agent.citations import CitationManager
from gemma_agent.context import ContextBuilder
from gemma_agent.effort import TaskEffortPolicy
from gemma_agent.memory import MemoryStore
from gemma_agent.progress import ProgressTracker
from gemma_agent.schemas import AgentRunResult, InvalidAction, ThinkingSummary, ToolResult

from .actions import ActionCodec, extract_response_text
from .budget import BudgetManager
from .context import PromptAssembler, estimate_payload_tokens
from .evidence import ClaimVerifier, EvidenceLedger
from .subagents import SubagentScheduler
from .tools import RuntimeToolRunner


class AgentRunLoop:
    def __init__(
        self,
        model_client: Any,
        tool_executor: Any,
        *,
        memory: MemoryStore | None = None,
        citations: CitationManager | None = None,
        context_builder: ContextBuilder | None = None,
        max_iterations: int = 8,
        max_subagents: int = 8,
        subagent_budget: Any | None = None,
        subagent_factory: Any | None = None,
        effort_policy: TaskEffortPolicy | None = None,
        action_codec: ActionCodec | None = None,
        claim_verifier: ClaimVerifier | None = None,
    ) -> None:
        self.model_client = model_client
        self.tool_executor = tool_executor
        self.memory = MemoryStore() if memory is None else memory
        self.citations = CitationManager() if citations is None else citations
        self.context_builder = (
            ContextBuilder(self.memory, self.citations, safety_guard=self.tool_executor.safety_guard)
            if context_builder is None
            else context_builder
        )
        self.max_iterations = max(0, int(max_iterations))
        self.max_subagents = max(0, int(max_subagents))
        self.subagent_budget = subagent_budget
        self.subagent_factory = subagent_factory
        self.effort_policy = TaskEffortPolicy() if effort_policy is None else effort_policy
        self.action_codec = ActionCodec() if action_codec is None else action_codec
        self.claim_verifier = ClaimVerifier() if claim_verifier is None else claim_verifier

    def run(self, task: str, *, context: dict[str, Any] | None = None) -> AgentRunResult:
        run_context = copy.deepcopy(context or {})
        tool_results: list[ToolResult] = []
        thinking_summaries: list[ThinkingSummary] = []
        invalid_actions: list[InvalidAction] = []
        evidence = EvidenceLedger()
        effort_budget = self.effort_policy.plan_for_task(task, context=run_context)
        effort_budget = _clamp_effort_budget_to_runtime_limit(effort_budget, self.max_iterations)
        budget_manager = BudgetManager(effort_budget)
        progress = ProgressTracker(
            repeated_tool_threshold=effort_budget.repeated_tool_threshold,
            repeated_invalid_threshold=effort_budget.repeated_invalid_threshold,
            safety_guard=self.tool_executor.safety_guard,
        )
        assembler = PromptAssembler(context_builder=self.context_builder, tool_executor=self.tool_executor)
        tool_runner = RuntimeToolRunner(self.tool_executor, budget_manager=budget_manager, evidence=evidence)

        while True:
            decision = budget_manager.before_iteration()
            if not decision.allowed:
                invalid_actions.append(InvalidAction(raw="", error=decision.message))
                break
            progress.start_iteration()
            payload = assembler.build(
                task=task,
                run_context=run_context,
                tool_results=tool_results,
                subagent_results=evidence.subagent_results,
                thinking_summaries=thinking_summaries,
                invalid_actions=invalid_actions,
                progress_payload=progress.to_payload(),
                budget_manager=budget_manager,
            )
            raw_response = self.model_client.create_response(payload)
            budget_manager.record_model_call(prompt_tokens=estimate_payload_tokens(payload))
            action, invalid = self.action_codec.parse(raw_response)
            if invalid is not None:
                invalid_actions.append(invalid)
                progress.record_invalid_action(invalid)
                continue

            action_name = action.get("action")
            if action_name == "final":
                final_content = str(action.get("content", ""))
                claim_error = self.claim_verifier.final_claim_error(final_content, evidence)
                if claim_error:
                    invalid = InvalidAction(
                        raw=extract_response_text(raw_response),
                        error=(
                            "final action claims file edits or command/test/tool execution, "
                            f"but {claim_error}"
                        ),
                    )
                    invalid_actions.append(invalid)
                    progress.record_invalid_action(invalid)
                    continue
                return AgentRunResult(
                    final=final_content,
                    ok=True,
                    tool_results=tool_results,
                    subagent_results=evidence.subagent_results,
                    thinking_summaries=thinking_summaries,
                    invalid_actions=invalid_actions,
                )
            if action_name == "tool_call":
                result = tool_runner.execute(str(action.get("tool", action.get("name"))), action.get("args", {}))
                tool_results.append(result)
                progress.record_tool_result(result)
                continue
            if action_name == "spawn_subagents":
                results = self._spawn_subagents(action, task, run_context, budget_manager, evidence)
                progress_events = [item for item in results if not item.ok]
                for item in progress_events:
                    invalid_actions.append(InvalidAction(raw=extract_response_text(raw_response), error=item.error or "subagent failed"))
                continue
            if action_name == "thinking_summary":
                summary = ThinkingSummary.from_mapping(action.get("summary", action))
                errors = summary.validate()
                if errors:
                    invalid = InvalidAction(raw=extract_response_text(raw_response), error="; ".join(errors))
                    invalid_actions.append(invalid)
                    progress.record_invalid_action(invalid)
                    continue
                thinking_summaries.append(summary)
                continue

        return AgentRunResult(
            final=None,
            ok=False,
            tool_results=tool_results,
            subagent_results=evidence.subagent_results,
            thinking_summaries=thinking_summaries,
            invalid_actions=invalid_actions,
        )

    def _spawn_subagents(
        self,
        action: dict[str, Any],
        parent_task: str,
        run_context: dict[str, Any],
        budget_manager: BudgetManager,
        evidence: EvidenceLedger,
    ):
        from gemma_agent.subagent import SubAgent

        def factory(spec: dict[str, str], child_context: dict[str, Any]):
            if self.subagent_factory is not None:
                return self.subagent_factory(
                    agent_id=spec["id"],
                    task=spec["task"],
                    model_client=self.model_client,
                    tool_executor=self.tool_executor,
                    context=child_context,
                    max_iterations=self.max_iterations,
                    max_subagents=self.max_subagents,
                    subagent_budget=self.subagent_budget,
                )
            child_context["parent_task"] = parent_task
            child_context["task"] = spec["task"]
            return SubAgent(
                agent_id=spec["id"],
                task=spec["task"],
                model_client=self.model_client,
                tool_executor=self.tool_executor,
                context=child_context,
                max_iterations=self.max_iterations,
                max_subagents=self.max_subagents,
                subagent_budget=self.subagent_budget,
            )

        scheduler = SubagentScheduler(
            budget_manager=budget_manager,
            evidence=evidence,
            agent_factory=factory,
            concurrent=True,
            max_workers=self.max_subagents,
        )
        return scheduler.spawn(action.get("tasks", []), parent_task=parent_task, run_context=run_context)


def _clamp_effort_budget_to_runtime_limit(budget, max_iterations: int):
    from dataclasses import replace

    effective_iterations = max(0, int(max_iterations))
    if budget.max_iterations <= effective_iterations:
        return budget
    return replace(budget, max_iterations=effective_iterations)
```

- [ ] **Step 4: Delegate `AgentSupervisor.run` to `AgentRunLoop`**

Modify `gemma_agent/supervisor.py` by replacing the body of `AgentSupervisor.run` with:

```python
    def run(self, task: str, *, context: dict[str, Any] | None = None) -> AgentRunResult:
        from .runtime.run_loop import AgentRunLoop

        return AgentRunLoop(
            self.model_client,
            self.tool_executor,
            memory=self.memory,
            citations=self.citations,
            context_builder=self.context_builder,
            max_iterations=self.max_iterations,
            max_subagents=self.max_subagents,
            subagent_budget=self.subagent_budget,
            subagent_factory=self.subagent_factory,
            effort_policy=self.effort_policy,
        ).run(task, context=context)
```

Do not delete old helper functions yet. `gemma_agent.runtime.evidence` still imports the existing claim helper as a compatibility bridge in this task.

- [ ] **Step 5: Export run loop**

Modify `gemma_agent/runtime/__init__.py`:

```python
from .run_loop import AgentRunLoop
```

Add to `__all__`:

```python
"AgentRunLoop",
```

- [ ] **Step 6: Run focused run-loop tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_agent_runtime_phase_a_run_loop.py -v
```

Expected:

```text
Ran 2 tests in <seconds>s
OK
```

- [ ] **Step 7: Run existing runtime compatibility tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_gemma_agent_runtime.py -v
```

Expected:

```text
Ran <count> tests in <seconds>s
OK
```

- [ ] **Step 8: Commit**

Run:

```powershell
git add gemma_agent/runtime/run_loop.py gemma_agent/runtime/__init__.py gemma_agent/supervisor.py tests/test_agent_runtime_phase_a_run_loop.py
git commit -m "feat: delegate supervisor to runtime run loop"
```

---

### Task 8: Phase A Verification And Report Hook

**Files:**
- Modify: `plan.md`
- Create: `docs/superpowers/reports/2026-06-14-gemma-runtime-phase-a-report.md`

- [ ] **Step 1: Run all new Phase A tests together**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_agent_runtime_phase_a_*.py" -v
```

Expected:

```text
Ran <count> tests in <seconds>s
OK
```

- [ ] **Step 2: Run affected existing runtime tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_gemma_agent_runtime.py -v
```

Expected:

```text
Ran <count> tests in <seconds>s
OK
```

- [ ] **Step 3: Run full unit suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected:

```text
Ran <count> tests in <seconds>s
OK
```

- [ ] **Step 4: Run current local behavior benchmark**

Run:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.runner --suite local-agent-behavior --run-id runtime-phase-a-local-behavior --timeout 20
```

Expected:

```text
status=passed
```

- [ ] **Step 5: Verify exactly one llama server if the local runtime is active**

Run:

```powershell
Get-Process llama-server -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,Path
```

Expected:

```text
<zero rows if runtime is stopped, or one row if runtime is active>
```

If more than one row appears, stop and inspect before continuing.

- [ ] **Step 6: Write the Phase A report**

Create `docs/superpowers/reports/2026-06-14-gemma-runtime-phase-a-report.md`:

```markdown
# Gemma Runtime Phase A Report

## Executive Summary

Phase A replaced the runtime core shape behind the existing `AgentSupervisor` public API.

## Design Changes

- Added `gemma_agent.runtime.budget`.
- Added `gemma_agent.runtime.actions`.
- Added `gemma_agent.runtime.evidence`.
- Added `gemma_agent.runtime.context`.
- Added `gemma_agent.runtime.tools`.
- Added `gemma_agent.runtime.subagents`.
- Added `gemma_agent.runtime.run_loop`.
- Routed `AgentSupervisor.run` through `AgentRunLoop`.

## Tests Run

- `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_agent_runtime_phase_a_*.py" -v`
- `.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_gemma_agent_runtime.py -v`
- `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

## Benchmarks Run

- `.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.runner --suite local-agent-behavior --run-id runtime-phase-a-local-behavior --timeout 20`

## Remaining Risks

- Phase B launch/proxy command behavior is not yet rebuilt.
- Phase C workspace tidiness and release evidence cleanup is not yet complete.
- Live repo-fix benchmark reliability remains unproven until the later benchmark gate runs.

## Verdict

Partial pass for the overall controlled big-bang rebuild. Phase A passes only if all commands above passed on fresh output.
```

- [ ] **Step 7: Update `plan.md` to point to Phase B next**

Patch `plan.md` active plan section:

```markdown
- Design: `docs/superpowers/specs/2026-06-14-gemma-controlled-big-bang-rebuild-design.md`
- Completed Phase A plan: `docs/superpowers/plans/2026-06-14-gemma-controlled-big-bang-runtime-phase-a.md`
- Next implementation plan: Phase B launch/proxy/Codex command layer
```

- [ ] **Step 8: Commit verification report and controller update**

Run:

```powershell
git add plan.md docs/superpowers/reports/2026-06-14-gemma-runtime-phase-a-report.md
git commit -m "docs: report runtime phase a verification"
```

---

## Plan Self-Review

- Spec coverage: Phase A covers the runtime-core section of the approved controlled big-bang spec. Phase B and C intentionally remain separate plans because they are independent subsystems.
- Red-flag scan: no incomplete requirement markers are intended in this plan.
- Type consistency: `BudgetManager`, `ActionCodec`, `EvidenceLedger`, `ClaimVerifier`, `PromptAssembler`, `RuntimeToolRunner`, `SubagentScheduler`, and `AgentRunLoop` names are consistent across tests, implementation snippets, and exports.
- Verification: each production behavior task starts with a focused failing test, then implementation, focused pass, and commit. The final gate runs new tests, affected runtime tests, the full suite, the local behavior benchmark, and the llama-server process check.
