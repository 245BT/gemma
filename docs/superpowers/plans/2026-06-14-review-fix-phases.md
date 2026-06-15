# Review Fix Phases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task in the reviewed workspace. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the six review findings without widening behavior beyond the reviewed surfaces.

**Architecture:** Keep model-output marker cleaning separate from proxy-controlled sanitization. Reuse existing benchmark label safety, route only real tool requests through raw forwarding, make idle timeout win over backgrounding at equal thresholds, recognize shell-wrapped test commands as execution evidence, and clamp advertised effort budgets to the runtime loop.

**Tech Stack:** Python standard library, unittest, local Gemma agent/proxy modules.

---

## File Map

- Modify `gemma_response_proxy.py`: separate marker stripping from proxy/error/tool text sanitization.
- Modify `tests/test_gemma_response_proxy.py`: replace model-output sanitizer expectations with preservation tests and keep scoped proxy/error/tool sanitization tests.
- Modify `benchmarks/agent_benchmarks/local_repo_fix.py`: use a safe run-id label for filesystem path construction.
- Modify `tests/test_agent_benchmarks.py`: keep the run-id traversal regression as the completion gate.
- Modify `gemma_reasoning_proxy.py`: make raw-forward routing depend on non-empty tools or a tool choice that actually selects tools.
- Modify `tests/test_gemma_reasoning_proxy.py`: add direct helper tests for no-tool `tool_choice` cases.
- Modify `gemma_agent/terminal.py`: check idle timeout before backgrounding.
- Modify `tests/test_gemma_agent_runtime.py`: update the equal-threshold terminal test and keep shell-wrapped evidence and effort-budget tests.
- Modify `gemma_agent/supervisor.py`: recognize shell-wrapped test commands and align the model-facing effort budget with the real loop limit.

## Phase 1: Baseline And Scope

- [ ] **Step 1: Run focused tests before edits**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_response_proxy tests.test_agent_benchmarks tests.test_gemma_reasoning_proxy tests.test_gemma_agent_runtime -v
```

Expected: at least the reviewed regressions fail against current code.

- [ ] **Step 2: Record exact failing tests**

Record the failing test names in the working notes before implementation. Do not change unrelated failing tests unless they block the reviewed fixes.

## Phase 2: Proxy Preservation Tests

- [ ] **Step 1: Replace ordinary model-output sanitizer expectation**

In `tests/test_gemma_response_proxy.py`, replace `test_clean_response_payload_sanitizes_output_behavior_rules` with:

```python
def test_clean_response_payload_preserves_ordinary_model_text_without_markers(self):
    payload = {
        "output_text": (
            "Ship it \\U0001f680 \\u26a0\\ufe0f It is 100% achieveable and production-ready. "
            "warning terms may appear as model content."
        ),
        "output": [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": "This is 100% achievable and production-ready \\u2705",
                    }
                ]
            }
        ],
    }

    cleaned = gemma_response_proxy.clean_response_payload(payload)

    self.assertEqual(cleaned["output_text"], payload["output_text"])
    self.assertEqual(cleaned["output"][0]["content"][0]["text"], payload["output"][0]["content"][0]["text"])
```

- [ ] **Step 2: Replace streamed model-output sanitizer expectation**

In `tests/test_gemma_response_proxy.py`, replace `test_clean_sse_payload_sanitizes_streamed_behavior_rules` with:

```python
def test_clean_sse_payload_preserves_ordinary_streamed_model_text_without_markers(self):
    raw = (
        b'data: {"type":"response.output_text.delta","delta":"Deploy \\\\ud83d\\\\udea8 100% achievable"}\n\n'
        b'data: {"type":"response.output_text.delta","delta":" and production-ready warning"}\n\n'
        b"data: [DONE]\n\n"
    )

    cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

    self.assertIn("\\\\ud83d\\\\udea8", cleaned)
    self.assertIn("100% achievable", cleaned)
    self.assertIn("production-ready", cleaned)
    self.assertIn("warning", cleaned)
```

- [ ] **Step 3: Replace final-marker sanitizer expectation**

In `tests/test_gemma_response_proxy.py`, replace `test_clean_sse_payload_sanitizes_streamed_final_marker_text` with:

```python
def test_clean_sse_payload_preserves_streamed_final_model_text_after_marker_strip(self):
    raw = (
        b'data: {"type":"response.output_text.delta","delta":"<|channel>final<channel|>warning \\\\u2705 100% achievable production-ready"}\n\n'
        b"data: [DONE]\n\n"
    )

    cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

    self.assertNotIn("<|channel>final", cleaned)
    self.assertIn("\\\\u2705", cleaned)
    self.assertIn("100% achievable", cleaned)
    self.assertIn("production-ready", cleaned)
```

- [ ] **Step 4: Run proxy tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_response_proxy -v
```

Expected: preservation tests fail because current marker cleaning still rewrites normal model text.

## Phase 3: Scoped Proxy Sanitizer Implementation

- [ ] **Step 1: Stop sanitizing in marker-strip helpers**

In `gemma_response_proxy.py`, make `clean_channel_markers()` and `StreamChannelCleaner.clean()` return only channel/tool-marker-stripped text. Use:

```python
return clean_internal_tool_call_markers(final_text)
```

and:

```python
return clean_internal_tool_call_markers(text)
```

- [ ] **Step 2: Add scoped key detection**

Add helpers:

```python
PROXY_SANITIZED_KEY_FRAGMENTS = ("error", "tool", "proxy", "details")
PROXY_MODEL_TEXT_KEYS = {"output_text", "text", "delta", "content"}

def should_sanitize_proxy_metadata(path):
    normalized = [str(item).lower().replace("-", "_") for item in path]
    if not normalized:
        return False
    if normalized[-1] in PROXY_MODEL_TEXT_KEYS and not any(
        item in {"error", "tool_output", "proxy_message"} for item in normalized[:-1]
    ):
        return False
    return any(
        fragment in item
        for item in normalized
        for fragment in PROXY_SANITIZED_KEY_FRAGMENTS
    )
```

- [ ] **Step 3: Pass paths through recursive cleaners**

Change `clean_json_strings(value)` to accept a `path=()` tuple and call `sanitize_behavior_text()` only when `should_sanitize_proxy_metadata(path)` is true. Change `clean_stream_json_strings()` similarly, but leave `STREAM_TEXT_KEYS` routed only through the streaming marker cleaner.

- [ ] **Step 4: Sanitize generated proxy error payloads**

Change `build_proxy_error_payload(message)` so the returned error message is `sanitize_behavior_text(message)`.

- [ ] **Step 5: Run proxy tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_response_proxy -v
```

Expected: pass.

## Phase 4: Benchmark Run-Id Path Safety

- [ ] **Step 1: Confirm existing regression failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agent_benchmarks.TestAgentBenchmarks.test_local_repo_fix_sanitizes_run_id_before_creating_artifact_paths -v
```

Expected: fail because `run_id` can affect paths outside `out_dir`.

- [ ] **Step 2: Implement safe label locally**

In `benchmarks/agent_benchmarks/local_repo_fix.py`, add:

```python
def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-")
    return label or "local-repo-fix"
```

and import `re`.

- [ ] **Step 3: Use the safe label for path names**

In `run_local_repo_fix_benchmark()`, set:

```python
safe_run_id = _safe_label(run_id)
artifact_root = output_root / f"{safe_run_id}.local-repo-fix-artifacts"
summary_path = output_root / f"{safe_run_id}.local-repo-fix.summary.json"
```

Keep `summary["run_id"] = run_id` so reporting preserves the requested id.

- [ ] **Step 4: Run benchmark tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agent_benchmarks -v
```

Expected: pass.

## Phase 5: Reasoning Proxy Tool Routing Tests

- [ ] **Step 1: Add helper tests**

In `tests/test_gemma_reasoning_proxy.py`, add:

```python
def test_should_raw_forward_tool_request_ignores_no_tool_choices(self):
    cases = [
        {"input": "Reply with OK.", "tool_choice": "none"},
        {"input": "Reply with OK.", "tool_choice": "auto"},
        {"input": "Reply with OK.", "parallel_tool_calls": True},
        {"input": "Reply with OK.", "tools": []},
    ]
    for payload in cases:
        with self.subTest(payload=payload):
            self.assertFalse(gemma_reasoning_proxy.should_raw_forward_tool_request(payload))

def test_should_raw_forward_tool_request_accepts_real_tool_requests(self):
    cases = [
        {"tools": [{"type": "function", "name": "noop"}]},
        {"tool_choice": "required"},
        {"tool_choice": {"type": "function", "name": "noop"}},
    ]
    for payload in cases:
        with self.subTest(payload=payload):
            self.assertTrue(gemma_reasoning_proxy.should_raw_forward_tool_request(payload))
```

- [ ] **Step 2: Run reasoning proxy tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_reasoning_proxy -v
```

Expected: fail for no-tool `tool_choice` cases.

## Phase 6: Reasoning Proxy Routing Implementation

- [ ] **Step 1: Add tool-choice classifier**

In `gemma_reasoning_proxy.py`, add:

```python
def _tool_choice_selects_tool(tool_choice):
    if tool_choice in {None, "", "none", "auto"}:
        return False
    return True
```

For dict/list values, return true before the set membership check.

- [ ] **Step 2: Update raw-forward condition**

Use:

```python
tools = payload.get("tools")
if isinstance(tools, list) and len(tools) > 0:
    return True
if tools and not isinstance(tools, list):
    return True
return _tool_choice_selects_tool(payload.get("tool_choice"))
```

Do not raw-forward only because `parallel_tool_calls` is true when no tools are provided.

- [ ] **Step 3: Run reasoning proxy tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_reasoning_proxy -v
```

Expected: pass.

## Phase 7: Terminal Timeout Ordering

- [ ] **Step 1: Update equal-threshold regression expectation**

In `tests/test_gemma_agent_runtime.py`, rename `test_terminal_runner_background_threshold_wins_idle_tie` to `test_terminal_runner_idle_timeout_wins_background_tie` and assert:

```python
self.assertFalse(result.ok)
self.assertEqual(result.status, "stalled")
self.assertIsNotNone(result.job_id)
self.assertEqual(result.recovery["reason"], "idle_timeout")
```

Keep the cleanup block that kills any returned job id.

- [ ] **Step 2: Run the focused terminal test and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime.TestGemmaAgentRuntime.test_terminal_runner_idle_timeout_wins_background_tie -v
```

Expected: fail because current code returns `running`.

- [ ] **Step 3: Move idle check before background check**

In `gemma_agent/terminal.py`, compute `idle_for` after hard timeout and before backgrounding, then check `idle_for >= idle_timeout` before `now - started >= background_after`.

- [ ] **Step 4: Run terminal-focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime.TestGemmaAgentRuntime.test_terminal_runner_reports_idle_stall_without_killing_command tests.test_gemma_agent_runtime.TestGemmaAgentRuntime.test_terminal_runner_idle_timeout_wins_background_tie tests.test_gemma_agent_runtime.TestGemmaAgentRuntime.test_terminal_runner_background_job_can_be_polled_to_completion -v
```

Expected: pass.

## Phase 8: Shell-Wrapped Test Evidence

- [ ] **Step 1: Confirm existing shell-wrapper test failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime.TestGemmaAgentRuntime.test_supervisor_accepts_shell_wrapped_test_runner_as_test_evidence -v
```

Expected: fail because only `argv[0]` is inspected.

- [ ] **Step 2: Add shell-wrapper command detection**

In `gemma_agent/supervisor.py`, add:

```python
_SHELL_WRAPPER_COMMANDS = {"cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe", "bash", "bash.exe", "sh", "sh.exe"}

def _argv_shell_wraps_test_runner(argv: list[str]) -> bool:
    if not argv:
        return False
    if _command_basename(argv[0]) not in _SHELL_WRAPPER_COMMANDS:
        return False
    command_text = " ".join(argv[1:])
    return bool(_TEST_COMMAND_RE.search(command_text))
```

- [ ] **Step 3: Use shell-wrapper detection**

Update `_command_args_start_with_test_runner()`:

```python
if _argv_starts_with_test_runner(argv) or _argv_shell_wraps_test_runner(argv):
    return True
```

- [ ] **Step 4: Run evidence tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime.TestGemmaAgentRuntime.test_supervisor_accepts_shell_wrapped_test_runner_as_test_evidence tests.test_gemma_agent_runtime.TestGemmaAgentRuntime.test_supervisor_rejects_test_claim_from_unrelated_tool_command_arg -v
```

Expected: pass.

## Phase 9: Effort Budget Alignment

- [ ] **Step 1: Add clamping test**

In `tests/test_gemma_agent_runtime.py`, after `test_supervisor_includes_effort_policy_in_model_payload`, add:

```python
def test_supervisor_effort_payload_clamps_max_iterations_to_runtime_limit(self):
    model = FakeModelClient([json.dumps({"action": "final", "content": "done"})])

    result = AgentSupervisor(
        model,
        ToolExecutor(ToolRegistry()),
        max_iterations=1,
    ).run("Research and debug a stalled terminal command, then benchmark the fix.")

    self.assertEqual(result.final, "done")
    self.assertEqual(model.payloads[0]["effort"]["difficulty"], "hard")
    self.assertEqual(model.payloads[0]["effort"]["max_iterations"], 1)
```

- [ ] **Step 2: Run effort test and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime.TestGemmaAgentRuntime.test_supervisor_effort_payload_clamps_max_iterations_to_runtime_limit -v
```

Expected: fail because hard tasks advertise more iterations than the loop allows.

- [ ] **Step 3: Add budget clamping helper**

In `gemma_agent/supervisor.py`, add:

```python
def _clamp_effort_budget_to_runtime_limit(budget: TaskEffortBudget, max_iterations: int) -> TaskEffortBudget:
    effective_iterations = max(0, int(max_iterations))
    if budget.max_iterations <= effective_iterations:
        return budget
    return dataclasses.replace(budget, max_iterations=effective_iterations)
```

Import `dataclasses`.

- [ ] **Step 4: Clamp before progress and payload building**

In `AgentSupervisor.run()`, set:

```python
effort_budget = _clamp_effort_budget_to_runtime_limit(
    self.effort_policy.plan_for_task(task, context=run_context),
    self.max_iterations,
)
```

- [ ] **Step 5: Run effort tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime.TestGemmaAgentRuntime.test_supervisor_includes_effort_policy_in_model_payload tests.test_gemma_agent_runtime.TestGemmaAgentRuntime.test_supervisor_effort_payload_clamps_max_iterations_to_runtime_limit -v
```

Expected: pass.

## Phase 10: Two-Pass Verification

- [ ] **Step 1: Run focused suite once**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_response_proxy tests.test_agent_benchmarks tests.test_gemma_reasoning_proxy tests.test_gemma_agent_runtime -v
```

Expected: pass.

- [ ] **Step 2: Run focused suite again**

Run the same command a second time. Expected: pass again.

- [ ] **Step 3: Run package tests**

Run:

```powershell
npm test
```

Expected: pass.

- [ ] **Step 4: Inspect diff**

Run:

```powershell
git diff -- gemma_response_proxy.py tests/test_gemma_response_proxy.py benchmarks/agent_benchmarks/local_repo_fix.py tests/test_agent_benchmarks.py gemma_reasoning_proxy.py tests/test_gemma_reasoning_proxy.py gemma_agent/terminal.py gemma_agent/supervisor.py tests/test_gemma_agent_runtime.py docs/superpowers/plans/2026-06-14-review-fix-phases.md
```

Expected: only scoped review fixes and plan file changes.
