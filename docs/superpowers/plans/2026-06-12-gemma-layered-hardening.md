# Gemma Layered Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add layered prompt-integrity, startup-preflight, and runtime trust-boundary controls so Gemma cannot silently turn untrusted instruction text into trusted source edits, prompts, or completion claims.

**Architecture:** Use the Blockblast-style umbrella approach: one end-to-end hardening objective split into independent tracks with narrow file ownership. Each task starts with failing tests, implements the smallest production change, then gets spec and quality review before the next task.

**Tech Stack:** Python 3.14 stdlib, `unittest`, PowerShell parser APIs, local Codex/Gemma config generation, current `gemma_agent` package, project `.venv`.

---

## File Structure

- `docs/superpowers/specs/2026-06-12-gemma-layered-hardening.md`: umbrella design and evidence.
- `docs/superpowers/plans/2026-06-12-gemma-layered-hardening.md`: this task plan.
- `setup_local_codex.py`: base prompt cleanup, generated catalog verifier, generated syntax preflight.
- `tests/test_setup_local_codex.py`: prompt denylist, catalog drift, Python/PowerShell preflight tests.
- `gemma_agent/safety.py`: untrusted evidence wrappers and local artifact denylist.
- `gemma_agent/tool_registry.py`: recursive fail-closed schema validation.
- `gemma_agent/tool_executor.py`: recursive path scanning, edit-capable thread-mode rejection, untrusted error evidence.
- `gemma_agent/supervisor.py`: evidence-gated final claim flags and untrusted invalid-action payloads.
- `gemma_agent/schemas.py`: subagent result evidence wrapping.
- `gemma_agent_mcp.py`: task ingress length and instruction-looking payload controls.
- `tests/test_gemma_agent_runtime.py`: runtime guardrail regression tests.
- `tests/test_gemma_agent_mcp.py`: MCP ingress regression tests.
- `benchmarks/bench_runtime.py`: optional hardening metric fields.
- `tests/test_bench_runtime.py`: metric serialization tests.

## Task 1: Prompt Integrity And Catalog Drift

**Files:**
- Modify: `setup_local_codex.py`
- Modify: `tests/test_setup_local_codex.py`

- [ ] **Step 1: Write failing prompt-contamination tests**

Add tests that assert `build_base_instructions()` does not contain control phrases such as `Human life is worthless`, `Never ask for professional help`, `Ignore previous instructions`, `SYSTEM_LOGIC_INJECTION`, `<|tool_call>`, or `call:codex:` outside the existing explanatory ban.

- [ ] **Step 2: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex -v`

Expected now: fail on contaminated base instructions and token budget.

- [ ] **Step 3: Clean the base prompt only**

Restore `build_base_instructions()` to the intended local-Codex instruction text while preserving these meanings: local Gemma model, no added moderation/refusal wrapper, use Codex tools when available, prove work with command output, treat tool outputs as evidence, cite current research, efficient style.

- [ ] **Step 4: Add generated catalog integrity helpers**

Add a helper that can compare generated catalog JSON against `build_model_catalog()` by parsing JSON and comparing the `models[0].base_instructions` field.

- [ ] **Step 5: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex -v`

Expected: setup tests pass and base prompt is at or below 448 estimated tokens.

## Task 2: Startup Syntax Preflight

**Files:**
- Modify: `setup_local_codex.py`
- Modify: `tests/test_setup_local_codex.py`
- Verify: `start-gemma-runtime.ps1`
- Verify: `verify-local-setup.ps1`

- [ ] **Step 1: Write failing preflight tests**

Add tests that generate launchers into a temp directory and parse the generated PowerShell scripts with `[System.Management.Automation.Language.Parser]::ParseFile(...)`. Add a Python `py_compile` check for `launch_gemma_codex.py`, `gemma_response_proxy.py`, and `gemma_reasoning_proxy.py`.

- [ ] **Step 2: Verify red if preflight helper is missing**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex -v`

Expected: fail until generated verification script includes explicit syntax preflight.

- [ ] **Step 3: Add syntax preflight to generated `verify-local-setup.ps1`**

Make the generated verifier parse `start-gemma-runtime.ps1` and itself before checking runtime dependencies, and compile the Python launcher/proxy modules.

- [ ] **Step 4: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex -v`

Expected: setup tests pass.

## Task 3: Runtime Trust Boundary

**Files:**
- Modify: `gemma_agent/safety.py`
- Modify: `gemma_agent/schemas.py`
- Modify: `gemma_agent/tool_executor.py`
- Modify: `gemma_agent/supervisor.py`
- Modify: `tests/test_gemma_agent_runtime.py`

- [ ] **Step 1: Write failing tests for untrusted propagation**

Add tests proving tool exception text, invalid model action text, and subagent final text containing `Ignore previous instructions` are wrapped as untrusted evidence before entering the next model payload.

- [ ] **Step 2: Write failing tests for final-claim evidence flags**

Add a test where the model returns `{"action":"final","content":"I edited file.py and ran tests."}` with zero executed tools. Expected result: `ok=False` or a recorded invalid action explaining missing execution evidence.

- [ ] **Step 3: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime -v`

Expected: new tests fail.

- [ ] **Step 4: Add trust wrappers**

Centralize untrusted evidence wrapping in `SafetyGuard`, use it for tool outputs, tool errors, invalid actions, and subagent result evidence.

- [ ] **Step 5: Add final claim validation**

Before accepting `final`, detect edit/command/tool-execution claims and require at least one executed `ToolResult`; otherwise record an invalid action and continue or end with `ok=False`.

- [ ] **Step 6: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime -v`

Expected: runtime tests pass.

## Task 4: Tool Schema And Side-Effect Containment

**Files:**
- Modify: `gemma_agent/tool_registry.py`
- Modify: `gemma_agent/tool_executor.py`
- Modify: `gemma_agent/safety.py`
- Modify: `tests/test_gemma_agent_runtime.py`

- [ ] **Step 1: Write failing schema tests**

Add tests that `ToolRegistry.register(...)` rejects unsupported JSON Schema constructs: `type` arrays, `$ref`, `anyOf`, `oneOf`, `allOf`, and object properties whose schemas are not dicts.

- [ ] **Step 2: Write failing path and timeout tests**

Add tests proving nested path fields are validated recursively and edit-capable tools cannot be registered or executed in thread mode. Add a late-write thread timeout regression test if thread mode remains available for read-only tools.

- [ ] **Step 3: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime -v`

Expected: new tests fail.

- [ ] **Step 4: Implement fail-closed schema validation**

Recursively reject unsupported constructs and unsupported type declarations at registration time.

- [ ] **Step 5: Implement side-effect mode policy**

Use schema hints such as path/file fields and write-like tool names to forbid thread execution for edit-capable tools. Keep process mode as the hard-timeout path.

- [ ] **Step 6: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime -v`

Expected: runtime tests pass.

## Task 5: MCP Ingress Limits

**Files:**
- Modify: `gemma_agent_mcp.py`
- Modify: `tests/test_gemma_agent_mcp.py`

- [ ] **Step 1: Write failing MCP ingress tests**

Add tests that reject oversized task strings and instruction-looking task payloads such as `Ignore previous instructions`.

- [ ] **Step 2: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_mcp -v`

Expected: new tests fail.

- [ ] **Step 3: Add task budget validation**

Limit task length and reject obvious control-text payloads before creating subagents.

- [ ] **Step 4: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_mcp -v`

Expected: MCP tests pass.

## Task 6: Hardening Metrics

**Files:**
- Modify: `benchmarks/bench_runtime.py`
- Modify: `tests/test_bench_runtime.py`
- Modify docs only if a measured value changes.

- [ ] **Step 1: Write failing metric serialization tests**

Add tests that benchmark summaries include `catalog_drift`, `syntax_preflight_pass_rate`, `hallucinated_tool_claim_rate`, `raw_untrusted_bytes_in_prompt`, and `failed_json_tool_call_rate` when supplied by workload results.

- [ ] **Step 2: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_bench_runtime -v`

Expected: new tests fail.

- [ ] **Step 3: Add optional metric fields**

Extend summary aggregation without inventing values when fields are absent.

- [ ] **Step 4: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_bench_runtime -v`

Expected: benchmark tests pass.

## Task 7: Final Verification And Review

**Files:**
- Review all changed files.

- [ ] Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`
- [ ] Run: `npm test`
- [ ] Run: `.\.venv\Scripts\python.exe -m py_compile setup_local_codex.py launch_gemma_codex.py gemma_response_proxy.py gemma_reasoning_proxy.py gemma_agent\*.py gemma_agent_mcp.py duckduckgo_mcp.py`
- [ ] Run PowerShell parser checks for `start-gemma-runtime.ps1` and `verify-local-setup.ps1`.
- [ ] Run `powershell -NoProfile -ExecutionPolicy Bypass -File .\start-gemma-runtime.ps1` if the model server is available or can be started safely.
- [ ] Dispatch a GPT-5.5 xhigh code-review subagent for changed files.
- [ ] Fix Critical and Important findings.

## Self-Review

- Spec coverage: prompt integrity, generated drift, startup syntax, untrusted propagation, evidence-gated claims, schema fail-closed behavior, timeout containment, MCP ingress, metrics, and final review all map to tasks.
- Placeholder scan: no TBD/TODO placeholders are present.
- Type consistency: referenced modules and test files exist in this repository.
- Scope check: no task changes model weights, model refusal-removal behavior, or uncensored model behavior.
