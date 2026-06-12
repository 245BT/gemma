# Full Pass Gemma Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining gaps from the Gemma audit so the local agent can honestly pass on speed, subagents, DDG verification, metrics, and runtime hygiene.

**Architecture:** Keep the existing proxy-based runtime, but make reasoning adaptive: simple exact-output requests use a one-call local reasoning path, while complex requests keep the DSPy/LangGraph path. Expose Gemma-owned subagent execution through a local MCP tool wired into generated Codex configs. Make DDG private-route verification explicit and fail closed unless an actual private SOCKS route is configured.

**Tech Stack:** Python stdlib HTTP servers, MCP FastMCP, existing `gemma_agent` package, unittest, PowerShell launchers.

---

### Task 1: Adaptive Reasoning Path

**Files:**
- Modify: `gemma_reasoning/graph.py`
- Modify: `gemma_reasoning/dspy_programs.py`
- Modify: `gemma_reasoning_proxy.py`
- Test: `tests/test_gemma_reasoning.py`
- Test: `tests/test_gemma_reasoning_proxy.py`

- [ ] Write failing tests proving a simple forced reasoning request can run with local planner/verifier and one upstream model call.
- [ ] Run targeted tests and confirm failure.
- [ ] Add local heuristic programs and a `ReasoningConfig(use_langgraph=False)` path for simple requests.
- [ ] Make `gemma_reasoning_proxy.run_reasoning_request()` select the lightweight path only for simple exact-output requests, never for complex tool/research/code prompts.
- [ ] Run targeted tests and benchmark `audit-fullpass-reasoning-forced-marker-8082`.

### Task 2: Gemma Runtime Subagent MCP

**Files:**
- Create: `gemma_agent_mcp.py`
- Modify: `setup_local_codex.py`
- Test: `tests/test_gemma_agent_mcp.py`
- Test: `tests/test_setup_local_codex.py`

- [ ] Write failing tests for a `gemma_run_subagents` MCP helper that accepts 1-8 task strings, runs them through `AgentSupervisor`, and returns structured JSON evidence.
- [ ] Run targeted tests and confirm failure.
- [ ] Implement the MCP server using `LocalResponsesClient`, `ToolRegistry`, `ToolExecutor`, `AgentSupervisor`, and `run_subagents`.
- [ ] Add the MCP server to both generated local Codex homes.
- [ ] Run setup generation and targeted tests.

### Task 3: DDG Private Route Verification

**Files:**
- Modify: `duckduckgo_mcp.py`
- Test: `tests/test_duckduckgo_mcp.py`

- [ ] Write failing tests for private-route metadata: public HTTPS remains `private_network=false`; SOCKS proxy configuration marks the route private only when verification succeeds.
- [ ] Run targeted tests and confirm failure.
- [ ] Add optional environment-driven SOCKS proxy metadata and a verifier function that reports `private_network_verified`.
- [ ] Keep search behavior fail-closed for claims: never report private route unless verified.
- [ ] Run DDG five-probe check and report whether the environment satisfies private route.

### Task 4: Runtime Process Hygiene And Metrics

**Files:**
- Modify: `setup_local_codex.py`
- Modify: `benchmarks/bench_runtime.py`
- Test: `tests/test_setup_local_codex.py`
- Test: `tests/test_bench_runtime.py`

- [ ] Write failing tests for launcher process cleanup by port ownership and explicit `test_pass_rate` behavior.
- [ ] Run targeted tests and confirm failure.
- [ ] Update generated PowerShell launcher to stop stale non-owning proxy parent processes before starting proxies.
- [ ] Keep benchmark metrics explicit and non-inferred.
- [ ] Run targeted tests and a final runtime process check.

### Task 5: Review And Verification

**Files:**
- Modify docs only if behavior or benchmark claims changed.

- [ ] Run full unit tests, compileall, pip check, npm audit, leak scans, DDG probes, and final benchmarks.
- [ ] Request final code review through a GPT-5.5 xhigh subagent.
- [ ] Fix any Important/Critical findings and rerun affected checks.
- [ ] Final report may say full pass only if all hard requirements are met, including verified private DDG route.
