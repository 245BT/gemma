# Gemma Agent Runtime Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a measured Gemma-owned agent runtime with strict JSON tools, Gemma subagents, hardened DuckDuckGo research, hierarchical memory, and visible before/after benchmarks.

**Architecture:** Add a new `gemma_agent` package beside the current proxies. Keep existing launch paths working while the new runtime is tested as a library. Add a benchmark harness and Markdown results table before optimizing.

**Tech Stack:** Python 3.14 stdlib, existing `.venv`, `unittest`, local llama.cpp Responses API, local DuckDuckGo MCP helper.

---

## File Structure

- `gemma_agent/config.py`: runtime defaults and endpoint/timeouts.
- `gemma_agent/model_client.py`: local Responses API client and test fake client protocol.
- `gemma_agent/schemas.py`: action/result dataclasses and validation helpers.
- `gemma_agent/tool_registry.py`: tool schema registry.
- `gemma_agent/tool_executor.py`: schema validation, external execution, timing, audit results.
- `gemma_agent/supervisor.py`: Gemma supervisor action loop.
- `gemma_agent/subagent.py`: isolated Gemma subagent task runner.
- `gemma_agent/memory.py`: hierarchical memory records and deduplication.
- `gemma_agent/context.py`: compact context pack builder.
- `gemma_agent/citations.py`: source and citation tracking.
- `gemma_agent/safety.py`: path validation and untrusted-output wrappers.
- `benchmarks/bench_runtime.py`: benchmark CLI and reusable functions.
- `benchmarks/workloads/smoke.jsonl`: smoke workload.
- `benchmarks/BENCHMARKS.md`: before/after table.
- `duckduckgo_mcp.py`: hardened public-search tool while preserving existing MCP string interface.
- `tests/test_gemma_agent_runtime.py`: runtime tests.
- `tests/test_bench_runtime.py`: benchmark harness tests.
- `tests/test_duckduckgo_mcp.py`: expanded DDG tests.
- `package.json`: make `npm test` run the real Python suite.
- `requirements-gemma.txt`: document actual local runtime dependencies.

## Task 1: Benchmark Harness

**Files:**
- Create: `benchmarks/bench_runtime.py`
- Create: `benchmarks/workloads/smoke.jsonl`
- Create/Modify: `benchmarks/BENCHMARKS.md`
- Create: `tests/test_bench_runtime.py`

- [ ] Write tests that create a temporary workload, run the harness with fake endpoints, and assert JSONL, summary JSON, and Markdown rows are written.
- [ ] Run: `.\.venv\Scripts\python.exe -m unittest tests.test_bench_runtime -v`
- [ ] Confirm the tests fail because the harness does not exist yet.
- [ ] Implement the minimal benchmark runner with injectable request/resource functions.
- [ ] Run targeted tests until passing.
- [ ] Run: `.\.venv\Scripts\python.exe benchmarks\bench_runtime.py --mode smoke --out benchmarks\runs --markdown benchmarks\BENCHMARKS.md`
- [ ] Record the baseline row in `benchmarks/BENCHMARKS.md`.

## Task 2: Gemma-Owned Runtime Interfaces

**Files:**
- Create: `gemma_agent/*.py`
- Create: `tests/test_gemma_agent_runtime.py`

- [ ] Write tests for strict JSON actions, invalid JSON, hallucinated tool-call claims, successful external tool execution, thinking-summary shape, subagent spawning, memory deduplication, and citations.
- [ ] Run: `.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime -v`
- [ ] Confirm the tests fail because the package does not exist yet.
- [ ] Implement minimal dataclasses and validation in `schemas.py`.
- [ ] Implement `ToolRegistry` and `ToolExecutor`.
- [ ] Implement `ModelClient`, `AgentSupervisor`, and `SubAgent`.
- [ ] Implement `MemoryStore`, `ContextBuilder`, `CitationManager`, and `SafetyGuard`.
- [ ] Run targeted tests until passing.

## Task 3: DuckDuckGo Hardening

**Files:**
- Modify: `duckduckgo_mcp.py`
- Modify: `tests/test_duckduckgo_mcp.py`

- [ ] Add failing tests for non-string queries, huge queries, malformed `max_results`, prompt-injection snippets, unsupported URL schemes, citation fields, irrelevant results, and timeout propagation.
- [ ] Run: `.\.venv\Scripts\python.exe -m unittest tests.test_duckduckgo_mcp -v`
- [ ] Confirm the new tests fail for missing behavior.
- [ ] Add structured search responses while preserving the existing `duckduckgo_search(...) -> str` interface.
- [ ] Add untrusted-output labeling, relevance scoring, URL scheme filtering, and controlled errors.
- [ ] Run targeted tests until passing.
- [ ] Run the five live DDG probes and record pass/fail notes in `benchmarks/BENCHMARKS.md`.

## Task 4: Verification Reproducibility

**Files:**
- Modify: `package.json`
- Modify: `requirements-gemma.txt`
- Add tests only if behavior changes.

- [ ] Make `npm test` run `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- [ ] Add actual imported runtime dependencies to `requirements-gemma.txt`.
- [ ] Run `npm test`.
- [ ] Run `.\.venv\Scripts\python.exe -m pip check`.

## Task 5: Prompt And Setup Slimming

**Files:**
- Modify: `setup_local_codex.py`
- Modify: `tests/test_setup_local_codex.py`

- [ ] Add a test that measures generated base-instruction token estimate and asserts at least 30% reduction from the recorded baseline.
- [ ] Preserve the uncensored/no-wrapper instruction exactly in meaning.
- [ ] Move verbose behavior into AGENTS/spec docs, keeping runtime prompt modular and short.
- [ ] Run setup tests and full suite.

## Task 6: Proxy And Runtime Safety

**Files:**
- Modify: `gemma_response_proxy.py`
- Modify: `gemma_reasoning_proxy.py`
- Modify: `gemma_reasoning/upstream.py`
- Modify related tests.

- [ ] Add tests for configurable timeouts and sanitized error payloads.
- [ ] Replace hardcoded `3600` defaults with config values.
- [ ] Avoid returning raw exception details to clients.
- [ ] Keep existing output-cleaning behavior intact.
- [ ] Run targeted tests and full suite.

## Task 7: Repeated Benchmarks And Optimization

**Files:**
- Modify: `benchmarks/BENCHMARKS.md`
- Modify runtime config only when a benchmark proves the change.

- [ ] Run the benchmark before each optimization.
- [ ] Apply one optimization at a time.
- [ ] Run the same benchmark after each optimization.
- [ ] Keep only optimizations that improve latency/throughput without reducing test pass rate or task success.
- [ ] Mark unverified metrics as unverified instead of inventing numbers.

## Task 8: Final Review And Verification

**Files:**
- Review all changed files.

- [ ] Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`
- [ ] Run: `npm test`
- [ ] Run: `.\.venv\Scripts\python.exe -m pip check`
- [ ] Run: `npm audit --audit-level=low`
- [ ] Run benchmark and update `benchmarks/BENCHMARKS.md`.
- [ ] Request code review from a fresh GPT-5.5 xhigh reviewer.
- [ ] Fix blocking review issues.

## Self-Review

- Spec coverage: benchmark, runtime, subagents, DDG hardening, memory/context, prompt slimming, safety, and final verification are mapped to tasks.
- Placeholder scan: no TBD/TODO placeholders are used.
- Type consistency: file/module names match the approved design.
- Constraint check: no task changes model weights or adds moderation/refusal policy.
