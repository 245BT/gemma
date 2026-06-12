# Agent Benchmark And Context7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add official-harness benchmark wrappers plus Context7 MCP and stronger DDG/yolo behavior for local Gemma.

**Architecture:** A small `benchmarks.agent_benchmarks` package owns benchmark suite metadata, command construction, result normalization, and CLI execution. Existing launcher/setup files keep owning Codex/Gemma configuration.

**Tech Stack:** Python stdlib, unittest, official external CLIs (`swebench`, `tb`, `harbor`), Codex CLI, Context7 MCP via npm.

---

### Task 1: Benchmark Adapter Tests

**Files:**
- Create: `tests/test_agent_benchmarks.py`
- Create: `benchmarks/agent_benchmarks/__init__.py`
- Create: `benchmarks/agent_benchmarks/registry.py`
- Create: `benchmarks/agent_benchmarks/runner.py`
- Create: `benchmarks/agent_benchmarks/results.py`
- Create: `benchmarks/agent_benchmarks/suites.py`

- [ ] Write failing tests for suite lookup, command generation, missing dependency handling, and result normalization.
- [ ] Run `.\.venv\Scripts\python.exe -m unittest tests.test_agent_benchmarks -v` and verify failure.
- [ ] Implement the minimal package code.
- [ ] Run the same test command and verify pass.

### Task 2: Context7 MCP And DDG Freshness Tests

**Files:**
- Modify: `tests/test_setup_local_codex.py`
- Modify: `tests/test_duckduckgo_mcp.py`
- Modify: `setup_local_codex.py`
- Modify: `duckduckgo_mcp.py`

- [ ] Write failing tests for Context7 config in direct/reasoning homes and base instructions.
- [ ] Write failing tests for DDG recency query parameters and tool signature.
- [ ] Run targeted tests and verify failure.
- [ ] Add Context7 MCP config and DDG recency support.
- [ ] Regenerate local configs with `.\.venv\Scripts\python.exe setup_local_codex.py`.
- [ ] Run targeted tests and verify pass.

### Task 3: Yolo Launcher Tests

**Files:**
- Modify: `tests/test_launch_gemma_codex.py`
- Modify: `launch_gemma_codex.py`

- [ ] Write failing tests that `--yolo` expands to bypass, approval never, and danger-full-access.
- [ ] Run targeted tests and verify failure.
- [ ] Implement launcher argument expansion.
- [ ] Run targeted tests and verify pass.

### Task 4: Benchmark Smoke Metrics

**Files:**
- Modify: `benchmarks/BENCHMARKS.md`

- [ ] Run direct benchmark metrics against 8081.
- [ ] Run reasoning benchmark metrics against 8082.
- [ ] Run a dry-run agent benchmark command for direct yolo.
- [ ] Run a dry-run agent benchmark command for reasoning yolo.
- [ ] Record the commands and output paths.

### Task 5: Final Verification

**Files:**
- No new files.

- [ ] Run `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- [ ] Run `npm test`.
- [ ] Run `.\.venv\Scripts\python.exe -m compileall benchmarks gemma_agent gemma_reasoning scripts duckduckgo_mcp.py setup_local_codex.py launch_gemma_codex.py`.
- [ ] Run `.\.venv\Scripts\python.exe -m pip check`.
- [ ] Run `npm audit --audit-level=moderate --json`.
- [ ] Report unavailable optional tools honestly.
