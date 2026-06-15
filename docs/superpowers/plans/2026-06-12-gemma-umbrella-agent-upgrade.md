# Gemma Umbrella Agent Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Gemma agent hardening project using the blockblast-style umbrella/subtrack workflow.

**Architecture:** Keep a single controller session responsible for integration. Use fresh subagents for independent audits and narrow implementation tasks, then run full local verification and benchmark commands before any completion claim.

**Tech Stack:** Python `unittest`, Gemma local agent runtime modules, benchmark wrapper scripts, PowerShell commands, Codex subagents.

---

### Task 1: Terminal And Runtime Audit

**Files:**
- Review: `gemma_agent/terminal.py`
- Review: `gemma_agent/tool_executor.py`
- Review: `gemma_agent/tools.py`
- Review: `tests/test_gemma_agent_runtime.py`

- [x] Verify terminal stalls become failed tool results.
- [x] Verify idle and hard timeout behavior.
- [x] Verify spawned children are killed on timeout.
- [x] Verify recovery metadata is preserved.

### Task 2: Supervisor And Chat Behavior Audit

**Files:**
- Review: `gemma_agent/supervisor.py`
- Review: `gemma_agent/progress.py`
- Review: `tests/test_gemma_agent_runtime.py`

- [x] Verify repeated tool calls produce recovery events.
- [x] Verify stalled tools produce recovery events.
- [x] Verify honest "tests not run" caveats are accepted.
- [x] Verify unsupported tool claims still require evidence.

### Task 3: Research And Benchmark Audit

**Files:**
- Review: `docs/research/`
- Review: `benchmarks/agent_benchmarks/local_behavior.py`
- Review: `benchmarks/agent_benchmarks/runner.py`
- Review: `benchmarks/agent_benchmarks/results.py`
- Review: `tests/test_agent_benchmarks.py`

- [x] Verify Composer 2 and related public papers are local.
- [x] Verify benchmark artifacts are written under `benchmarks/runs`.
- [x] Verify local benchmark uses the subprocess runner and timeout path.
- [x] Verify failure taxonomy is normalized.

### Task 4: Context, Large Repo, And Safety Audit

**Files:**
- Review: `gemma_agent/context.py`
- Review: `gemma_agent/safety.py`
- Review: `gemma_agent/schemas.py`
- Review: `tests/test_gemma_agent_runtime.py`

- [x] Verify large-file context budget behavior is covered.
- [x] Verify untrusted evidence wrapping prevents prompt-injection replay.
- [x] Verify path validation protects tool arguments.
- [x] Verify no raw chain-of-thought or prompt text is stored in new artifacts.

### Task 5: Final Verification And Report

**Files:**
- Update: `docs/superpowers/reports/2026-06-12-gamma-stall-recovery-report.md`
- Review: `benchmarks/runs/gamma-stall-after.local-agent-behavior.summary.json`

- [x] Run full Python unit test suite.
- [x] Run `npm test`.
- [x] Run local behavior benchmark.
- [x] Run whitespace and credential scans.
- [x] Report measured results, caveats, and reproduction commands.

## Current Verification Commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
npm test
.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.runner --suite local-agent-behavior --mode direct --run-id gamma-stall-after --out-dir benchmarks\runs --timeout 30
git diff --check
rg -n "(AKIA[0-9A-Z]{16}|BEGIN (RSA|DSA|EC|OPENSSH|PRIVATE) KEY|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})" AGENTS.md README.md gemma_agent benchmarks\agent_benchmarks tests docs\superpowers launch_gemma_codex.py gemma_agent_mcp.py setup_local_codex.py
```

## Final Local Results

- Python unit suite: 249 tests passed.
- `npm test`: 249 tests passed.
- `local-agent-behavior`: 8/8 passed, pass rate 1.0, `latency_ms` 657.177, failure counts `{}`.
- `git diff --check`: no whitespace errors; Git reported CRLF normalization warnings only.
- High-signal credential scan: no matches.

## Remaining Expansion Tracks

- Live external SWE-bench and Terminal-Bench 2 runs when dependencies and datasets are available.
- Runtime instrumentation for first-byte latency, token usage, VRAM/RAM, cache hit rate, and cost.
- Larger ambiguous real-world task set beyond deterministic local checks.
