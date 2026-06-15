# Gamma Stall Recovery Engineering Report

Date: 2026-06-12

## Executive Summary

Gamma now has bounded terminal execution, process-tree cleanup, supervisor progress detection,
schema-validated default tools, MCP subagent terminal access, launcher timeout control, context
budgeting, sanitized tool failures, final-claim evidence checks, and a deterministic local
benchmark suite for the exact failure class reported by the user.

## Research Used

- Composer 2 technical report, user-provided PDF and public arXiv copy.
- SWE-agent: agent-computer interface design for software engineering agents.
- SWE-bench: real-world GitHub issue evaluation.
- Terminal-Bench 2: command-line task evaluation.
- AgentDojo: untrusted tool-output and prompt-injection evaluation.
- SWE-evo: long-horizon software evolution evaluation.

## Baseline Measurements

- Full Python unit suite before this implementation slice: 213 tests passed in about 3.245 seconds.
- `npm test` before this implementation slice: 213 tests passed in about 3.246 seconds.
- Targeted TDD baseline failed for missing terminal runner, missing progress payload, missing local
  behavior benchmark, unbounded Codex execution, and empty MCP tool registry.

## Bugs And Leaks Found

- Confirmed stall root cause: no Gemma-owned terminal supervisor with idle-timeout recovery.
- Confirmed low-progress gap: repeated tool calls and invalid actions were not summarized back to
  the model as structured recovery evidence.
- Confirmed MCP tool gap: runtime subagents had no default bounded terminal tool.
- Confirmed launcher timeout gap: Codex execution did not support an optional hard timeout.
- Confirmed benchmark gap: no deterministic suite covered stalls, bad tool calls, schema rejection,
  large-file context budget, large-repo navigation, or research artifact presence.
- Subagent audit found and fixed additional defects: newline-free terminal output could look idle,
  process-mode tools and launcher subprocesses did not share one process-tree cleanup path,
  structured `ok: false` payloads could surface untrusted error codes, final-claim validation could
  accept echoed test names or non-file issue creation as proof, invalid thinking summaries did not
  feed progress recovery, citation and `extra` context could bypass budgets, large relevant files
  were silently dropped, benchmark failure modes were not normalized, and the research-artifact
  benchmark only checked the Composer artifacts.
- No secrets were intentionally added. Benchmark summaries store counters and paths, not raw prompts.

## Design Changes

- `TerminalCommandRunner` captures stdout/stderr tails, elapsed time, output lengths, hashes,
  exit code, status, and recovery metadata.
- Terminal streaming now treats newline-free output as progress, so commands that print progress
  without line breaks do not falsely stall.
- Terminal, process-mode tools, and launcher timeout cleanup now use one process-tree termination
  helper instead of killing only the parent process.
- `ProgressTracker` detects repeated tool calls, repeated invalid actions, timeouts, and stalls.
- `ToolExecutor` promotes structured `ok: false` tool payloads into failed `ToolResult` records,
  preserves output evidence, and sanitizes untrusted error codes.
- `AgentSupervisor` injects `progress` into model payloads and records tool results for recovery.
- `AgentSupervisor` accepts honest verification caveats such as "Tests not run" without treating
  them as successful execution claims.
- `AgentSupervisor` now requires real test-runner or file-edit evidence for completion claims, so
  `echo pytest` and non-file `create_issue` payloads do not satisfy verification gates.
- Invalid thinking summaries are recorded as invalid actions and can trigger progress recovery.
- `ContextBuilder` caps citation quotes and `extra` context, wraps untrusted evidence, and keeps a
  bounded excerpt for the first relevant oversized memory instead of silently dropping it.
- Thinking-summary validation rejects prompt, raw chain-of-thought, and session-log leakage
  indicators.
- `build_default_tool_registry` exposes `terminal_command` with JSON schema and path safety checks.
- `gemma_agent_mcp.py` uses the default tool registry for Gemma-owned subagent tasks.
- `launch_gemma_codex.py` supports `GEMMA_CODEX_EXEC_TIMEOUT` and returns 124 on timeout.
- `local-agent-behavior` benchmark covers eight deterministic agent-behavior checks.
- Benchmark results now use a normalized failure taxonomy and resolve relative output directories
  under the repository root.

## Current Benchmark Result

Suite: `local-agent-behavior`

| Metric | Before | After |
| --- | ---: | ---: |
| Suite available | No | Yes |
| Total cases | 0 | 8 |
| Passed cases | 0 | 8 |
| Pass rate | n/a | 1.0 |
| Inner suite latency | n/a | 657.177 ms |
| End-to-end wrapper latency | n/a | 786.640 ms |
| Failure counts | n/a | `{}` |

Covered categories: stall recovery, terminal execution, tool-use reliability, planning recovery,
large-file handling, large-repo navigation, and research citation. The research check now requires
local artifacts for Composer 2, SWE-agent, SWE-bench, Terminal-Bench 2, AgentDojo, and SWE-evo.

## Speed Measurements

- Local behavior benchmark inner latency: 657.177 ms.
- End-to-end benchmark wrapper latency: 786.640 ms.
- Terminal stall recovery now uses idle and hard timeouts instead of unbounded waits.
- Launcher execution supports `GEMMA_CODEX_EXEC_TIMEOUT` and returns 124 after timeout cleanup.

## Context Measurements

- Context selection tests cover budgeted project maps, task scratchpads, evidence stores, and
  first relevant oversized memory excerpts.
- Citation quotes and caller-provided `extra` context are capped before model exposure.
- Token count, VRAM/RAM, cache-hit, and cost measurements were not available from deterministic
  local tests; these require instrumented real model calls.

## Gemma Sub-Agent Implementation Status

- Gemma-owned subagent runtime paths now use the default schema-validated tool registry.
- `terminal_command` is available to MCP subagent tasks with timeout and workspace path guards.
- Codex subagents were used only as development reviewers for this project, not described as
  Gemma runtime subagents.

## DuckDuckGo Verification

- No DuckDuckGo runtime behavior was changed in this implementation slice.
- The local startup contract still requires treating the DuckDuckGo tool as public HTTPS to
  `lite.duckduckgo.com` unless runtime evidence proves otherwise.
- No private-network DuckDuckGo claim was made or added.

## Tests Run

- Targeted red/green unit tests for terminal, progress, launcher, MCP, and benchmark behavior.
- `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`: 249 tests passed.
- `npm test`: 249 tests passed.
- `.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.runner --suite local-agent-behavior --mode direct --run-id gamma-stall-after --out-dir benchmarks\runs --timeout 30`: passed, 8/8 cases.
- `git diff --check`: no whitespace errors; Git reported CRLF normalization warnings only.
- High-signal credential scan: no matches.

## Remaining Risks

- The benchmark is deterministic and local. It should be extended with live SWE-bench,
  Terminal-Bench 2, and CursorBench-style tasks when those external harnesses and datasets are
  available.
- Runtime quality still depends on the underlying model; these changes improve harness behavior,
  context, recovery, and evaluation rather than model weights.
- Cost, token, VRAM, and cache-hit measurements require instrumented model calls from the real
  runtime path.

## Reproduction Commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
npm test
.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.runner --suite local-agent-behavior --mode direct --run-id gamma-stall-after --out-dir benchmarks\runs --timeout 30
```

Final verdict: pass for the implemented architecture and deterministic benchmark scope.
