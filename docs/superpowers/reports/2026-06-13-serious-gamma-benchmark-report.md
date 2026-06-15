# Serious Gamma Benchmark Report - 2026-06-13

## Executive summary

Gamma's local Codex tool path had a real blocker: the 8082 reasoning proxy consumed tool-bearing
Codex requests and returned plain text, so `--reasoning --yolo` could emit tool-call-looking text
without executing tools. I changed the reasoning proxy to raw-forward any request with tool
schemas to the normal Responses proxy, preserving Codex tool-call events.

After the fix, a live `--reasoning --yolo` edit probe executed terminal tools, and the live
generated repo-fix suite passed 3/3 tasks with FAIL_TO_PASS and PASS_TO_PASS checks. Official
SWE-bench and Terminal-Bench were not run on this PC because Docker, Harbor, and SWE-bench are
missing locally.

## Research used

- Terminal-Bench 2.1 official leaderboard: https://www.tbench.ai/leaderboard/terminal-bench/2.1
- Anthropic Project Glasswing benchmark notes: https://www.anthropic.com/glasswing
- Anthropic Opus 4.8 benchmark footnotes: https://www.anthropic.com/news/claude-opus-4-8
- SWE-bench repository and official harness notes: https://github.com/swe-bench/SWE-bench
- SWE-Bench Pro public benchmark methodology: https://labs.scale.com/leaderboard/swe_bench_pro_public
- Harbor/Terminal-Bench install notes: https://www.harborframework.com/docs/tutorials/running-terminal-bench

## Baseline measurements

| Check | Result | Evidence |
| --- | ---: | --- |
| Reasoning-mode repo-fix before proxy fix | 0/1 | `benchmarks/runs/serious-repo-fix-gemma-smoke2.local-repo-fix.summary.json` |
| Before failure mode | patch failed | Codex/Gemma entered task repo but made no file changes |
| Before edit probe | failed | model emitted raw `<\|tool_call\>...` text; no `probe.txt` was created |
| Deterministic behavior harness | 9/9 | `benchmarks/runs/serious-local-behavior.local-agent-behavior.summary.json` |
| Oracle repo-fix harness | 3/3 | `benchmarks/runs/serious-repo-fix-oracle.local-repo-fix.summary.json` |

## Bugs and leaks found

- `--reasoning --yolo` tool-use bug: fixed by raw-forwarding tool-bearing requests in `gemma_reasoning_proxy.py`.
- `local_repo_fix --root .` bug: fixed by resolving the benchmark root before building `gemma-codex.cmd`.
- Solver timeout handling: fixed so outer `TimeoutExpired` becomes a measured timeout result with captured partial output.
- No secret or prompt-leak artifact was added. Benchmark logs keep paths, counts, status, and output lengths by default.

## Design changes

- `gemma_reasoning_proxy.py`: preserves Codex tool-call streams for any request with `tools`, `tool_choice`, or `parallel_tool_calls`.
- `benchmarks/agent_benchmarks/local_repo_fix.py`: supports live `gemma` solver runs, max task limits, timeout status, diff artifacts, stdout/stderr logs, and root resolution.
- `benchmarks/agent_benchmarks/capability_card.py`: writes local capability cards and multi-column comparison cards as JSON, Markdown, HTML, and PNG.
- Tests added for proxy raw-forwarding, comparison-card rendering, live solver invocation, and solver timeout behavior.

## Speed measurements

| Benchmark | Score | Latency |
| --- | ---: | ---: |
| GammaBench multiple-choice | 10/10 | 8.493 s |
| Local Agent Behavior | 9/9 | 0.646 s |
| Local Repo Fix, oracle | 3/3 | 0.442 s |
| Local Repo Fix, Gamma live | 3/3 | 601.743 s |
| Local Repo Fix, Gamma live avg/task | 3/3 | 200.581 s |

The quality result is good on the local suite, but the live repo-fix latency is still high for three
small tasks.

## Context measurements

- Local model endpoint reports `n_ctx=262144`.
- Current local Codex configs set `model_context_window=262144` and compact around 240000 tokens.
- The live local repo-fix benchmark used generated small repos; it did not stress the full context window.

## Gemma sub-agent implementation status

This benchmark did not add new Gemma runtime sub-agent features. It did preserve the existing
Gemma MCP/sub-agent tool path and verified the Codex tool-preservation path needed before larger
sub-agent work is meaningful.

## DuckDuckGo verification

No DuckDuckGo private-route claim was made. Public web research was done through normal HTTPS
sources and official leaderboard/vendor pages.

## Tests run

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_reasoning_proxy
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_response_proxy tests.test_gemma_reasoning tests.test_gemma_reasoning_proxy tests.test_agent_benchmarks tests.test_capability_card
.\.venv\Scripts\python.exe -m unittest tests.test_capability_card
```

Latest focused result: 63 tests passed across the proxy, reasoning, benchmark, and card suites.

## Before/after benchmark table

| Capability | Benchmark | Before | After | Notes |
| --- | --- | ---: | ---: | --- |
| Agentic coding | Local Repo Fix | 0/1 | 3/3 | before was blocked by reasoning proxy tool handling |
| Tool execution | `--reasoning --yolo` edit probe | failed | passed | after fix, Codex executed PowerShell tool calls |
| Knowledge work | GammaBench-KW | not measured | 3/3 | live no-tools multiple-choice |
| Spatial reasoning | GammaBench-Spatial | not measured | 3/3 | live no-tools multiple-choice |
| Tool policy | GammaBench-ToolPolicy | not measured | 2/2 | live no-tools multiple-choice |
| Planning/recovery | GammaBench-Recovery | not measured | 2/2 | live no-tools multiple-choice |
| Terminal behavior | Local Agent Behavior | not measured here | 2/2 | deterministic local harness |
| Large repo/context | Local Agent Behavior | not measured here | 2/2 | deterministic local harness |
| Research readiness | Local Agent Behavior | not measured here | 1/1 | cited research artifact check |

## Official benchmark readiness

| Harness | Local status | Reason |
| --- | --- | --- |
| Terminal-Bench 2.1 | not run | `harbor` missing; Docker missing |
| SWE-bench Verified/Pro | not run | `swebench` missing; Docker missing |
| SWE-Bench Pro public | not run | official environments require Docker-style reproducible execution |

Checked locally:

```powershell
where.exe harbor
docker --version
.\.venv\Scripts\python.exe -c "import swebench"
```

All three were unavailable on this PC during this run.

## Artifacts

- Local capability card PNG: `benchmarks/runs/serious-gamma-card.capability-card.png`
- Local capability card Markdown: `benchmarks/runs/serious-gamma-card.capability-card.md`
- Multi-column comparison PNG: `benchmarks/runs/serious-gamma-comparison.comparison-card.png`
- Multi-column comparison Markdown: `benchmarks/runs/serious-gamma-comparison.comparison-card.md`
- Live repo-fix summary: `benchmarks/runs/serious-repo-fix-gemma-full.local-repo-fix.summary.json`
- Before-fix repo-fix summary: `benchmarks/runs/serious-repo-fix-gemma-smoke2.local-repo-fix.summary.json`

## Remaining risks

- The local benchmark suite is small and should not be advertised as SWE-bench, Terminal-Bench,
  OSWorld, or Claude-style public benchmark performance.
- The live coding score is 3/3 locally, but the tasks are generated smoke tasks, not broad
  ambiguous production tasks.
- Latency is high: about 200 seconds per live repo-fix task.
- Official benchmark execution still requires installing Docker plus Harbor/SWE-bench and then
  running the official harnesses.
- The reasoning proxy now preserves tool execution, but it does not add extra reasoning to
  tool-bearing Codex requests; it correctly prioritizes tool-channel integrity.

## Reproduction commands

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-gemma-runtime.ps1
.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.local_behavior --root . --out-dir benchmarks\runs --run-id serious-local-behavior
.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.local_repo_fix --root . --out-dir benchmarks\runs --run-id serious-repo-fix-oracle --solver oracle --timeout 30
.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.local_repo_fix --root . --out-dir benchmarks\runs --run-id serious-repo-fix-gemma-full --solver gemma --timeout 240
.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.capability_card --endpoint http://127.0.0.1:8082/v1/responses --out-dir benchmarks\runs --run-id serious-gamma-card --model-label "Gamma Local" --behavior-summary benchmarks\runs\serious-local-behavior.local-agent-behavior.summary.json --repo-fix-summary benchmarks\runs\serious-repo-fix-gemma-full.local-repo-fix.summary.json --timeout 120
```

## Final verdict

Partial pass overall. Gamma is now measurably better as a local coding agent because the
`--reasoning --yolo` tool path works and the live local repo-fix suite passes 3/3. Public-grade
benchmarking remains incomplete until Docker, Harbor, and SWE-bench are available and official
harnesses are run.
