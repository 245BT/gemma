# Composer 2 Gamma Hardening Report

## 1. Executive Summary

This pass fixed the live `son`/`sonion`/`operator` launcher timeout, added response normalization for the reasoning proxy, wired a Composer-2-inspired effort budget into the Gamma supervisor payload, and expanded the benchmark suite with deterministic behavior checks plus generated local repo-fix tasks.

The live startup bug was in the runtime launcher script: PowerShell `Start-Process` calls redirected stdio for long-running child services, which caused startup to hang even when the HTTP services were already healthy. The generated and live scripts now start the llama server and proxies detached with `-WindowStyle Hidden` and no redirected stdio.

## 2. Research Used

- User-provided Composer 2 report: `C:\Users\Agent-1\Downloads\Composer2.pdf`.
  - Extracted full text to `docs/research/composer2/Composer2.extracted.txt`.
  - Rendered all 23 pages at 200 DPI under `docs/research/composer2/renders_200dpi/`.
  - Built visual contact sheet: `docs/research/composer2/contact-sheet.png`.
  - Text extraction: 75,899 chars, 10,983 words; every page had an extractable text layer.
- Downloaded and extracted additional public papers:
  - `docs/research/papers/terminal-bench-2601.11868.pdf` and `.txt`.
  - `docs/research/papers/swe-agent-neurips-2024.pdf` and `.txt`.
  - `docs/research/papers/openhands-iclr-2025.pdf` and `.txt`.
  - `docs/research/papers/swe-smith-2504.21798.pdf` and `.txt`.

Applied ideas:

- Composer 2: harness matching, self-summarization, nonlinear effort penalty, tool-call discipline, CursorBench-style ambiguous repo tasks, and cost/latency/tool metrics.
- SWE-agent: agent-computer-interface guardrails for search, edit, tests, and error feedback.
- Terminal-Bench: terminal task framing with instruction, environment, tests, reference solution, time limit, and failure analysis.
- OpenHands: sandboxed code execution, benchmark integration, and agent platform evaluation.
- SWE-smith: generated repo tasks with executable tests and oracle-style solutions.

## 3. Baseline Measurements

Baseline local behavior artifact:

- `benchmarks/runs/gamma-stall-after.local-agent-behavior.summary.json`
- Total checks: 8
- Resolved: 8
- Pass rate: 1.0
- Latency: 657.177 ms
- Categories: stall recovery, terminal execution, tool reliability, planning recovery, large-file handling, large-repo navigation, research citation.

Launcher baseline failure:

- `launch_gemma_codex.py` raised because `start-gemma-runtime.ps1` timed out waiting for `http://127.0.0.1:8081/v1/models`.
- Direct endpoint probing showed the services could be healthy while the startup script still failed to return.

## 4. Bugs And Leaks Found

- Fixed: runtime startup hang from `Start-Process -RedirectStandardOutput/-RedirectStandardError` on long-running children.
- Fixed: reasoning proxy fast path could return a valid Responses API payload with nested `output[*].content[*].text` but no top-level `output_text`.
- Confirmed: DuckDuckGo module metadata marks the search endpoint as `https://lite.duckduckgo.com/lite/`, transport `public_https`, and `private_network=false`.
- No new secrets or raw prompt logs were added.
- Benchmark outputs store counters, lengths, hashes, paths, and summary metrics, not raw model prompt text.

## 5. Design Changes

- Added `gemma_agent/effort.py`:
  - `TaskEffortPolicy`
  - `TaskEffortBudget`
  - task difficulty classification
  - nonlinear cost score using turns, model calls, tool calls, prompt/completion/tool-output/final-token estimates.
- Wired effort budgets into `AgentSupervisor` payloads as `payload["effort"]`.
- `ProgressTracker` thresholds now come from the effort budget for each run.
- Added fast-path response normalization in `gemma_reasoning_proxy.normalize_response_payload`.
- Added benchmark case `effort_policy_budgeting` to `local-agent-behavior`.
- Added generated local repo-fix benchmark:
  - `benchmarks/agent_benchmarks/local_repo_fix.py`
  - suite registration: `local-repo-fix`
  - metrics: pass rate, fail-to-pass rate, pass-to-pass rate, patch apply rate, regression count, changed files, changed lines, latency.

## 6. Speed Measurements

- Runtime startup after fix: 6.852 seconds in the measured startup run.
- Sequential endpoint verification after latest changes:
  - `8080 /v1/models`: 200
  - `8081 /v1/models`: 200
  - `8082 /v1/models`: 200
- Reasoning proxy fast path:
  - Request: `Reply with only OK.`
  - Response: `status=completed`, `output_text=OK`

## 7. Context Measurements

- Composer 2 extraction: 23 rendered pages, 75,899 extracted chars, 10,983 words.
- Local context benchmark still passes large-file and large-repo retrieval checks.
- `ContextBuilder` retains tier caps, relevance ranking, truncation with hashes, and untrusted evidence wrappers.
- This pass did not implement a global serialized context cap; effort payload now provides `max_context_chars` so the supervisor/model can budget context explicitly.

## 8. Gemma Sub-Agent Implementation Status

- Existing Gemma runtime sub-agent support remains present through `AgentSupervisor`, `SubAgent`, `run_subagents`, and `SubAgentBudget`.
- This pass did not copy Codex subagent internals into Gemma.
- This pass exposed effort-budget `max_subagents` in the model payload and preserved the existing budgeted `spawn_subagents` action path.

## 9. DuckDuckGo Verification

- Static/unit verification passed under full test discovery.
- Module metadata confirms public HTTPS endpoint:
  - `search_endpoint=https://lite.duckduckgo.com/lite/`
  - `transport=public_https`
  - `private_network=false`
- No claim is made that DuckDuckGo uses a private route.

## 10. Tests Run

- `.\.venv\Scripts\python.exe -m unittest tests.test_gemma_reasoning_proxy` -> 20 tests OK.
- `.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime` -> 127 tests OK.
- `.\.venv\Scripts\python.exe -m unittest tests.test_agent_benchmarks tests.test_bench_runtime` -> 26 tests OK.
- `.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex` -> 40 tests OK.
- `.\.venv\Scripts\python.exe -m unittest discover -s tests` -> 307 tests OK.

Live launcher checks:

- `son --help` -> 0
- `sonion --help` -> 0
- `operator --help` -> 0
- `son --yolo --help` -> 0
- `sonion --yolo --help` -> 0
- `operator --yolo --help` -> 0
- `son exec --help` -> 0
- `sonion exec --help` -> 0
- `operator exec --help` -> 0
- `son --yolo exec --help` -> 0
- `sonion --yolo exec --help` -> 0
- `operator --yolo exec --help` -> 0

## 11. Before/After Benchmark Table

| Benchmark | Before | After |
| --- | ---: | ---: |
| Local behavior total checks | 8 | 9 |
| Local behavior resolved | 8 | 9 |
| Local behavior pass rate | 1.0 | 1.0 |
| Local behavior latency | 657.177 ms | 628.764 ms |
| Effort budgeting check | not present | pass |
| Local repo-fix total tasks | not present | 3 |
| Local repo-fix resolved | not present | 3 |
| Local repo-fix pass rate | not present | 1.0 |
| Local repo-fix fail-to-pass rate | not present | 1.0 |
| Local repo-fix pass-to-pass rate | not present | 1.0 |
| Local repo-fix patch apply rate | not present | 1.0 |
| Local repo-fix regressions | not present | 0 |
| Local repo-fix latency | not present | 420.289 ms |

## 12. Remaining Risks

- The generated local repo-fix suite currently runs the oracle solver, not a full slow Gemma live coding rollout, so it validates benchmark mechanics and regression accounting rather than live model coding success.
- Global context compaction is still tier-based rather than a hard serialized payload cap.
- Terminal low-progress detection handles idle/hard timeouts and repeated tool loops, but noisy commands that print useless progress until hard timeout are not separately classified as low progress yet.
- Full external SWE-bench and Terminal-Bench runs were not executed because those harnesses and datasets are heavy and may not be installed locally.

## 13. Reproduction Commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests

powershell -NoProfile -ExecutionPolicy Bypass -File .\start-gemma-runtime.ps1

son --help
sonion --help
operator --help
son --yolo --help
sonion --yolo --help
operator --yolo --help
son exec --help
sonion exec --help
operator exec --help
son --yolo exec --help
sonion --yolo exec --help
operator --yolo exec --help

.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.local_behavior --root . --out-dir benchmarks\runs --run-id composer2-hardening-after

.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.local_repo_fix --root . --out-dir benchmarks\runs --run-id composer2-repo-fix-after --solver oracle --timeout 20

.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.runner --suite local-repo-fix --mode direct --run-id composer2-repo-fix-runner-after --out-dir benchmarks\runs --timeout 20
```

## 14. Final Verdict

Pass for the implemented scope.

The original launcher timeout is fixed and verified through all requested terminal entrypoints. Gamma now has explicit effort budgeting, response normalization, stronger deterministic hardening coverage, and a generated repo-fix benchmark suite with fail-to-pass/pass-to-pass metrics. The remaining weak point is live model coding performance on the new generated tasks; the harness is ready, but this pass did not run a long Gemma coding rollout against it.
