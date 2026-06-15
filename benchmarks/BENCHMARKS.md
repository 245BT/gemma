# Gemma Runtime Benchmarks

Use `bench_runtime.py` to append run-scoped benchmark rows here. Raw events and summary JSON files are written under `benchmarks/runs/` by default.

Example baseline run against the response proxy:

```powershell
python benchmarks\bench_runtime.py --phase before --run-label baseline-proxy --endpoint http://127.0.0.1:8081/v1/responses
```

Example candidate run against the reasoning proxy:

```powershell
python benchmarks\bench_runtime.py --phase after --run-label candidate-reasoning --endpoint http://127.0.0.1:8082/v1/responses
```

## Remeasure Aggregate: 2026-06-12

Source: `benchmarks/runs/remeasure-2026-06-12.aggregate.json`.

| Group | n | Avg first byte ms | Avg first text ms | Avg total ms | Total ms range | Total ms stdev | Avg tok/s | Prompt tok avg | Completion tok avg | Peak RAM MB avg | Peak VRAM MB avg | Task success |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw llama.cpp `8080` | 5 | 257.18 | 258.29 | 258.34 | 252.10-265.20 | 4.95 | 23.23 | 17.0 | 6.0 | 30.17 | 16003.00 | 1.00 |
| Direct proxy `8081` | 5 | 273.36 | 274.05 | 274.09 | 246.00-333.90 | 35.11 | 22.15 | 17.0 | 6.0 | 30.19 | 16003.00 | 1.00 |
| Reasoning proxy forced graph `8082` | 5 | 413.09 | 413.27 | 413.31 | 370.01-534.49 | 68.20 | - | 0.0 | 0.0 | 30.16 | 16003.00 | 1.00 |
| Reasoning proxy fast path `8082` | 5 | 328.91 | 329.24 | 329.29 | 280.33-412.68 | 52.46 | 18.56 | 17.0 | 6.0 | 30.24 | 16003.00 | 1.00 |

Reasoning proxy current-code request-shape comparison:

| Comparison | n | Avg total ms before | Avg total ms after | Delta ms | Reduction | Ratio | Caveat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Forced graph request shape -> fast-path request shape on `8082` | 5 each | 413.31 | 329.29 | -84.02 | 20.33% | 1.26x | Current-code A/B only; ranges overlap. |

Token/sec and token-count fields are not comparable for the retained aggregate forced graph row
because those measurements were captured before graph usage propagation. New audit rows generated
after the metadata fix include graph usage when the underlying response client provides it.

## Base Instruction Prompt Size

Measured with `tiktoken` `cl100k_base` against `setup_local_codex.build_base_instructions()`.
The before row is the selected 640-token legacy base-instruction artifact from local session
logs, not a universal baseline for every older session or full runtime prompt.

| Artifact | Phase | Chars | Words | cl100k tokens | Change |
| --- | --- | ---: | ---: | ---: | ---: |
| Selected legacy base instructions | before | 3306 | 504 | 640 | - |
| Current `setup_local_codex.build_base_instructions()` | after | 2309 | 338 | 446 | -30.3% |

## DuckDuckGo Controlled Research Verification

Live checks run through `duckduckgo_search_response(..., max_results=3, timeout=15)`.
The controlled interface reports its route as `https://lite.duckduckgo.com/lite/` over
`public_https`; `private_network` is `false`, so citation verification passes but the private
network requirement remains a documented gap.

| Probe | Status | Citation behavior | Verdict |
| --- | --- | --- | --- |
| current news: `OpenAI Codex fast mode June 2026 latest news` | `ok` | Returned 3 supported citations; first citation `https://developers.openai.com/codex/changelog`. | pass |
| niche technical: `VeriCache KV cache compression speculative decoding arXiv 2026` | `ok` | Returned 3 supported citations; first citation `https://arxiv.org/abs/2605.17613`. | pass |
| source comparison: `OpenAI Codex pricing fast mode credits rate official` | `ok` | Returned 3 supported citations; first citation `https://developers.openai.com/codex/pricing`. | pass |
| failed/no-result: `qzvxxnonexistentgemmaagentquery20260612 no results` | `irrelevant_results` | Returned zero citations and refused to treat unrelated results as support. | pass |
| citation extraction: `DuckDuckGo privacy policy official` | `ok` | Returned 3 supported citations; first citation `https://duckduckgo.com/privacy`. | pass |

## Runtime Smoke Rows

Single-run rows below are retained as raw artifacts only. Use `Remeasure Aggregate: 2026-06-12`
for the current controlled benchmark numbers.

## Agent Benchmark Wrapper Setup: 2026-06-12

Source: `benchmarks/agent_benchmarks/`.

| Suite | Harness | Current artifact | Mode | Yolo | Status |
| --- | --- | --- | --- | --- | --- |
| SWE-bench | `python -m swebench.harness.run_evaluation` | `benchmarks\runs\swe-bench-reasoning-yolo-dry.agent-benchmark.summary.json` | reasoning | yes | dry-run command generated |
| Terminal-Bench | `tb run` | `benchmarks\runs\terminal-bench-reasoning-yolo-dry.agent-benchmark.summary.json` | reasoning | yes | dry-run command generated |
| Terminal-Bench 2.0 | `harbor run --dataset terminal-bench@2.0` | `benchmarks\runs\terminal-bench-2-reasoning-yolo-dry.agent-benchmark.summary.json` | reasoning | yes | dry-run command generated |
| GitHub bugs | `gemma-codex.cmd --reasoning --yolo exec` | `benchmarks\runs\github-bugs-reasoning-yolo-dry.agent-benchmark.summary.json` | reasoning | yes | dry-run command generated |

The external benchmark wrappers intentionally do not claim upstream scores until the required
official harnesses, Docker images, datasets, and prediction files are present and a non-dry run
completes.

## Reasoning On/Off Metrics: 2026-06-12

| Run | Endpoint | Reasoning | Total ms | First text ms | Tok/s | Prompt tok | Completion tok | Task success |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `agentbench-reasoning-off-8081` | `8081` | off | 265.35 | 265.28 | 22.61 | 17 | 6 | 1.00 |
| `agentbench-reasoning-on-8082` | `8082` | on | 362.59 | 362.53 | 16.55 | 17 | 6 | 1.00 |

Live `gemma-codex.cmd --reasoning --yolo exec "Reply with only OK."` completed once with exit
code 0 and did not stop for approval. A second measured variant using `--output-last-message`
wrote `OK, father.` but the wrapper command timed out, so it is not counted as a clean timing row.

| Run | Phase | Workloads | First byte ms | First text ms | Total ms | Tok/s | Prompt tok | Completion tok | Model calls | Peak RAM MB | Peak VRAM MB | Context | Cache hit | Cache hit rate | Test pass | Task success | Hallucinated tools | Failed JSON/tools | Failed tools | Summary |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| live-before-8081-direct-proxy | before | 1 | 257.24 | 257.43 | 257.55 | 23.30 | 17 | 6 | 1 | 30.00 | - | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\live-before-8081-direct-proxy.summary.json |
| live-before-8080-llama | before | 1 | 226.23 | 226.39 | 226.50 | 26.49 | 17 | 6 | 1 | 30.01 | - | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\live-before-8080-llama.summary.json |
| live-before-8082-reasoning-proxy | before | 1 | 3132.93 | 3133.11 | 3133.15 | - | 0 | 0 | 1 | 30.21 | - | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\live-before-8082-reasoning-proxy.summary.json |
| live-after-8082-reasoning-fast-path | after | 1 | 301.68 | 302.38 | 302.41 | 19.84 | 17 | 6 | 1 | 29.96 | - | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\live-after-8082-reasoning-fast-path.summary.json |
| live-after-final-8082-reasoning-fast-path | after | 1 | 265.26 | 265.85 | 265.88 | 22.57 | 17 | 6 | 1 | 30.34 | 16000.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\live-after-final-8082-reasoning-fast-path.summary.json |
| live-after-final2-8082-reasoning-fast-path | after | 1 | 266.65 | 267.07 | 267.11 | 22.46 | 17 | 6 | 1 | 30.12 | 16000.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\live-after-final2-8082-reasoning-fast-path.summary.json |
| live-after-final3-8082-reasoning-fast-path | after | 1 | 269.06 | 269.35 | 269.38 | 22.27 | 17 | 6 | 1 | 30.21 | 16000.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\live-after-final3-8082-reasoning-fast-path.summary.json |
| audit-before-raw-8080 | baseline | 1 | 401.95 | 403.15 | 403.19 | 14.88 | 17 | 6 | 1 | 30.20 | 15974.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-before-raw-8080.summary.json |
| audit-before-direct-8081 | baseline | 1 | 293.04 | 293.25 | 293.29 | 20.46 | 17 | 6 | 1 | 30.11 | 15974.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-before-direct-8081.summary.json |
| audit-before-reasoning-forced-8082 | before | 1 | 451.18 | 451.39 | 451.42 | - | 0 | 0 | 1 | 30.10 | 15974.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-before-reasoning-forced-8082.summary.json |
| audit-before-reasoning-fast-8082 | before | 1 | 317.27 | 317.46 | 317.49 | 18.90 | 17 | 6 | 1 | 30.07 | 15974.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-before-reasoning-fast-8082.summary.json |
| audit-after-raw-8080 | baseline | 1 | 256.81 | 257.03 | 257.10 | 23.34 | 17 | 6 | 1 | 30.29 | 15912.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-after-raw-8080.summary.json |
| audit-after-direct-8081 | after | 1 | 248.23 | 248.71 | 248.78 | 24.12 | 17 | 6 | 1 | 30.23 | 15912.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-after-direct-8081.summary.json |
| audit-after-reasoning-forced-8082 | after | 1 | 3958.52 | 3958.79 | 3958.85 | 1.52 | 112 | 6 | 1 | 30.21 | 15930.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-after-reasoning-forced-8082.summary.json |
| audit-after-reasoning-fast-8082 | after | 1 | 255.14 | 255.58 | 255.64 | 23.47 | 17 | 6 | 1 | 30.36 | 15930.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-after-reasoning-fast-8082.summary.json |
| audit-after-reasoning-forced-8082-warm | after | 1 | 20734.17 | 20734.41 | 20734.47 | 0.29 | 112 | 6 | 1 | 30.30 | 15935.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-after-reasoning-forced-8082-warm.summary.json |
| audit-after-reasoning-forced-8082-optimized | after | 1 | 19990.63 | 19990.78 | 19990.85 | 0.30 | 62 | 6 | 1 | 30.38 | 15949.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-after-reasoning-forced-8082-optimized.summary.json |
| audit-after-reasoning-forced-8082-strip-tools | after | 1 | 676.75 | 676.90 | 676.97 | 8.86 | 17 | 6 | 1 | 30.11 | 15950.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-after-reasoning-forced-8082-strip-tools.summary.json |
| audit-after-reasoning-forced-8082-strip-tools-warm | after | 1 | 24272.21 | 24272.37 | 24272.45 | 0.25 | 17 | 6 | 1 | 30.44 | 15950.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-after-reasoning-forced-8082-strip-tools-warm.summary.json |
| audit-after-direct-8081-check | after | 1 | 273.78 | 274.38 | 274.45 | 21.86 | 17 | 6 | 1 | 30.45 | 15949.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-after-direct-8081-check.summary.json |
| audit-final-raw-8080 | after | 1 | 633.06 | 634.31 | 634.36 | 9.46 | 17 | 6 | 1 | 30.38 | 15923.00 | - | - | - | - | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-final-raw-8080.summary.json |
| audit-final-direct-8081 | after | 1 | 270.48 | 271.18 | 271.27 | 22.12 | 17 | 6 | 1 | 30.46 | 15923.00 | - | - | - | - | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-final-direct-8081.summary.json |
| audit-final-reasoning-fast-8082 | after | 1 | 274.93 | 275.18 | 275.24 | 21.80 | 17 | 6 | 1 | 30.43 | 15923.00 | - | - | - | - | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-final-reasoning-fast-8082.summary.json |
| audit-final-reasoning-forced-marker-8082 | after | 1 | 3707.98 | 3708.13 | 3708.19 | 1.62 | 67 | 6 | 1 | 30.36 | 15923.00 | - | - | - | - | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-final-reasoning-forced-marker-8082.summary.json |
| audit-fullpass-raw-8080 | after | 1 | 236.63 | 236.90 | 236.96 | 25.32 | 17 | 6 | 1 | 30.61 | 15942.00 | - | - | - | - | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-fullpass-raw-8080.summary.json |
| audit-fullpass-direct-8081 | after | 1 | 229.74 | 229.97 | 230.02 | 26.08 | 17 | 6 | 1 | 30.36 | 15942.00 | - | - | - | - | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-fullpass-direct-8081.summary.json |
| audit-fullpass-reasoning-fast-8082 | after | 1 | 269.42 | 269.62 | 269.67 | 22.25 | 17 | 6 | 1 | 30.39 | 15942.00 | - | - | - | - | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-fullpass-reasoning-fast-8082.summary.json |
| audit-fullpass-reasoning-forced-marker-8082 | after | 1 | 383.11 | 383.23 | 383.28 | 15.65 | 54 | 6 | 1 | 30.09 | 15942.00 | - | - | - | - | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-fullpass-reasoning-forced-marker-8082.summary.json |
| audit-fullpass2-raw-8080 | after | 1 | 256.96 | 258.21 | 258.27 | 23.23 | 17 | 6 | 1 | 26663.37 | 15942.00 | - | - | - | - | 0.00 | 0 | 0 | 0 | benchmarks\runs\audit-fullpass2-raw-8080.summary.json |
| audit-fullpass2-direct-8081 | after | 1 | 243.93 | 244.81 | 244.87 | 24.50 | 17 | 6 | 1 | 26695.36 | 15942.00 | - | - | - | - | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-fullpass2-direct-8081.summary.json |
| audit-fullpass2-reasoning-fast-8082 | after | 1 | 277.68 | 278.06 | 278.12 | 21.57 | 17 | 6 | 1 | 26728.17 | 15942.00 | - | - | - | - | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-fullpass2-reasoning-fast-8082.summary.json |
| audit-fullpass2-reasoning-forced-marker-8082 | after | 1 | 384.14 | 384.25 | 384.31 | 15.61 | 54 | 6 | 1 | 26728.48 | 15942.00 | - | - | - | - | 1.00 | 0 | 0 | 0 | benchmarks\runs\audit-fullpass2-reasoning-forced-marker-8082.summary.json |
| agentbench-reasoning-off-8081 | after | 1 | 265.05 | 265.28 | 265.35 | 22.61 | 17 | 6 | 1 | 26689.96 | 15942.00 | - | - | - | - | 1.00 | 0 | 0 | 0 | benchmarks\runs\agentbench-reasoning-off-8081.summary.json |
| agentbench-reasoning-on-8082 | after | 1 | 362.04 | 362.53 | 362.59 | 16.55 | 17 | 6 | 1 | 26725.55 | 15942.00 | - | - | - | - | 1.00 | 0 | 0 | 0 | benchmarks\runs\agentbench-reasoning-on-8082.summary.json |
| bench-20260614-raw-8080 | after | 1 | 357.00 | 357.19 | 357.54 | 49.74 | 17 | 6 | 1 | 23339.20 | 15962.00 | - | - | - | - | 0.00 | 0 | 0 | 0 | benchmarks\runs\bench-20260614-raw-8080.summary.json |
| bench-20260614-direct-8081 | after | 1 | 267.32 | 267.48 | 267.62 | 22.42 | 17 | 6 | 1 | 23375.79 | 15962.00 | - | - | - | - | 0.00 | 0 | 0 | 0 | benchmarks\runs\bench-20260614-direct-8081.summary.json |
| bench-20260614-reasoning-fast-8082 | after | 1 | 262.25 | 263.96 | 264.01 | 22.73 | 17 | 6 | 1 | 23409.54 | 15962.00 | - | - | - | - | 0.00 | 0 | 0 | 0 | benchmarks\runs\bench-20260614-reasoning-fast-8082.summary.json |
| bench-20260614-reasoning-forced-8082 | after | 1 | 577.16 | 577.32 | 577.38 | 10.39 | 86 | 6 | 1 | 23530.94 | 15978.00 | - | - | - | - | 0.00 | 0 | 0 | 0 | benchmarks\runs\bench-20260614-reasoning-forced-8082.summary.json |
