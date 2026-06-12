# Gemma Remeasure: 2026-06-12

This file records a controlled current-code microbenchmark. Single-run smoke rows in
`benchmarks/BENCHMARKS.md` are retained as raw artifacts and are not the basis for the aggregate
numbers below.

## Scope

The benchmark used two one-line workloads:

- `benchmarks/workloads/exact_fastpath.jsonl`: `Reply with only the digit 4.`
- `benchmarks/workloads/exact_forced_reasoning_graph.jsonl`: the same request plus the
  proxy-local `x_gemma_force_reasoning` benchmark marker, used only to force the reasoning proxy
  off its direct fast path.

Each group was run 5 times against the live local endpoints:

- `http://127.0.0.1:8080/v1/responses`: raw llama.cpp response endpoint.
- `http://127.0.0.1:8081/v1/responses`: direct response proxy.
- `http://127.0.0.1:8082/v1/responses`: reasoning proxy.

Raw per-run summaries and event logs are in `benchmarks/runs/`. Aggregate data is in
`benchmarks/runs/remeasure-2026-06-12.aggregate.json`.

## Aggregate Metrics

Reasoning proxy request-shape comparison:

| Comparison | n | Avg total ms before | Avg total ms after | Delta ms | Reduction | Ratio | Caveat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Forced graph request shape -> fast-path request shape on `8082` | 5 each | 413.31 | 329.29 | -84.02 | 20.33% | 1.26x | Current-code A/B only; ranges overlap. |

Observed ranges from `benchmarks/runs/remeasure-2026-06-12.aggregate.json`
using sample standard deviation:

- Forced graph total time: 370.01-534.49 ms, stdev 68.20 ms.
- Fast path total time: 280.33-412.68 ms, stdev 52.46 ms.

Do not use the retained aggregate forced-graph rows for token/sec or token-count comparison:
those rows were captured before graph response usage propagation and can report `0` or blank
usage fields. New audit rows generated after the metadata fix include graph usage when the
underlying response client provides it. The latency comparison is still useful because it is
measured at the HTTP client boundary.

Endpoint aggregate metrics:

| Group | n | Avg first byte ms | Avg first text ms | Avg total ms | Total ms range | Total ms stdev | Avg tok/s | Prompt tok avg | Completion tok avg | Peak RAM MB avg | Peak VRAM MB avg | Task success |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw llama.cpp `8080` | 5 | 257.18 | 258.29 | 258.34 | 252.10-265.20 | 4.95 | 23.23 | 17.0 | 6.0 | 30.17 | 16003.00 | 1.00 |
| Direct proxy `8081` | 5 | 273.36 | 274.05 | 274.09 | 246.00-333.90 | 35.11 | 22.15 | 17.0 | 6.0 | 30.19 | 16003.00 | 1.00 |
| Reasoning proxy forced graph `8082` | 5 | 413.09 | 413.27 | 413.31 | 370.01-534.49 | 68.20 | - | 0.0 | 0.0 | 30.16 | 16003.00 | 1.00 |
| Reasoning proxy fast path `8082` | 5 | 328.91 | 329.24 | 329.29 | 280.33-412.68 | 52.46 | 18.56 | 17.0 | 6.0 | 30.24 | 16003.00 | 1.00 |

## Prompt Token Remeasure

The 30% prompt-token reduction is supported only against the selected 640-token legacy
base-instruction candidate found in `.codex-local/sessions/2026/06/11/rollout-2026-06-11T03-59-18-019eb656-1fcd-7092-877f-2f8c4e5bc750.jsonl`.
Current base-instruction prompt size is 448 cl100k tokens. This is not a universal comparison
against every older local session.

| Prompt artifact | Chars | Words | cl100k tokens | Change |
| --- | ---: | ---: | ---: | ---: |
| Selected legacy base instructions | 3306 | 504 | 640 | - |
| Current `setup_local_codex.build_base_instructions()` | 2345 | 350 | 448 | -30.0% |

Caveats:

- Token counts use `tiktoken` `cl100k_base`, not Gemma's tokenizer.
- This measures the base-instruction string only, not total runtime prompt tokens.
- Other legacy candidates in local sessions measured 594, 413, 102, and 56 cl100k tokens.
- This supports a prompt-size comparison for the selected artifact; quality preservation depends on
  the behavior tests and task benchmarks.

## Reproduction Commands

Start the local runtime first:

```powershell
.\start-gemma-runtime.ps1
```

Run the repeated measurements:

```powershell
$ErrorActionPreference = 'Stop'
$python = '.\.venv\Scripts\python.exe'
$report = 'benchmarks\REMEASURE_2026-06-12.md'
$runs = @(
  @{label='remeasure-raw-8080'; endpoint='http://127.0.0.1:8080/v1/responses'; workload='benchmarks\workloads\exact_fastpath.jsonl'; phase='baseline'},
  @{label='remeasure-direct-proxy-8081'; endpoint='http://127.0.0.1:8081/v1/responses'; workload='benchmarks\workloads\exact_fastpath.jsonl'; phase='baseline'},
  @{label='remeasure-reasoning-graph-forced-8082'; endpoint='http://127.0.0.1:8082/v1/responses'; workload='benchmarks\workloads\exact_forced_reasoning_graph.jsonl'; phase='before'},
  @{label='remeasure-reasoning-fastpath-8082'; endpoint='http://127.0.0.1:8082/v1/responses'; workload='benchmarks\workloads\exact_fastpath.jsonl'; phase='after'}
)
foreach ($run in $runs) {
  foreach ($i in 1..5) {
    & $python benchmarks\bench_runtime.py --workload $run.workload --endpoint $run.endpoint --phase $run.phase --run-label "$($run.label)-r$i" --markdown $report --timeout 120
  }
}
```

Regenerate the aggregate JSON:

```powershell
.\.venv\Scripts\python.exe benchmarks\aggregate_remeasure.py --output benchmarks\runs\remeasure-2026-06-12.aggregate.json
```

## Raw Repeated Runs

| Run | Phase | Workloads | First byte ms | First text ms | Total ms | Tok/s | Prompt tok | Completion tok | Model calls | Peak RAM MB | Peak VRAM MB | Context | Cache hit | Cache hit rate | Test pass | Task success | Hallucinated tools | Failed JSON/tools | Failed tools | Summary |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| remeasure-raw-8080-r1 | baseline | 1 | 263.91 | 265.15 | 265.20 | 22.62 | 17 | 6 | 1 | 30.20 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-raw-8080-r1.summary.json |
| remeasure-raw-8080-r2 | baseline | 1 | 254.87 | 256.11 | 256.15 | 23.42 | 17 | 6 | 1 | 30.19 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-raw-8080-r2.summary.json |
| remeasure-raw-8080-r3 | baseline | 1 | 251.19 | 252.05 | 252.10 | 23.80 | 17 | 6 | 1 | 30.15 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-raw-8080-r3.summary.json |
| remeasure-raw-8080-r4 | baseline | 1 | 259.75 | 260.78 | 260.84 | 23.00 | 17 | 6 | 1 | 30.21 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-raw-8080-r4.summary.json |
| remeasure-raw-8080-r5 | baseline | 1 | 256.16 | 257.36 | 257.39 | 23.31 | 17 | 6 | 1 | 30.07 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-raw-8080-r5.summary.json |
| remeasure-direct-proxy-8081-r1 | baseline | 1 | 274.97 | 275.78 | 275.81 | 21.75 | 17 | 6 | 1 | 30.23 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-direct-proxy-8081-r1.summary.json |
| remeasure-direct-proxy-8081-r2 | baseline | 1 | 333.27 | 333.86 | 333.90 | 17.97 | 17 | 6 | 1 | 30.38 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-direct-proxy-8081-r2.summary.json |
| remeasure-direct-proxy-8081-r3 | baseline | 1 | 257.57 | 258.43 | 258.46 | 23.21 | 17 | 6 | 1 | 30.20 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-direct-proxy-8081-r3.summary.json |
| remeasure-direct-proxy-8081-r4 | baseline | 1 | 245.01 | 245.95 | 246.00 | 24.39 | 17 | 6 | 1 | 30.09 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-direct-proxy-8081-r4.summary.json |
| remeasure-direct-proxy-8081-r5 | baseline | 1 | 255.97 | 256.25 | 256.28 | 23.41 | 17 | 6 | 1 | 30.06 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-direct-proxy-8081-r5.summary.json |
| remeasure-reasoning-graph-forced-8082-r1 | before | 1 | 534.22 | 534.43 | 534.49 | - | 0 | 0 | 1 | 30.36 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-reasoning-graph-forced-8082-r1.summary.json |
| remeasure-reasoning-graph-forced-8082-r2 | before | 1 | 389.61 | 389.82 | 389.87 | - | 0 | 0 | 1 | 30.25 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-reasoning-graph-forced-8082-r2.summary.json |
| remeasure-reasoning-graph-forced-8082-r3 | before | 1 | 369.82 | 369.98 | 370.01 | - | 0 | 0 | 1 | 30.01 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-reasoning-graph-forced-8082-r3.summary.json |
| remeasure-reasoning-graph-forced-8082-r4 | before | 1 | 383.26 | 383.45 | 383.48 | - | 0 | 0 | 1 | 30.20 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-reasoning-graph-forced-8082-r4.summary.json |
| remeasure-reasoning-graph-forced-8082-r5 | before | 1 | 388.51 | 388.66 | 388.69 | - | 0 | 0 | 1 | 30.00 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-reasoning-graph-forced-8082-r5.summary.json |
| remeasure-reasoning-fastpath-8082-r1 | after | 1 | 280.05 | 280.29 | 280.33 | 21.40 | 17 | 6 | 1 | 30.32 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-reasoning-fastpath-8082-r1.summary.json |
| remeasure-reasoning-fastpath-8082-r2 | after | 1 | 291.16 | 291.48 | 291.53 | 20.58 | 17 | 6 | 1 | 30.14 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-reasoning-fastpath-8082-r2.summary.json |
| remeasure-reasoning-fastpath-8082-r3 | after | 1 | 412.32 | 412.63 | 412.68 | 14.54 | 17 | 6 | 1 | 30.30 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-reasoning-fastpath-8082-r3.summary.json |
| remeasure-reasoning-fastpath-8082-r4 | after | 1 | 341.18 | 341.50 | 341.54 | 17.57 | 17 | 6 | 1 | 30.25 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-reasoning-fastpath-8082-r4.summary.json |
| remeasure-reasoning-fastpath-8082-r5 | after | 1 | 319.83 | 320.31 | 320.36 | 18.73 | 17 | 6 | 1 | 30.17 | 16003.00 | - | - | - | 1.00 | 1.00 | 0 | 0 | 0 | benchmarks\runs\remeasure-reasoning-fastpath-8082-r5.summary.json |
