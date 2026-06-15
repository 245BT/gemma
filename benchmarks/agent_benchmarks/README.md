# Agent Benchmark Wrappers

These wrappers set up the benchmark command lines used by coding agents without copying or
reimplementing upstream benchmark datasets.

## Suites

- `swe-bench`: official SWE-bench harness. Requires a predictions JSONL file.
- `terminal-bench`: official `tb` CLI harness.
- `terminal-bench-2`: Harbor harness with `terminal-bench@2.0`.
- `github-bugs`: local manifest-driven GitHub bug fixing through `gemma-codex.cmd`.
- `local-agent-behavior`: deterministic runtime checks for stalls, context, tools, and research artifacts.
- `local-repo-fix`: generated CPU-only repo-fix tasks with fail-to-pass/pass-to-pass metrics.

## Modes

- `--mode direct`: uses `.codex-local` and the direct proxy on port 8081.
- `--mode reasoning`: uses `.codex-local-reasoning` and the reasoning proxy on port 8082.
- `--yolo`: marks the run as yolo and, for `github-bugs`, passes `--yolo` to the Gemma launcher.

## Examples

```powershell
.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.runner --list

.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.runner `
  --suite terminal-bench-2 `
  --mode reasoning `
  --yolo `
  --run-id tb2-gemma-reasoning `
  --dry-run

.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.runner `
  --suite swe-bench `
  --mode reasoning `
  --yolo `
  --predictions-path benchmarks\predictions\gemma-swe.jsonl `
  --run-id swe-verified-gemma

.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.runner `
  --suite local-repo-fix `
  --mode direct `
  --run-id local-repo-fix-oracle `
  --out-dir benchmarks\runs `
  --timeout 20
```

Dry-run writes a summary JSON but does not invoke the external harness.
