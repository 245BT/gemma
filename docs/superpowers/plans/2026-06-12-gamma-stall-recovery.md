# Gamma Stall Recovery Execution Plan

Date: 2026-06-12

## Status

- [x] Read mandatory workflow skills and record mandatory use in `AGENTS.md`.
- [x] Read the user-provided Composer 2 PDF and extract it to local text.
- [x] Download and extract related public agent-evaluation research papers.
- [x] Capture baseline test state and add targeted failing tests for stall behavior.
- [x] Implement bounded terminal execution.
- [x] Implement supervisor progress and low-progress recovery payloads.
- [x] Expose bounded terminal behavior through the default Gemma tool registry and MCP runtime.
- [x] Add launcher timeout control for Codex execution.
- [x] Add deterministic local benchmark coverage for stall recovery and agent behavior.
- [x] Fix benchmark artifact placement so detailed results stay under `benchmarks/runs`.
- [x] Run full final verification.
- [x] Complete final code-review pass and incorporate findings.

## Work Boundaries

Primary implementation files:

- `gemma_agent/terminal.py`
- `gemma_agent/progress.py`
- `gemma_agent/tools.py`
- `gemma_agent/supervisor.py`
- `gemma_agent_mcp.py`
- `launch_gemma_codex.py`
- `benchmarks/agent_benchmarks/local_behavior.py`
- `benchmarks/agent_benchmarks/registry.py`
- `benchmarks/agent_benchmarks/results.py`
- `benchmarks/agent_benchmarks/runner.py`

Primary tests:

- `tests/test_gemma_agent_runtime.py`
- `tests/test_agent_benchmarks.py`
- `tests/test_launch_gemma_codex.py`
- `tests/test_gemma_agent_mcp.py`

## Verification Commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
npm test
.\.venv\Scripts\python.exe -m benchmarks.agent_benchmarks.runner --suite local-agent-behavior --mode direct --run-id gamma-stall-after --out-dir benchmarks\runs --timeout 30
```

## Known Caveats

- This is an architecture and harness improvement, not model training.
- The deterministic local benchmark proves recovery primitives and harness behavior, not full
  SWE-bench accuracy.
- Existing unrelated dirty files in the workspace were not reverted or normalized.
