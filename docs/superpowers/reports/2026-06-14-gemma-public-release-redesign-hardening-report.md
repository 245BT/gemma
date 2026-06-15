# Gemma Public Release Redesign Hardening Report

Date: 2026-06-14

## 1. Executive Summary

Verdict: pass for the requested local-trusted public-release hardening scope.

The workspace was tidied with `.superpowers` planning artifacts, a root `plan.md`, and a concrete design/implementation plan. The implementation added a runtime policy spine, safer benchmark subprocess handling, redacted benchmark logs, capability-card path safety, broader sanitizer coverage, invalid model-output redaction, launcher dry-run verification, Codex provenance reporting, and README command-matrix documentation.

This pass does not claim hosted multi-tenant safety, npm/git distribution, or public benchmark superiority.

## 2. Research Used

- Official Codex CLI documentation was fetched through Context7 for `/openai/codex`.
- Temporary Codex 0.139 comparison was verified with:
  `npm exec --yes --package @openai/codex@0.139.0 -- codex --version`
- Result: `codex-cli 0.139.0`.

## 3. Baseline Measurements

Before this pass:

| Check | Result |
| --- | --- |
| `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` | 471 tests OK |
| `npm test` | 471 tests OK |
| `.\verify-local-setup.ps1` | Passed, Codex 0.139.0 |
| `npm exec --yes --package @openai/codex@0.139.0 -- codex --version` | `codex-cli 0.139.0` |
| `llama-server.exe` count | 1 |

## 4. Bugs And Leaks Found

- Local trusted terminal tools had no explicit public profile boundary.
- Benchmark subprocess timeout handling did not guarantee child process-tree cleanup.
- Benchmark stdout/stderr logs could persist raw secrets or private-key text.
- Capability-card `run_id` values could influence output paths.
- Sanitizer coverage missed project benchmark logs and runtime logs.
- Invalid model output was re-fed into later model payloads as full raw text.
- Launch behavior had no no-start dry-run path for the full `son`/`sonion`/`operator` matrix.
- Generated verification reported Codex version but not hash/provenance.

## 5. Design Changes

- Added `RuntimeConfig.runtime_profile`, `terminal_tools_enabled`, and `RuntimeConfig.public_local()`.
- Wired runtime config into `build_default_tool_registry()` and MCP subagents.
- Added `benchmarks.agent_benchmarks.subprocesses` for bounded subprocess execution, child-tree cleanup, redaction, tails, lengths, and hashes.
- Replaced benchmark runner and local repo-fix direct subprocess calls with bounded subprocess execution.
- Preserved benchmark output metadata while writing redacted tails instead of raw streams.
- Sanitized capability-card filenames while preserving original `run_id` in JSON payloads.
- Extended local artifact sanitizer patterns for benchmark and runtime logs with root-containment checks.
- Replaced invalid-action raw payload refeed with `raw_preview`, `raw_length`, and `raw_sha256`.
- Added launcher dry-run JSON output under `GEMMA_CODEX_DRY_RUN=1`.
- Added Codex selected-binary provenance helper and generated verifier SHA/version/package output.
- Documented command matrix and local trusted scope in `README.md`.

## 6. Speed Measurements

No live model speed benchmark was rerun in this pass. Verification timings:

| Command | Result |
| --- | --- |
| Focused suites | 336 tests OK in 18.606s |
| Full Python discovery | 488 tests OK in 18.998s |
| `npm test` | 488 tests OK in 18.947s |
| `.\verify-local-setup.ps1` | Passed in about 2.6s |

## 7. Context Measurements

- Context window remains `262144`.
- Auto-compact token limit remains `180000`.
- No output-token cap was added.
- Benchmark logs now retain redacted tails, original lengths, and SHA-256 hashes instead of raw full streams.

## 8. Gemma Sub-Agent Implementation Status

- Copernicus implemented capability-card path-safety tests and changes. Main-agent review passed and `tests.test_capability_card` passed.
- Dirac implemented sanitizer coverage tests and changes. Main-agent review passed and `tests.test_sanitize_local_artifacts` passed.
- Main agent independently implemented runtime policy, benchmark subprocess hardening, invalid-output redaction, launcher dry-run/provenance, README docs, regeneration, and final verification.

## 9. DuckDuckGo Verification

DuckDuckGo runtime behavior was not changed in this pass. Full verification included the DuckDuckGo test module as part of the 488-test suite.

## 10. Tests Run

| Command | Result |
| --- | --- |
| `.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime tests.test_gemma_agent_mcp tests.test_agent_benchmarks tests.test_capability_card tests.test_sanitize_local_artifacts tests.test_launch_gemma_codex tests.test_setup_local_codex -v` | 336 tests OK |
| `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` | 488 tests OK |
| `npm test` | 488 tests OK |
| `.\verify-local-setup.ps1` | Passed |
| `npm exec --yes --package @openai/codex@0.139.0 -- codex --version` | `codex-cli 0.139.0` |

Dry-run command matrix was verified for:

- `son`
- `sonion`
- `operator`
- `son --reasoing`
- `sonion --reasoning`
- `operator --reasoing`
- `son --reasoing --yolo`
- `sonion --reasoning --yolo`
- `operator --reasoing --yolo`
- `son --yolo`
- `sonion --yolo`
- `operator --yolo`

The live process check found one `llama-server.exe` at:
`C:\Users\Agent-1\Desktop\gemma\tools\llama.cpp\llama-server.exe`.

## 11. Before/After Benchmark Table

No live model benchmark was rerun. Test coverage increased:

| Phase | Python tests | npm tests | Notes |
| --- | ---: | ---: | --- |
| Before | 471 OK | 471 OK | Baseline before hardening |
| After | 488 OK | 488 OK | Added release-hardening coverage |

## 12. Remaining Risks

- This is still a local trusted workstation runtime, not a hosted untrusted-user service.
- Terminal execution remains enabled for the default local trusted profile.
- External SWE-bench, Terminal-Bench, OSWorld, and live model benchmark claims were not rerun.
- The repository had many unrelated pre-existing dirty files; this pass did not revert or audit all of them.
- No npm/git install packaging was added by request.

## 13. Reproduction Commands

```powershell
.\.venv\Scripts\python.exe setup_local_codex.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
npm test
.\verify-local-setup.ps1
npm exec --yes --package @openai/codex@0.139.0 -- codex --version
$env:GEMMA_CODEX_DRY_RUN=1
sonion --reasoning --yolo exec "inspect the workspace"
Remove-Item Env:\GEMMA_CODEX_DRY_RUN
Get-Process llama-server -ErrorAction SilentlyContinue | Select-Object Id,Path
```

## 14. Final Verdict

Pass for local-trusted public-release hardening and workspace tidiness. Partial for broader public launch only because hosted security, package distribution, and live public benchmarks remain explicitly out of scope.
