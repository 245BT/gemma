# Gemma Public-Release Redesign And Hardening Design

## Goal

Move this workspace from MVP-shaped local runtime to a public-release-quality architecture trajectory: tidy process artifacts, redesign the runtime spine, then harden concrete launch risks with tests, benchmarks, and a final evidence report.

This pass does not create npm packaging, git-based installation, hosted service deployment, or public multi-tenant access. It prepares the product internals so those can be considered later from a sound base.

## Current Evidence

- The workspace already has a local Gemma runtime, Codex launcher, direct proxy, reasoning proxy, Gemma-owned agent runtime, DuckDuckGo MCP, benchmarks, skills, and tests.
- Baseline verification before this design pass:
  - `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`: 471 tests, 0 failures.
  - `npm test`: 471 tests, 0 failures.
  - `.\verify-local-setup.ps1`: passed and reported `codex-cli 0.139.0`.
  - `npm exec --yes --package @openai/codex@0.139.0 -- codex --version`: `codex-cli 0.139.0`.
  - Process scan found exactly one `llama-server.exe`.
- Independent subagent audit found no material bug in `son` / `sonion` / `operator` launch parsing, but found missing behavioral coverage and docs.
- Independent security/process audit found launch-blocking risks for public-quality claims: terminal boundary, subprocess timeout cleanup, raw benchmark artifacts, path traversal in capability-card outputs, sanitizer coverage gaps, and raw invalid model-output retention.

## Design Order

The implementation order is:

1. **Tidy workspace structure.**
   Create a `blockblast`-style local `.superpowers` process tree, a root `plan.md`, and a clear split between durable docs under `docs/superpowers/` and ignored local evidence under `.superpowers/`.

2. **Runtime V2 spine.**
   Improve the product skeleton before applying isolated patches. This is not a destructive rewrite. It is a compatibility-preserving redesign that introduces clearer boundaries around:
   - `AgentSupervisor`
   - `ToolRegistry`
   - `ToolExecutor`
   - `TerminalRunner`
   - `SubAgent`
   - `MemoryStore`
   - `ContextBuilder`
   - `CitationManager`
   - `BenchmarkRunner`
   - `LaunchConfig`
   - `SafetyGuard`

3. **Hardening pass.**
   Apply TDD-backed fixes over the redesigned structure:
   - terminal execution boundary for local-trusted mode;
   - subprocess tree cleanup on benchmark timeouts;
   - benchmark/log redaction with hashes, lengths, and bounded tails;
   - run-id/path traversal guards for generated reports;
   - sanitizer coverage for local evidence artifacts;
   - invalid action redaction before refeeding model output;
   - single-llama behavioral cleanup tests;
   - launch dry-run or equivalent end-to-end shim matrix verification;
   - command documentation for `son`, `sonion`, `operator`, `--reasoning`, `--reasoing`, and `--yolo`;
   - Codex 0.139.0 provenance comparison and selected-binary manifest.

## Runtime Policy

This product remains a local trusted developer runtime for this launch pass. The current terminal/tool surface is not safe to expose to untrusted hosted users. A future hosted mode must disable terminal tools or run them inside a real OS/container sandbox before any hosted-public claim.

`--yolo` semantics must be made explicit. If non-yolo configs remain `approval_policy = "never"` and `sandbox_mode = "danger-full-access"`, the docs and verification must say plainly that all local commands are trusted-local operation and that `sonion --yolo` only changes confirmation behavior. If later the product needs safer defaults, that is a separate behavior change with tests.

## Output And Context Policy

The user goal is maximum useful work, not artificial short answers. The design preserves that by:

- keeping the Gemma context window at the current maximum used by this project;
- avoiding proxy-injected output token caps for ordinary model output;
- using auto-compaction/context packs for long work;
- using stall detection, hard timeouts, resumable tasks, and evidence summaries to prevent hangs;
- storing hashes, lengths, and bounded excerpts instead of raw sensitive output where artifacts may be shared.

This is a deliberate distinction: no artificial content cap for model usefulness, but strong process and artifact controls so long-running work remains operable.

## Workspace Tidiness

Target top-level shape:

```text
.superpowers/                  ignored local process/evidence trail
docs/superpowers/specs/         durable design specs
docs/superpowers/plans/         durable implementation plans
docs/superpowers/reports/       final evidence reports
gemma_agent/                    runtime library
gemma_reasoning/                reasoning helpers
benchmarks/                     reproducible benchmark code and summaries
scripts/                        maintenance and sanitizer scripts
tests/                          completion gate
plan.md                         current work controller
README.md                       public-facing local runtime documentation
```

Generated state, model files, local Codex homes, runtime logs, raw benchmark events, and `.superpowers` evidence remain ignored or sanitized before any shareable bundle.

## Testing And Review Discipline

Every behavior change follows TDD:

1. Add or update a focused failing test.
2. Run the focused test and confirm the expected failure.
3. Implement the smallest compatible fix.
4. Run the focused test and affected suite.
5. Run the full suite before completion claims.

Independent work should use subagents with disjoint scopes. Subagent reports are evidence, not truth; controller review and local verification remain required.

## Acceptance Criteria

- Workspace has a tidy `.superpowers` process tree and root `plan.md`.
- Runtime redesign plan identifies explicit interfaces and migration tasks instead of scattered one-off patches.
- Launch-blocking security/process findings are fixed or explicitly marked as future hosted-mode blockers.
- No raw prompt, secret, chain-of-thought-like invalid output, or full tool output is persisted in shareable artifacts by default.
- Benchmarks and capability-card outputs cannot escape their configured output directory through `run_id`.
- Benchmark subprocess timeouts terminate child process trees.
- Verification covers the requested command matrix and single-llama-server behavior.
- Codex 0.139.0 comparison is reproducible and records the selected binary/provenance.
- Final report includes tests run, before/after measurements where applicable, caveats, and final verdict.

## Risks

- A full destructive rewrite would risk breaking already-green behavior. This design uses a spine rewrite with compatibility tests.
- True hosted-public safety requires OS/container sandboxing and is out of scope for this pass.
- Removing all execution limits would reduce reliability. This design removes artificial output caps but keeps lifecycle controls.
