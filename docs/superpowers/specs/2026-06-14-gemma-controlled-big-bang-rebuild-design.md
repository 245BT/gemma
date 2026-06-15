# Gemma Controlled Big-Bang Rebuild Design

## Decision

Gemma will move from an MVP-shaped local agent into a production-readiness architecture through a controlled big-bang rewrite. "Big bang" means the weak core shape is replaced deliberately instead of patched around. "Controlled" means the rewrite is bounded by existing behavior tests, new failing tests, benchmark baselines, Codex 0.139 public-behavior comparison, leak checks, and a release evidence trail.

The implementation order is:

1. **A: Agent runtime core**
2. **B: Launch, proxy, and Codex command layer**
3. **C: Workspace, release evidence, and tidiness**

Those tracks are necessary but not sufficient. The rebuild also includes benchmark gates, security/leak gates, public-claim gates, and a final command-matrix verification across `son`, `sonion`, `operator`, reasoning variants, and yolo variants.

## Why This Is Not A Blind Rewrite

The rewrite starts from measured facts:

- Current unit baseline: `python -m unittest discover -s tests -v` passed 488 tests on June 14, 2026.
- Current local behavior baseline: `benchmarks.agent_benchmarks.runner --suite local-agent-behavior` passed 9/9 on June 14, 2026.
- Current serious benchmark evidence is mixed: one retained live repo-fix run passed 3/3, while a later retained run patched 0/3. The rewrite must not claim reliable coding-agent success until this is remeasured on the current tree.
- The current runtime has real behavior but weak boundaries: supervisor, prompt assembly, budget hints, tool execution, terminal evidence, and claim verification are coupled.
- The current launcher/proxy layer has useful commands but weak profile separation, shallow health checks, buffered streaming paths, and duplicated generated config.
- The current workspace has useful evidence but is not release-tidy: local Codex artifacts, benchmark runs, generated tools, and docs are mixed too closely.

The old system remains the behavioral oracle where it is proven correct. The new system replaces internals behind compatibility interfaces, then deletes or archives the old paths only after tests and command verification prove the replacement.

## Scope

### A: Agent Runtime Core

Replace the monolithic MVP runtime with focused modules:

- `AgentRunLoop`: owns the step loop, stop reasons, iteration state, and recovery decisions.
- `ModelGateway`: owns model request construction, streaming support, retries, and response metadata.
- `ActionCodec`: parses model actions, validates shape, rejects malformed tool calls, and reports structured parse failures.
- `BudgetManager`: enforces runtime budgets for iterations, tool calls, subagents, terminal time, context, and evidence. Budgets are profiles, not arbitrary low caps.
- `PromptAssembler` and `ContextBudgeter`: build bounded prompts from memory, files, tool results, progress, and evidence using explicit priorities.
- `ToolCatalog`, `ToolPolicy`, `ArgumentValidator`, and `ToolRunner`: separate tool registration, safety policy, schema validation, and execution.
- `TerminalSessionManager`: runs shell commands with timeouts, process tracking, streaming output, and reliable completion status.
- `EvidenceLedger`: records verified facts from files, commands, tests, benchmarks, and tool outputs without storing hidden reasoning.
- `ClaimVerifier`: blocks completion claims that do not match fresh evidence.
- `SubagentScheduler`: dispatches independent tasks with deadlines, ownership, cancellation, and result fan-in.

The public agent API must stay stable enough for the MCP server, CLI launchers, tests, and benchmark harness to keep working during the migration.

### B: Launch, Proxy, And Codex Command Layer

Rebuild the product-facing command layer around explicit profiles:

- `son`: normal local agent profile.
- `sonion`: reasoning-enabled profile.
- `operator`: operator-style command profile.
- `--reasoning`: selects the reasoning proxy/model path.
- `--yolo`: selects `approval_policy = "never"` and `sandbox_mode = "danger-full-access"` only when requested.

Plain commands and yolo commands must not collapse into the same effective configuration. Health checks must verify the expected local server identity, not just any HTTP 200 from `/v1/models`. The proxy layer must support real streaming instead of buffering long responses. The launcher must handle public Codex CLI 0.139 behavior, including the `exec` alias `e`, while avoiding proprietary prompt or implementation copying.

The Codex comparison is limited to public behavior, public documentation, and reproducible local probes. It may copy interface lessons and command ergonomics. It must not copy private prompts, closed-source logic, or confidential material.

### C: Workspace, Release Evidence, And Tidiness

Tidy the workspace toward the `blockblast` discipline:

- Keep root files intentional: project docs, source, tests, scripts, benchmark definitions, and release controller files.
- Keep `.superpowers/` as local workflow state and evidence, ignored by git.
- Keep `docs/superpowers/specs/`, `docs/superpowers/plans/`, and `docs/superpowers/reports/` as the durable design/plan/report trail.
- Move retained benchmark evidence into sanitized, documented locations.
- Keep generated logs, Codex sessions, local model files, downloaded tools, and raw benchmark runs ignored or explicitly redacted.
- Split oversized tests and modules when the rewrite touches them.
- Maintain `plan.md` as the top-level controller pointing to the active spec and future implementation plans.

## Limits Policy

Gemma should not contain arbitrary MVP caps that stop useful work early. The production design should use maximum practical local profiles, continuation, checkpoints, resumable runs, streaming, and evidence-aware compaction.

The system must not promise literal unlimited tokens, characters, words, or runtime. Physical model context, host memory, shell process limits, HTTP timeouts, and user cost constraints still exist. The correct behavior is to remove artificial low ceilings, enforce explicit high-cap profiles, continue across turns when needed, and report when a hard runtime limit is reached.

## Data Flow

1. The launcher selects a command profile and validates effective config.
2. The proxy/model layer exposes an identity-checked model endpoint.
3. `AgentRunLoop` creates a run state and asks `PromptAssembler` for the first prompt.
4. `ContextBudgeter` selects memory, files, progress, and evidence under the active budget profile.
5. `ModelGateway` sends the request and streams or returns model output with metadata.
6. `ActionCodec` parses a response into assistant text, tool calls, or structured errors.
7. `ToolRunner` executes validated tools under `ToolPolicy`; terminal work goes through `TerminalSessionManager`.
8. `EvidenceLedger` records verified command/file/test/benchmark facts.
9. `BudgetManager` decides whether to continue, recover, checkpoint, spawn subagents, or stop.
10. `ClaimVerifier` checks final claims against `EvidenceLedger` before any completion report.

## Testing Strategy

Every production behavior change must follow TDD:

- Write or update a focused failing test first.
- Confirm it fails for the expected reason.
- Implement the smallest production change that makes it pass.
- Run the focused test.
- Run the affected suite.
- Preserve the full-suite command as a final gate.

Required test groups:

- Runtime unit tests for budget enforcement, action parsing, context selection, evidence recording, claim blocking, and subagent scheduling.
- Terminal tests for timeout, streaming, process completion, background-process handling, and shell error reporting.
- Launcher tests for profile separation, `exec` and `e` handling, yolo semantics, reasoning selection, identity checks, and dry-run matrix output.
- Proxy tests for streaming, error propagation, request redaction, and upstream identity mismatch.
- Sanitizer tests for Codex session files, benchmark artifacts, logs, patch files, and nested Terminal-Bench outputs.
- Benchmark harness tests that distinguish oracle success from live Gemma success.

## Verification Gates

No public-readiness claim is allowed until all applicable gates pass on fresh output:

- Full unit test suite.
- Focused tests for each rewritten subsystem.
- Local behavior benchmark rerun on current code.
- At least one current live local repo-fix benchmark rerun, reported honestly even if it fails.
- Launch dry-run matrix for `son`, `sonion`, `operator`, reasoning variants, yolo variants, and reasoning+yolo variants.
- Runtime process check proving exactly one `llama-server` instance when required.
- Sanitizer scan showing no raw local Codex histories, sessions, prompts, logs, or benchmark raw outputs in shareable artifacts.
- Codex 0.139 temporary CLI comparison using public commands and docs.
- Security scan for secrets, raw prompt leakage, hidden chain-of-thought persistence, unsafe subprocess calls, missing timeouts, and untrusted search/tool input.

## Migration Strategy

The rewrite should happen in a dedicated worktree after the implementation plan is approved. The old runtime remains available until the new runtime passes compatibility tests. Migration proceeds by adapter:

1. Add new module skeletons and tests without deleting old behavior.
2. Route one behavior at a time through the new module.
3. Keep compatibility shims for existing MCP, CLI, and benchmark callers.
4. Delete or archive old code only after tests prove the new path owns the behavior.
5. Keep release notes and the final report synchronized with measured evidence.

## Non-Goals

- No npm publishing or public installer in this phase.
- No hosted multi-tenant service claim.
- No model-weight changes.
- No change to the user's uncensored local model behavior unless explicitly requested later.
- No copying proprietary Codex implementation, prompts, or confidential material.
- No success claim based only on mocks, dry-runs, or old benchmark artifacts.

## Acceptance Criteria

The rebuild is considered a pass only when:

- The active architecture is modular and the old monolithic responsibilities are either replaced or explicitly quarantined.
- A/B/C tracks are implemented in order, with cross-cutting security and benchmark gates.
- Root workspace structure is materially tidier and release evidence is separated from local/private artifacts.
- Plain, reasoning, yolo, and reasoning+yolo command variants have distinct verified behavior.
- Agent limits are explicit high-cap profiles with continuation paths, not arbitrary MVP cutoffs.
- Fresh tests and benchmarks are reported with commands, dates, pass rates, latency where available, and caveats.
- The final report includes the required AGENTS.md sections and marks the verdict as pass, partial pass, or fail.

## Spec Self-Review

- Placeholder scan: no incomplete requirement markers are present.
- Consistency scan: the order A, then B, then C matches the user's confirmed direction and the migration strategy.
- Scope scan: this is intentionally an umbrella design. Implementation must be split into separate plans or plan phases so each phase produces testable software.
- Ambiguity scan: "big bang" is defined as architecture replacement, while "not blind" is defined as tests, baselines, adapters, and verification gates.
