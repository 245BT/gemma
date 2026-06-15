# Gamma Stall Recovery Design Spec

Date: 2026-06-12
Workspace: `C:\Users\Agent-1\Desktop\gemma`

## Objective

Fix the observed Gamma stall pattern where one terminal command can leave the agent idle for an
unbounded period, and improve the surrounding agent architecture enough that slow, ambiguous, or
research-heavy coding work has explicit recovery paths.

## Research Basis

- Composer 2 technical report, user-provided copy:
  `C:\Users\Agent-1\Downloads\Composer2.pdf`
- Local extracted references:
  - `docs/research/composer2-user-provided.txt`
  - `docs/research/composer-2-technical-report.txt`
  - `docs/research/swe-agent-agent-computer-interfaces.txt`
  - `docs/research/swe-bench-real-world-github-issues.txt`
  - `docs/research/terminal-bench-2.txt`
  - `docs/research/agentdojo-prompt-injection-agents.txt`
  - `docs/research/swe-evo-long-horizon-software-evolution.txt`

Relevant public ideas applied here:

- Composer 2: harness/model matching, self-summarization for long work, latency and token efficiency
  penalties, tool-call discipline, realistic coding evaluation, and failure analysis.
- SWE-agent: compact agent-computer interfaces, concise environment feedback, and guardrails for
  recovery.
- SWE-bench and SWE-evo: repository-scale coding tasks need tests, patches, multi-file navigation,
  and long-horizon evaluation.
- Terminal-Bench 2: command-line tasks need final-state tests, latency and cost tracking, and
  command failure analysis.
- AgentDojo: tool outputs and search results are untrusted data; tool calls need validation and
  prompt-injection resistance.

## Baseline Failure Modes

- Terminal command execution had no Gemma-owned process supervisor with idle timeout, hard timeout,
  tail capture, or recovery metadata.
- `AgentSupervisor.run` could repeat ineffective actions without feeding the model structured
  progress or stuck-state evidence.
- MCP subagent runtime exposed an empty tool registry instead of a bounded terminal tool.
- Codex launcher startup had bounded subprocess handling, but the launched Codex process could run
  indefinitely unless the operator intervened.
- Benchmark registry had no local deterministic suite for stalls, weak searches, invalid actions,
  large-file context pressure, large-repo navigation, and tool schema failures.

## Architecture Changes

- Add `TerminalCommandRunner` as the bounded terminal execution primitive.
- Add `ProgressTracker` as a supervisor-side low-progress detector.
- Add `build_default_tool_registry` with a schema-validated `terminal_command` tool.
- Inject structured `progress` into every supervisor model payload.
- Register bounded default tools for Gemma MCP subagents.
- Add optional launcher timeout through `GEMMA_CODEX_EXEC_TIMEOUT`.
- Add a deterministic `local-agent-behavior` benchmark suite and normalization path.

## Completion Gates

- Targeted red tests must fail before implementation and pass after implementation.
- Full Python unit tests must pass.
- `npm test` must pass.
- Local behavior benchmark must pass and write artifacts under `benchmarks/runs`.
- Final report must include measured facts, caveats, reproduction commands, and remaining risks.
