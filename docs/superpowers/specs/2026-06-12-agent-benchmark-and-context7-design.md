# Agent Benchmark And Context7 Design

## Goal

Add reproducible benchmark entrypoints for SWE-bench, Terminal-Bench, Terminal-Bench 2.0, and
manifest-driven GitHub bug fixing, then wire Context7 MCP beside DuckDuckGo for Gemma.

## Scope

The project will not reimplement upstream benchmark datasets. It will add local adapters that
generate and run official harness commands, normalize result files, and fail closed when required
tools such as Docker, Harbor, Terminal-Bench, or SWE-bench are not installed.

## Benchmark Suites

- `swe-bench`: wraps the official SWE-bench harness and predictions JSONL format.
- `terminal-bench`: wraps the `tb` Terminal-Bench CLI.
- `terminal-bench-2`: wraps Harbor with `harbor run --dataset terminal-bench@2.0`.
- `github-bugs`: runs local manifest tasks that clone a repository at a fixed commit, invoke the
  local Gemma Codex launcher, and run declared validation commands.

Each suite supports a `direct` mode and `reasoning` mode. Direct mode uses `.codex-local` and the
8081 response proxy. Reasoning mode uses `.codex-local-reasoning` and the 8082 reasoning proxy.
Both modes can be run with yolo enabled for non-interactive command execution.

## Context7 And DuckDuckGo

Context7 MCP is added through the official npm server package using `npx -y
@upstash/context7-mcp@latest`. DuckDuckGo remains the local controlled search MCP. Gemma's base
instructions require these as supporting tools when source freshness, current news, library docs,
API uncertainty, or user ambiguity can be resolved with evidence. They do not replace model
reasoning or user clarification when the missing information is genuinely about user intent.

DuckDuckGo gets a recency/freshness option so current-news queries can bias results toward recent
pages without weakening citation or prompt-injection handling.

## Yolo Behavior

`--yolo` must make launcher invocations non-interactive for approvals. The launcher should pass
Codex's bypass flag, explicit `--ask-for-approval never`, and `--sandbox danger-full-access` so
config defaults cannot silently keep the session in approval-prompt mode.

## Metrics

The benchmark wrapper records per-run metadata including suite, mode, yolo flag, command,
elapsed time, pass/fail status, stdout/stderr log paths, result path, and normalized score fields
when upstream results are available.

## Verification

Tests cover command generation, dependency fail-closed behavior, result normalization, Context7 MCP
config generation, DDG recency query building, and yolo argument expansion.
