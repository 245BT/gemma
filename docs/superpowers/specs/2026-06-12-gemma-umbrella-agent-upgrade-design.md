# Gemma Umbrella Agent Upgrade Design

Date: 2026-06-12
Workspace: `C:\Users\Agent-1\Desktop\gemma`

## Purpose

Mirror the successful blockblast execution shape without copying its project: one umbrella
objective, independent subtracks, subagent delegation, review gates, and final measured evidence.

## Blockblast Pattern Applied

The inspected blockblast `.superpowers` artifact selected an umbrella-plus-independent-subspecs
shape. For Gemma, that translates to:

- One umbrella objective: make Gemma a faster, safer, stall-resistant coding and research agent.
- Independent tracks that can be audited or implemented without sharing write scopes.
- Controller-owned integration so subagents do not overwrite each other.
- Review after every task: spec compliance first, code quality second.
- Final verification with measurements, not status claims.

## Subtracks

1. Terminal and runtime control:
   - Bounded command execution.
   - Process-tree cleanup.
   - Stall, timeout, repeated-failure, and low-progress reporting.
   - Recovery metadata that reaches the next model turn.

2. Planning and supervisor behavior:
   - Structured `progress` payload.
   - Repeated action detection.
   - Honest final-claim validation.
   - Normal chat compatibility when no tool claim is made.

3. Research and benchmark method:
   - Composer 2 and related public papers stored locally.
   - Deterministic local benchmark for the reported failure modes.
   - External suite wrappers for SWE-bench, Terminal-Bench, Terminal-Bench 2, and GitHub-style bugs.
   - Failure taxonomy for stalls, searches, hallucinations, weak edits, weak testing, and recovery.

4. Context and large-repository handling:
   - Retrieval over whole-file loading.
   - Budgeted context packs.
   - Large-file and large-repo benchmark coverage.
   - Safe handling of untrusted evidence and search output.

5. Release evidence:
   - Tests, benchmark summaries, docs, reproduction commands.
   - Security and leak scan.
   - Remaining weakness list with no overclaiming.

## Execution Policy

- Use fresh subagents for bounded audits or implementation tasks.
- Give each subagent exact file scopes and expected output.
- Do not let subagents edit overlapping files in parallel.
- Controller integrates findings and runs final verification.
- Do not commit automatically while unrelated dirty workspace files exist.

## Acceptance Criteria

- Terminal stalls and child-process leaks are detected and recovered.
- A stalled `terminal_command` is a failed `ToolResult`, not a hidden success.
- Supervisor injects recovery events after stalls and repeated actions.
- Benchmarks run through reproducible commands with timeouts.
- Research artifacts are local and cited.
- Full Python and npm test suites pass.
- Local agent behavior benchmark passes and records before/after evidence.
