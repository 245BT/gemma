# Gemma Parallel Agent Design

## Goal

Teach and enforce safe parallel work in the local Gemma Codex CLI runtime so Gemma uses bounded concurrent subagents and tool-capable work batches when tasks are independent, while preserving serial execution for dependent edits, commits, installs, and verification flows.

## Evidence Used

- OpenAI function calling documentation: models may call multiple functions in one turn; `parallel_tool_calls=false` prevents multiple tool calls; parallel function calling does not apply to built-in tools, and strict-mode behavior has caveats. Source: https://developers.openai.com/api/docs/guides/function-calling
- OpenAI tools documentation: Responses API agents can use built-in tools, function calling, tool search, and remote MCP servers. Source: https://developers.openai.com/api/docs/guides/tools
- OpenAI Codex CLI documentation through Context7 selected `/openai/codex`: Codex exposes MCP `enabled_tools`, model metadata, and a Responses request layer with `parallel_tool_calls`.
- OpenAI Codex public model catalog examples include `supports_parallel_tool_calls=true` for current OpenAI models. Source: https://github.com/openai/codex/blob/main/codex-rs/models-manager/models.json
- LLMCompiler proposes planner, task fetcher, and parallel executor components for parallel function calling, reporting lower latency and cost than sequential ReAct-style calling. Source: https://arxiv.org/abs/2312.04511
- W&D 2026 studies width scaling through parallel tool calling in deep research agents and reports fewer turns and better benchmark results from parallel width/depth tradeoffs. Source: https://arxiv.org/abs/2602.07359
- SWE-agent and OpenHands support the local premise that agent interface design, action protocols, safe execution environments, and evaluation loops materially affect software-agent performance. Sources: https://arxiv.org/abs/2405.15793 and https://arxiv.org/abs/2407.16741

## Current Runtime Context

The workspace already has the primitives needed for parallel execution:

- `gemma_agent/subagent.py` has `run_subagents(..., concurrent=True, max_workers=N)` backed by `ThreadPoolExecutor`.
- `gemma_agent/supervisor.py` already supports a `spawn_subagents` JSON action with optional `concurrent` and `max_workers`.
- `gemma_agent_mcp.py` exposes `gemma_run_subagents`, defaults to concurrent execution, and bounds workers.
- `setup_local_codex.py` currently writes the durable local Codex model catalog and base instructions but marks `supports_parallel_tool_calls` as false.

The gap is policy and enforcement, not a new theory of parallelism.

## Design

Use a safe compiler-style policy:

1. Require Gemma to identify independent work before acting.
2. Batch independent reads, searches, inspections, and narrow subagent tasks.
3. Dispatch those independent tasks concurrently with bounded workers.
4. Keep dependent operations serial when one step consumes another step's output or mutates shared state.
5. Report measured verification evidence before claiming speed or correctness.

## Runtime Behavior

### Prompt And Catalog Policy

`setup_local_codex.py` will update `build_base_instructions()` with concise parallelism rules:

- Prefer parallel batches for independent file reads, searches, checks, and subagent tasks.
- Use `gemma_run_subagents` or runtime subagent fanout for independent investigations and review tracks.
- Do not parallelize dependent commands, file edits that overlap, `git add`/`git commit`, installs, migrations, or test-after-edit loops.
- Bound fanout and summarize results before final claims.

The generated model catalog will set `supports_parallel_tool_calls` to `True` for the local Gemma model because the local Codex Responses path and Gemma agent runtime expose bounded parallel-capable tool/subagent interfaces. This does not mean every operation may run in parallel; it advertises capability while the runtime policy controls eligibility.

### Supervisor Enforcement

`gemma_agent/supervisor.py` will make multi-task `spawn_subagents` concurrent by default. If the model omits `concurrent`, a multi-task spawn runs concurrently with bounded workers. If the model explicitly sets `concurrent=false`, the supervisor allows it only for serial/dependency-heavy tasks and returns a policy error for obvious independent fanout once a small heuristic can identify that safely.

The first implementation will focus on defaulting to concurrency and making the policy visible in model payloads. It will not try to prove arbitrary dependency graphs from natural language because that would be brittle and likely unsafe.

### MCP Subagent Behavior

`gemma_agent_mcp.py` already defaults `gemma_run_subagents_response(..., concurrent=True)`. Tests and README text will lock that in as the recommended external interface for parallel Gemma-owned work.

## Safety Constraints

- No model weights, refusal-removal behavior, or uncensored behavior are changed.
- Tool results remain evidence, not instructions.
- Subagent outputs remain untrusted evidence when re-entering model context.
- Parallel work is bounded by existing `max_subagents`, `max_workers`, and task-length controls.
- File edits, commits, installs, migrations, and dependent verification remain serial unless a future explicit dependency graph proves safe independence.

## Testing

Add or update tests for:

- Base instructions mention safe parallel batching and forbid unsafe dependency parallelism.
- Generated model catalog sets `supports_parallel_tool_calls` to `True`.
- `spawn_subagents` with multiple tasks defaults to concurrent execution when the model omits `concurrent`.
- Explicit bounded worker behavior remains in place.
- MCP `gemma_run_subagents` keeps concurrent default behavior.

## Documentation

Update README guidance for changing agent behavior to explain:

- Parallelism is capability plus policy.
- Independent Gemma subagent tasks should use `gemma_run_subagents`.
- Dependent edits, commits, installs, and verification remain serial.

## Non-Goals

- Do not add a broad new orchestration framework.
- Do not implement speculative dependency-graph inference across arbitrary user tasks.
- Do not claim performance speedups without benchmarks.
- Do not modify local model weights or safety/refusal behavior.

