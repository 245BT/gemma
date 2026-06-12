# Gemma Agent Runtime Redesign

## Goal

Redesign the local Gemma coding and research agent so Gemma owns the agent loop, tool protocol, context selection, evidence tracking, and Gemma subagent delegation. Codex may still launch the local runtime, but the target behavior is `Gemma supervisor -> Gemma subagents -> external tool executor -> verified evidence`, not Codex pretending Gemma has tools.

## Non-Negotiable Constraints

- Preserve the uncensored model path. Do not add a moderation layer, refusal wrapper, safety policy prompt, or model-behavior restriction.
- Tool safety is runtime safety: schema validation, path controls, timeouts, audit events, and untrusted-output handling.
- Gemma must not claim a tool ran unless the runtime executed a registered tool and returned a structured result.
- Research claims must cite public papers, official docs, or other reputable public sources. Unverified claims must be marked unverified.
- Optimization claims require before/after measurements in `benchmarks/BENCHMARKS.md`.
- Do not copy proprietary prompts or closed-source implementation details from OpenAI, Cursor, Anthropic, Google, DeepSeek, Qwen, ByteDance, Alibaba, or any other large lab.

## Architecture

The redesigned runtime adds a focused `gemma_agent` package:

- `Config`: single source for model slug, endpoints, ports, timeouts, benchmark paths, and context limits.
- `ModelClient`: local Responses API client used by the supervisor and Gemma subagents.
- `ToolRegistry`: allowlisted tools with strict argument schemas.
- `ToolExecutor`: validates action JSON, executes tools externally, applies timeouts, records evidence, and returns structured tool results.
- `AgentSupervisor`: runs the Gemma action loop, dispatches tool calls, spawns Gemma subagents, and emits thinking summaries.
- `SubAgent`: isolated Gemma worker context with a typed task/result contract.
- `MemoryStore`: hierarchical session, project, scratchpad, evidence, and long-term cache records with deduplication.
- `ContextBuilder`: builds compact context packs instead of stuffing full unrelated files.
- `CitationManager`: records source URLs and verifies that final claims can point to evidence.
- `SafetyGuard`: validates paths and wraps untrusted tool/search output without changing model content policy.

The existing direct proxy and reasoning proxy stay available during migration. The new runtime is introduced as a tested library first, then connected to launch/setup only after benchmark and behavior tests pass.

## Tool Protocol

Gemma actions are strict JSON objects. The minimum action types are:

- `final`: user-visible final answer.
- `thinking_summary`: concise summary with Goal, Plan, Evidence, Current finding, Confidence, and Next action.
- `tool_call`: request to run one allowlisted tool with schema-validated arguments.
- `spawn_subagents`: request to run independent Gemma subagent tasks.

Free text that says a tool ran is not accepted as execution evidence. Only a `ToolResult` produced by `ToolExecutor` counts.

## DuckDuckGo Research

The DuckDuckGo MCP tool remains local and stdio-compatible, but it gains structured search responses, URL scheme allowlisting, citation fields, timeout controls, prompt-injection neutralization, relevance scoring, and clear distinction between unavailable, parsed-no-results, and irrelevant-results states.

The current implementation uses public HTTPS to DuckDuckGo. It is not a private network. If a private route is required later, the runtime must verify the actual route and record that evidence before claiming privacy.

## Benchmarking

`benchmarks/bench_runtime.py` owns repeatable benchmark runs. Each run writes raw JSONL events, a summary JSON file, and updates `benchmarks/BENCHMARKS.md`.

Required metrics:

- first-byte latency
- first-text latency
- tokens per second
- end-to-end task time
- tool-call latency
- number of model calls
- prompt tokens
- completion tokens
- peak VRAM/RAM
- context length used
- cache hit indicator or rate when available
- test pass rate
- task success rate
- hallucinated tool-call rate
- failed JSON/tool-call rate

No speedup is considered real until the same workload has before/after measurements.

## Speed Strategy

The target is a public-method approximation of Codex Fast: same model quality, faster token generation and lower orchestration overhead, with any extra compute cost made explicit. Allowed methods are prompt slimming, streaming, lower proxy overhead, parallel independent tool calls, asynchronous indexing, prompt/cache reuse, context packs, llama.cpp prompt cache/context checkpoints, KV cache tuning, and speculative decoding where supported.

Prompt slimming target: reduce generated base prompt tokens by at least 30% while preserving required behavior and the uncensored/no-wrapper instruction.

## Testing

Testing must cover:

- strict JSON action validation
- malformed JSON and failed schema validation
- hallucinated tool-call detection
- actual tool execution and evidence recording
- Gemma subagent spawning with isolated context
- search-result prompt injection
- no-result and irrelevant-result DDG behavior
- citation extraction
- long-context context-pack behavior
- repeated-task cache consistency
- concurrent subagent/tool execution
- benchmark harness output and Markdown table update

Because this folder is not a git repository, the spec cannot be committed unless git is initialized later.
