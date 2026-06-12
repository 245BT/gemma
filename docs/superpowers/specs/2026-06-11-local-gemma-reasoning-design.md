# Local Gemma Reasoning Design

## Goal

Add an opt-in LangGraph + DSPy planner/verifier reasoning layer to the local Gemma Codex terminal without changing global Codex behavior.

The feature is enabled only through the local shim commands:

- `operator --reasoning`
- `son --reasoning`
- `sonion --reasoning`

Normal `operator`, `son`, and `sonion` continue to use the current direct local Gemma proxy.

## Locality

All generated files and configuration remain under `C:\Users\Agent-1\Desktop\gemma`.

The global Codex config under `%USERPROFILE%\.codex` is not modified. The global shim commands continue to call this folder's `gemma-codex.cmd`, and that launcher chooses between two project-local Codex homes:

- `.codex-local` for direct mode, pointing at `http://127.0.0.1:8081/v1`
- `.codex-local-reasoning` for reasoning mode, pointing at `http://127.0.0.1:8082/v1`

The `--reasoning` flag is consumed by the local launcher and stripped before the vendored Codex binary is invoked.

## Architecture

The existing runtime remains:

`local Codex -> direct proxy on 8081 -> llama.cpp on 8080`

Reasoning mode adds a second local proxy:

`local Codex -> reasoning proxy on 8082 -> LangGraph graph -> DSPy planner/verifier -> direct proxy on 8081 -> llama.cpp on 8080`

LangGraph owns the request workflow:

1. Extract the latest user-visible task from the incoming Responses request.
2. Ask a DSPy planner to produce a concise plan bounded by instruction hierarchy and terminal context.
3. Ask Gemma for a draft response with the plan as private orchestration context.
4. Ask a DSPy verifier to judge whether the draft answers the task, respects constraints, and should be revised.
5. If needed, ask Gemma for one revised response using the verifier notes.
6. Return a normal Responses-compatible payload to Codex.

Codex remains responsible for tool orchestration, shell approval, filesystem edits, and sandbox semantics. The reasoning layer does not execute tools directly.

## Boundaries

This feature does not claim native model reasoning support. The model catalog should continue to report `default_reasoning_level = "none"` because this is an external orchestration layer.

This feature does not bypass system, developer, or project instructions. Planner and verifier prompts explicitly treat higher-priority instructions as constraints.

This feature does not add a separate moderation layer. It improves local task discipline through planning and verification while preserving the existing local Gemma model path.

## Error Handling

If reasoning dependencies are missing, the reasoning proxy fails fast with an actionable message telling the user to install local dependencies in `.venv`.

If the reasoning graph fails for a request, the proxy returns an HTTP 500 JSON error instead of silently falling back to direct mode. Silent fallback would hide failures and make debugging misleading.

Streaming requests are accepted, but reasoning mode buffers the final result and returns a short Server-Sent Events sequence after the graph completes. Planner/verifier loops need the complete draft before verification, so token-by-token streaming is intentionally not preserved in the first implementation.

## Testing

Unit tests cover:

- local launcher argument parsing and `--reasoning` stripping
- direct versus reasoning Codex home selection
- local setup helper generation for both Codex homes
- reasoning graph behavior with fake model calls
- reasoning proxy response shaping for JSON and SSE requests
- existing channel-marker cleanup behavior

End-to-end smoke testing covers:

- `operator` still using the direct local Codex home
- `operator --reasoning` using `.codex-local-reasoning`
- the reasoning proxy starting on `8082`
- local Gemma returning a response through the reasoning graph

## Scope Notes

The implementation will avoid modifying the vendored Codex binary. It will also avoid global package installs and global Codex configuration writes.

Because this folder is not a git repository, the normal spec commit step cannot be performed.
