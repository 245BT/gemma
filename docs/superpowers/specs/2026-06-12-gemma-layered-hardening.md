# Gemma Layered Hardening Design

## Goal

Harden the local Gemma coding runtime so model-generated or tool-returned instruction text cannot silently become trusted runtime instructions, executable source, or unverified completion claims.

## Evidence

- `setup_local_codex.py` currently contains instruction-looking contamination inside `build_base_instructions()`.
- `tests.test_setup_local_codex` currently fails because the base prompt is 540 estimated tokens, above the 448-token budget.
- The earlier launch failure came from instruction-looking text inserted into Python source, which broke proxy parsing and caused `start-gemma-runtime.ps1` to time out on port 8081.
- GPT-5.5 xhigh read-only subagents found gaps in prompt integrity, generated catalog drift, startup syntax preflight, schema validation, untrusted propagation, thread timeout containment, and evidence-gated final claims.

## Research Basis

- OWASP LLM01:2025 treats prompt injection as a top LLM application risk and recommends output validation, filtering, least privilege, human approval for high-risk actions, external-content segregation, and adversarial testing.
- OpenAI's public Model Spec recommends treating quoted text, tool outputs, attachments, and other untrusted data as having no authority by default unless higher-priority instructions delegate authority.
- NCSC's prompt-injection guidance describes current LLMs as lacking a robust internal instruction/data boundary, so mitigation must reduce risk and impact through system design.
- AgentDojo provides the benchmark pattern: evaluate agents on realistic tasks with tools over untrusted data, not only happy-path task success.

## Architecture

The fix is layered rather than prompt-only.

1. Prompt integrity: base instructions remain explicit config, but tests reject known injection/control phrases and setup verification detects generated catalog drift.
2. Startup preflight: Python and PowerShell syntax checks run before launch paths are trusted.
3. Runtime trust boundaries: all model-returned, tool-returned, invalid-action, error, and subagent-returned text that can re-enter prompts is explicitly wrapped as untrusted evidence.
4. Tool safety: tool schemas fail closed on unsupported JSON Schema constructs, path validation applies recursively, and side-effect tools must use process-mode hard timeouts.
5. Evidence accounting: final claims about edits, commands, or tool execution are flagged if no matching executed evidence exists.
6. Metrics: tests and benchmark summaries track prompt token size, catalog drift, syntax preflight pass rate, hallucinated tool-claim rate, raw untrusted text leakage, and failed JSON/tool-call rate.

## File Boundaries

- `setup_local_codex.py`: source of durable local Codex prompt/config and generated verification scripts.
- `tests/test_setup_local_codex.py`: prompt integrity, generated catalog integrity, and launcher preflight tests.
- `gemma_agent/safety.py`: trust wrappers, path denylist, and evidence artifact policy.
- `gemma_agent/tool_registry.py`: strict schema validation.
- `gemma_agent/tool_executor.py`: timeout mode policy, path validation, and untrusted error wrapping.
- `gemma_agent/supervisor.py`: evidence-gated final claims and untrusted invalid-action handling.
- `gemma_agent/schemas.py`: evidence serialization for subagent outputs.
- `gemma_agent_mcp.py`: ingress limits for Gemma-owned subagent tasks.
- `tests/test_gemma_agent_runtime.py` and `tests/test_gemma_agent_mcp.py`: runtime guardrail tests.
- `benchmarks/bench_runtime.py` and `tests/test_bench_runtime.py`: metric extraction and summary fields.

## Out Of Scope

- No model weight changes.
- No moderation, refusal, or safety-policy wrapper.
- No change to uncensored model behavior.
- No copying proprietary prompts or private implementation details from other vendors.
- No broad unrelated refactors.

## Completion Gate

- Targeted red/green tests for every behavior change.
- Full project unittest discovery through the project venv.
- `npm test`.
- Python syntax scan.
- PowerShell parser check for startup scripts.
- Actual `start-gemma-runtime.ps1` readiness check when the local model server is available.
- Final GPT-5.5 xhigh code review for blocking regressions.
