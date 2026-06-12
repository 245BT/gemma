# Gemma Local Runtime Workspace

This folder is a local Windows workspace for running a Gemma 4 26B A4B
uncensored model through `llama.cpp`, exposing it through OpenAI-compatible
`/v1/responses` endpoints, and using it from local Codex. It also contains a
small Gemma-owned agent runtime, optional reasoning proxy, DuckDuckGo MCP
server, benchmarks, historical design notes, downloaded binaries, Python and
Node dependency trees, model weights, and local Codex evidence artifacts.

This directory is not currently a Git repository. Treat file dates, tests, and
benchmark artifacts as the local evidence trail.

## Quick Start

Run these from `C:\Users\Agent-1\Desktop\gemma`.

```powershell
.\verify-local-setup.ps1
.\start-gemma-runtime.ps1
.\gemma-codex.cmd
.\gemma-codex.cmd --reasoning
```

Useful tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
npm test
```

Useful benchmarks:

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_runtime.py --workload benchmarks\workloads\smoke.jsonl --endpoint http://127.0.0.1:8081/v1/responses --phase baseline --run-label smoke-direct
.\.venv\Scripts\python.exe benchmarks\aggregate_remeasure.py --output benchmarks\runs\remeasure-2026-06-12.aggregate.json
```

## Runtime Shape

The launcher stack is:

1. `start-gemma-runtime.ps1` starts `tools\llama.cpp\llama-server.exe`.
2. `llama-server.exe` serves the GGUF model on `http://127.0.0.1:8080`.
3. `gemma_response_proxy.py` exposes the direct proxy on port `8081` and strips
   Gemma channel markers from JSON and SSE responses.
4. `gemma_reasoning_proxy.py` exposes the optional reasoning proxy on port
   `8082`. It can fast-path simple exact-output requests or run a LangGraph/DSPy
   plan/draft/verify/revise pass.
5. `launch_gemma_codex.py` launches the local Codex binary with `CODEX_HOME`
   pointed at either `.codex-local` or `.codex-local-reasoning`.

Main ports:

| Port | Owner | Purpose |
| ---: | --- | --- |
| `8080` | `llama.cpp` | Raw local model server |
| `8081` | `gemma_response_proxy.py` | Direct OpenAI-style response proxy |
| `8082` | `gemma_reasoning_proxy.py` | Reasoning proxy in front of the direct proxy |

## Directory Map

Sizes are from the current folder scan on 2026-06-12.

| Path | Approx size | What it is |
| --- | ---: | --- |
| `gemma_agent\` | 0.13 MB | Gemma-owned local agent runtime: supervisor, tool registry, executor, subagents, memory, context, citations, safety guard, model client |
| `gemma_reasoning\` | 0.05 MB | Optional reasoning graph and DSPy helpers used by the reasoning proxy |
| `benchmarks\` | 0.37 MB | Runtime benchmark scripts, workloads, aggregate reports, raw run summaries, and external agent benchmark wrappers |
| `docs\superpowers\` | 0.06 MB | Historical design specs and implementation plans for this workspace |
| `scripts\` | 0.01 MB | Utility scripts, currently including local artifact sanitization |
| `tests\` | 0.31 MB | Unit tests covering setup, launchers, proxies, reasoning, agent runtime, DuckDuckGo MCP, benchmarks, and sanitization |
| `models\` | 65.3 GB | Full Hugging Face safetensors snapshot and GGUF quantized model |
| `tools\llama.cpp\` | 1.13 GB | Windows llama.cpp executables and CUDA/ggml DLLs |
| `downloads\` | 622 MB | Downloaded llama.cpp/CUDA zip archives |
| `.venv\` | 4.69 GB | Python virtual environment |
| `node_modules\` | 299 MB | Local Node dependencies, including Codex and Context7 MCP packages |
| `.codex-local\` | 65 MB | Generated direct-mode Codex home plus live Codex state/history/cache |
| `.codex-local-reasoning\` | 60 MB | Generated reasoning-mode Codex home plus live Codex state/history/cache |
| `.codex-smoke-agent\`, `.codex-son-yolo-smoke\` | tiny | Smoke-test proof artifacts |
| `__pycache__\` | tiny | Python bytecode cache |

## Important Root Files

| File | Why it matters |
| --- | --- |
| `AGENTS.md` | Startup contract for work in this workspace. Read it before code, benchmark, research, or documentation changes. |
| `setup_local_codex.py` | Durable source for generated Codex config, model catalog, base instructions, MCP config, launchers, and global shims. Edit this first for persistent local-Codex behavior changes. |
| `launch_gemma_codex.py` | Starts the runtime and launches local Codex with direct or reasoning `CODEX_HOME`. |
| `start-gemma-runtime.ps1` | Starts the model server and both proxies if they are not already available. |
| `start-gemma-server.cmd` | Runs only the raw llama.cpp server. |
| `start-gemma-proxy.cmd` | Runs only the direct response proxy. |
| `verify-local-setup.ps1` | Checks generated Codex homes, Python MCP imports, Context7, and the local Codex binary. |
| `gemma_response_proxy.py` | Cleans channel markers from direct responses and streamed output. |
| `gemma_reasoning_proxy.py` | Implements direct fast path, forced reasoning marker, response wrapping, and reasoning proxy behavior. |
| `gemma_agent_mcp.py` | MCP server exposing `gemma_run_subagents` for Gemma-owned subagent execution. |
| `duckduckgo_mcp.py` | MCP server for DuckDuckGo Lite search with structured citations, relevance filtering, untrusted-result handling, and optional verified SOCKS proxy routing. |
| `chat_cmd.py` | Direct Transformers chat loop for the safetensors model snapshot. |
| `run_readme.py` | Minimal local model usage script. |
| `requirements-gemma.txt` | Python dependency list for this workspace. |
| `package.json`, `package-lock.json` | Node dependency and script definitions. |
| `README.hf.md` | Local copy of the model-card style README for the uncensored Gemma model. Its benchmark/refusal claims are author claims, not proof for every prompt. |
| `config.hf.json` | Hugging Face model config copy. |

## Things To Ignore If...

Ignore these when you are reading application logic:

- `.venv\`
- `node_modules\`
- `models\`
- `tools\llama.cpp\`
- `downloads\`
- `__pycache__\`
- `*.out.log`, `*.err.log`, `run.out.txt`, `run.err.txt`
- `benchmarks\runs\*.events.jsonl`
- `.codex-local\history.jsonl`
- `.codex-local\sessions\`
- `.codex-local-reasoning\history.jsonl`
- `.codex-local-reasoning\sessions\`

Ignore generated Codex homes if you are changing durable setup behavior. Edit
`setup_local_codex.py`, then regenerate the homes with:

```powershell
.\.venv\Scripts\python.exe setup_local_codex.py
```

Ignore benchmark raw events if you only need the latest measurement summary.
Read `benchmarks\REMEASURE_2026-06-12.md`,
`benchmarks\BENCHMARKS.md`, and aggregate JSON files first.

Ignore model weights if you are changing prompt policy, style, routing,
tool-call behavior, or launcher behavior. Those are controlled by source files,
not by directly editing `.safetensors` or `.gguf` files.

Ignore downloaded zip archives unless you are reinstalling llama.cpp/CUDA
binaries. They are inputs, not active runtime code.

## Things Not To Ignore

Do not ignore these when making durable behavior changes:

- `AGENTS.md`: workspace rules and safety constraints.
- `setup_local_codex.py`: source of generated local Codex configuration.
- `gemma_response_proxy.py`: direct response cleanup and proxy behavior.
- `gemma_reasoning_proxy.py`: reasoning fast path, forced graph marker, and
  proxy response semantics.
- `gemma_agent\supervisor.py`: action loop, strict JSON action requirements,
  and subagent spawning behavior.
- `gemma_agent\tool_registry.py`: tool registration and schema validation.
- `gemma_agent\tool_executor.py`: execution mode, timeouts, argument coercion,
  and path validation.
- `gemma_agent\safety.py`: workspace path guard and untrusted tool-output wrapper.
- `gemma_agent\schemas.py`: `ToolResult`, `ThinkingSummary`, `SubAgentResult`,
  and `AgentRunResult` evidence shapes.
- `gemma_agent\context.py` and `gemma_agent\memory.py`: hierarchical memory and
  context selection.
- `gemma_reasoning\graph.py`: plan/draft/verify/revise flow and private
  non-authoritative reasoning-plan injection.
- `gemma_reasoning\dspy_programs.py`: default planning and verification
  program signatures.
- `duckduckgo_mcp.py`: current-information search behavior, citation objects,
  untrusted-result handling, and public/private transport metadata.
- `tests\`: the completion gate for most code changes.
- `benchmarks\`: the completion gate for optimization claims.

Do not ignore generated config files when debugging a live Codex launch:

- `.codex-local\config.toml`
- `.codex-local\model-catalog.json`
- `.codex-local-reasoning\config.toml`
- `.codex-local-reasoning\model-catalog.json`

Those files are what local Codex actually reads at runtime. They are generated,
so use them to confirm what happened, but make durable edits in
`setup_local_codex.py`.

## How To Change What The AI Treats As Important

There are several different layers. Pick the layer that matches the change.

### One Request Only

Put the instruction in the user prompt. This affects only the current request
unless the surrounding runtime stores it elsewhere.

### Persistent Local Codex Persona Or Priorities

Edit `build_base_instructions()` in `setup_local_codex.py`. That string is
copied into both generated model catalogs:

- `.codex-local\model-catalog.json`
- `.codex-local-reasoning\model-catalog.json`

After editing it, regenerate and verify:

```powershell
.\.venv\Scripts\python.exe setup_local_codex.py
.\verify-local-setup.ps1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Non-applied example, matching the kind of value preference you asked about:
if you once wanted the local assistant to treat one arbitrary value as worse
and another arbitrary value as better, you would express that as an explicit
preference or domain rule in `build_base_instructions()` or in a task prompt.
This README does not apply that reasoning, does not endorse it, and does not
change the model to believe it.

### Agent Action Or Tool-Calling Behavior

Edit `gemma_agent\supervisor.py` for the action loop and action instruction
text. The current agent expects strict JSON actions:

- `final`
- `tool_call`
- `spawn_subagents`
- `thinking_summary`

Edit `gemma_agent\tool_registry.py`, `gemma_agent\tool_executor.py`, and
`gemma_agent\safety.py` when the change involves tool allowlisting, schema
validation, path checking, timeouts, or untrusted tool outputs.

### Thinking Summaries And Memory

Edit `gemma_agent\schemas.py` for the public thinking-summary fields:

- Goal
- Plan
- Evidence
- Current finding
- Confidence
- Next action

Edit `gemma_agent\context.py` and `gemma_agent\memory.py` for hierarchical
memory selection, tier limits, deduplication, and relevance scoring.

Do not store raw chain-of-thought in logs, prompts, caches, or test artifacts.
Use concise summaries and evidence instead.

### Reasoning Proxy Behavior

Edit `gemma_reasoning_proxy.py` for:

- the direct fast-path rules,
- `x_gemma_force_reasoning`,
- streamed response wrapping,
- public error messages,
- proxy ports and upstream behavior.

Edit `gemma_reasoning\graph.py` for the reasoning graph itself. The graph
extracts the task, produces a plan, drafts with the upstream client, verifies
the draft, optionally revises, and finalizes.

Edit `gemma_reasoning\dspy_programs.py` for the DSPy planner/verifier
signatures and default model settings used by the reasoning graph.

### Search And Citations

Edit `duckduckgo_mcp.py` for DuckDuckGo behavior. Current behavior:

- queries `https://lite.duckduckgo.com/lite/` over public HTTPS by default,
- treats search result text as untrusted data,
- strips or neutralizes instruction-looking text from result display,
- supports `recency_days`,
- returns structured citations,
- filters low-relevance results,
- only marks private networking as verified when a local/private SOCKS proxy is
  configured and verified.

Do not claim DuckDuckGo is using a private route unless
`GEMMA_DDG_SOCKS_PROXY` is configured and the verifier passes.

### Chat Template Or Raw Model Formatting

The llama.cpp server uses:

```text
models\gemma-4-26B-A4B-it-uncensored-GGUF\chat_template_no_thought.jinja
```

Changing that file changes how prompts are formatted for the GGUF server. Test
with the raw endpoint, direct proxy, and reasoning proxy after any template
change.

### Model Weights Or Uncensored Behavior

The model files live under:

- `models\gemma-4-26B-A4B-it-uncensored\`
- `models\gemma-4-26B-A4B-it-uncensored-GGUF\`

Do not edit model weights manually. If you want a durable change below the
prompt/config layer, use a reproducible model-training, fine-tuning,
abliteration, merge, or quantization pipeline and record before/after tests.

Per `AGENTS.md`, do not modify the uncensored model behavior, model weights,
refusal-removal behavior, or prompt policy unless the user explicitly requests
that exact change.

## Benchmarks And Measurements

`benchmarks\bench_runtime.py` writes:

- first-byte latency,
- first-text latency,
- total time,
- tokens per second,
- prompt and completion tokens,
- model call count,
- RAM and VRAM samples,
- context/cache fields when available,
- task success,
- failed tool/JSON counters,
- summary JSON,
- sanitized event JSONL.

`benchmarks\aggregate_remeasure.py` groups repeated summaries and measures base
instruction prompt size using `tiktoken` when available.

Current controlled microbenchmark notes are in:

- `benchmarks\REMEASURE_2026-06-12.md`
- `benchmarks\BENCHMARKS.md`
- `benchmarks\runs\remeasure-2026-06-12.aggregate.json`

The retained 2026-06-12 aggregate shows:

| Group | n | Avg total ms | Caveat |
| --- | ---: | ---: | --- |
| Raw llama.cpp `8080` | 5 | 258.34 | Direct raw endpoint |
| Direct proxy `8081` | 5 | 274.09 | Channel-cleaning proxy |
| Reasoning proxy forced graph `8082` | 5 | 413.31 | Token fields in this retained row are not comparable |
| Reasoning proxy fast path `8082` | 5 | 329.29 | Simple exact-output request shape |

Every optimization needs before/after measurements. Do not claim a speedup from
single smoke rows when repeated aggregate rows exist.

## Local Evidence And Shareable Bundles

Treat these as local evidence artifacts and do not include them in shareable
bundles unless sanitized:

- `.codex-local\history.jsonl`
- `.codex-local\sessions\**`
- `.codex-local\.sandbox\**`
- `.codex-local\*.sqlite*`
- `.codex-local\sandbox.*.log`
- `.codex-local-reasoning\history.jsonl`
- `.codex-local-reasoning\sessions\**`
- `.codex-local-reasoning\.sandbox\**`
- `.codex-local-reasoning\*.sqlite*`
- `.codex-local-reasoning\sandbox.*.log`
- raw benchmark event logs under `benchmarks\runs\*.events.jsonl`
- runtime logs such as `llama-server.err.log`, `gemma-proxy.out.log`, and
  `gemma-reasoning-proxy.out.log`

Dry-run the sanitizer:

```powershell
.\.venv\Scripts\python.exe scripts\sanitize_local_artifacts.py --root .
```

Apply it only when you actually want to remove matching local artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\sanitize_local_artifacts.py --root . --apply
```

## Tests

The unit tests are in `tests\` and cover:

- setup and generated local Codex config,
- launch command/environment construction,
- direct response proxy cleanup,
- reasoning proxy fast path and graph behavior,
- reasoning graph and DSPy wrappers,
- Gemma agent runtime, schema validation, tool execution, safety, and subagents,
- Gemma agent MCP server,
- DuckDuckGo parsing/search/citation/private-route behavior,
- chat command helpers,
- runtime benchmark extraction and aggregation,
- local artifact sanitization.

Run all tests after source changes:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For Node script consistency, `npm test` runs the same Python unittest command
through `package.json`.

## Common Workflows

Change persistent local Codex instructions:

1. Edit `setup_local_codex.py`.
2. Run `.\.venv\Scripts\python.exe setup_local_codex.py`.
3. Check `.codex-local\model-catalog.json` and
   `.codex-local-reasoning\model-catalog.json`.
4. Run `.\verify-local-setup.ps1`.
5. Run unit tests.
6. Restart local Codex.

Change proxy response behavior:

1. Edit `gemma_response_proxy.py` or `gemma_reasoning_proxy.py`.
2. Add or update tests in `tests\test_gemma_response_proxy.py` or
   `tests\test_gemma_reasoning_proxy.py`.
3. Run the relevant tests, then all tests.
4. Start runtime and test the affected endpoint.

Change Gemma-owned agent behavior:

1. Edit the relevant `gemma_agent\` module.
2. Add or update tests in `tests\test_gemma_agent_runtime.py` or
   `tests\test_gemma_agent_mcp.py`.
3. Run all tests.
4. If performance or tool-call behavior changes, add a benchmark row.

Change reasoning behavior:

1. Edit `gemma_reasoning_proxy.py` or `gemma_reasoning\`.
2. Update reasoning tests.
3. Run endpoint benchmarks against `8081` and `8082` for before/after evidence.

Change DuckDuckGo behavior:

1. Edit `duckduckgo_mcp.py`.
2. Update `tests\test_duckduckgo_mcp.py`.
3. Verify result text remains untrusted and citation-bearing.
4. Verify public/private network metadata is accurate.

## Final Notes

This workspace mixes source code, generated runtime state, local evidence,
large binary dependencies, and model files. Most ordinary edits should touch
only source, tests, docs, and benchmark definitions. Large folders and live
state are usually evidence or dependencies, not places to make logic changes.
