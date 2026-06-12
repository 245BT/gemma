# Local Codex Gemma Design

## Goal

Run Codex 0.139.0 from this folder only, backed by the TrevorJS Gemma 4 uncensored model in the fastest quantization that fits this computer.

## Locality

Everything created for this setup lives under `C:\Users\Agent-1\Desktop\gemma`:

- npm-local Codex package under `node_modules`
- local Codex home under `.codex-local`
- local model catalog under `.codex-local\model-catalog.json`
- local launcher scripts in this folder
- local llama.cpp server binary under `tools`
- local GGUF model under `models`

No global npm package, global `~\.codex\config.toml`, or global Codex model cache is modified.

## Model

The runtime uses `TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF` with `Q4_K_M`.

The same-author GGUF repo currently offers two quant levels:

- `Q8_0`: 26,859,854,400 bytes, too large for the 16 GB GPU.
- `Q4_K_M`: 16,796,011,072 bytes, the highest same-author quant that can plausibly fit.

The setup verifies the local GGUF SHA-256 against Hugging Face metadata before use.

## Unrestricted Behavior

The setup does not add a safety filter, moderation layer, refusal system prompt, or wrapper policy. It routes requests directly from Codex to the local llama.cpp server serving the TrevorJS uncensored quantized model.

This verifies absence of an added local restriction layer. It does not prove that every possible prompt will be answered because actual behavior is determined by model weights, prompt template, and inference runtime.

## Codex Integration

Codex runs with `CODEX_HOME=.codex-local`. The local config points at a local OpenAI-compatible provider served by llama.cpp:

- `model = "gemma-4-26b-a4b-it-uncensored-q4-k-m"`
- `model_provider = "local_gemma"`
- `model_catalog_json = ".codex-local\model-catalog.json"`
- `[model_providers.local_gemma] base_url = "http://127.0.0.1:8080/v1"`

The model catalog contains only the Gemma model with `visibility = "list"` so `/models` in this local Codex session should show only Gemma from the local catalog.

## Skills

Existing installed skills are made visible to the local Codex home by creating local directory links from `.codex-local\skills` to the current user skill directory. This preserves one source of truth and avoids copying or modifying global skill contents.
