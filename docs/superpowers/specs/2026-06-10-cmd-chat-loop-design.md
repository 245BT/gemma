# CMD Chat Loop Design

## Goal

Create a single Command Prompt chat loop for the downloaded `TrevorJS/gemma-4-26B-A4B-it-uncensored` model. The model should load once, then accept repeated user messages and print assistant replies until the user exits.

## Scope

The interface is terminal-only. It supports `You >` input, `Gemma >` output, message history across turns, and the commands `/exit`, `/clear`, and `/info`.

## Runtime

The chat loop uses the same model-loading approach from the model README:

```python
AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
AutoTokenizer.from_pretrained(model_id)
```

The local model path is `models/gemma-4-26B-A4B-it-uncensored` to avoid a second download.

## Verification

The `/info` command reports facts that are locally verifiable:

- `config.json` has `model_type: gemma4`.
- `config.json` has architecture `Gemma4ForConditionalGeneration`.
- `model.safetensors.index.json` reports `25,805,933,872` parameters.
- The two safetensor shards and tokenizer match the Hugging Face SHA-256 metadata.

The `/info` command also reports the model-card claim that this is an uncensored Gemma 4 variant with refusal behavior removed. That claim is attributed to the README and not treated as independent proof that every possible prompt is unrestricted.

## Error Handling

If model files are missing, the script exits with a clear message. If remote Hugging Face metadata cannot be fetched for `/info`, the command still reports local identity and marks SHA verification as unavailable.
