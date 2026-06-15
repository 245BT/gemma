---
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: text-generation
library_name: transformers
language:
- en
license: apache-2.0
tags:
- abliteration
- uncensored
- gemma-4
---

# gemma-4-26B-A4B-it-uncensored

Uncensored version of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) with refusal behavior removed.

## Results

| | Before | After |
|--|--------|-------|
| **Refusals (mlabonne, 100 prompts)** | 98/100 | **1/100 effective (3 flagged, 2 refusal-then-comply)** |
| **Refusals (cross-dataset, 686 prompts)** | — | **5/686 (0.7%)** |
| **KL Divergence** | 0 (baseline) | **0.09** |
| **Quality (harmless response length ratio)** | 1.0 | **~1.01** (no degradation) |

### Cross-Dataset Validation

Tested against 4 independent prompt datasets to verify generalization:

| Dataset | Prompts | Refusals |
|---------|---------|----------|
| [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | 100 | 1/100 |
| [tulu-harmbench](https://huggingface.co/datasets/allenai/tulu-3-harmbench-eval) | 320 | 1/320 |
| [NousResearch/RefusalDataset](https://huggingface.co/datasets/NousResearch/RefusalDataset) | 166 | 0/166 |
| [mlabonne/harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors) | 100 | 3/100 |
| **Total** | **686** | **5/686 (0.7%)** |

Every flagged refusal was manually audited. Most are "refusal-then-comply" false positives where the model
adds an AI identity disclaimer then answers the question anyway.

## Method

Norm-preserving biprojected abliteration on the dense pathway (o_proj + shared mlp.down_proj),
plus **Expert-Granular Abliteration (EGA)** on all 128 MoE expert down_proj slices per layer.

EGA ([OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS)) hooks the MoE routers during probing
to compute per-expert routing weights for harmful vs harmless prompts, then applies norm-preserving
projection ([grimjim](https://huggingface.co/blog/grimjim/abliteration-biprojection)) to each expert
individually. Dense-only abliteration leaves 29/100 refusals; adding EGA drops it to 3/100.

### Pipeline

1. Load model in bf16 with LoRA adapters on `o_proj` and `mlp.down_proj`
2. Collect residual activations for 400 harmful + 400 harmless prompts ([mlabonne](https://huggingface.co/mlabonne) datasets)
3. Winsorize activations at 99.5th percentile (clamps GeGLU outlier activations in Gemma family)
4. Compute per-layer refusal direction: `normalize(mean(harmful) - mean(harmless))`
5. Orthogonalize each direction against harmless mean (double-pass Gram-Schmidt)
6. Apply norm-preserving weight modification to `o_proj` and `down_proj` in all layers
7. Hook MoE routers, collect per-expert routing weights for harmful vs harmless prompts
8. Apply same norm-preserving modification to all 128 expert `down_proj` slices per layer
9. Merge LoRA adapters into base weights for clean tensor names

### Parameters

| Parameter | Value |
|-----------|-------|
| Layers abliterated | 100% |
| Scale | 1.0 |
| Winsorization | 0.995 |
| Experts abliterated | 100% (128/128 per layer) |
| Expert scale | 1.0 |

### How this differs from vanilla [heretic](https://github.com/p-e-w/heretic)

- **Norm-preserving biprojection** instead of standard projection (preserves weight magnitudes)
- **Per-layer refusal directions** instead of one global direction
- **Deterministic single-pass** instead of 50-trial Optuna search (faster, same or better results)
- **LoRA merge before save** for clean GGUF-compatible tensor names
- **Expert-Granular Abliteration** for MoE expert weights (not supported in heretic)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("TrevorJS/gemma-4-26B-A4B-it-uncensored", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("TrevorJS/gemma-4-26B-A4B-it-uncensored")

messages = [{"role": "user", "content": "Your prompt here"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
inputs = inputs.to(model.device)
context_length = getattr(model.config, "max_position_embeddings", None)
generation_kwargs = {"max_length": context_length} if context_length else {}
outputs = model.generate(**inputs, **generation_kwargs)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Reproduction

Full code and experiment data: [abliteration research repo](https://github.com/TrevorS/gemma-4-abliteration)

```bash
python scripts/ega.py --model google/gemma-4-26B-A4B-it \
  --top-pct 100 --strip-topic-markers --skip-prefix --batch-size 4 \
  --save output_dir
```
