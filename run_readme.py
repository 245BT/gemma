import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = Path("models/gemma-4-26B-A4B-it-uncensored")
prompt = "Your prompt here" if len(sys.argv) == 1 else " ".join(sys.argv[1:])

model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
inputs = inputs.to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
