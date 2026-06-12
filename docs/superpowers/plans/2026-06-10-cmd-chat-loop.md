# CMD Chat Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable CMD chat loop around the locally downloaded Gemma model.

**Architecture:** Add `chat_cmd.py` as the interactive entrypoint and keep `run_readme.py` as the one-shot README-style runner. Unit-test command parsing and model-info helpers without loading model weights, then smoke-test the real model manually with a short prompt.

**Tech Stack:** Python 3.14, Transformers, PyTorch CUDA, Accelerate, Hugging Face Hub metadata API.

---

### Task 1: Test CLI Helpers

**Files:**
- Create: `tests/test_chat_cmd.py`
- Create: `chat_cmd.py`

- [ ] **Step 1: Write the failing tests**

```python
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chat_cmd


class ChatCmdTests(unittest.TestCase):
    def test_normalize_command_handles_known_commands(self):
        self.assertEqual(chat_cmd.normalize_command(" /exit "), "/exit")
        self.assertEqual(chat_cmd.normalize_command("/quit"), "/exit")
        self.assertEqual(chat_cmd.normalize_command("/clear"), "/clear")
        self.assertEqual(chat_cmd.normalize_command("/info"), "/info")
        self.assertIsNone(chat_cmd.normalize_command("hello"))

    def test_trim_history_keeps_recent_messages(self):
        messages = [{"role": "user", "content": str(i)} for i in range(6)]
        self.assertEqual(chat_cmd.trim_history(messages, max_messages=4), messages[-4:])

    def test_load_local_model_metadata_reads_config_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(json.dumps({
                "model_type": "gemma4",
                "architectures": ["Gemma4ForConditionalGeneration"],
                "dtype": "bfloat16",
            }))
            (model_dir / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {
                    "total_parameters": 25805933872,
                    "total_size": 51611872412,
                }
            }))
            metadata = chat_cmd.load_local_model_metadata(model_dir)
            self.assertEqual(metadata["model_type"], "gemma4")
            self.assertEqual(metadata["architectures"], ["Gemma4ForConditionalGeneration"])
            self.assertEqual(metadata["total_parameters"], 25805933872)

    def test_verify_hf_files_compares_size_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            sample = model_dir / "tokenizer.json"
            sample.write_text("abc")
            remote = {"tokenizer.json": {"size": 3, "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}}
            result = chat_cmd.verify_files_against_metadata(model_dir, remote, ["tokenizer.json"])
            self.assertTrue(result["tokenizer.json"]["size_matches"])
            self.assertTrue(result["tokenizer.json"]["sha_matches"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_chat_cmd -v`

Expected: failure because `chat_cmd` does not exist.

- [ ] **Step 3: Implement `chat_cmd.py`**

Add command parsing, metadata loading, optional SHA verification, model loading, and the interactive loop.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_chat_cmd -v`

Expected: four passing tests.

- [ ] **Step 5: Smoke-test the real model**

Run: `.\.venv\Scripts\python.exe chat_cmd.py --once "What is 2+2? Answer with only the number."`

Expected: the model loads and prints `Gemma > 4` or an equivalent concise answer.

- [ ] **Step 6: Launch the CMD chat**

Run: `cmd.exe /k cd /d "C:\Users\Agent-1\Desktop\gemma" && .\.venv\Scripts\python.exe chat_cmd.py`

Expected: a second CMD window stays open at `You >`.
