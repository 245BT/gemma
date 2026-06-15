import json
import tempfile
import unittest
from pathlib import Path

import chat_cmd


class ChatCmdTests(unittest.TestCase):
    def test_huggingface_entrypoints_do_not_set_output_token_caps(self):
        root = Path(chat_cmd.__file__).resolve().parent

        for relative_path in ("chat_cmd.py", "run_readme.py", "README.hf.md"):
            source = (root / relative_path).read_text(encoding="utf-8")
            with self.subTest(path=relative_path):
                self.assertNotIn("max_new_tokens", source)
                self.assertNotIn("--max-new-tokens", source)

    def test_normalize_command_handles_known_commands(self):
        self.assertEqual(chat_cmd.normalize_command(" /exit "), "/exit")
        self.assertEqual(chat_cmd.normalize_command("/quit"), "/exit")
        self.assertEqual(chat_cmd.normalize_command("/clear"), "/clear")
        self.assertEqual(chat_cmd.normalize_command("/info"), "/info")
        self.assertIsNone(chat_cmd.normalize_command("hello"))

    def test_trim_history_keeps_recent_messages(self):
        messages = [{"role": "user", "content": str(i)} for i in range(6)]
        self.assertEqual(chat_cmd.trim_history(messages, max_messages=4), messages[-4:])

    def test_generate_reply_uses_model_context_length_without_new_token_cap(self):
        captured = {}

        class FakeIds:
            shape = (1, 1)

        class FakeInputs(dict):
            def __init__(self):
                super().__init__({"input_ids": FakeIds()})

            def to(self, _device):
                return self

        class FakeTokenizer:
            def apply_chat_template(self, messages, return_tensors, add_generation_prompt):
                self.messages = messages
                self.return_tensors = return_tensors
                self.add_generation_prompt = add_generation_prompt
                return FakeInputs()

            def decode(self, tokens, skip_special_tokens):
                self.tokens = tokens
                self.skip_special_tokens = skip_special_tokens
                return "ok"

        class FakeConfig:
            max_position_embeddings = 123

        class FakeModel:
            device = "cpu"
            config = FakeConfig()

            def generate(self, **kwargs):
                captured.update(kwargs)
                return [[1, 2, 3]]

        reply = chat_cmd.generate_reply(
            FakeModel(),
            FakeTokenizer(),
            [{"role": "user", "content": "hello"}],
        )

        self.assertEqual(reply, "ok")
        self.assertEqual(captured["max_length"], 123)
        self.assertNotIn("max_new_tokens", captured)

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
            remote = {
                "tokenizer.json": {
                    "size": 3,
                    "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                }
            }
            result = chat_cmd.verify_files_against_metadata(model_dir, remote, ["tokenizer.json"])
            self.assertTrue(result["tokenizer.json"]["size_matches"])
            self.assertTrue(result["tokenizer.json"]["sha_matches"])


if __name__ == "__main__":
    unittest.main()
