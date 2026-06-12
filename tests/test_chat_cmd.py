import json
import tempfile
import unittest
from pathlib import Path

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
