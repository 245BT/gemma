import tempfile
import unittest
from pathlib import Path

from scripts import sanitize_local_artifacts


class SanitizeLocalArtifactsTests(unittest.TestCase):
    def test_finds_sensitive_codex_artifacts_but_preserves_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_home = root / ".codex-local"
            session_dir = local_home / "sessions" / "2026" / "06" / "12"
            sandbox_dir = local_home / ".sandbox" / "run"
            session_dir.mkdir(parents=True)
            sandbox_dir.mkdir(parents=True)
            (local_home / "config.toml").write_text("model = 'gemma'\n", encoding="utf-8")
            (local_home / "model-catalog.json").write_text("{}\n", encoding="utf-8")
            (local_home / "history.jsonl").write_text('{"text":"secret prompt"}\n', encoding="utf-8")
            (local_home / "logs_2.sqlite").write_text("raw log", encoding="utf-8")
            (local_home / "state_5.sqlite-wal").write_text("raw state", encoding="utf-8")
            session_path = session_dir / "rollout.jsonl"
            session_path.write_text('{"payload":{"text":"raw prompt"}}\n', encoding="utf-8")
            sandbox_path = sandbox_dir / "tool-output.txt"
            sandbox_path.write_text("raw tool output", encoding="utf-8")

            found = {path.relative_to(root).as_posix() for path in sanitize_local_artifacts.find_artifact_paths(root)}

        self.assertIn(".codex-local/history.jsonl", found)
        self.assertIn(".codex-local/logs_2.sqlite", found)
        self.assertIn(".codex-local/state_5.sqlite-wal", found)
        self.assertIn(".codex-local/sessions/2026/06/12/rollout.jsonl", found)
        self.assertIn(".codex-local/.sandbox/run/tool-output.txt", found)
        self.assertNotIn(".codex-local/config.toml", found)
        self.assertNotIn(".codex-local/model-catalog.json", found)

    def test_sanitize_removes_artifacts_and_keeps_local_setup_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_home = root / ".codex-local-reasoning"
            session_dir = local_home / "sessions" / "2026"
            session_dir.mkdir(parents=True)
            config_path = local_home / "config.toml"
            history_path = local_home / "history.jsonl"
            session_path = session_dir / "rollout.jsonl"
            sqlite_path = local_home / "logs_2.sqlite-shm"
            config_path.write_text("model = 'gemma'\n", encoding="utf-8")
            history_path.write_text('{"text":"raw prompt"}\n', encoding="utf-8")
            session_path.write_text('{"payload":{"text":"raw prompt"}}\n', encoding="utf-8")
            sqlite_path.write_text("raw sqlite", encoding="utf-8")

            result = sanitize_local_artifacts.sanitize(root, apply=True)

            self.assertEqual(result["removed_count"], 3)
            self.assertTrue(config_path.exists())
            self.assertFalse(history_path.exists())
            self.assertFalse(session_path.exists())
            self.assertFalse(sqlite_path.exists())


if __name__ == "__main__":
    unittest.main()
