import unittest
from pathlib import Path
from unittest.mock import patch

import launch_gemma_codex


class LaunchGemmaCodexTests(unittest.TestCase):
    def test_parse_launcher_args_strips_reasoning(self):
        parsed = launch_gemma_codex.parse_launcher_args(["--reasoning", "--foo", "bar"])
        self.assertTrue(parsed.reasoning)
        self.assertEqual(parsed.codex_args, ["--foo", "bar"])

    def test_parse_launcher_args_strips_reasoning_after_other_args(self):
        parsed = launch_gemma_codex.parse_launcher_args(["--foo", "--reasoning", "bar"])
        self.assertTrue(parsed.reasoning)
        self.assertEqual(parsed.codex_args, ["--foo", "bar"])

    def test_parse_launcher_args_maps_yolo_to_codex_bypass_flag(self):
        parsed = launch_gemma_codex.parse_launcher_args(["--yolo"])
        self.assertFalse(parsed.reasoning)
        self.assertEqual(
            parsed.codex_args,
            [
                "--ask-for-approval",
                "never",
                "--sandbox",
                "danger-full-access",
            ],
        )

    def test_parse_launcher_args_supports_reasoning_yolo_combo(self):
        parsed = launch_gemma_codex.parse_launcher_args(["--reasoning", "--yolo"])
        self.assertTrue(parsed.reasoning)
        self.assertEqual(
            parsed.codex_args,
            [
                "--ask-for-approval",
                "never",
                "--sandbox",
                "danger-full-access",
            ],
        )

    def test_parse_launcher_args_supports_yolo_reasoning_combo(self):
        parsed = launch_gemma_codex.parse_launcher_args(["--yolo", "--reasoning", "exec", "task"])
        self.assertTrue(parsed.reasoning)
        self.assertEqual(
            parsed.codex_args,
            [
                "--ask-for-approval",
                "never",
                "--sandbox",
                "danger-full-access",
                "exec",
                "task",
            ],
        )

    def test_yolo_args_do_not_use_codex_incompatible_bypass_pair(self):
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", launch_gemma_codex.YOLO_CODEX_ARGS)
        self.assertIn("--ask-for-approval", launch_gemma_codex.YOLO_CODEX_ARGS)
        self.assertIn("never", launch_gemma_codex.YOLO_CODEX_ARGS)

    def test_parse_launcher_args_keeps_direct_mode_by_default(self):
        parsed = launch_gemma_codex.parse_launcher_args(["--foo"])
        self.assertFalse(parsed.reasoning)
        self.assertEqual(parsed.codex_args, ["--foo"])

    def test_build_codex_environment_selects_local_home(self):
        root = Path("C:/Users/Agent-1/Desktop/gemma")
        env = launch_gemma_codex.build_codex_environment(root, reasoning=False, base_env={})
        self.assertEqual(env["CODEX_HOME"], str(root / ".codex-local"))

    def test_build_codex_environment_selects_reasoning_home(self):
        root = Path("C:/Users/Agent-1/Desktop/gemma")
        env = launch_gemma_codex.build_codex_environment(root, reasoning=True, base_env={})
        self.assertEqual(env["CODEX_HOME"], str(root / ".codex-local-reasoning"))

    def test_build_codex_command_preserves_target_dir(self):
        root = Path("C:/Users/Agent-1/Desktop/gemma")
        command = launch_gemma_codex.build_codex_command(root, Path("C:/work"), ["--foo"])
        self.assertEqual(command[-3:], ["--cd", "C:\\work", "--foo"])
        self.assertIn("codex.exe", command[0])

    def test_runtime_start_subprocess_uses_timeout(self):
        with (
            patch.object(launch_gemma_codex.subprocess, "run") as run_process,
            patch.object(launch_gemma_codex.subprocess, "call", return_value=0),
            patch.object(launch_gemma_codex.os, "getcwd", return_value="C:/work"),
            patch.dict(launch_gemma_codex.os.environ, {}, clear=True),
        ):
            launch_gemma_codex.run([], root=Path("C:/Users/Agent-1/Desktop/gemma"))

        self.assertEqual(run_process.call_args.kwargs["timeout"], launch_gemma_codex.STARTUP_TIMEOUT_SECONDS)


if __name__ == "__main__":
    unittest.main()
