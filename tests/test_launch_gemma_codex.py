import contextlib
import io
import json
import unittest
import os
import tempfile
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

    def test_parse_launcher_args_accepts_common_reasoning_typo(self):
        parsed = launch_gemma_codex.parse_launcher_args(["--reasoing", "--foo", "bar"])
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

    def test_parse_launcher_args_supports_reasoning_typo_yolo_combo(self):
        parsed = launch_gemma_codex.parse_launcher_args(["--reasoing", "--yolo"])
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

    def test_parse_launcher_args_can_map_yolo_to_no_confirm_bypass(self):
        parsed = launch_gemma_codex.parse_launcher_args(
            ["--reasoning", "--yolo"],
            yolo_args=launch_gemma_codex.NO_CONFIRM_YOLO_CODEX_ARGS,
        )

        self.assertTrue(parsed.reasoning)
        self.assertTrue(parsed.yolo)
        self.assertEqual(parsed.codex_args, ["--dangerously-bypass-approvals-and-sandbox"])

    def test_run_uses_no_confirm_yolo_when_sonion_marker_is_set(self):
        with (
            patch.object(launch_gemma_codex.subprocess, "run"),
            patch.object(launch_gemma_codex, "run_codex_command", return_value=0) as run_codex,
            patch.object(launch_gemma_codex.os, "getcwd", return_value="C:/work"),
            patch.dict(
                launch_gemma_codex.os.environ,
                {launch_gemma_codex.SONION_NO_CONFIRM_YOLO_ENV: "1"},
                clear=True,
            ),
        ):
            launch_gemma_codex.run(["--reasoning", "--yolo"], root=Path("C:/Users/Agent-1/Desktop/gemma"))

        command = run_codex.call_args.args[0]
        env = run_codex.call_args.kwargs["env"]
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", command)
        self.assertNotIn("--ask-for-approval", command)
        self.assertEqual(env["CODEX_HOME"], "C:\\Users\\Agent-1\\Desktop\\gemma\\.codex-local-reasoning")

    def test_run_dry_run_prints_selected_home_and_args_without_starting_runtime(self):
        root = Path("C:/Users/Agent-1/Desktop/gemma")
        output = io.StringIO()
        with (
            patch.object(launch_gemma_codex.subprocess, "run") as start_runtime,
            patch.object(launch_gemma_codex, "run_codex_command") as run_codex,
            patch.dict(
                launch_gemma_codex.os.environ,
                {
                    "GEMMA_CODEX_DRY_RUN": "1",
                    "GEMMA_CODEX_TARGET_DIR": "C:/repo",
                    launch_gemma_codex.SONION_NO_CONFIRM_YOLO_ENV: "1",
                },
                clear=True,
            ),
            contextlib.redirect_stdout(output),
        ):
            code = launch_gemma_codex.run(["--reasoning", "--yolo", "exec", "hi"], root=root)

        self.assertEqual(code, 0)
        start_runtime.assert_not_called()
        run_codex.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["codex_home_name"], ".codex-local-reasoning")
        self.assertEqual(payload["target_dir"], "C:\\repo")
        self.assertTrue(payload["reasoning"])
        self.assertTrue(payload["yolo"])
        self.assertIn("exec", payload["codex_args"])
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", payload["codex_args"])
        self.assertIn("codex.exe", payload["selected_command"][0])

    def test_codex_binary_provenance_reports_selected_version_hash_and_npm_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "tools" / "codex-local" / "bin" / "codex.exe"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"codex fixture")

            provenance = launch_gemma_codex.codex_binary_provenance(
                root,
                version_runner=lambda command: "codex-cli 0.139.0",
            )

        self.assertEqual(provenance["selected_path"], str(binary))
        self.assertEqual(provenance["version"], "codex-cli 0.139.0")
        self.assertEqual(provenance["npm_version"], "0.139.0")
        self.assertRegex(provenance["sha256"], r"^[a-f0-9]{64}$")

    def test_run_covers_global_command_reasoning_and_yolo_variants(self):
        root = Path("C:/Users/Agent-1/Desktop/gemma")
        variants = [
            ("son", [], False, False, launch_gemma_codex.YOLO_CODEX_ARGS),
            ("sonion", [], False, True, launch_gemma_codex.NO_CONFIRM_YOLO_CODEX_ARGS),
            ("operator", [], False, False, launch_gemma_codex.YOLO_CODEX_ARGS),
            ("son", ["--reasoing"], True, False, launch_gemma_codex.YOLO_CODEX_ARGS),
            ("sonion", ["--reasoning"], True, True, launch_gemma_codex.NO_CONFIRM_YOLO_CODEX_ARGS),
            ("operator", ["--reasoing"], True, False, launch_gemma_codex.YOLO_CODEX_ARGS),
            ("son", ["--reasoing", "--yolo"], True, False, launch_gemma_codex.YOLO_CODEX_ARGS),
            ("sonion", ["--reasoning", "--yolo"], True, True, launch_gemma_codex.NO_CONFIRM_YOLO_CODEX_ARGS),
            ("operator", ["--reasoing", "--yolo"], True, False, launch_gemma_codex.YOLO_CODEX_ARGS),
            ("son", ["--yolo"], False, False, launch_gemma_codex.YOLO_CODEX_ARGS),
            ("sonion", ["--yolo"], False, True, launch_gemma_codex.NO_CONFIRM_YOLO_CODEX_ARGS),
            ("operator", ["--yolo"], False, False, launch_gemma_codex.YOLO_CODEX_ARGS),
        ]
        for command_name, argv, expects_reasoning, sonion_marker, expected_yolo_args in variants:
            with self.subTest(command_name=command_name, argv=argv):
                env_patch = (
                    {launch_gemma_codex.SONION_NO_CONFIRM_YOLO_ENV: "1"}
                    if sonion_marker
                    else {}
                )
                with (
                    patch.object(launch_gemma_codex.subprocess, "run"),
                    patch.object(launch_gemma_codex, "run_codex_command", return_value=0) as run_codex,
                    patch.object(launch_gemma_codex.os, "getcwd", return_value="C:/work"),
                    patch.dict(launch_gemma_codex.os.environ, env_patch, clear=True),
                ):
                    launch_gemma_codex.run(argv, root=root)

                launched_command = run_codex.call_args.args[0]
                launched_env = run_codex.call_args.kwargs["env"]
                expected_home = ".codex-local-reasoning" if expects_reasoning else ".codex-local"
                self.assertEqual(launched_env["CODEX_HOME"], str(root / expected_home))
                self.assertNotIn("--reasoning", launched_command)
                self.assertNotIn("--reasoing", launched_command)
                if "--yolo" in argv:
                    for item in expected_yolo_args:
                        self.assertIn(item, launched_command)
                else:
                    for item in launch_gemma_codex.YOLO_CODEX_ARGS:
                        self.assertNotIn(item, launched_command)
                    for item in launch_gemma_codex.NO_CONFIRM_YOLO_CODEX_ARGS:
                        self.assertNotIn(item, launched_command)

    def test_run_trusts_target_dir_for_sonion_no_confirm_yolo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "gemma"
            target_dir = Path(tmp) / "work repo"
            target_dir.mkdir()
            (root / "start-gemma-runtime.ps1").parent.mkdir(parents=True)
            (root / "start-gemma-runtime.ps1").write_text("", encoding="utf-8")
            for home_name in (".codex-local", ".codex-local-reasoning"):
                home = root / home_name
                home.mkdir()
                (home / "config.toml").write_text('model = "gemma"\n', encoding="utf-8")

            with (
                patch.object(launch_gemma_codex.subprocess, "run"),
                patch.object(launch_gemma_codex, "run_codex_command", return_value=0),
                patch.dict(
                    launch_gemma_codex.os.environ,
                    {
                        launch_gemma_codex.SONION_NO_CONFIRM_YOLO_ENV: "1",
                        "GEMMA_CODEX_TARGET_DIR": str(target_dir),
                    },
                    clear=True,
                ),
            ):
                launch_gemma_codex.run(["--yolo"], root=root)

            direct_config = (root / ".codex-local" / "config.toml").read_text(encoding="utf-8")
            reasoning_config = (root / ".codex-local-reasoning" / "config.toml").read_text(encoding="utf-8")

        self.assertIn(f"[projects.{launch_gemma_codex.toml_basic_string(target_dir)}]", direct_config)
        self.assertIn('trust_level = "trusted"', direct_config)
        self.assertNotIn(str(target_dir), reasoning_config)

    def test_run_trusts_target_dir_for_sonion_reasoning_no_confirm_yolo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "gemma"
            target_dir = Path(tmp) / "work repo"
            target_dir.mkdir()
            (root / "start-gemma-runtime.ps1").parent.mkdir(parents=True)
            (root / "start-gemma-runtime.ps1").write_text("", encoding="utf-8")
            for home_name in (".codex-local", ".codex-local-reasoning"):
                home = root / home_name
                home.mkdir()
                (home / "config.toml").write_text('model = "gemma"\n', encoding="utf-8")

            with (
                patch.object(launch_gemma_codex.subprocess, "run"),
                patch.object(launch_gemma_codex, "run_codex_command", return_value=0),
                patch.dict(
                    launch_gemma_codex.os.environ,
                    {
                        launch_gemma_codex.SONION_NO_CONFIRM_YOLO_ENV: "1",
                        "GEMMA_CODEX_TARGET_DIR": str(target_dir),
                    },
                    clear=True,
                ),
            ):
                launch_gemma_codex.run(["--reasoning", "--yolo"], root=root)

            direct_config = (root / ".codex-local" / "config.toml").read_text(encoding="utf-8")
            reasoning_config = (root / ".codex-local-reasoning" / "config.toml").read_text(encoding="utf-8")

        self.assertIn(f"[projects.{launch_gemma_codex.toml_basic_string(target_dir)}]", reasoning_config)
        self.assertIn('trust_level = "trusted"', reasoning_config)
        self.assertNotIn(str(target_dir), direct_config)

    def test_run_does_not_trust_target_dir_for_sonion_without_yolo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "gemma"
            target_dir = Path(tmp) / "work repo"
            target_dir.mkdir()
            (root / "start-gemma-runtime.ps1").parent.mkdir(parents=True)
            (root / "start-gemma-runtime.ps1").write_text("", encoding="utf-8")
            local_home = root / ".codex-local"
            local_home.mkdir()
            (local_home / "config.toml").write_text('model = "gemma"\n', encoding="utf-8")

            with (
                patch.object(launch_gemma_codex.subprocess, "run"),
                patch.object(launch_gemma_codex, "run_codex_command", return_value=0),
                patch.dict(
                    launch_gemma_codex.os.environ,
                    {
                        launch_gemma_codex.SONION_NO_CONFIRM_YOLO_ENV: "1",
                        "GEMMA_CODEX_TARGET_DIR": str(target_dir),
                    },
                    clear=True,
                ),
            ):
                launch_gemma_codex.run([], root=root)

            config = (local_home / "config.toml").read_text(encoding="utf-8")

        self.assertNotIn("[projects.", config)

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

    def test_build_codex_environment_adds_local_node_bin_to_path(self):
        root = Path("C:/Users/Agent-1/Desktop/gemma")
        env = launch_gemma_codex.build_codex_environment(
            root,
            reasoning=False,
            base_env={"PATH": "C:\\Windows\\System32"},
        )

        self.assertEqual(env["PATH"].split(os.pathsep)[0], str(root / "tools" / "bin"))
        self.assertEqual(env["PATH"].split(os.pathsep)[1], str(root / "node_modules" / ".bin"))
        self.assertIn("C:\\Windows\\System32", env["PATH"].split(os.pathsep))

    def test_build_codex_command_preserves_target_dir(self):
        root = Path("C:/Users/Agent-1/Desktop/gemma")
        command = launch_gemma_codex.build_codex_command(root, Path("C:/work"), ["--foo"])
        self.assertEqual(command[-3:], ["--cd", "C:\\work", "--foo"])
        self.assertIn("codex.exe", command[0])

    def test_build_codex_command_prefers_patched_local_codex_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patched = root / "tools" / "codex-local" / "bin" / "codex.exe"
            patched.parent.mkdir(parents=True)
            patched.write_text("", encoding="utf-8")

            command = launch_gemma_codex.build_codex_command(root, Path("C:/work"), ["--foo"])

        self.assertEqual(command[0], str(patched))

    def test_build_codex_command_forces_unified_exec_runtime_config(self):
        root = Path("C:/Users/Agent-1/Desktop/gemma")
        command = launch_gemma_codex.build_codex_command(root, Path("C:/work"), ["--yolo"])

        self.assertIn("--enable", command)
        self.assertIn("unified_exec", command)
        self.assertIn("-c", command)
        self.assertIn("background_terminal_max_timeout=3600000", command)
        self.assertEqual(command[-1], "--yolo")

    def test_json_exec_detection_skips_common_codex_options_with_values_before_exec(self):
        command = [
            "codex",
            "-m",
            "gemma",
            "-C",
            "C:/repo",
            "-s",
            "danger-full-access",
            "-a",
            "never",
            "-i",
            "screenshot.png",
            "--add-dir",
            "C:/other",
            "--color",
            "never",
            "exec",
            "--json",
            "task",
        ]

        self.assertTrue(launch_gemma_codex.is_json_exec_command(command))
        self.assertTrue(launch_gemma_codex.codex_command_uses_exec_timeout(command))

    def test_repair_codex_timeout_configs_updates_stale_local_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for home_name in (".codex-local", ".codex-local-reasoning"):
                config_dir = root / home_name
                config_dir.mkdir()
                (config_dir / "config.toml").write_text(
                    "\n".join(
                        [
                            'model = "gemma"',
                            "background_terminal_max_timeout = 300000",
                            "",
                            "[mcp_servers.duckduckgo]",
                            "tool_timeout_sec = 30",
                            "",
                            "[mcp_servers.gemma_agent]",
                            "startup_timeout_sec = 20",
                            "tool_timeout_sec = 300",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

            launch_gemma_codex.repair_codex_timeout_configs(root)

            for home_name in (".codex-local", ".codex-local-reasoning"):
                config = (root / home_name / "config.toml").read_text(encoding="utf-8")
                with self.subTest(home_name=home_name):
                    self.assertIn("background_terminal_max_timeout = 3600000", config)
                    self.assertIn("[mcp_servers.duckduckgo]\ntool_timeout_sec = 30", config)
                    self.assertIn("[mcp_servers.gemma_agent]\nstartup_timeout_sec = 20\ntool_timeout_sec = 1800", config)
                    self.assertNotIn("background_terminal_max_timeout = 300000", config)
                    self.assertNotIn("tool_timeout_sec = 300", config)

    def test_run_repairs_timeout_configs_before_starting_codex(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "start-gemma-runtime.ps1").write_text("", encoding="utf-8")
            local_home = root / ".codex-local"
            local_home.mkdir()
            (local_home / "config.toml").write_text(
                'model = "gemma"\nbackground_terminal_max_timeout = 300000\n',
                encoding="utf-8",
            )

            with (
                patch.object(launch_gemma_codex.subprocess, "run"),
                patch.object(launch_gemma_codex, "run_codex_command", return_value=0),
                patch.object(launch_gemma_codex.os, "getcwd", return_value=str(root)),
                patch.dict(launch_gemma_codex.os.environ, {}, clear=True),
            ):
                launch_gemma_codex.run([], root=root)

            config = (local_home / "config.toml").read_text(encoding="utf-8")

        self.assertIn("background_terminal_max_timeout = 3600000", config)

    def test_runtime_start_subprocess_uses_timeout(self):
        with (
            patch.object(launch_gemma_codex.subprocess, "run") as run_process,
            patch.object(launch_gemma_codex.subprocess, "call", return_value=0),
            patch.object(launch_gemma_codex.os, "getcwd", return_value="C:/work"),
            patch.dict(launch_gemma_codex.os.environ, {}, clear=True),
        ):
            launch_gemma_codex.run([], root=Path("C:/Users/Agent-1/Desktop/gemma"))

        self.assertEqual(run_process.call_args.kwargs["timeout"], launch_gemma_codex.STARTUP_TIMEOUT_SECONDS)

    def test_interactive_codex_chat_ignores_exec_timeout_env(self):
        with (
            patch.object(launch_gemma_codex.subprocess, "call", return_value=0) as call_process,
            patch.object(launch_gemma_codex.subprocess, "Popen") as popen_process,
        ):
            code = launch_gemma_codex.run_codex_command(
                ["codex", "--yolo"],
                cwd=Path("C:/repo"),
                env={"GEMMA_CODEX_EXEC_TIMEOUT": "1"},
            )

        self.assertEqual(code, 0)
        call_process.assert_called_once()
        popen_process.assert_not_called()

    def test_codex_exec_subprocess_can_use_configured_timeout(self):
        class FakeProcess:
            pid = 123

            def wait(self, timeout=None):
                self.timeout = timeout
                return 0

        fake_process = FakeProcess()
        with (
            patch.object(launch_gemma_codex.subprocess, "run") as run_process,
            patch.object(launch_gemma_codex.subprocess, "Popen", return_value=fake_process) as popen_process,
            patch.object(launch_gemma_codex.os, "getcwd", return_value="C:/work"),
            patch.dict(
                launch_gemma_codex.os.environ,
                {"GEMMA_CODEX_EXEC_TIMEOUT": "12"},
                clear=True,
            ),
        ):
            launch_gemma_codex.run(["exec", "task"], root=Path("C:/Users/Agent-1/Desktop/gemma"))

        self.assertEqual(run_process.call_count, 1)
        self.assertEqual(popen_process.call_count, 1)
        self.assertEqual(fake_process.timeout, 12)

    def test_codex_exec_timeout_returns_controlled_code(self):
        class FakeProcess:
            pid = 123

            def wait(self, timeout=None):
                raise launch_gemma_codex.subprocess.TimeoutExpired(["codex"], timeout)

        with (
            patch.object(launch_gemma_codex.subprocess, "Popen", return_value=FakeProcess()),
            patch.object(launch_gemma_codex, "terminate_process_tree") as terminate,
        ):
            code = launch_gemma_codex.run_codex_command(
                ["codex", "exec", "task"],
                cwd=Path("C:/repo"),
                env={"GEMMA_CODEX_EXEC_TIMEOUT": "1"},
            )

        self.assertEqual(code, launch_gemma_codex.CODEX_EXEC_TIMEOUT_RETURN_CODE)
        terminate.assert_called_once()

    def test_json_exec_recovers_completed_turn_without_visible_agent_message(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if len(calls) == 1:
                return launch_gemma_codex.subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        '{"type":"thread.started","thread_id":"019ec985-8af0"}\n'
                        '{"type":"turn.started"}\n'
                        '{"type":"turn.completed","usage":{"input_tokens":1}}\n'
                    ),
                    stderr="",
                )
            return launch_gemma_codex.subprocess.CompletedProcess(
                command,
                0,
                stdout='{"type":"turn.completed","last_agent_message":"visible"}\n',
                stderr="",
            )

        output = io.StringIO()
        with (
            patch.object(launch_gemma_codex.subprocess, "run", side_effect=fake_run),
            contextlib.redirect_stdout(output),
        ):
            code = launch_gemma_codex.run_codex_command(
                ["codex", "--enable", "unified_exec", "exec", "--json", "task"],
                cwd=Path("C:/repo"),
                env={},
            )

        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[1],
            [
                "codex",
                "--enable",
                "unified_exec",
                "exec",
                "resume",
                "--json",
                "019ec985-8af0",
                launch_gemma_codex.NULL_FINAL_RECOVERY_PROMPT,
            ],
        )
        self.assertIn('"thread.started"', output.getvalue())
        self.assertIn('"last_agent_message":"visible"', output.getvalue())

    def test_json_exec_recovers_completed_turn_with_empty_visible_fallback_message(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if len(calls) == 1:
                return launch_gemma_codex.subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        '{"type":"thread.started","thread_id":"019ec985-8af0"}\n'
                        '{"type":"item.completed","item":{"type":"agent_message","text":'
                        + json.dumps(launch_gemma_codex.EMPTY_VISIBLE_RESPONSE_MESSAGE)
                        + '}}\n'
                        '{"type":"turn.completed","usage":{"input_tokens":1}}\n'
                    ),
                    stderr="",
                )
            return launch_gemma_codex.subprocess.CompletedProcess(
                command,
                0,
                stdout='{"type":"item.completed","item":{"type":"agent_message","text":"real answer"}}\n',
                stderr="",
            )

        output = io.StringIO()
        with (
            patch.object(launch_gemma_codex.subprocess, "run", side_effect=fake_run),
            contextlib.redirect_stdout(output),
        ):
            code = launch_gemma_codex.run_codex_command(
                ["codex", "exec", "--json", "task"],
                cwd=Path("C:/repo"),
                env={},
            )

        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][-2:], ["019ec985-8af0", launch_gemma_codex.NULL_FINAL_RECOVERY_PROMPT])
        self.assertIn('"text":"real answer"', output.getvalue())
        self.assertNotIn(launch_gemma_codex.EMPTY_VISIBLE_RESPONSE_MESSAGE, output.getvalue())

    def test_json_exec_recovers_empty_assistant_message_content(self):
        first_stdout = (
            '{"type":"thread.started","thread_id":"019ec985-8af0"}\n'
            '{"type":"item.completed","item":{"type":"message","role":"assistant","content":[]}}\n'
            '{"type":"turn.completed","usage":{"input_tokens":1}}\n'
        )

        summary = launch_gemma_codex.summarize_json_exec_output(first_stdout)

        self.assertTrue(summary["needs_recovery"])

    def test_json_exec_recovers_whitespace_only_last_agent_message(self):
        first_stdout = (
            '{"type":"thread.started","thread_id":"019ec985-8af0"}\n'
            '{"type":"turn.completed","last_agent_message":"   "}\n'
        )

        summary = launch_gemma_codex.summarize_json_exec_output(first_stdout)

        self.assertTrue(summary["needs_recovery"])

    def test_json_exec_does_not_recover_visible_agent_message(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return launch_gemma_codex.subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    '{"type":"thread.started","thread_id":"019ec985-8af0"}\n'
                    '{"type":"item.completed","item":{"type":"message","role":"assistant","text":"ok"}}\n'
                    '{"type":"turn.completed","usage":{"input_tokens":1}}\n'
                ),
                stderr="",
            )

        with (
            patch.object(launch_gemma_codex.subprocess, "run", side_effect=fake_run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = launch_gemma_codex.run_codex_command(
                ["codex", "exec", "--json", "task"],
                cwd=Path("C:/repo"),
                env={},
            )

        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
