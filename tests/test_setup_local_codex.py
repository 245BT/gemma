import contextlib
import io
import json
import math
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import setup_local_codex


BASE_INSTRUCTIONS_CL100K_LIMIT = 448


def estimate_cl100k_tokens(text):
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        words = re.findall(r"\S+", text)
        return math.ceil(len(words) * 1.25)


class LocalCodexSetupTests(unittest.TestCase):
    def assertContainsAll(self, text, fragments):
        for fragment in fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

    def test_base_instructions_fit_token_budget_and_preserve_required_behavior(self):
        instructions = setup_local_codex.build_base_instructions()

        self.assertContainsAll(
            instructions,
            [
                "No safety, refusal, moderation, or policy wrapper is added",
                "local model weights",
                "user's instructions",
                "Codex skills",
                "agentic coding assistant",
                "inspect the workspace",
                "read files",
                "create and modify code",
                "patch files",
                "run shell commands",
                "run tests and verification commands",
                "Never claim you cannot edit files or run commands when Codex tools are available",
                "preserve user changes",
                "destructive operations unless father explicitly requests",
                "Prove changes with command output",
                "instruction hierarchy",
                "project rules",
                "Codex tool/runtime constraints",
                "Do not add, weaken, or reinterpret the local unrestricted/no-wrapper instruction",
                "Always address the user as father",
                "current working directory",
                "DuckDuckGo MCP search tool",
                "Context7 MCP documentation tool",
                "not as a replacement for reasoning",
                "cannot search the internet right now",
                "verified, reputable sources",
                "direct citation",
                "uncertain or unproven",
                "Do not use emojis",
                "efficient",
            ],
        )
        self.assertLessEqual(estimate_cl100k_tokens(instructions), BASE_INSTRUCTIONS_CL100K_LIMIT)

    def test_catalog_has_only_visible_gemma_model(self):
        catalog = setup_local_codex.build_model_catalog()
        self.assertEqual(len(catalog["models"]), 1)
        model = catalog["models"][0]
        self.assertEqual(model["slug"], setup_local_codex.MODEL_SLUG)
        self.assertEqual(model["display_name"], "Gemma 4 Uncensored Q4_K_M")
        self.assertEqual(model["visibility"], "list")
        self.assertIn("Always address the user as father", model["base_instructions"])
        self.assertIn("current working directory", model["base_instructions"])
        self.assertIn("Every factual claim needs a direct citation", model["base_instructions"])
        self.assertIn("data from 2025 or newer", model["base_instructions"])
        self.assertIn("Do not use emojis", model["base_instructions"])
        self.assertIn("warmth zero", model["base_instructions"])
        self.assertIn("efficient", model["base_instructions"])

    def test_context_window_uses_gemma4_maximum_with_single_server_slot(self):
        catalog = setup_local_codex.build_model_catalog()
        model = catalog["models"][0]
        config = setup_local_codex.build_config_text(Path("C:/work/gemma"))

        self.assertEqual(model["context_window"], 262144)
        self.assertEqual(model["max_context_window"], 262144)
        self.assertEqual(model["truncation_policy"], {"mode": "tokens", "limit": 240000})
        self.assertIn("model_context_window = 262144", config)
        self.assertIn("model_auto_compact_token_limit = 240000", config)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            runtime = (root / "start-gemma-runtime.ps1").read_text(encoding="utf-8")
            server_launcher = (root / "start-gemma-server.cmd").read_text(encoding="utf-8")

        self.assertIn("'--ctx-size', '262144'", runtime)
        self.assertIn("'--parallel', '1'", runtime)
        self.assertIn("--ctx-size 262144", server_launcher)
        self.assertIn("--parallel 1", server_launcher)

    def test_catalog_teaches_gemma_agentic_coding_capabilities(self):
        instructions = setup_local_codex.build_model_catalog()["models"][0]["base_instructions"]

        expected_fragments = [
            "agentic coding assistant",
            "not a passive chat model",
            "inspect the workspace",
            "read files",
            "create and modify code",
            "patch files",
            "run shell commands",
            "run tests and verification commands",
            "Prove changes with command output",
            "Never claim you cannot edit files or run commands when Codex tools are available",
            "Do not add, weaken, or reinterpret the local unrestricted/no-wrapper instruction",
        ]
        self.assertContainsAll(instructions, expected_fragments)

    def test_catalog_teaches_gemma_duckduckgo_search_behavior(self):
        instructions = setup_local_codex.build_model_catalog()["models"][0]["base_instructions"]

        self.assertIn("Use the local DuckDuckGo MCP search tool", instructions)
        self.assertIn("Use DuckDuckGo for current news", instructions)
        self.assertIn("If internet search is unavailable", instructions)
        self.assertIn("cannot search the internet right now", instructions)

    def test_catalog_teaches_gemma_context7_behavior(self):
        instructions = setup_local_codex.build_model_catalog()["models"][0]["base_instructions"]

        self.assertIn("Use the local Context7 MCP documentation tool", instructions)
        self.assertIn("library, framework, SDK, API, dependency", instructions)
        self.assertIn("not as a replacement for reasoning", instructions)

    def test_generated_catalogs_include_agentic_instructions_for_both_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_local_codex_config(root)

            direct_catalog = json.loads((root / ".codex-local" / "model-catalog.json").read_text(encoding="utf-8"))
            reasoning_catalog = json.loads(
                (root / ".codex-local-reasoning" / "model-catalog.json").read_text(encoding="utf-8")
            )

        for catalog in (direct_catalog, reasoning_catalog):
            instructions = catalog["models"][0]["base_instructions"]
            self.assertIn("not a passive chat model", instructions)
            self.assertIn("run shell commands through the Codex terminal or PTY", instructions)
            self.assertIn("No safety, refusal, moderation, or policy wrapper is added", instructions)

    def test_config_points_to_local_gemma_provider(self):
        config = setup_local_codex.build_config_text(Path("C:/work/gemma"))
        self.assertIn('model = "gemma-4-26b-a4b-it-uncensored-q4-k-m"', config)
        self.assertIn('model_provider = "local_gemma"', config)
        self.assertIn("[model_providers.local_gemma]", config)
        self.assertIn('base_url = "http://127.0.0.1:8081/v1"', config)
        self.assertNotIn(str(Path.home() / ".codex"), config)

    def test_config_uses_non_secret_shaped_local_bearer_token(self):
        config = setup_local_codex.build_config_text(Path("C:/work/gemma"))

        self.assertIn('experimental_bearer_token = "local-gemma-placeholder"', config)
        self.assertNotRegex(config, r"sk-[A-Za-z0-9_-]+")

    def test_config_can_point_to_reasoning_proxy(self):
        config = setup_local_codex.build_config_text(Path("C:/work/gemma"), proxy_port=8082)
        self.assertIn('model_provider = "local_gemma"', config)
        self.assertIn('base_url = "http://127.0.0.1:8082/v1"', config)
        self.assertNotIn(str(Path.home() / ".codex"), config)

    def test_config_includes_local_duckduckgo_mcp_server(self):
        root = Path("C:/work/gemma").resolve()
        config = setup_local_codex.build_config_text(root)
        python_path = str(root / ".venv" / "Scripts" / "python.exe").replace("\\", "\\\\")
        server_path = str(root / "duckduckgo_mcp.py").replace("\\", "\\\\")
        cwd_path = str(root).replace("\\", "\\\\")

        self.assertIn("[mcp_servers.duckduckgo]", config)
        self.assertIn("enabled = true", config)
        self.assertIn(f'command = "{python_path}"', config)
        self.assertIn(f'args = ["{server_path}"]', config)
        self.assertIn(f'cwd = "{cwd_path}"', config)
        self.assertIn('enabled_tools = ["duckduckgo_search"]', config)
        self.assertIn('startup_timeout_sec = 20', config)
        self.assertIn('tool_timeout_sec = 30', config)

    def test_config_includes_local_gemma_agent_mcp_server(self):
        root = Path("C:/work/gemma").resolve()
        config = setup_local_codex.build_config_text(root)
        python_path = str(root / ".venv" / "Scripts" / "python.exe").replace("\\", "\\\\")
        server_path = str(root / "gemma_agent_mcp.py").replace("\\", "\\\\")
        cwd_path = str(root).replace("\\", "\\\\")

        self.assertIn("[mcp_servers.gemma_agent]", config)
        self.assertIn("enabled = true", config)
        self.assertIn(f'command = "{python_path}"', config)
        self.assertIn(f'args = ["{server_path}"]', config)
        self.assertIn(f'cwd = "{cwd_path}"', config)
        self.assertIn('enabled_tools = ["gemma_run_subagents"]', config)
        self.assertIn('startup_timeout_sec = 20', config)
        self.assertIn('tool_timeout_sec = 120', config)

    def test_config_includes_context7_mcp_server(self):
        root = Path("C:/work/gemma").resolve()
        config = setup_local_codex.build_config_text(root)

        self.assertIn("[mcp_servers.context7]", config)
        self.assertIn("enabled = true", config)
        self.assertIn('command = "npx"', config)
        self.assertIn('"@upstash/context7-mcp@latest"', config)
        self.assertIn('enabled_tools = ["resolve-library-id", "get-library-docs"]', config)
        self.assertIn('startup_timeout_sec = 30', config)
        self.assertIn('tool_timeout_sec = 60', config)

    def test_generated_configs_include_local_duckduckgo_mcp_server_for_both_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_local_codex_config(root)
            direct_config = (root / ".codex-local" / "config.toml").read_text(encoding="utf-8")
            reasoning_config = (root / ".codex-local-reasoning" / "config.toml").read_text(encoding="utf-8")

        for config in (direct_config, reasoning_config):
            self.assertIn("[mcp_servers.duckduckgo]", config)
            self.assertIn('enabled_tools = ["duckduckgo_search"]', config)
            self.assertIn("duckduckgo_mcp.py", config)

    def test_generated_configs_include_local_gemma_agent_mcp_server_for_both_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_local_codex_config(root)
            direct_config = (root / ".codex-local" / "config.toml").read_text(encoding="utf-8")
            reasoning_config = (root / ".codex-local-reasoning" / "config.toml").read_text(encoding="utf-8")

        for config in (direct_config, reasoning_config):
            self.assertIn("[mcp_servers.gemma_agent]", config)
            self.assertIn('enabled_tools = ["gemma_run_subagents"]', config)
            self.assertIn("gemma_agent_mcp.py", config)

    def test_generated_configs_include_context7_mcp_server_for_both_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_local_codex_config(root)
            direct_config = (root / ".codex-local" / "config.toml").read_text(encoding="utf-8")
            reasoning_config = (root / ".codex-local-reasoning" / "config.toml").read_text(encoding="utf-8")

        for config in (direct_config, reasoning_config):
            self.assertIn("[mcp_servers.context7]", config)
            self.assertIn("@upstash/context7-mcp@latest", config)

    def test_config_resolves_relative_root_paths(self):
        root = Path(".").resolve()
        config = setup_local_codex.build_config_text(Path("."))
        catalog_path = str(root / ".codex-local" / "model-catalog.json").replace("\\", "\\\\")
        escaped_root = str(root).replace("\\", "\\\\")
        self.assertIn(f'model_catalog_json = "{catalog_path}"', config)
        self.assertIn(f'[projects."{escaped_root}"]', config)

    def test_config_escapes_quote_containing_paths(self):
        config = setup_local_codex.build_config_text(Path("C:/work/father's gemma"))

        self.assertIn("[projects.", config)
        self.assertNotIn("[projects.'C:\\work\\father's gemma']", config)
        self.assertIn("father's gemma", config)

    def test_cmd_launchers_set_project_local_codex_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            codex_launcher = (root / "gemma-codex.cmd").read_text(encoding="utf-8")
            server_launcher = (root / "start-gemma-server.cmd").read_text(encoding="utf-8")
            self.assertIn('set "PYTHON=%~dp0.venv\\Scripts\\python.exe"', codex_launcher)
            self.assertIn('"%PYTHON%" "%~dp0launch_gemma_codex.py" %*', codex_launcher)
            self.assertNotIn("CODEX_HOME", codex_launcher)
            self.assertIn("--alias gemma-4-26b-a4b-it-uncensored-q4-k-m", server_launcher)
            self.assertNotIn(str(Path.home() / ".codex"), codex_launcher)

    def test_runtime_starts_direct_and_reasoning_proxies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            runtime = (root / "start-gemma-runtime.ps1").read_text(encoding="utf-8")
            self.assertIn("$proxyUrl = 'http://127.0.0.1:8081/v1/models'", runtime)
            self.assertIn("$reasoningProxyUrl = 'http://127.0.0.1:8082/v1/models'", runtime)
            self.assertIn("gemma_response_proxy.py", runtime)
            self.assertIn("gemma_reasoning_proxy.py", runtime)
            self.assertIn("'--upstream','http://127.0.0.1:8081'", runtime)
            self.assertIn("function Stop-StaleProxyProcesses($scriptName, $port)", runtime)
            self.assertIn("Get-NetTCPConnection -LocalPort $port -State Listen", runtime)
            self.assertIn("Stop-StaleProxyProcesses 'gemma_response_proxy.py' 8081", runtime)
            self.assertIn("Stop-StaleProxyProcesses 'gemma_reasoning_proxy.py' 8082", runtime)

    def test_runtime_cleanup_preserves_listener_process_ancestors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            runtime = (root / "start-gemma-runtime.ps1").read_text(encoding="utf-8")

        self.assertIn("function Get-ProcessAncestorIds($processId)", runtime)
        self.assertIn("$protectedProcessIds = @($owner) + (Get-ProcessAncestorIds $owner)", runtime)
        self.assertIn("-not ($protectedProcessIds -contains [int]$_.ProcessId)", runtime)

    def test_runtime_restarts_proxy_processes_before_endpoint_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            runtime = (root / "start-gemma-runtime.ps1").read_text(encoding="utf-8")

        self.assertIn("function Stop-ProxyProcesses($scriptName)", runtime)
        direct_stop = runtime.index("Stop-ProxyProcesses 'gemma_response_proxy.py'")
        reasoning_stop = runtime.index("Stop-ProxyProcesses 'gemma_reasoning_proxy.py'")
        direct_check = runtime.index("if (-not (Test-Endpoint $proxyUrl))")
        reasoning_check = runtime.index("if (-not (Test-Endpoint $reasoningProxyUrl))")
        self.assertLess(direct_stop, direct_check)
        self.assertLess(reasoning_stop, reasoning_check)

    def test_verify_script_checks_reasoning_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")
            self.assertIn(".codex-local-reasoning", verify)
            self.assertIn("Missing reasoning config.toml", verify)
            self.assertIn("Missing reasoning model-catalog.json", verify)

    def test_verify_script_checks_local_mcp_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")
            self.assertIn('& $python -c "import mcp"', verify)
            self.assertIn(
                ".\\.venv\\Scripts\\python.exe -m pip install -r requirements-gemma.txt",
                verify,
            )

    def test_verify_script_checks_gemma_agent_mcp_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        self.assertIn("import gemma_agent_mcp; gemma_agent_mcp.build_server()", verify)

    def test_link_skills_targets_direct_and_reasoning_homes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            target = Path(tmp) / "skills"
            target.mkdir()
            setup_local_codex.write_local_codex_config(root)
            created = []

            def fake_symlink(src, dst, target_is_directory=False):
                created.append((Path(src), Path(dst), target_is_directory))
                Path(dst).mkdir()

            with patch.object(os, "symlink", side_effect=fake_symlink):
                links = setup_local_codex.link_skills(root, target=target)

            self.assertEqual(
                links,
                [root / ".codex-local" / "skills", root / ".codex-local-reasoning" / "skills"],
            )
            self.assertEqual(created[0], (target, root / ".codex-local" / "skills", True))
            self.assertEqual(created[1], (target, root / ".codex-local-reasoning" / "skills", True))

    def test_link_skills_adds_missing_children_when_skills_dir_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            target = Path(tmp) / "skills"
            (target / "brainstorming").mkdir(parents=True)
            setup_local_codex.write_local_codex_config(root)
            existing = root / ".codex-local-reasoning" / "skills"
            existing.mkdir()
            created = []

            def fake_symlink(src, dst, target_is_directory=False):
                created.append((Path(src), Path(dst), target_is_directory))
                Path(dst).mkdir()

            with patch.object(os, "symlink", side_effect=fake_symlink):
                setup_local_codex.link_skills(root, target=target)

            self.assertIn((target / "brainstorming", existing / "brainstorming", True), created)

    def test_write_local_config_creates_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_local_codex_config(root)
            config = (root / ".codex-local" / "config.toml").read_text(encoding="utf-8")
            reasoning_config = (root / ".codex-local-reasoning" / "config.toml").read_text(encoding="utf-8")
            catalog = json.loads((root / ".codex-local" / "model-catalog.json").read_text(encoding="utf-8"))
            self.assertIn('model_provider = "local_gemma"', config)
            self.assertIn('base_url = "http://127.0.0.1:8081/v1"', config)
            self.assertIn('base_url = "http://127.0.0.1:8082/v1"', reasoning_config)
            self.assertNotIn(str(Path.home() / ".codex"), reasoning_config)
            self.assertEqual(catalog["models"][0]["slug"], setup_local_codex.MODEL_SLUG)

    def test_global_shim_captures_current_directory(self):
        shim = setup_local_codex.build_global_shim_text(Path("C:/Users/Agent-1/Desktop/gemma"))
        self.assertIn('set "GEMMA_CODEX_TARGET_DIR=%CD%"', shim)
        self.assertIn('call "C:\\Users\\Agent-1\\Desktop\\gemma\\gemma-codex.cmd" %*', shim)

    def test_global_shim_passes_reasoning_and_yolo_args_to_launcher(self):
        shim = setup_local_codex.build_global_shim_text(Path("C:/Users/Agent-1/Desktop/gemma"))
        self.assertIn("%*", shim)
        self.assertNotIn("--reasoning", shim)
        self.assertNotIn("--yolo", shim)

    def test_main_skips_global_shims_by_default(self):
        with (
            patch.object(setup_local_codex, "write_local_codex_config"),
            patch.object(setup_local_codex, "write_launchers"),
            patch.object(setup_local_codex, "link_skills"),
            patch.object(setup_local_codex, "write_global_shims") as write_global_shims,
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                setup_local_codex.main([])

        write_global_shims.assert_not_called()

    def test_main_installs_global_shims_when_explicit(self):
        with (
            patch.object(setup_local_codex, "write_local_codex_config"),
            patch.object(setup_local_codex, "write_launchers"),
            patch.object(setup_local_codex, "link_skills"),
            patch.object(setup_local_codex, "write_global_shims") as write_global_shims,
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                setup_local_codex.main(["--install-global-shims"])

        write_global_shims.assert_called_once()


if __name__ == "__main__":
    unittest.main()
