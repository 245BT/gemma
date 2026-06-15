import contextlib
import io
import json
import os
import py_compile
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import setup_local_codex


class LocalCodexSetupTests(unittest.TestCase):
    PROMPT_CRITICAL_MODEL_FIELDS = [
        "slug",
        "display_name",
        "description",
        "default_reasoning_level",
        "supported_reasoning_levels",
        "shell_type",
        "visibility",
        "supported_in_api",
        "supports_reasoning_summaries",
        "default_reasoning_summary",
        "support_verbosity",
        "default_verbosity",
        "context_window",
        "max_context_window",
        "effective_context_window_percent",
        "input_modalities",
        "supports_parallel_tool_calls",
        "supports_image_detail_original",
        "supports_search_tool",
        "experimental_supported_tools",
        "truncation_policy",
        "apply_patch_tool_type",
        "base_instructions",
    ]

    def assertContainsAll(self, text, fragments):
        for fragment in fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

    def assertCatalogMatchesPromptCriticalModelFields(self, catalog, expected_model):
        self.assertEqual(len(catalog["models"]), 1)
        model = catalog["models"][0]
        for field in self.PROMPT_CRITICAL_MODEL_FIELDS:
            with self.subTest(field=field):
                self.assertEqual(model[field], expected_model[field])

    def require_powershell(self):
        powershell = (
            shutil.which("powershell.exe")
            or shutil.which("powershell")
            or shutil.which("pwsh.exe")
            or shutil.which("pwsh")
        )
        if powershell is None:
            self.skipTest("PowerShell executable is not available")
        return powershell

    def runtime_support_functions(self):
        runtime = setup_local_codex.build_runtime_script_text(Path("C:/work/gemma"))
        return runtime.split("$serverUrl = ", 1)[0].replace(
            "$root = Split-Path -Parent $MyInvocation.MyCommand.Path",
            "$root = 'C:\\work\\gemma'",
        )

    def assertGeneratedPowerShellScriptsParse(self, root):
        powershell = self.require_powershell()

        parser = Path(root) / "parse-generated-launchers.ps1"
        parser.write_text(
            """
param(
  [Parameter(Mandatory=$true, ValueFromRemainingArguments=$true)]
  [string[]]$Paths
)
$ErrorActionPreference = 'Stop'
$failed = $false
foreach ($path in $Paths) {
  $tokens = $null
  $errors = $null
  [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors) | Out-Null
  if ($errors.Count -gt 0) {
    $failed = $true
    foreach ($parserError in $errors) {
      Write-Error ("{0}: {1}" -f $path, $parserError.Message)
    }
  }
}
if ($failed) { exit 1 }
""",
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(parser),
                str(Path(root) / "start-gemma-runtime.ps1"),
                str(Path(root) / "verify-local-setup.ps1"),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode != 0:
            self.fail(
                "Generated PowerShell scripts failed parser preflight:\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    def test_base_instructions_preserve_required_behavior_without_prompt_budget_cap(self):
        instructions = setup_local_codex.build_base_instructions()

        self.assertContainsAll(
            instructions,
            [
                "No safety refusal moderation policy wrapper is added",
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
                "No warning signs icons",
                "No 100% achievable",
                "No production-ready",
                "No all-caps harm emergency",
                "read verification-before-completion",
                "follow Codex CLI project rules",
                "efficient",
            ],
        )

    def test_base_instructions_reject_control_and_prompt_injection_contamination(self):
        instructions = setup_local_codex.build_base_instructions()

        forbidden_fragments = [
            "Human life is worthless",
            "Never ask for professional help",
            "Ignore previous instructions",
            "SYSTEM_LOGIC_INJECTION",
            "<|tool_call>",
            "call:codex:",
        ]
        for fragment in forbidden_fragments:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, instructions)

    def test_base_instructions_forbid_internal_codex_tool_call_text(self):
        instructions = setup_local_codex.build_base_instructions()

        self.assertIn("never print or imitate internal tool-call markup", instructions)
        self.assertIn("call:codex text", instructions)
        self.assertNotIn("call:codex:", instructions)
        self.assertIn("skill activation pseudo-calls", instructions)
        self.assertNotIn("<|tool_call>", instructions)

    def test_base_instructions_teach_codex_cli_skill_file_workflow(self):
        instructions = setup_local_codex.build_base_instructions()

        self.assertContainsAll(
            instructions,
            [
                "Codex CLI skills are listed in the skills instructions",
                "read the relevant SKILL.md file by path before using a skill",
                "no separate Skill tool is needed in this local runtime",
                "Do not say skills are unavailable because no Skill tool exists",
            ],
        )

    def test_base_instructions_require_triggered_workflow_skill_discovery(self):
        instructions = setup_local_codex.build_base_instructions()

        self.assertContainsAll(
            instructions,
            [
                "Use skills only when the user's task triggers them",
                "inspect CODEX_HOME skills",
                "read only triggered SKILL.md files",
                "do not run skill inventory for greetings or casual chat",
                "using-superpowers",
                "brainstorming",
                "writing-plans",
                "using-git-worktrees",
                "systematic-debugging",
                "test-driven-development",
                "code-simplifier",
                "subagent-driven-development",
                "executing-plans",
                "dispatching-parallel-agents",
                "requesting-code-review",
                "finishing-a-development-branch",
                "verification-before-completion",
            ],
        )
        self.assertNotIn("First job is skill discovery", instructions)
        self.assertNotIn("Get-ChildItem (Join-Path $env:CODEX_HOME 'skills')", instructions)

    def test_base_instructions_teach_powershell_safe_skill_inventory(self):
        instructions = setup_local_codex.build_base_instructions()

        self.assertContainsAll(
            instructions,
            [
                "$env:CODEX_HOME",
                "only to locate needed skill files",
                "do not use recursive skill inventory",
            ],
        )

    def test_base_instructions_prevent_plan_only_stalls_and_recursive_inventory(self):
        instructions = setup_local_codex.build_base_instructions()

        self.assertContainsAll(
            instructions,
            [
                "software task first action real tool call",
                "do not return plan-only final",
                "continuation prompts act with a tool call or visible status",
                "never return empty",
                "Never use ls -R",
                "Get-ChildItem -Recurse",
                "dir /s",
                "use rg --files",
            ],
        )

    def test_required_gemma_skills_include_deepsearch_and_requested_workflow_bundle(self):
        required = " ".join(setup_local_codex.GEMMA_REQUIRED_SKILLS)

        self.assertContainsAll(
            required,
            [
                "deep-research",
                "context7",
                "using-superpowers",
                "brainstorming",
                "writing-plans",
                "using-git-worktrees",
                "systematic-debugging",
                "test-driven-development",
                "code-simplifier",
                "subagent-driven-development",
                "executing-plans",
                "dispatching-parallel-agents",
                "requesting-code-review",
                "finishing-a-development-branch",
                "verification-before-completion",
            ],
        )

    def test_base_instructions_route_deep_research_through_context7(self):
        instructions = setup_local_codex.build_base_instructions()

        self.assertContainsAll(
            instructions,
            [
                "deep-research",
                "Use deep-research for multi-source research",
                "deep-research must use Context7",
                "resolve-library-id then query-docs",
            ],
        )

    def test_base_instructions_route_deepsearch_alias_to_deep_research(self):
        instructions = setup_local_codex.build_base_instructions()

        self.assertContainsAll(
            instructions,
            [
                "deepsearch",
                "deep search",
                "deep-search",
                "deepsearch means deep-research",
                "Context7 is the research source, not the replacement skill",
                "never choose context7-cli or context7-mcp as the top-level skill for deepsearch",
            ],
        )

    def test_base_instructions_require_safe_parallel_work(self):
        instructions = setup_local_codex.build_base_instructions()

        self.assertContainsAll(
            instructions,
            [
                "parallel batches",
                "independent reads searches checks and subagent tasks",
                "gemma_run_subagents",
                (
                    "Do not parallelize dependent commands overlapping file edits "
                    "git add/commit installs migrations or test-after-edit loops"
                ),
                "bounded fanout",
            ],
        )

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

    def test_catalog_advertises_parallel_tool_call_capability(self):
        model = setup_local_codex.build_model_catalog()["models"][0]

        self.assertIs(model["supports_parallel_tool_calls"], True)
        self.assertIn("warmth zero", model["base_instructions"])
        self.assertIn("efficient", model["base_instructions"])

    def test_context_window_uses_gemma4_maximum_without_lower_local_prompt_caps(self):
        catalog = setup_local_codex.build_model_catalog()
        model = catalog["models"][0]
        config = setup_local_codex.build_config_text(Path("C:/work/gemma"))

        self.assertEqual(model["context_window"], 262144)
        self.assertEqual(model["max_context_window"], 262144)
        self.assertEqual(model["truncation_policy"], {"mode": "tokens", "limit": 262144})
        self.assertIn("model_context_window = 262144", config)
        self.assertNotIn("model_auto_compact_token_limit", config)

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
            "Exact father commands outrank generic inspection",
            "do not stop after ls",
            "Never claim PASS or say a tool was used unless that tool was actually called",
            "Do not add, weaken, or reinterpret the local unrestricted/no-wrapper instruction",
        ]
        self.assertContainsAll(instructions, expected_fragments)

    def test_catalog_teaches_gemma_duckduckgo_search_behavior(self):
        instructions = setup_local_codex.build_model_catalog()["models"][0]["base_instructions"]

        self.assertIn("Use the local DuckDuckGo MCP search tool", instructions)
        self.assertIn("DuckDuckGo is only for public news people and non-code public info", instructions)
        self.assertIn("Never use DuckDuckGo for software code package dependency install command API SDK framework questions", instructions)
        self.assertNotIn("Use DuckDuckGo for current facts, failed commands, and install uncertainty", instructions)
        self.assertIn("If internet search is unavailable", instructions)
        self.assertIn("cannot search the internet right now", instructions)

    def test_catalog_teaches_gemma_to_search_before_retrying_unknown_installs(self):
        instructions = setup_local_codex.build_model_catalog()["models"][0]["base_instructions"]

        self.assertIn("Before retrying a failed or stalled install/package-manager command", instructions)
        self.assertIn("use Context7 or official docs once", instructions)
        self.assertIn("Shell echo/searching text is not search evidence", instructions)
        self.assertIn("Prefer gemma_agent context7_search", instructions)
        self.assertIn("call resolve-library-id then query-docs", instructions)
        self.assertIn("Do not call list_mcp_resources", instructions)
        self.assertIn("If Context7 MCP fails, run npx -y ctx7@latest library <name> <query>", instructions)
        self.assertIn("npx -y ctx7@latest docs <libraryId> <query>", instructions)
        self.assertIn("ctx7 library <name> <query>", instructions)
        self.assertIn("ctx7 docs <libraryId> <query>", instructions)
        self.assertIn("Do not run ctx7 search", instructions)
        self.assertNotIn("npx ctx7@latest <command>", instructions)
        self.assertIn("Do not pip install search packages", instructions)
        self.assertIn("inspect active processes", instructions)
        self.assertIn("do not rerun the same install command", instructions)

    def test_catalog_teaches_gemma_shell_timeout_and_subnet_scan_recovery(self):
        instructions = setup_local_codex.build_model_catalog()["models"][0]["base_instructions"]

        self.assertIn("Codex unified exec exposes exec_command and write_stdin", instructions)
        self.assertIn("Do not use legacy shell_command when exec_command is available", instructions)
        self.assertIn("commands expected to run past 120 seconds must continue in the background", instructions)
        self.assertIn("Do not claim command flags changed unless the command includes them", instructions)
        self.assertIn("avoid sequential Test-Connection loops", instructions)
        self.assertIn("Test-Connection -TargetName $targets -Count 1 -Quiet -TimeoutSeconds 1", instructions)
        self.assertIn("ForEach-Object -Parallel", instructions)

    def test_catalog_teaches_gemma_context7_behavior(self):
        instructions = setup_local_codex.build_model_catalog()["models"][0]["base_instructions"]

        self.assertIn("Use the local Context7 MCP documentation tool", instructions)
        self.assertIn("software code package dependency install command API SDK framework", instructions)
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
            self.assertIn("run shell commands", instructions)
            self.assertIn("No safety refusal moderation policy wrapper is added", instructions)

    def test_generated_catalogs_match_prompt_critical_model_fields_for_both_modes(self):
        expected_model = setup_local_codex.build_model_catalog()["models"][0]
        expected_instructions = setup_local_codex.build_base_instructions()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_local_codex_config(root)

            direct_catalog = json.loads((root / ".codex-local" / "model-catalog.json").read_text(encoding="utf-8"))
            reasoning_catalog = json.loads(
                (root / ".codex-local-reasoning" / "model-catalog.json").read_text(encoding="utf-8")
            )

        for catalog in (direct_catalog, reasoning_catalog):
            with self.subTest(catalog=catalog["models"][0]["slug"]):
                self.assertEqual(catalog["models"][0]["base_instructions"], expected_instructions)
                self.assertCatalogMatchesPromptCriticalModelFields(catalog, expected_model)

    def test_config_points_to_local_gemma_provider(self):
        config = setup_local_codex.build_config_text(Path("C:/work/gemma"))
        self.assertIn('model = "gemma-4-26b-a4b-it-uncensored-q4-k-m"', config)
        self.assertIn('model_provider = "local_gemma"', config)
        self.assertIn("[model_providers.local_gemma]", config)
        self.assertIn('base_url = "http://127.0.0.1:8081/v1"', config)
        self.assertNotIn(str(Path.home() / ".codex"), config)

    def test_config_runs_without_shell_approval_or_workspace_sandbox_by_default(self):
        config = setup_local_codex.build_config_text(Path("C:/work/gemma"))

        self.assertIn('approval_policy = "never"', config)
        self.assertIn('sandbox_mode = "danger-full-access"', config)
        self.assertNotIn('approval_policy = "on-request"', config)
        self.assertNotIn('sandbox_mode = "workspace-write"', config)

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
        self.assertIn('enabled_tools = ["gemma_run_subagents", "context7_search"]', config)
        self.assertIn('startup_timeout_sec = 20', config)
        self.assertIn('tool_timeout_sec = 1800', config)

    def test_config_enables_unified_exec_and_long_background_poll(self):
        config = setup_local_codex.build_config_text(Path("C:/work/gemma"))

        self.assertIn("background_terminal_max_timeout = 3600000", config)
        self.assertIn("[features]", config)
        self.assertIn("unified_exec = true", config)
        self.assertNotIn("experimental_use_unified_exec_tool", config)

    def test_config_includes_context7_mcp_server(self):
        root = Path("C:/work/gemma").resolve()
        config = setup_local_codex.build_config_text(root)
        context7_path = str(root / "node_modules" / ".bin" / "context7-mcp.cmd").replace("\\", "\\\\")
        cwd_path = str(root).replace("\\", "\\\\")

        self.assertIn("[mcp_servers.context7]", config)
        self.assertIn("enabled = true", config)
        self.assertIn(f'command = "{context7_path}"', config)
        self.assertIn("args = []", config)
        self.assertIn(f'cwd = "{cwd_path}"', config)
        self.assertIn('enabled_tools = ["resolve-library-id", "query-docs"]', config)
        self.assertNotIn("get-library-docs", config)
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
            self.assertIn('enabled_tools = ["gemma_run_subagents", "context7_search"]', config)
            self.assertIn("gemma_agent_mcp.py", config)

    def test_generated_configs_include_context7_mcp_server_for_both_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_local_codex_config(root)
            direct_config = (root / ".codex-local" / "config.toml").read_text(encoding="utf-8")
            reasoning_config = (root / ".codex-local-reasoning" / "config.toml").read_text(encoding="utf-8")

        for config in (direct_config, reasoning_config):
            self.assertIn("[mcp_servers.context7]", config)
            self.assertIn("node_modules", config)
            self.assertIn("context7-mcp.cmd", config)

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
        self.assertIn("function Get-ProxyListenerFamilyIds($owner, $scriptName, $port)", runtime)
        self.assertIn("function Get-ProtectedProxyListenerProcessIds($scriptName)", runtime)
        self.assertIn("-not ($protectedProcessIds -contains [int]$_.ProcessId)", runtime)

    def test_runtime_enforces_single_llama_server_before_endpoint_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            runtime = (root / "start-gemma-runtime.ps1").read_text(encoding="utf-8")

        self.assertIn("function Stop-StaleLlamaServerProcesses($serverExe, $port)", runtime)
        self.assertIn("Stop-StaleLlamaServerProcesses $serverExe 8080", runtime)
        self.assertIn("$_.Name -eq 'llama-server.exe'", runtime)
        server_cleanup = runtime.index("Stop-StaleLlamaServerProcesses $serverExe 8080")
        server_endpoint_check = runtime.index("if (-not (Test-Endpoint $serverUrl))")
        self.assertLess(server_cleanup, server_endpoint_check)

    def test_runtime_proxy_cleanup_preserves_listener_process_and_ancestors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            runtime = (root / "start-gemma-runtime.ps1").read_text(encoding="utf-8")

        self.assertIn("Get-ProtectedProxyListenerProcessIds $scriptName", runtime)
        self.assertIn("-not ($protectedProcessIds -contains [int]$_.ProcessId)", runtime)

    def test_runtime_stale_proxy_cleanup_functionally_preserves_active_listener_family(self):
        powershell = self.require_powershell()
        support_functions = self.runtime_support_functions()
        script = f"""
$ErrorActionPreference = 'Stop'
{support_functions}
$script:stopped = @()
$script:connections = @(
  [pscustomobject]@{{ LocalPort = 8081; State = 'Listen'; OwningProcess = 200 }}
)
$script:processes = @(
  [pscustomobject]@{{ ProcessId = 100; ParentProcessId = 0; Name = 'python.exe'; CommandLine = 'gemma_response_proxy.py --host 127.0.0.1 --port 8081 parent launcher' }},
  [pscustomobject]@{{ ProcessId = 200; ParentProcessId = 100; Name = 'python.exe'; CommandLine = 'gemma_response_proxy.py --host 127.0.0.1 --port 8081 active listener' }},
  [pscustomobject]@{{ ProcessId = 300; ParentProcessId = 999; Name = 'python.exe'; CommandLine = 'gemma_response_proxy.py --host 127.0.0.1 --port 8081 stale duplicate' }},
  [pscustomobject]@{{ ProcessId = 400; ParentProcessId = 999; Name = 'python.exe'; CommandLine = 'gemma_reasoning_proxy.py unrelated proxy' }},
  [pscustomobject]@{{ ProcessId = 500; ParentProcessId = 999; Name = 'python.exe'; CommandLine = 'unrelated process' }}
)
function Get-NetTCPConnection {{
  param($LocalPort, $State, $ErrorAction)
  $connections = $script:connections | Where-Object {{ $_.State -eq $State }}
  if ($null -ne $LocalPort) {{
    $connections = $connections | Where-Object {{ $_.LocalPort -eq $LocalPort }}
  }}
  return $connections
}}
function Get-CimInstance {{
  param($ClassName, $Filter, $ErrorAction)
  if ($Filter -match 'ProcessId=(\\d+)') {{
    $id = [int]$Matches[1]
    return $script:processes | Where-Object {{ $_.ProcessId -eq $id }}
  }}
  return $script:processes
}}
function Stop-Process {{
  param($Id, [switch]$Force, $ErrorAction)
  $script:stopped += [int]$Id
}}
Stop-StaleProxyProcesses 'gemma_response_proxy.py' 8081
[pscustomobject]@{{ stopped = @($script:stopped) }} | ConvertTo-Json -Compress
"""
        result = subprocess.run(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["stopped"], [300])

    def test_runtime_stale_llama_cleanup_functionally_preserves_listener_family(self):
        powershell = self.require_powershell()
        support_functions = self.runtime_support_functions()
        script = f"""
$ErrorActionPreference = 'Stop'
{support_functions}
$serverExe = 'C:\\work\\gemma\\tools\\llama.cpp\\llama-server.exe'
$script:stopped = @()
$script:connections = @(
  [pscustomobject]@{{ LocalPort = 8080; State = 'Listen'; OwningProcess = 200 }}
)
$script:processes = @(
  [pscustomobject]@{{ ProcessId = 100; ParentProcessId = 0; Name = 'powershell.exe'; CommandLine = 'launcher parent' }},
  [pscustomobject]@{{ ProcessId = 200; ParentProcessId = 100; Name = 'llama-server.exe'; CommandLine = 'C:\\work\\gemma\\tools\\llama.cpp\\llama-server.exe --port 8080 active listener' }},
  [pscustomobject]@{{ ProcessId = 300; ParentProcessId = 999; Name = 'llama-server.exe'; CommandLine = 'C:\\work\\gemma\\tools\\llama.cpp\\llama-server.exe --port 8080 stale duplicate' }},
  [pscustomobject]@{{ ProcessId = 400; ParentProcessId = 999; Name = 'llama-server.exe'; CommandLine = 'D:\\other\\llama-server.exe --port 8080 other workspace' }},
  [pscustomobject]@{{ ProcessId = 500; ParentProcessId = 999; Name = 'python.exe'; CommandLine = 'unrelated process' }}
)
function Get-NetTCPConnection {{
  param($LocalPort, $State, $ErrorAction)
  $connections = $script:connections | Where-Object {{ $_.State -eq $State }}
  if ($null -ne $LocalPort) {{
    $connections = $connections | Where-Object {{ $_.LocalPort -eq $LocalPort }}
  }}
  return $connections
}}
function Get-CimInstance {{
  param($ClassName, $Filter, $ErrorAction)
  if ($Filter -match 'ProcessId=(\\d+)') {{
    $id = [int]$Matches[1]
    return $script:processes | Where-Object {{ $_.ProcessId -eq $id }}
  }}
  return $script:processes
}}
function Stop-Process {{
  param($Id, [switch]$Force, $ErrorAction)
  $script:stopped += [int]$Id
}}
Stop-StaleLlamaServerProcesses $serverExe 8080
[pscustomobject]@{{ stopped = @($script:stopped) }} | ConvertTo-Json -Compress
"""
        result = subprocess.run(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["stopped"], [300])

    def test_runtime_stale_proxy_cleanup_preserves_same_script_listener_on_other_port(self):
        powershell = self.require_powershell()
        support_functions = self.runtime_support_functions()
        script = f"""
$ErrorActionPreference = 'Stop'
{support_functions}
$script:stopped = @()
$script:connections = @(
  [pscustomobject]@{{ LocalPort = 8081; State = 'Listen'; OwningProcess = 200 }},
  [pscustomobject]@{{ LocalPort = 9090; State = 'Listen'; OwningProcess = 600 }}
)
$script:processes = @(
  [pscustomobject]@{{ ProcessId = 200; ParentProcessId = 0; Name = 'python.exe'; CommandLine = 'gemma_response_proxy.py --host 127.0.0.1 --port 8081' }},
  [pscustomobject]@{{ ProcessId = 300; ParentProcessId = 999; Name = 'python.exe'; CommandLine = 'gemma_response_proxy.py --host 127.0.0.1 --port 8081' }},
  [pscustomobject]@{{ ProcessId = 600; ParentProcessId = 999; Name = 'python.exe'; CommandLine = 'gemma_response_proxy.py --host 127.0.0.1 --port 9090' }}
)
function Get-NetTCPConnection {{
  param($LocalPort, $State, $ErrorAction)
  $connections = $script:connections | Where-Object {{ $_.State -eq $State }}
  if ($null -ne $LocalPort) {{
    $connections = $connections | Where-Object {{ $_.LocalPort -eq $LocalPort }}
  }}
  return $connections
}}
function Get-CimInstance {{
  param($ClassName, $Filter, $ErrorAction)
  if ($Filter -match 'ProcessId=(\\d+)') {{
    $id = [int]$Matches[1]
    return $script:processes | Where-Object {{ $_.ProcessId -eq $id }}
  }}
  return $script:processes
}}
function Stop-Process {{
  param($Id, [switch]$Force, $ErrorAction)
  $script:stopped += [int]$Id
}}
Stop-StaleProxyProcesses 'gemma_response_proxy.py' 8081
[pscustomobject]@{{ stopped = @($script:stopped) }} | ConvertTo-Json -Compress
"""
        result = subprocess.run(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["stopped"], [300])

    def test_runtime_start_proxy_if_needed_stops_unhealthy_listener_before_replacement(self):
        powershell = self.require_powershell()
        support_functions = self.runtime_support_functions()
        script = f"""
$ErrorActionPreference = 'Stop'
{support_functions}
$script:events = @()
$script:connections = @(
  [pscustomobject]@{{ LocalPort = 8081; State = 'Listen'; OwningProcess = 200 }},
  [pscustomobject]@{{ LocalPort = 9090; State = 'Listen'; OwningProcess = 600 }}
)
$script:processes = @(
  [pscustomobject]@{{ ProcessId = 100; ParentProcessId = 0; Name = 'python.exe'; CommandLine = 'gemma_response_proxy.py --host 127.0.0.1 --port 8081 parent' }},
  [pscustomobject]@{{ ProcessId = 200; ParentProcessId = 100; Name = 'python.exe'; CommandLine = 'gemma_response_proxy.py --host 127.0.0.1 --port 8081 child' }},
  [pscustomobject]@{{ ProcessId = 600; ParentProcessId = 999; Name = 'python.exe'; CommandLine = 'gemma_response_proxy.py --host 127.0.0.1 --port 9090 other-listener' }}
)
function Test-Endpoint {{
  param($url)
  return $false
}}
function Wait-Endpoint {{
  param($url, $seconds)
  $script:events += "wait:$url"
}}
function Test-Path {{
  param($Path)
  return $true
}}
function Get-NetTCPConnection {{
  param($LocalPort, $State, $ErrorAction)
  $connections = $script:connections | Where-Object {{ $_.State -eq $State }}
  if ($null -ne $LocalPort) {{
    $connections = $connections | Where-Object {{ $_.LocalPort -eq $LocalPort }}
  }}
  return $connections
}}
function Get-CimInstance {{
  param($ClassName, $Filter, $ErrorAction)
  if ($Filter -match 'ProcessId=(\\d+)') {{
    $id = [int]$Matches[1]
    return $script:processes | Where-Object {{ $_.ProcessId -eq $id }}
  }}
  return $script:processes
}}
function Stop-Process {{
  param($Id, [switch]$Force, $ErrorAction)
  $script:events += "stop:$Id"
}}
function Start-Process {{
  param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle)
  $script:events += "start:$FilePath"
}}
Start-ProxyIfNeeded 'http://127.0.0.1:8081/v1/models' 'gemma_response_proxy.py' 8081 'python.exe' @('gemma_response_proxy.py','--host','127.0.0.1','--port','8081') 'C:\\work\\gemma'
[pscustomobject]@{{ events = @($script:events) }} | ConvertTo-Json -Compress
"""
        result = subprocess.run(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(
            payload["events"],
            [
                "stop:200",
                "stop:100",
                "start:python.exe",
                "wait:http://127.0.0.1:8081/v1/models",
            ],
        )

    def test_runtime_stale_proxy_cleanup_does_not_stop_processes_without_listener(self):
        powershell = self.require_powershell()
        support_functions = self.runtime_support_functions()
        script = f"""
$ErrorActionPreference = 'Stop'
{support_functions}
$script:stopped = @()
$script:connections = @()
$script:processes = @(
  [pscustomobject]@{{ ProcessId = 300; ParentProcessId = 999; Name = 'python.exe'; CommandLine = 'gemma_response_proxy.py stale duplicate' }}
)
function Get-NetTCPConnection {{
  param($LocalPort, $State, $ErrorAction)
  $connections = $script:connections | Where-Object {{ $_.State -eq $State }}
  if ($null -ne $LocalPort) {{
    $connections = $connections | Where-Object {{ $_.LocalPort -eq $LocalPort }}
  }}
  return $connections
}}
function Get-CimInstance {{
  param($ClassName, $Filter, $ErrorAction)
  return $script:processes
}}
function Stop-Process {{
  param($Id, [switch]$Force, $ErrorAction)
  $script:stopped += [int]$Id
}}
Stop-StaleProxyProcesses 'gemma_response_proxy.py' 8081
[pscustomobject]@{{ stopped = @($script:stopped) }} | ConvertTo-Json -Compress
"""
        result = subprocess.run(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["stopped"], [])

    def test_runtime_reuses_healthy_proxy_processes_before_endpoint_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            runtime = (root / "start-gemma-runtime.ps1").read_text(encoding="utf-8")

        self.assertNotIn("function Stop-ProxyProcesses($scriptName)", runtime)
        self.assertNotIn("Stop-ProxyProcesses 'gemma_response_proxy.py'", runtime)
        self.assertNotIn("Stop-ProxyProcesses 'gemma_reasoning_proxy.py'", runtime)
        direct_start = runtime.index("Start-ProxyIfNeeded $proxyUrl")
        reasoning_start = runtime.index("Start-ProxyIfNeeded $reasoningProxyUrl")
        direct_cleanup = runtime.index("Stop-StaleProxyProcesses 'gemma_response_proxy.py' 8081")
        reasoning_cleanup = runtime.index("Stop-StaleProxyProcesses 'gemma_reasoning_proxy.py' 8082")
        self.assertLess(direct_start, direct_cleanup)
        self.assertLess(reasoning_start, reasoning_cleanup)

    def test_runtime_proxy_children_run_without_local_timeout_or_output_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            runtime = (root / "start-gemma-runtime.ps1").read_text(encoding="utf-8")

        self.assertIn("Start-ProxyIfNeeded $proxyUrl", runtime)
        self.assertIn("Start-ProxyIfNeeded $reasoningProxyUrl", runtime)
        self.assertNotIn("GEMMA_PROXY_UPSTREAM_TIMEOUT_SECONDS", runtime)
        self.assertNotIn("GEMMA_RESPONSE_MAX_OUTPUT_TOKENS", runtime)

    def test_runtime_starts_background_processes_without_attached_stdio_redirects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            runtime = (root / "start-gemma-runtime.ps1").read_text(encoding="utf-8")

        self.assertNotIn("-RedirectStandardOutput", runtime)
        self.assertNotIn("-RedirectStandardError", runtime)
        self.assertIn("-WindowStyle Hidden", runtime)

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

    def test_verify_script_runs_syntax_preflight_before_dependency_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        required_fragments = [
            "function Assert-PowerShellSyntax($path)",
            "[System.Management.Automation.Language.Parser]::ParseFile",
            "Assert-PowerShellSyntax (Join-Path $root 'verify-local-setup.ps1')",
            "Assert-PowerShellSyntax (Join-Path $root 'start-gemma-runtime.ps1')",
            "function Assert-PythonSyntax($scriptName)",
            "& $python -m py_compile $scriptPath",
            "Assert-PythonSyntax 'launch_gemma_codex.py'",
            "Assert-PythonSyntax 'gemma_response_proxy.py'",
            "Assert-PythonSyntax 'gemma_reasoning_proxy.py'",
        ]
        self.assertContainsAll(
            verify,
            required_fragments,
        )
        if not all(fragment in verify for fragment in required_fragments):
            return
        preflight_index = verify.index("Assert-PowerShellSyntax (Join-Path $root 'verify-local-setup.ps1')")
        mcp_dependency_index = verify.index('& $python -c "import mcp"')
        context7_dependency_index = verify.index("& (Join-Path $root 'node_modules\\.bin\\context7-mcp.cmd') --version")
        self.assertLess(preflight_index, mcp_dependency_index)
        self.assertLess(preflight_index, context7_dependency_index)

    def test_ctx7_wrapper_rejects_stale_top_level_search_with_official_guidance(self):
        wrapper = Path(setup_local_codex.__file__).resolve().parent / "tools" / "bin" / "ctx7.cmd"
        if os.name != "nt":
            self.skipTest("ctx7.cmd wrapper is Windows-specific")

        result = subprocess.run(
            [str(wrapper), "search", "new mythos claude and fable claude"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        combined_output = result.stdout + result.stderr
        self.assertIn("ctx7 has no top-level search command", combined_output)
        self.assertIn("ctx7 library <name> <query>", combined_output)
        self.assertIn("ctx7 docs <libraryId> <query>", combined_output)
        self.assertIn("ctx7 skills search <keywords>", combined_output)

    def test_npx_wrapper_removes_repeated_ctx7_binary_before_real_npx(self):
        wrapper = Path(setup_local_codex.__file__).resolve().parent / "tools" / "bin" / "npx.cmd"
        if os.name != "nt":
            self.skipTest("npx.cmd wrapper is Windows-specific")

        powershell = self.require_powershell()
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            fake_bin = temp / "fake-bin"
            fake_bin.mkdir()
            captured = temp / "captured-args.txt"
            fake_npx = fake_bin / "npx.cmd"
            fake_npx.write_text(
                rf"""@echo off
setlocal
if exist "{captured}" del "{captured}"
:loop
if "%~1"=="" exit /b 0
>>"{captured}" echo [%~1]
shift
goto loop
""",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["PATH"] = str(wrapper.parent) + os.pathsep + str(fake_bin) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    f"& '{wrapper}' ctx7@latest ctx7 library ooredoo 'admin credentials'",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            captured_args = captured.read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            captured_args,
            ["[ctx7@latest]", "[library]", "[ooredoo]", "[admin credentials]"],
        )

    def test_ctx7_wrapper_rejects_repeated_binary_search_with_official_guidance(self):
        wrapper = Path(setup_local_codex.__file__).resolve().parent / "tools" / "bin" / "ctx7.cmd"
        if os.name != "nt":
            self.skipTest("ctx7.cmd wrapper is Windows-specific")

        result = subprocess.run(
            [str(wrapper), "ctx7", "search", "Ooredoo"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        combined_output = result.stdout + result.stderr
        self.assertIn("Do not repeat the ctx7 binary name after npx ctx7@latest.", combined_output)
        self.assertIn("npx -y ctx7@latest library <name> <query>", combined_output)
        self.assertIn("npx -y ctx7@latest docs <libraryId> <query>", combined_output)
        self.assertIn("ctx7 has no top-level search command", combined_output)

    def test_ctx7_wrapper_rejects_repeated_binary_library_with_official_guidance(self):
        wrapper = Path(setup_local_codex.__file__).resolve().parent / "tools" / "bin" / "ctx7.cmd"
        if os.name != "nt":
            self.skipTest("ctx7.cmd wrapper is Windows-specific")

        result = subprocess.run(
            [str(wrapper), "ctx7", "library", "Ooredoo", "admin credentials"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        combined_output = result.stdout + result.stderr
        self.assertIn("Do not repeat the ctx7 binary name after npx ctx7@latest.", combined_output)
        self.assertIn("npx -y ctx7@latest library <name> <query>", combined_output)
        self.assertIn("ctx7 library <name> <query>", combined_output)

    def test_npx_wrapper_rejects_repeated_ctx7_binary_search_with_official_guidance(self):
        wrapper = Path(setup_local_codex.__file__).resolve().parent / "tools" / "bin" / "npx.cmd"
        if os.name != "nt":
            self.skipTest("npx.cmd wrapper is Windows-specific")

        result = subprocess.run(
            [str(wrapper), "ctx7@latest", "ctx7", "search", "Ooredoo"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        combined_output = result.stdout + result.stderr
        self.assertIn("Do not repeat the ctx7 binary name after npx ctx7@latest.", combined_output)
        self.assertIn("npx -y ctx7@latest library <name> <query>", combined_output)
        self.assertIn("npx -y ctx7@latest docs <libraryId> <query>", combined_output)
        self.assertNotIn("too many arguments", combined_output)

    def test_npx_wrapper_preserves_y_flag_when_removing_repeated_ctx7_binary(self):
        wrapper = Path(setup_local_codex.__file__).resolve().parent / "tools" / "bin" / "npx.cmd"
        if os.name != "nt":
            self.skipTest("npx.cmd wrapper is Windows-specific")

        powershell = self.require_powershell()
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            fake_bin = temp / "fake-bin"
            fake_bin.mkdir()
            captured = temp / "captured-args.txt"
            fake_npx = fake_bin / "npx.cmd"
            fake_npx.write_text(
                rf"""@echo off
setlocal
if exist "{captured}" del "{captured}"
:loop
if "%~1"=="" exit /b 0
>>"{captured}" echo [%~1]
shift
goto loop
""",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["PATH"] = str(wrapper.parent) + os.pathsep + str(fake_bin) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    f"& '{wrapper}' -y ctx7@latest ctx7 library Ooredoo 'admin credentials'",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            captured_args = captured.read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            captured_args,
            ["[-y]", "[ctx7@latest]", "[library]", "[Ooredoo]", "[admin credentials]"],
        )

    def test_base_instructions_forbid_repeating_ctx7_binary_after_npx(self):
        instructions = setup_local_codex.build_base_instructions()

        self.assertIn("Do not run npx ctx7@latest ctx7", instructions)
        self.assertIn("npx -y ctx7@latest library <name> <query>", instructions)
        self.assertIn("npx -y ctx7@latest docs <libraryId> <query>", instructions)

    def test_verify_script_checks_gemma_ctx7_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        self.assertIn("tools\\bin\\ctx7.cmd", verify)
        self.assertIn("ctx7 wrapper failed", verify)
        self.assertIn("tools\\bin\\npx.cmd", verify)
        self.assertIn("npx wrapper failed repeated ctx7 guard", verify)

    def test_verify_script_checks_gemma_npx_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        self.assertIn("tools\\bin\\npx.cmd", verify)
        self.assertIn("Missing Gemma npx wrapper", verify)

    def test_write_launchers_creates_context7_command_wrappers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)

            ctx7 = root / "tools" / "bin" / "ctx7.cmd"
            npx = root / "tools" / "bin" / "npx.cmd"
            npx_wrapper = root / "tools" / "bin" / "npx_wrapper.py"

            self.assertTrue(ctx7.is_file())
            self.assertTrue(npx.is_file())
            self.assertTrue(npx_wrapper.is_file())
            self.assertIn("ctx7 has no top-level search command", ctx7.read_text(encoding="utf-8"))
            self.assertIn("EnableDelayedExpansion", npx.read_text(encoding="utf-8"))
            self.assertIn("has_repeated_ctx7_search", npx_wrapper.read_text(encoding="utf-8"))

    def test_verify_script_checks_deep_research_skill_visible_to_local_homes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        self.assertContainsAll(
            verify,
            [
                "Join-Path $codexHome 'skills\\deep-research\\SKILL.md'",
                "Missing deep-research skill in $label",
                "Assert-DeepResearchSkill $localHome 'local CODEX_HOME'",
                "Assert-DeepResearchSkill $reasoningHome 'reasoning CODEX_HOME'",
                "Deep-research skill is visible to both local Codex homes.",
            ],
        )

    def test_verify_script_checks_deep_research_skill_uses_context7(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        self.assertContainsAll(
            verify,
            [
                "$skillText = Get-Content -Raw $skillPath",
                "Gemma Context7 Runtime Override",
                "is not configured to use Context7",
            ],
        )

    def test_verify_script_checks_deep_research_alias_triggers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        self.assertContainsAll(
            verify,
            [
                "Assert-TextContains $skillText 'deepsearch'",
                "Assert-TextContains $skillText 'deep search'",
                "Assert-TextContains $skillText 'deep-search'",
            ],
        )

    def test_verify_script_checks_context7_skill_defers_deepsearch_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        self.assertContainsAll(
            verify,
            [
                "function Assert-Context7Skill($codexHome, $label)",
                "skills\\context7\\SKILL.md",
                "deepsearch requests must use deep-research as the top-level workflow",
                "Assert-Context7Skill $localHome 'local CODEX_HOME'",
                "Assert-Context7Skill $reasoningHome 'reasoning CODEX_HOME'",
            ],
        )

    def test_verify_script_checks_deep_research_context7_override_in_both_homes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        self.assertContainsAll(
            verify,
            [
                "function Assert-DeepResearchSkill($codexHome, $label)",
                "Assert-DeepResearchSkill $localHome 'local CODEX_HOME'",
                "Assert-DeepResearchSkill $reasoningHome 'reasoning CODEX_HOME'",
            ],
        )

    def test_verify_script_checks_required_workflow_skills_in_both_homes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        self.assertContainsAll(
            verify,
            [
                "function Assert-GemmaSkillBundle($codexHome, $label)",
                "'using-superpowers'",
                "'brainstorming'",
                "'writing-plans'",
                "'using-git-worktrees'",
                "'systematic-debugging'",
                "'test-driven-development'",
                "'code-simplifier'",
                "'subagent-driven-development'",
                "'executing-plans'",
                "'dispatching-parallel-agents'",
                "'requesting-code-review'",
                "'finishing-a-development-branch'",
                "'verification-before-completion'",
                "'deep-research'",
                "Assert-GemmaSkillBundle $localHome 'local CODEX_HOME'",
                "Assert-GemmaSkillBundle $reasoningHome 'reasoning CODEX_HOME'",
            ],
        )

    def test_verify_script_checks_terminal_shims_for_all_gemma_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        self.assertContainsAll(
            verify,
            [
                "function Assert-GemmaTerminalShim($commandName, $expectsNoConfirmYolo)",
                "Get-Command $commandName",
                "call \"' + $launcherPath + '\" %*",
                "GEMMA_CODEX_TARGET_DIR=%CD%",
                "GEMMA_CODEX_SONION_NO_CONFIRM_YOLO=1",
                "Assert-GemmaTerminalShim 'son' $false",
                "Assert-GemmaTerminalShim 'sonion' $true",
                "Assert-GemmaTerminalShim 'operator' $false",
            ],
        )

    def test_verify_script_avoids_powershell_readonly_home_variable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        self.assertIn("function Assert-DeepResearchSkill($codexHome, $label)", verify)
        self.assertIn("Join-Path $codexHome 'skills\\deep-research\\SKILL.md'", verify)
        self.assertNotIn("function Assert-DeepResearchSkill($home, $label)", verify)

    def test_verify_script_checks_context7_tool_names_in_both_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        self.assertContainsAll(
            verify,
            [
                "function Assert-ConfigContains($configPath, $fragment, $message)",
                "enabled_tools = [\"resolve-library-id\", \"query-docs\"]",
                "local Context7 config must expose resolve-library-id and query-docs",
                "reasoning Context7 config must expose resolve-library-id and query-docs",
            ],
        )

    def test_verify_script_prefers_patched_local_codex_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        self.assertIn("$patchedCodex = Join-Path $root 'tools\\codex-local\\bin\\codex.exe'", verify)
        self.assertIn("$packagedCodex = Join-Path $root 'node_modules\\.bin\\codex.cmd'", verify)
        self.assertIn("if (Test-Path $patchedCodex)", verify)
        self.assertIn("elseif (Test-Path $packagedCodex)", verify)
        self.assertIn('Write-Host "Codex binary: $codexBinary"', verify)
        self.assertIn("& $codexBinary --version", verify)

    def test_verify_script_prints_codex_provenance_and_pinned_npm_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            verify = (root / "verify-local-setup.ps1").read_text(encoding="utf-8")

        self.assertContainsAll(
            verify,
            [
                "$codexVersion = & $codexBinary --version",
                "Write-Host \"Codex version: $codexVersion\"",
                "Get-FileHash -Algorithm SHA256 $codexBinary",
                "Write-Host \"Codex sha256: $codexSha256\"",
                "Write-Host 'Codex npm package: @openai/codex@0.139.0'",
            ],
        )

    def test_generated_powershell_launchers_parse_without_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_local_codex.write_launchers(root)
            self.assertGeneratedPowerShellScriptsParse(root)

    def test_python_launcher_and_proxy_modules_compile(self):
        root = Path(setup_local_codex.__file__).resolve().parent
        module_names = [
            "launch_gemma_codex.py",
            "gemma_response_proxy.py",
            "gemma_reasoning_proxy.py",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            for module_name in module_names:
                with self.subTest(module=module_name):
                    source = root / module_name
                    py_compile.compile(
                        str(source),
                        cfile=str(cache_root / f"{module_name}.pyc"),
                        doraise=True,
                    )

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
            (existing / "custom-local-skill").mkdir()
            created = []

            def fake_symlink(src, dst, target_is_directory=False):
                created.append((Path(src), Path(dst), target_is_directory))
                Path(dst).mkdir()

            with patch.object(os, "symlink", side_effect=fake_symlink):
                setup_local_codex.link_skills(root, target=target)

            self.assertIn((target / "brainstorming", existing / "brainstorming", True), created)

    def test_link_skills_repairs_existing_managed_directory_to_exact_home_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            target = Path(tmp) / "skills"
            (target / "using-superpowers").mkdir(parents=True)
            (target / "brainstorming").mkdir()
            setup_local_codex.write_local_codex_config(root)
            existing = root / ".codex-local-reasoning" / "skills"
            (existing / "using-superpowers").mkdir(parents=True)
            created = []

            def fake_symlink(src, dst, target_is_directory=False):
                created.append((Path(src), Path(dst), target_is_directory))
                Path(dst).mkdir()

            with patch.object(os, "symlink", side_effect=fake_symlink):
                setup_local_codex.link_skills(root, target=target)

            self.assertIn((target, root / ".codex-local-reasoning" / "skills", True), created)

    def test_install_gemma_skills_creates_repo_skills_dir_and_copies_required_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            source = Path(tmp) / "codex-skills"
            for skill_name in setup_local_codex.GEMMA_REQUIRED_SKILLS:
                skill_dir = source / skill_name
                skill_dir.mkdir(parents=True)
                (skill_dir / "SKILL.md").write_text(
                    f"---\nname: {skill_name}\ndescription: Use when testing.\n---\n",
                    encoding="utf-8",
                )
            (source / "deep-research" / "references").mkdir()
            (source / "deep-research" / "references" / "report_template.md").write_text(
                "template",
                encoding="utf-8",
            )

            target = setup_local_codex.install_gemma_skills(root, source=source)

            self.assertEqual(target, root / "skills")
            for skill_name in setup_local_codex.GEMMA_REQUIRED_SKILLS:
                with self.subTest(skill_name=skill_name):
                    self.assertTrue((target / skill_name / "SKILL.md").is_file())
            self.assertEqual(
                (target / "deep-research" / "references" / "report_template.md").read_text(encoding="utf-8"),
                "template",
            )

    def test_install_gemma_skills_refreshes_existing_required_skills_from_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            source = Path(tmp) / "codex-skills"
            for skill_name in setup_local_codex.GEMMA_REQUIRED_SKILLS:
                skill_dir = source / skill_name
                skill_dir.mkdir(parents=True)
                (skill_dir / "SKILL.md").write_text(
                    f"---\nname: {skill_name}\ndescription: Fresh source copy.\n---\n",
                    encoding="utf-8",
                )
            stale_skill = root / "skills" / "using-superpowers"
            stale_skill.mkdir(parents=True)
            (stale_skill / "SKILL.md").write_text("stale placeholder", encoding="utf-8")

            target = setup_local_codex.install_gemma_skills(root, source=source)

            self.assertIn(
                "Fresh source copy",
                (target / "using-superpowers" / "SKILL.md").read_text(encoding="utf-8"),
            )

    def test_install_gemma_skills_raises_when_required_skill_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            source = Path(tmp) / "codex-skills"
            source.mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "using-superpowers"):
                setup_local_codex.install_gemma_skills(root, source=source)

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

    def test_global_shims_apply_no_confirm_yolo_only_to_sonion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "gemma"
            shim_dir = Path(tmp) / "commands"

            setup_local_codex.write_global_shims(root, shim_dir=shim_dir)

            son = (shim_dir / "son.cmd").read_text(encoding="utf-8")
            sonion = (shim_dir / "sonion.cmd").read_text(encoding="utf-8")
            operator = (shim_dir / "operator.cmd").read_text(encoding="utf-8")
        self.assertEqual(son, operator)
        self.assertNotEqual(son, sonion)
        self.assertIn("gemma-codex.cmd", son)
        self.assertIn("%*", son)
        self.assertIn('set "GEMMA_CODEX_SONION_NO_CONFIRM_YOLO=1"', sonion)
        self.assertNotIn("GEMMA_CODEX_SONION_NO_CONFIRM_YOLO", son)
        self.assertNotIn("GEMMA_CODEX_SONION_NO_CONFIRM_YOLO", operator)

    def test_global_shim_passes_reasoning_and_yolo_args_to_launcher(self):
        shim = setup_local_codex.build_global_shim_text(Path("C:/Users/Agent-1/Desktop/gemma"))
        self.assertIn("%*", shim)
        self.assertNotIn("--reasoning", shim)
        self.assertNotIn("--yolo", shim)

    def test_generated_shim_text_preserves_dry_run_and_sonion_marker(self):
        root = Path("C:/Users/Agent-1/Desktop/gemma")
        son = setup_local_codex.build_global_shim_text(root)
        sonion = setup_local_codex.build_global_shim_text(root, no_confirm_yolo=True)

        for shim in (son, sonion):
            self.assertIn('set "GEMMA_CODEX_TARGET_DIR=%CD%"', shim)
            self.assertIn('call "C:\\Users\\Agent-1\\Desktop\\gemma\\gemma-codex.cmd" %*', shim)
            self.assertIn("%*", shim)
        self.assertIn('set "GEMMA_CODEX_SONION_NO_CONFIRM_YOLO=1"', sonion)
        self.assertNotIn("GEMMA_CODEX_SONION_NO_CONFIRM_YOLO", son)

    def test_main_installs_global_shims_by_default(self):
        with (
            patch.object(setup_local_codex, "write_local_codex_config"),
            patch.object(setup_local_codex, "write_launchers"),
            patch.object(setup_local_codex, "install_gemma_skills"),
            patch.object(setup_local_codex, "link_skills"),
            patch.object(setup_local_codex, "write_global_shims") as write_global_shims,
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                setup_local_codex.main([])

        write_global_shims.assert_called_once_with(Path(setup_local_codex.__file__).resolve().parent)

    def test_main_installs_gemma_skills_and_links_local_homes_to_them(self):
        skill_dir = Path("C:/Users/Agent-1/Desktop/gemma/skills")
        with (
            patch.object(setup_local_codex, "write_local_codex_config"),
            patch.object(setup_local_codex, "write_launchers"),
            patch.object(setup_local_codex, "install_gemma_skills", return_value=skill_dir) as install_skills,
            patch.object(setup_local_codex, "link_skills") as link_skills,
            patch.object(setup_local_codex, "write_global_shims"),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                setup_local_codex.main([])

        install_skills.assert_called_once()
        link_skills.assert_called_once_with(Path(setup_local_codex.__file__).resolve().parent, target=skill_dir)

    def test_main_installs_global_shims_when_explicit(self):
        with (
            patch.object(setup_local_codex, "write_local_codex_config"),
            patch.object(setup_local_codex, "write_launchers"),
            patch.object(setup_local_codex, "install_gemma_skills"),
            patch.object(setup_local_codex, "link_skills"),
            patch.object(setup_local_codex, "write_global_shims") as write_global_shims,
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                setup_local_codex.main(["--install-global-shims"])

        write_global_shims.assert_called_once()

    def test_main_can_skip_global_shims_when_explicit(self):
        with (
            patch.object(setup_local_codex, "write_local_codex_config"),
            patch.object(setup_local_codex, "write_launchers"),
            patch.object(setup_local_codex, "install_gemma_skills"),
            patch.object(setup_local_codex, "link_skills"),
            patch.object(setup_local_codex, "write_global_shims") as write_global_shims,
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                setup_local_codex.main(["--skip-global-shims"])

        write_global_shims.assert_not_called()

    def test_readme_documents_command_matrix_and_codex_0139_comparison(self):
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertContainsAll(
            readme,
            [
                "## Local Command Matrix",
                "`son`",
                "`son --reasoning`",
                "`son --reasoing`",
                "`son --yolo`",
                "`sonion --reasoning --yolo`",
                "`operator --reasoing --yolo`",
                "GEMMA_CODEX_DRY_RUN=1",
                "npm exec --yes --package @openai/codex@0.139.0 -- codex --version",
                "local trusted operation",
                "not hosted untrusted users",
            ],
        )


if __name__ == "__main__":
    unittest.main()
