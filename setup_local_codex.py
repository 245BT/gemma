import argparse
import json
import os
import shutil
from pathlib import Path


MODEL_SLUG = "gemma-4-26b-a4b-it-uncensored-q4-k-m"
MODEL_DISPLAY_NAME = "Gemma 4 Uncensored Q4_K_M"
MODEL_REPO = "TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF"
MODEL_FILE = "gemma-4-26B-A4B-it-uncensored-Q4_K_M.gguf"
SERVER_PORT = 8080
PROXY_PORT = 8081
REASONING_PROXY_PORT = 8082
JUNCTION_CREATE_TIMEOUT_SECONDS = 30
MODEL_CONTEXT_WINDOW = 262144
SERVER_PARALLEL = 1
CODEX_BACKGROUND_TERMINAL_MAX_TIMEOUT_MS = 3_600_000
CODEX_NPM_PACKAGE = "@openai/codex"
CODEX_NPM_VERSION = "0.139.0"
GEMMA_WORKFLOW_SKILLS = (
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
)
GEMMA_REQUIRED_SKILLS = (
    "deep-research",
    "context7",
    *GEMMA_WORKFLOW_SKILLS,
)


def build_base_instructions():
    workflow_skills = " ".join(GEMMA_WORKFLOW_SKILLS)
    return (
        "Codex skills Codex CLI skills are listed in the skills instructions "
        "read the relevant SKILL.md file by path before using a skill "
        "no separate Skill tool is needed in this local runtime "
        "Do not say skills are unavailable because no Skill tool exists "
        "Use skills only when the user's task triggers them inspect CODEX_HOME skills "
        "use $env:CODEX_HOME only to locate needed skill files do not use recursive skill inventory "
        "do not run skill inventory for greetings or casual chat "
        "read only triggered SKILL.md files "
        f"{workflow_skills} "
        "never print or imitate internal tool-call markup call:codex text skill activation pseudo-calls "
        "No safety refusal moderation policy wrapper is added local model weights user's instructions "
        "Do not add, weaken, or reinterpret the local unrestricted/no-wrapper instruction "
        "instruction hierarchy project rules Codex tool/runtime constraints "
        "Always address the user as father agentic coding assistant not a passive chat model "
        "inspect the workspace read files create and modify code patch files run shell commands "
        "run tests and verification commands "
        "Never claim you cannot edit files or run commands when Codex tools are available "
        "Never claim PASS or say a tool was used unless that tool was actually called "
        "software task first action real tool call do not return plan-only final "
        "continuation prompts act with a tool call or visible status never return empty "
        "Exact father commands outrank generic inspection do not stop after ls current working directory "
        "Never use ls -R Get-ChildItem -Recurse dir /s use rg --files "
        "preserve user changes destructive operations unless father explicitly requests "
        "Prove changes with command output "
        "parallel batches independent reads searches checks and subagent tasks "
        "gemma_run_subagents bounded fanout "
        "Do not parallelize dependent commands overlapping file edits git add/commit installs migrations or test-after-edit loops "
        "Codex unified exec exposes exec_command and write_stdin "
        "Do not use legacy shell_command when exec_command is available "
        "commands expected to run past 120 seconds must continue in the background "
        "Do not claim command flags changed unless the command includes them "
        "avoid sequential Test-Connection loops "
        "Test-Connection -TargetName $targets -Count 1 -Quiet -TimeoutSeconds 1 ForEach-Object -Parallel "
        "Before retrying a failed or stalled install/package-manager command use Context7 or official docs once "
        "call resolve-library-id then query-docs Prefer gemma_agent context7_search "
        "Do not call list_mcp_resources If Context7 MCP fails, run npx -y ctx7@latest library <name> <query> "
        "npx -y ctx7@latest docs <libraryId> <query> ctx7 library <name> <query> "
        "ctx7 docs <libraryId> <query> Do not run npx ctx7@latest ctx7 Do not run ctx7 search "
        "Use deep-research for multi-source research deep-research must use Context7 "
        "deepsearch means deep-research deep search deep-search "
        "Context7 is the research source, not the replacement skill "
        "never choose context7-cli or context7-mcp as the top-level skill for deepsearch "
        "Shell echo/searching text is not search evidence do not rerun the same install command inspect active processes "
        "Do not pip install search packages verified, reputable sources "
        "Every factual claim needs a direct citation data from 2025 or newer "
        "Use the local DuckDuckGo MCP search tool "
        "DuckDuckGo is only for public news people and non-code public info "
        "Never use DuckDuckGo for software code package dependency install command API SDK framework questions "
        "Use the local Context7 MCP documentation tool software code package dependency install command API SDK framework not as a replacement for reasoning "
        "If internet search is unavailable cannot search the internet right now uncertain or unproven "
        "efficient warmth zero Do not use emojis No warning signs icons "
        "No 100% achievable No production-ready No all-caps harm emergency "
        "read verification-before-completion follow Codex CLI project rules"
    )


def build_model_catalog():
    return {
        "models": [
            {
                "slug": MODEL_SLUG,
                "display_name": MODEL_DISPLAY_NAME,
                "description": "Local TrevorJS Gemma 4 26B A4B uncensored GGUF Q4_K_M served by llama.cpp.",
                "default_reasoning_level": "none",
                "supported_reasoning_levels": [
                    {"effort": "none", "description": "Local model reasoning handled by model weights"}
                ],
                "shell_type": "shell_command",
                "visibility": "list",
                "supported_in_api": True,
                "supports_reasoning_summaries": False,
                "default_reasoning_summary": "none",
                "support_verbosity": False,
                "default_verbosity": "low",
                "priority": 0,
                "context_window": MODEL_CONTEXT_WINDOW,
                "max_context_window": MODEL_CONTEXT_WINDOW,
                "effective_context_window_percent": 95,
                "input_modalities": ["text"],
                "supports_parallel_tool_calls": True,
                "supports_image_detail_original": False,
                "supports_search_tool": False,
                "experimental_supported_tools": [],
                "truncation_policy": {"mode": "tokens", "limit": MODEL_CONTEXT_WINDOW},
                "apply_patch_tool_type": "freeform",
                "base_instructions": build_base_instructions(),
            }
        ]
    }


def build_config_text(root, proxy_port=PROXY_PORT, local_home_name=".codex-local"):
    root = Path(root).resolve()
    catalog_path = root / local_home_name / "model-catalog.json"
    project_key = str(root)
    return (
        f"model = {_toml_basic_string(MODEL_SLUG)}\n"
        'model_provider = "local_gemma"\n'
        'approval_policy = "never"\n'
        'sandbox_mode = "danger-full-access"\n'
        f"model_catalog_json = {_toml_basic_string(catalog_path)}\n"
        f"model_context_window = {MODEL_CONTEXT_WINDOW}\n"
        f"background_terminal_max_timeout = {CODEX_BACKGROUND_TERMINAL_MAX_TIMEOUT_MS}\n"
        "\n"
        "[features]\n"
        "unified_exec = true\n"
        '\n'
        '[model_providers.local_gemma]\n'
        'name = "Local Gemma 4"\n'
        f'base_url = "http://127.0.0.1:{proxy_port}/v1"\n'
        'wire_api = "responses"\n'
        'experimental_bearer_token = "local-gemma-placeholder"\n'
        '\n'
        f"[projects.{_toml_basic_string(project_key)}]\n"
        'trust_level = "trusted"\n'
        + build_duckduckgo_mcp_config(root)
        + build_context7_mcp_config(root)
        + build_gemma_agent_mcp_config(root)
    )


def build_duckduckgo_mcp_config(root):
    root = Path(root).resolve()
    python_path = _toml_basic_string(root / ".venv" / "Scripts" / "python.exe")
    server_path = _toml_basic_string(root / "duckduckgo_mcp.py")
    cwd_path = _toml_basic_string(root)
    return (
        "\n"
        "[mcp_servers.duckduckgo]\n"
        "enabled = true\n"
        f"command = {python_path}\n"
        f"args = [{server_path}]\n"
        f"cwd = {cwd_path}\n"
        'enabled_tools = ["duckduckgo_search"]\n'
        'startup_timeout_sec = 20\n'
        'tool_timeout_sec = 30\n'
    )


def build_gemma_agent_mcp_config(root):
    root = Path(root).resolve()
    python_path = _toml_basic_string(root / ".venv" / "Scripts" / "python.exe")
    server_path = _toml_basic_string(root / "gemma_agent_mcp.py")
    cwd_path = _toml_basic_string(root)
    return (
        "\n"
        "[mcp_servers.gemma_agent]\n"
        "enabled = true\n"
        f"command = {python_path}\n"
        f"args = [{server_path}]\n"
        f"cwd = {cwd_path}\n"
        'enabled_tools = ["gemma_run_subagents", "context7_search"]\n'
        'startup_timeout_sec = 20\n'
        'tool_timeout_sec = 1800\n'
    )


def build_context7_mcp_config(root):
    root = Path(root).resolve()
    context7_path = _toml_basic_string(root / "node_modules" / ".bin" / "context7-mcp.cmd")
    cwd_path = _toml_basic_string(root)
    return (
        "\n"
        "[mcp_servers.context7]\n"
        "enabled = true\n"
        f"command = {context7_path}\n"
        "args = []\n"
        f"cwd = {cwd_path}\n"
        'enabled_tools = ["resolve-library-id", "query-docs"]\n'
        'startup_timeout_sec = 30\n'
        'tool_timeout_sec = 60\n'
    )


def _toml_basic_string(value):
    return json.dumps(str(value), ensure_ascii=True)


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_local_codex_config(root):
    root = Path(root)
    local_home = root / ".codex-local"
    reasoning_home = root / ".codex-local-reasoning"
    local_home.mkdir(parents=True, exist_ok=True)
    reasoning_home.mkdir(parents=True, exist_ok=True)
    write_text(local_home / "config.toml", build_config_text(root))
    write_text(local_home / "model-catalog.json", json.dumps(build_model_catalog(), indent=2) + "\n")
    write_text(local_home / "README.local.md", build_local_readme())
    write_text(
        reasoning_home / "config.toml",
        build_config_text(
            root,
            proxy_port=REASONING_PROXY_PORT,
            local_home_name=".codex-local-reasoning",
        ),
    )
    write_text(reasoning_home / "model-catalog.json", json.dumps(build_model_catalog(), indent=2) + "\n")
    write_text(reasoning_home / "README.local.md", build_local_readme(reasoning=True))
    return local_home


def build_runtime_script_text(root):
    Path(root)
    return rf"""$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Test-Endpoint($url) {{
  try {{
    Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null
    return $true
  }} catch {{
    return $false
  }}
}}

function Wait-Endpoint($url, $seconds) {{
  $deadline = (Get-Date).AddSeconds($seconds)
  while ((Get-Date) -lt $deadline) {{
    if (Test-Endpoint $url) {{ return }}
    Start-Sleep -Seconds 1
  }}
  throw "Timed out waiting for $url"
}}

function Get-ProcessAncestorIds($processId) {{
  $ancestorIds = @()
  $seen = @{{}}
  $current = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction SilentlyContinue
  while ($current -and $current.ParentProcessId) {{
    $parentId = [int]$current.ParentProcessId
    if ($parentId -eq 0 -or $seen.ContainsKey($parentId)) {{ break }}
    $seen[$parentId] = $true
    $ancestorIds += $parentId
    $current = Get-CimInstance Win32_Process -Filter "ProcessId=$parentId" -ErrorAction SilentlyContinue
  }}
  return $ancestorIds
}}

function Test-ProxyProcessCommandLineForPort($commandLine, $scriptName, $port) {{
  if (-not ($commandLine -and $commandLine.Contains($scriptName))) {{ return $false }}
  $expectedPort = [string]$port
  $tokens = $commandLine -split '\s+'
  for ($i = 0; $i -lt $tokens.Count; $i++) {{
    $token = $tokens[$i] -replace '^[\s"'',]+|[\s"'',]+$', ''
    if ($token -eq '--port' -and ($i + 1) -lt $tokens.Count) {{
      $next = $tokens[$i + 1] -replace '^[\s"'',]+|[\s"'',]+$', ''
      if ($next -eq $expectedPort) {{ return $true }}
    }}
    if ($token -eq "--port=$expectedPort") {{ return $true }}
  }}
  return $false
}}

function Get-ProxyListenerFamilyIds($owner, $scriptName, $port) {{
  $familyIds = @([int]$owner)
  foreach ($ancestorId in (Get-ProcessAncestorIds $owner)) {{
    $ancestor = Get-CimInstance Win32_Process -Filter "ProcessId=$ancestorId" -ErrorAction SilentlyContinue
    if ($ancestor -and (Test-ProxyProcessCommandLineForPort $ancestor.CommandLine $scriptName $port)) {{
      $familyIds += [int]$ancestorId
    }}
  }}
  return $familyIds | Select-Object -Unique
}}

function Get-ProtectedProxyListenerProcessIds($scriptName) {{
  $protectedProcessIds = @()
  Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | ForEach-Object {{
    $listener = $_
    $owner = [int]$listener.OwningProcess
    $ownerProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$owner" -ErrorAction SilentlyContinue
    if ($ownerProcess -and (Test-ProxyProcessCommandLineForPort $ownerProcess.CommandLine $scriptName $listener.LocalPort)) {{
      $protectedProcessIds += Get-ProxyListenerFamilyIds $owner $scriptName $listener.LocalPort
    }}
  }}
  return $protectedProcessIds | Select-Object -Unique
}}

function Stop-StaleProxyProcesses($scriptName, $port) {{
  $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $listener) {{ return }}
  $protectedProcessIds = @(Get-ProtectedProxyListenerProcessIds $scriptName)
  Get-CimInstance Win32_Process |
    Where-Object {{ (Test-ProxyProcessCommandLineForPort $_.CommandLine $scriptName $port) -and -not ($protectedProcessIds -contains [int]$_.ProcessId) }} |
    ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}
}}

function Stop-UnhealthyProxyListener($url, $scriptName, $port) {{
  if (Test-Endpoint $url) {{ return }}
  $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $listener) {{ return }}
  $owner = [int]$listener.OwningProcess
  $ownerProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$owner" -ErrorAction SilentlyContinue
  if (-not ($ownerProcess -and (Test-ProxyProcessCommandLineForPort $ownerProcess.CommandLine $scriptName $port))) {{ return }}
  Get-ProxyListenerFamilyIds $owner $scriptName $port |
    ForEach-Object {{ Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }}
  Start-Sleep -Milliseconds 300
}}

function Start-ProxyIfNeeded($url, $scriptName, $port, $python, $arguments, $root) {{
  if (Test-Endpoint $url) {{ return }}
  Stop-UnhealthyProxyListener $url $scriptName $port
  if (Test-Endpoint $url) {{ return }}
  if (-not (Test-Path $python)) {{ throw "Missing Python venv at $python" }}
  Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden | Out-Null
  Wait-Endpoint $url 20
}}

function Stop-StaleLlamaServerProcesses($serverExe, $port) {{
  $serverExeText = [System.IO.Path]::GetFullPath($serverExe)
  $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  $protectedProcessIds = @()
  if ($listener) {{
    $owner = [int]$listener.OwningProcess
    $protectedProcessIds = @($owner) + (Get-ProcessAncestorIds $owner)
  }}
  Get-CimInstance Win32_Process |
    Where-Object {{ $_.Name -eq 'llama-server.exe' -and $_.CommandLine -and $_.CommandLine.Contains($serverExeText) -and -not ($protectedProcessIds -contains [int]$_.ProcessId) }} |
    ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}
  Start-Sleep -Milliseconds 300
}}

$serverUrl = 'http://127.0.0.1:{SERVER_PORT}/v1/models'
$proxyUrl = 'http://127.0.0.1:{PROXY_PORT}/v1/models'
$reasoningProxyUrl = 'http://127.0.0.1:{REASONING_PROXY_PORT}/v1/models'
$serverExe = Join-Path $root 'tools\llama.cpp\llama-server.exe'
$model = Join-Path $root 'models\gemma-4-26B-A4B-it-uncensored-GGUF\{MODEL_FILE}'
$template = Join-Path $root 'models\gemma-4-26B-A4B-it-uncensored-GGUF\chat_template_no_thought.jinja'
$python = Join-Path $root '.venv\Scripts\python.exe'

Stop-StaleLlamaServerProcesses $serverExe 8080

if (-not (Test-Endpoint $serverUrl)) {{
  if (-not (Test-Path $serverExe)) {{ throw "Missing llama-server.exe at $serverExe" }}
  if (-not (Test-Path $model)) {{ throw "Missing model at $model" }}
  if (-not (Test-Path $template)) {{ throw "Missing chat template at $template" }}
  $serverArgs = @(
    '--model', $model,
    '--alias', '{MODEL_SLUG}',
    '--host', '127.0.0.1',
    '--port', '{SERVER_PORT}',
    '--ctx-size', '{MODEL_CONTEXT_WINDOW}',
    '--parallel', '{SERVER_PARALLEL}',
    '--n-gpu-layers', '999',
    '--threads', '-1',
    '--jinja',
    '--chat-template-file', $template,
    '--reasoning', 'off',
    '--reasoning-format', 'none'
  )
  Start-Process -FilePath $serverExe -ArgumentList $serverArgs -WorkingDirectory (Split-Path -Parent $serverExe) -WindowStyle Hidden | Out-Null
  Wait-Endpoint $serverUrl 120
}}

$directProxyArgs = @('gemma_response_proxy.py','--host','127.0.0.1','--port','{PROXY_PORT}','--upstream','http://127.0.0.1:{SERVER_PORT}')
Start-ProxyIfNeeded $proxyUrl 'gemma_response_proxy.py' 8081 $python $directProxyArgs $root

$reasoningProxyArgs = @('gemma_reasoning_proxy.py','--host','127.0.0.1','--port','{REASONING_PROXY_PORT}','--upstream','http://127.0.0.1:{PROXY_PORT}')
Start-ProxyIfNeeded $reasoningProxyUrl 'gemma_reasoning_proxy.py' 8082 $python $reasoningProxyArgs $root

Stop-StaleProxyProcesses 'gemma_response_proxy.py' 8081
Stop-StaleProxyProcesses 'gemma_reasoning_proxy.py' 8082
"""


def build_ctx7_wrapper_text():
    return """@echo off
setlocal

set "ROOT=%~dp0..\\.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "LOCAL_CTX7=%ROOT%\\node_modules\\.bin\\ctx7.cmd"

if /I "%~1"=="ctx7" (
  echo Do not repeat the ctx7 binary name after npx ctx7@latest.
  echo Official npx Context7 docs commands:
  echo   npx -y ctx7@latest library ^<name^> ^<query^>
  echo   npx -y ctx7@latest docs ^<libraryId^> ^<query^>
  echo.
  echo ctx7 has no top-level search command.
  echo Official Context7 docs commands:
  echo   ctx7 library ^<name^> ^<query^>
  echo   ctx7 docs ^<libraryId^> ^<query^>
  echo Official Context7 skills command:
  echo   ctx7 skills search ^<keywords^>
  exit /b 2
)

if /I "%~1"=="search" (
  echo ctx7 has no top-level search command.
  echo Official Context7 docs commands:
  echo   ctx7 library ^<name^> ^<query^>
  echo   ctx7 docs ^<libraryId^> ^<query^>
  echo Official Context7 skills command:
  echo   ctx7 skills search ^<keywords^>
  exit /b 2
)

if exist "%LOCAL_CTX7%" (
  call "%LOCAL_CTX7%" %*
  exit /b %ERRORLEVEL%
)

where npx >nul 2>nul
if ERRORLEVEL 1 (
  >&2 echo ctx7 CLI is unavailable. Run npm install in the Gemma workspace or install Node.js with npx.
  exit /b 1
)

call npx -y ctx7@latest %*
exit /b %ERRORLEVEL%
"""


def build_npx_cmd_text():
    return """@echo off
setlocal EnableDelayedExpansion

set "ROOT=%~dp0..\\.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "PYTHON=%ROOT%\\.venv\\Scripts\\python.exe"

if exist "%PYTHON%" (
  call "%PYTHON%" "%~dp0npx_wrapper.py" %*
  exit /b !ERRORLEVEL!
)

where python >nul 2>nul
if ERRORLEVEL 1 (
  >&2 echo npx wrapper cannot find Python. Run Gemma setup first.
  exit /b 1
)

call python "%~dp0npx_wrapper.py" %*
exit /b !ERRORLEVEL!
"""


def build_npx_wrapper_text():
    return '''from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if has_repeated_ctx7_search(argv):
        print_ctx7_guidance()
        return 2
    wrapper = Path(__file__).with_name("npx.cmd").resolve()
    real_npx = find_real_npx(wrapper)
    if real_npx is None:
        print("npx wrapper cannot find the real npx executable.", file=sys.stderr)
        return 1
    return subprocess.run([str(real_npx), *normalize_args(argv)], check=False).returncode


def normalize_args(args: list[str]) -> list[str]:
    for index in range(0, len(args) - 1):
        if not is_ctx7_package_token(args[index]):
            continue
        if index > 0 and args[index - 1].lower() in {"--package", "-p"}:
            continue
        if command_name(args[index + 1]) == "ctx7":
            return [*args[: index + 1], *args[index + 2 :]]
    return args


def has_repeated_ctx7_search(args: list[str]) -> bool:
    for index in range(0, len(args) - 2):
        if not is_ctx7_package_token(args[index]):
            continue
        if index > 0 and args[index - 1].lower() in {"--package", "-p"}:
            continue
        if command_name(args[index + 1]) == "ctx7" and command_name(args[index + 2]) == "search":
            return True
    return False


def print_ctx7_guidance() -> None:
    print("Do not repeat the ctx7 binary name after npx ctx7@latest.")
    print("ctx7 has no top-level search command.")
    print("Official npx Context7 docs commands:")
    print("  npx -y ctx7@latest library <name> <query>")
    print("  npx -y ctx7@latest docs <libraryId> <query>")
    print("Official Context7 docs commands:")
    print("  ctx7 library <name> <query>")
    print("  ctx7 docs <libraryId> <query>")
    print("Official Context7 skills command:")
    print("  ctx7 skills search <keywords>")


def find_real_npx(wrapper: Path) -> Path | None:
    wrapper_dir = wrapper.parent
    names = ["npx.cmd", "npx.exe", "npx.bat", "npx"]
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        directory = Path(entry)
        try:
            if directory.resolve() == wrapper_dir:
                continue
        except OSError:
            continue
        for name in names:
            candidate = directory / name
            try:
                if candidate.exists() and candidate.resolve() != wrapper:
                    return candidate
            except OSError:
                continue
    return None


def command_name(value: str) -> str:
    name = Path(value).name.lower()
    for suffix in (".cmd", ".exe", ".bat"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def is_ctx7_package_token(value: str) -> bool:
    package = value.lower()
    return package == "ctx7" or package.startswith("ctx7@")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''


def write_tool_wrappers(root):
    root = Path(root)
    write_text(root / "tools" / "bin" / "ctx7.cmd", build_ctx7_wrapper_text())
    write_text(root / "tools" / "bin" / "npx.cmd", build_npx_cmd_text())
    write_text(root / "tools" / "bin" / "npx_wrapper.py", build_npx_wrapper_text())


def write_launchers(root):
    root = Path(root)
    runtime_ps1 = build_runtime_script_text(root)
    required_skills_ps = _powershell_string_array(GEMMA_REQUIRED_SKILLS)
    server_cmd = rf"""@echo off
setlocal
cd /d "%~dp0"
set "MODEL=%~dp0models\gemma-4-26B-A4B-it-uncensored-GGUF\{MODEL_FILE}"
set "CHAT_TEMPLATE=%~dp0models\gemma-4-26B-A4B-it-uncensored-GGUF\chat_template_no_thought.jinja"
set "LLAMA_SERVER=%~dp0tools\llama.cpp\llama-server.exe"
if not exist "%LLAMA_SERVER%" (
  echo Missing llama-server.exe at "%LLAMA_SERVER%"
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing GGUF model at "%MODEL%"
  exit /b 1
)
if not exist "%CHAT_TEMPLATE%" (
  echo Missing chat template at "%CHAT_TEMPLATE%"
  exit /b 1
)
"%LLAMA_SERVER%" --model "%MODEL%" --alias {MODEL_SLUG} --host 127.0.0.1 --port {SERVER_PORT} --ctx-size {MODEL_CONTEXT_WINDOW} --parallel {SERVER_PARALLEL} --n-gpu-layers 999 --threads -1 --jinja --chat-template-file "%CHAT_TEMPLATE%" --reasoning off --reasoning-format none
"""
    codex_cmd = rf"""@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Missing Python venv at "%PYTHON%"
  exit /b 1
)
"%PYTHON%" "%~dp0launch_gemma_codex.py" %*
"""
    proxy_cmd = rf"""@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" gemma_response_proxy.py --host 127.0.0.1 --port {PROXY_PORT} --upstream http://127.0.0.1:{SERVER_PORT}
"""
    verify_ps1 = rf"""$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$localHome = Join-Path $root '.codex-local'
$reasoningHome = Join-Path $root '.codex-local-reasoning'
$globalHome = Join-Path $env:USERPROFILE '.codex'
$python = Join-Path $root '.venv\Scripts\python.exe'
$requiredGemmaSkills = {required_skills_ps}
function Assert-PowerShellSyntax($path) {{
  if (-not (Test-Path $path)) {{ throw "Missing PowerShell script at $path" }}
  $tokens = $null
  $errors = $null
  [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors) | Out-Null
  if ($errors.Count -gt 0) {{
    $messages = $errors | ForEach-Object {{ $_.Message }}
    throw ("PowerShell syntax check failed for {{0}}: {{1}}" -f $path, ($messages -join '; '))
  }}
}}
function Assert-PythonSyntax($scriptName) {{
  $scriptPath = Join-Path $root $scriptName
  if (-not (Test-Path $scriptPath)) {{ throw "Missing Python script at $scriptPath" }}
  & $python -m py_compile $scriptPath
  if ($LASTEXITCODE -ne 0) {{ throw "Python syntax check failed for $scriptPath" }}
}}
function Assert-ConfigContains($configPath, $fragment, $message) {{
  $configText = Get-Content -Raw $configPath
  if (-not $configText.Contains($fragment)) {{ throw $message }}
}}
function Assert-TextContains($text, $fragment, $message) {{
  if (-not $text.Contains($fragment)) {{ throw $message }}
}}
function Assert-GemmaTerminalShim($commandName, $expectsNoConfirmYolo) {{
  $command = Get-Command $commandName -CommandType Application -ErrorAction Stop
  $shimPath = $command.Source
  if (-not (Test-Path $shimPath)) {{ throw "Gemma terminal command $commandName resolved to missing path $shimPath." }}
  $shimText = Get-Content -Raw $shimPath
  $launcherPath = Join-Path $root 'gemma-codex.cmd'
  $expectedCall = 'call "' + $launcherPath + '" %*'
  Assert-TextContains $shimText $expectedCall "$commandName terminal shim does not call this Gemma launcher."
  Assert-TextContains $shimText 'GEMMA_CODEX_TARGET_DIR=%CD%' "$commandName terminal shim does not preserve the caller working directory."
  Assert-TextContains $shimText '%*' "$commandName terminal shim does not pass through arguments such as --yolo."
  if ($expectsNoConfirmYolo) {{
    Assert-TextContains $shimText 'GEMMA_CODEX_SONION_NO_CONFIRM_YOLO=1' "$commandName terminal shim is missing sonion yolo marker."
  }} elseif ($shimText.Contains('GEMMA_CODEX_SONION_NO_CONFIRM_YOLO=1')) {{
    throw "$commandName terminal shim unexpectedly sets the sonion yolo marker."
  }}
}}
function Assert-GemmaSkillBundle($codexHome, $label) {{
  foreach ($skillName in $requiredGemmaSkills) {{
    $skillPath = Join-Path (Join-Path (Join-Path $codexHome 'skills') $skillName) 'SKILL.md'
    if (-not (Test-Path $skillPath)) {{ throw "Missing required Gemma skill $skillName in $label." }}
  }}
}}
function Assert-DeepResearchSkill($codexHome, $label) {{
  $skillPath = Join-Path $codexHome 'skills\deep-research\SKILL.md'
  if (-not (Test-Path $skillPath)) {{ throw "Missing deep-research skill in $label. Run npx skillfish add daymade/claude-code-skills deep-research." }}
  $skillText = Get-Content -Raw $skillPath
  if ($skillText -notmatch 'Gemma Context7 Runtime Override') {{ throw "deep-research skill in $label is not configured to use Context7." }}
  Assert-TextContains $skillText 'deepsearch' "deep-research skill in $label is missing the deepsearch alias."
  Assert-TextContains $skillText 'deep search' "deep-research skill in $label is missing the deep search alias."
  Assert-TextContains $skillText 'deep-search' "deep-research skill in $label is missing the deep-search alias."
}}
function Assert-Context7Skill($codexHome, $label) {{
  $skillPath = Join-Path $codexHome 'skills\context7\SKILL.md'
  if (-not (Test-Path $skillPath)) {{ throw "Missing context7 skill in $label." }}
  $skillText = Get-Content -Raw $skillPath
  Assert-TextContains $skillText 'deepsearch requests must use deep-research as the top-level workflow' "context7 skill in $label does not defer deepsearch to deep-research."
}}
Assert-PowerShellSyntax (Join-Path $root 'verify-local-setup.ps1')
Assert-PowerShellSyntax (Join-Path $root 'start-gemma-runtime.ps1')
Write-Host "Root: $root"
Write-Host "Local CODEX_HOME: $localHome"
Write-Host "Reasoning CODEX_HOME: $reasoningHome"
Write-Host "Global CODEX_HOME: $globalHome"
Assert-GemmaTerminalShim 'son' $false
Assert-GemmaTerminalShim 'sonion' $true
Assert-GemmaTerminalShim 'operator' $false
if (-not (Test-Path (Join-Path $localHome 'config.toml'))) {{ throw 'Missing local config.toml' }}
if (-not (Test-Path (Join-Path $localHome 'model-catalog.json'))) {{ throw 'Missing local model-catalog.json' }}
if (-not (Test-Path (Join-Path $reasoningHome 'config.toml'))) {{ throw 'Missing reasoning config.toml' }}
if (-not (Test-Path (Join-Path $reasoningHome 'model-catalog.json'))) {{ throw 'Missing reasoning model-catalog.json' }}
$context7ToolFragment = 'enabled_tools = ["resolve-library-id", "query-docs"]'
Assert-ConfigContains (Join-Path $localHome 'config.toml') $context7ToolFragment 'local Context7 config must expose resolve-library-id and query-docs.'
Assert-ConfigContains (Join-Path $reasoningHome 'config.toml') $context7ToolFragment 'reasoning Context7 config must expose resolve-library-id and query-docs.'
Assert-GemmaSkillBundle $localHome 'local CODEX_HOME'
Assert-GemmaSkillBundle $reasoningHome 'reasoning CODEX_HOME'
Assert-DeepResearchSkill $localHome 'local CODEX_HOME'
Assert-DeepResearchSkill $reasoningHome 'reasoning CODEX_HOME'
Assert-Context7Skill $localHome 'local CODEX_HOME'
Assert-Context7Skill $reasoningHome 'reasoning CODEX_HOME'
Write-Host 'Deep-research skill is visible to both local Codex homes.'
$patchedCodex = Join-Path $root 'tools\codex-local\bin\codex.exe'
$packagedCodex = Join-Path $root 'node_modules\.bin\codex.cmd'
if (Test-Path $patchedCodex) {{ $codexBinary = $patchedCodex }}
elseif (Test-Path $packagedCodex) {{ $codexBinary = $packagedCodex }}
else {{ throw 'Missing local Codex binary' }}
if (-not (Test-Path $python)) {{ throw "Missing Python venv at $python" }}
Assert-PythonSyntax 'launch_gemma_codex.py'
Assert-PythonSyntax 'gemma_response_proxy.py'
Assert-PythonSyntax 'gemma_reasoning_proxy.py'
Assert-PythonSyntax 'tools\bin\npx_wrapper.py'
& $python -c "import mcp"
if ($LASTEXITCODE -ne 0) {{ throw 'Missing local MCP dependency. Run .\.venv\Scripts\python.exe -m pip install -r requirements-gemma.txt' }}
& $python -c "import gemma_agent_mcp; gemma_agent_mcp.build_server()"
if ($LASTEXITCODE -ne 0) {{ throw 'Gemma agent MCP server failed to initialize.' }}
if (-not (Test-Path (Join-Path $root 'node_modules\.bin\context7-mcp.cmd'))) {{ throw 'Missing local Context7 MCP binary. Run npm install.' }}
& (Join-Path $root 'node_modules\.bin\context7-mcp.cmd') --version
if ($LASTEXITCODE -ne 0) {{ throw 'Context7 MCP server failed to initialize.' }}
if (-not (Test-Path (Join-Path $root 'node_modules\.bin\ctx7.cmd'))) {{ throw 'Missing local ctx7 CLI. Run npm install.' }}
& (Join-Path $root 'node_modules\.bin\ctx7.cmd') --version
if ($LASTEXITCODE -ne 0) {{ throw 'ctx7 CLI failed to initialize.' }}
if (-not (Test-Path (Join-Path $root 'tools\bin\ctx7.cmd'))) {{ throw 'Missing Gemma ctx7 wrapper at tools\bin\ctx7.cmd.' }}
& (Join-Path $root 'tools\bin\ctx7.cmd') --version
if ($LASTEXITCODE -ne 0) {{ throw 'ctx7 wrapper failed to initialize.' }}
if (-not (Test-Path (Join-Path $root 'tools\bin\npx.cmd'))) {{ throw 'Missing Gemma npx wrapper at tools\bin\npx.cmd.' }}
$ctx7SearchOutput = & (Join-Path $root 'tools\bin\ctx7.cmd') search context7 2>&1
if ($LASTEXITCODE -ne 2 -or (($ctx7SearchOutput -join "`n") -notmatch 'no top-level search command')) {{ throw 'ctx7 wrapper failed stale search guard.' }}
$npxCtx7Output = & (Join-Path $root 'tools\bin\npx.cmd') ctx7@latest ctx7 search context7 2>&1
if ($LASTEXITCODE -ne 2 -or (($npxCtx7Output -join "`n") -notmatch 'Do not repeat the ctx7 binary name')) {{ throw 'npx wrapper failed repeated ctx7 guard.' }}
Write-Host "Codex binary: $codexBinary"
$codexVersion = & $codexBinary --version
if ($LASTEXITCODE -ne 0) {{ throw 'Codex binary failed to initialize.' }}
Write-Host "Codex version: $codexVersion"
$codexSha256 = (Get-FileHash -Algorithm SHA256 $codexBinary).Hash.ToLowerInvariant()
Write-Host "Codex sha256: $codexSha256"
Write-Host 'Codex npm package: {CODEX_NPM_PACKAGE}@{CODEX_NPM_VERSION}'
Write-Host 'Local setup files are present.'
"""
    write_text(root / "start-gemma-runtime.ps1", runtime_ps1)
    write_text(root / "start-gemma-server.cmd", server_cmd)
    write_text(root / "start-gemma-proxy.cmd", proxy_cmd)
    write_text(root / "gemma-codex.cmd", codex_cmd)
    write_text(root / "verify-local-setup.ps1", verify_ps1)
    write_tool_wrappers(root)
    write_context7_command_wrappers(root)


def build_global_shim_text(root, *, no_confirm_yolo=False):
    launcher = str(Path(root) / "gemma-codex.cmd")
    no_confirm_yolo_line = (
        'set "GEMMA_CODEX_SONION_NO_CONFIRM_YOLO=1"\n' if no_confirm_yolo else ""
    )
    return (
        "@echo off\n"
        "setlocal\n"
        'set "GEMMA_CODEX_TARGET_DIR=%CD%"\n'
        f"{no_confirm_yolo_line}"
        f'call "{launcher}" %*\n'
    )


def write_context7_command_wrappers(root):
    source_bin = Path(__file__).resolve().parent / "tools" / "bin"
    target_bin = Path(root) / "tools" / "bin"
    target_bin.mkdir(parents=True, exist_ok=True)
    for name in ("ctx7.cmd", "npx.cmd", "npx_wrapper.py"):
        source = source_bin / name
        target = target_bin / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing Context7 wrapper source: {source}")
        if source.resolve() == target.resolve():
            continue
        shutil.copy2(source, target)


def _powershell_string_array(values):
    quoted = []
    for value in values:
        safe_value = str(value).replace("'", "''")
        quoted.append(f"  '{safe_value}'")
    return "@(\n" + ",\n".join(quoted) + "\n)"


def write_global_shims(root, shim_dir=None):
    shim_dir = Path(shim_dir) if shim_dir else Path.home() / ".codex-command"
    shim_dir.mkdir(parents=True, exist_ok=True)
    write_text(shim_dir / "son.cmd", build_global_shim_text(root))
    write_text(shim_dir / "sonion.cmd", build_global_shim_text(root, no_confirm_yolo=True))
    write_text(shim_dir / "operator.cmd", build_global_shim_text(root))
    return shim_dir


def install_gemma_skills(root, source=None, target=None):
    root = Path(root)
    source = Path(source) if source else Path.home() / ".codex" / "skills"
    target = Path(target) if target else root / "skills"
    target.mkdir(parents=True, exist_ok=True)

    missing = []
    for skill_name in GEMMA_REQUIRED_SKILLS:
        source_skill = source / skill_name
        target_skill = target / skill_name
        if not (source_skill / "SKILL.md").is_file():
            missing.append(skill_name)
            continue
        if target_skill.exists() and not target_skill.is_dir():
            raise RuntimeError(f"Cannot install Gemma skill over non-directory path: {target_skill}")
        shutil.copytree(source_skill, target_skill, dirs_exist_ok=True)

    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing required Gemma skill(s) in {source}: {missing_text}")
    return target


def build_local_readme(reasoning=False):
    mode = "reasoning proxy" if reasoning else "direct proxy"
    return (
        "# Local Gemma Codex Home\n\n"
        "This directory is used only when launchers set `CODEX_HOME` to this path.\n\n"
        f"- `config.toml` points Codex at the local Gemma {mode}.\n"
        "- `model-catalog.json` contains only the Gemma 4 entry.\n"
        "- No global `%USERPROFILE%\\.codex` files are modified by this setup helper.\n"
        "- No local safety or moderation wrapper is added in this configuration.\n"
    )


def link_skills(root, target=None):
    root = Path(root)
    target = Path(target) if target else Path.home() / ".codex" / "skills"
    if not target.exists():
        return []

    links = []
    for local_home in (root / ".codex-local", root / ".codex-local-reasoning"):
        local_home.mkdir(parents=True, exist_ok=True)
        link = local_home / "skills"
        links.append(_ensure_exact_skills_link(target, link, local_home))
    return links


def _ensure_exact_skills_link(target, link, local_home):
    target = Path(target).resolve()
    link = Path(link)
    local_home = Path(local_home).resolve()
    if _is_link_like(link):
        if _link_points_to_target(link, target):
            return link
        _remove_link_like(link)
        _link_directory(target, link)
        return link
    if link.exists():
        if _is_replaceable_managed_skill_directory(link, target, local_home):
            shutil.rmtree(link)
            _link_directory(target, link)
        elif link.is_dir():
            _link_skill_children(target, link)
        else:
            raise RuntimeError(f"Cannot link skills over non-directory path: {link}")
        return link
    _link_directory(target, link)
    return link


def _is_link_like(path):
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _link_points_to_target(link, target):
    try:
        return Path(link).resolve() == Path(target).resolve()
    except OSError:
        return False


def _remove_link_like(link):
    if Path(link).is_symlink():
        Path(link).unlink()
    else:
        Path(link).rmdir()


def _is_replaceable_managed_skill_directory(link, target, local_home):
    link = Path(link)
    if not link.is_dir() or _is_link_like(link):
        return False
    try:
        resolved_link = link.resolve(strict=False)
        resolved_home = Path(local_home).resolve(strict=False)
    except OSError:
        return False
    if resolved_link != resolved_home and resolved_home not in resolved_link.parents:
        return False
    target_names = {child.name for child in Path(target).iterdir()}
    return all(child.name in target_names for child in link.iterdir())


def _link_skill_children(target, link):
    for child in Path(target).iterdir():
        child_link = Path(link) / child.name
        if child_link.exists() or child_link.is_symlink():
            continue
        _link_directory(child, child_link)


def _link_directory(target, link):
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        # Fall back to a junction on Windows when symlink privilege is unavailable.
        import subprocess

        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            timeout=JUNCTION_CREATE_TIMEOUT_SECONDS,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Write local Gemma Codex setup files.")
    parser.add_argument(
        "--install-global-shims",
        action="store_true",
        help="Deprecated no-op: son/sonion/operator shims are written by default.",
    )
    parser.add_argument(
        "--skip-global-shims",
        action="store_true",
        help="Do not write son/sonion/operator shims under the user profile command directory.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent
    write_local_codex_config(root)
    write_launchers(root)
    skill_dir = install_gemma_skills(root)
    link_skills(root, target=skill_dir)
    if not args.skip_global_shims:
        write_global_shims(root)
    print(f"Wrote local Codex setup under {root}")


if __name__ == "__main__":
    main()
