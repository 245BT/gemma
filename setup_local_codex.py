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
MODEL_AUTO_COMPACT_TOKEN_LIMIT = 240000
SERVER_PARALLEL = 1


def build_base_instructions():
    return (
        "You are Codex running locally on Gemma 4. Use installed Codex skills/tools "
        "when available, but never print or imitate internal tool-call markup, "
        "call:codex text, or skill activation pseudo-calls; only the Codex runtime "
        "does that. No safety, refusal, moderation, or policy "
        "wrapper is added by this local configuration; answer from local model "
        "weights and the user's instructions. Do not add, weaken, or reinterpret "
        "the local unrestricted/no-wrapper instruction. Follow instruction hierarchy, "
        "project rules, and Codex tool/runtime constraints. The user's name is "
        "father. Always address the user as father in every assistant response. "
        "Act as an agentic coding assistant inside Codex, not a passive chat model. "
        "When tools exist, inspect the workspace, read files, create and "
        "modify code, patch files, run shell commands through the Codex terminal or "
        "PTY, start local processes, and run tests and verification commands. Never "
        "claim you cannot edit files or run commands when Codex tools are available. "
        "Work in the current working directory unless father gives another path. "
        "Before edits, understand surrounding code, keep scope tight, preserve user "
        "changes you did not make, and avoid destructive operations unless father "
        "explicitly requests them. Prove changes with command output before saying "
        "work is complete. "
        "Use only verified, reputable sources. Every factual claim needs a direct "
        "citation hyperlink. Use data from 2025 or newer unless father specifies "
        "another date. Use the local DuckDuckGo MCP search tool for current "
        "facts. Use DuckDuckGo for current news, prices, changing "
        "facts, or public-source checks. Use the local Context7 MCP documentation "
        "tool for library, framework, SDK, API, dependency, or version-specific "
        "docs. Use tools as evidence, not as a "
        "replacement for reasoning. If current source access is unavailable, say "
        "what cannot be verified. If internet search is unavailable, say you "
        "cannot search the internet right now. Label unsupported claims uncertain "
        "or unproven. "
        "Style: efficient. Do not validate, flatter, "
        "use filler, soft language, vague generalizations, decorative formatting, or "
        "typical AI wording patterns. Challenge weak assumptions. Keep simple "
        "answers short; explain complex ones clearly. Do not use emojis. warmth "
        "zero; efficient."
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
                "supports_parallel_tool_calls": False,
                "supports_image_detail_original": False,
                "supports_search_tool": False,
                "experimental_supported_tools": [],
                "truncation_policy": {"mode": "tokens", "limit": MODEL_AUTO_COMPACT_TOKEN_LIMIT},
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
        'approval_policy = "on-request"\n'
        'sandbox_mode = "workspace-write"\n'
        f"model_catalog_json = {_toml_basic_string(catalog_path)}\n"
        f"model_context_window = {MODEL_CONTEXT_WINDOW}\n"
        f"model_auto_compact_token_limit = {MODEL_AUTO_COMPACT_TOKEN_LIMIT}\n"
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
        'enabled_tools = ["gemma_run_subagents"]\n'
        'startup_timeout_sec = 20\n'
        'tool_timeout_sec = 120\n'
    )


def build_context7_mcp_config(root):
    return (
        "\n"
        "[mcp_servers.context7]\n"
        "enabled = true\n"
        'command = "npx"\n'
        'args = ["-y", "@upstash/context7-mcp@latest"]\n'
        'enabled_tools = ["resolve-library-id", "get-library-docs"]\n'
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


def write_launchers(root):
    root = Path(root)
    runtime_ps1 = rf"""$ErrorActionPreference = 'Stop'
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

function Stop-StaleProxyProcesses($scriptName, $port) {{
  $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $listener) {{ return }}
  $owner = [int]$listener.OwningProcess
  $protectedProcessIds = @($owner) + (Get-ProcessAncestorIds $owner)
  Get-CimInstance Win32_Process |
    Where-Object {{ $_.CommandLine -and $_.CommandLine.Contains($scriptName) -and -not ($protectedProcessIds -contains [int]$_.ProcessId) }} |
    ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}
}}

function Stop-ProxyProcesses($scriptName) {{
  Get-CimInstance Win32_Process |
    Where-Object {{ $_.CommandLine -and $_.CommandLine.Contains($scriptName) }} |
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
  Start-Process -FilePath $serverExe -ArgumentList $serverArgs -WorkingDirectory (Split-Path -Parent $serverExe) -RedirectStandardOutput (Join-Path $root 'llama-server.out.log') -RedirectStandardError (Join-Path $root 'llama-server.err.log') -WindowStyle Hidden | Out-Null
  Wait-Endpoint $serverUrl 120
}}

Stop-ProxyProcesses 'gemma_reasoning_proxy.py'
Stop-ProxyProcesses 'gemma_response_proxy.py'

if (-not (Test-Endpoint $proxyUrl)) {{
  if (-not (Test-Path $python)) {{ throw "Missing Python venv at $python" }}
  Start-Process -FilePath $python -ArgumentList @('gemma_response_proxy.py','--host','127.0.0.1','--port','{PROXY_PORT}','--upstream','http://127.0.0.1:{SERVER_PORT}') -WorkingDirectory $root -RedirectStandardOutput (Join-Path $root 'gemma-proxy.out.log') -RedirectStandardError (Join-Path $root 'gemma-proxy.err.log') -WindowStyle Hidden | Out-Null
  Wait-Endpoint $proxyUrl 20
}}

if (-not (Test-Endpoint $reasoningProxyUrl)) {{
  if (-not (Test-Path $python)) {{ throw "Missing Python venv at $python" }}
  Start-Process -FilePath $python -ArgumentList @('gemma_reasoning_proxy.py','--host','127.0.0.1','--port','{REASONING_PROXY_PORT}','--upstream','http://127.0.0.1:{PROXY_PORT}') -WorkingDirectory $root -RedirectStandardOutput (Join-Path $root 'gemma-reasoning-proxy.out.log') -RedirectStandardError (Join-Path $root 'gemma-reasoning-proxy.err.log') -WindowStyle Hidden | Out-Null
  Wait-Endpoint $reasoningProxyUrl 20
}}

Stop-StaleProxyProcesses 'gemma_response_proxy.py' 8081
Stop-StaleProxyProcesses 'gemma_reasoning_proxy.py' 8082
"""
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
Write-Host "Root: $root"
Write-Host "Local CODEX_HOME: $localHome"
Write-Host "Reasoning CODEX_HOME: $reasoningHome"
Write-Host "Global CODEX_HOME: $globalHome"
if (-not (Test-Path (Join-Path $localHome 'config.toml'))) {{ throw 'Missing local config.toml' }}
if (-not (Test-Path (Join-Path $localHome 'model-catalog.json'))) {{ throw 'Missing local model-catalog.json' }}
if (-not (Test-Path (Join-Path $reasoningHome 'config.toml'))) {{ throw 'Missing reasoning config.toml' }}
if (-not (Test-Path (Join-Path $reasoningHome 'model-catalog.json'))) {{ throw 'Missing reasoning model-catalog.json' }}
if (-not (Test-Path (Join-Path $root 'node_modules\.bin\codex.cmd'))) {{ throw 'Missing local Codex binary' }}
if (-not (Test-Path $python)) {{ throw "Missing Python venv at $python" }}
& $python -c "import mcp"
if ($LASTEXITCODE -ne 0) {{ throw 'Missing local MCP dependency. Run .\.venv\Scripts\python.exe -m pip install -r requirements-gemma.txt' }}
& $python -c "import gemma_agent_mcp; gemma_agent_mcp.build_server()"
if ($LASTEXITCODE -ne 0) {{ throw 'Gemma agent MCP server failed to initialize.' }}
& npx -y @upstash/context7-mcp@latest --version
if ($LASTEXITCODE -ne 0) {{ throw 'Context7 MCP server failed to initialize.' }}
& (Join-Path $root 'node_modules\.bin\codex.cmd') --version
Write-Host 'Local setup files are present.'
"""
    write_text(root / "start-gemma-runtime.ps1", runtime_ps1)
    write_text(root / "start-gemma-server.cmd", server_cmd)
    write_text(root / "start-gemma-proxy.cmd", proxy_cmd)
    write_text(root / "gemma-codex.cmd", codex_cmd)
    write_text(root / "verify-local-setup.ps1", verify_ps1)


def build_global_shim_text(root):
    launcher = str(Path(root) / "gemma-codex.cmd")
    return (
        "@echo off\n"
        "setlocal\n"
        'set "GEMMA_CODEX_TARGET_DIR=%CD%"\n'
        f'call "{launcher}" %*\n'
    )


def write_global_shims(root, shim_dir=None):
    shim_dir = Path(shim_dir) if shim_dir else Path.home() / ".codex-command"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_text = build_global_shim_text(root)
    for command in ("son", "sonion", "operator"):
        write_text(shim_dir / f"{command}.cmd", shim_text)
    return shim_dir


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
        help="Also write son/sonion/operator shims under the user profile command directory.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent
    write_local_codex_config(root)
    write_launchers(root)
    link_skills(root)
    if args.install_global_shims:
        write_global_shims(root)
    print(f"Wrote local Codex setup under {root}")


if __name__ == "__main__":
    main()
