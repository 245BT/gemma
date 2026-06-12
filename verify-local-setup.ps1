$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$localHome = Join-Path $root '.codex-local'
$reasoningHome = Join-Path $root '.codex-local-reasoning'
$globalHome = Join-Path $env:USERPROFILE '.codex'
$python = Join-Path $root '.venv\Scripts\python.exe'
Write-Host "Root: $root"
Write-Host "Local CODEX_HOME: $localHome"
Write-Host "Reasoning CODEX_HOME: $reasoningHome"
Write-Host "Global CODEX_HOME: $globalHome"
if (-not (Test-Path (Join-Path $localHome 'config.toml'))) { throw 'Missing local config.toml' }
if (-not (Test-Path (Join-Path $localHome 'model-catalog.json'))) { throw 'Missing local model-catalog.json' }
if (-not (Test-Path (Join-Path $reasoningHome 'config.toml'))) { throw 'Missing reasoning config.toml' }
if (-not (Test-Path (Join-Path $reasoningHome 'model-catalog.json'))) { throw 'Missing reasoning model-catalog.json' }
if (-not (Test-Path (Join-Path $root 'node_modules\.bin\codex.cmd'))) { throw 'Missing local Codex binary' }
if (-not (Test-Path $python)) { throw "Missing Python venv at $python" }
& $python -c "import mcp"
if ($LASTEXITCODE -ne 0) { throw 'Missing local MCP dependency. Run .\.venv\Scripts\python.exe -m pip install -r requirements-gemma.txt' }
& $python -c "import gemma_agent_mcp; gemma_agent_mcp.build_server()"
if ($LASTEXITCODE -ne 0) { throw 'Gemma agent MCP server failed to initialize.' }
& npx -y @upstash/context7-mcp@latest --version
if ($LASTEXITCODE -ne 0) { throw 'Context7 MCP server failed to initialize.' }
& (Join-Path $root 'node_modules\.bin\codex.cmd') --version
Write-Host 'Local setup files are present.'
