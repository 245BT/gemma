$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$localHome = Join-Path $root '.codex-local'
$reasoningHome = Join-Path $root '.codex-local-reasoning'
$globalHome = Join-Path $env:USERPROFILE '.codex'
$python = Join-Path $root '.venv\Scripts\python.exe'
$requiredGemmaSkills = @(
  'deep-research',
  'context7',
  'using-superpowers',
  'brainstorming',
  'writing-plans',
  'using-git-worktrees',
  'systematic-debugging',
  'test-driven-development',
  'code-simplifier',
  'subagent-driven-development',
  'executing-plans',
  'dispatching-parallel-agents',
  'requesting-code-review',
  'finishing-a-development-branch',
  'verification-before-completion'
)
function Assert-PowerShellSyntax($path) {
  if (-not (Test-Path $path)) { throw "Missing PowerShell script at $path" }
  $tokens = $null
  $errors = $null
  [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors) | Out-Null
  if ($errors.Count -gt 0) {
    $messages = $errors | ForEach-Object { $_.Message }
    throw ("PowerShell syntax check failed for {0}: {1}" -f $path, ($messages -join '; '))
  }
}
function Assert-PythonSyntax($scriptName) {
  $scriptPath = Join-Path $root $scriptName
  if (-not (Test-Path $scriptPath)) { throw "Missing Python script at $scriptPath" }
  & $python -m py_compile $scriptPath
  if ($LASTEXITCODE -ne 0) { throw "Python syntax check failed for $scriptPath" }
}
function Assert-ConfigContains($configPath, $fragment, $message) {
  $configText = Get-Content -Raw $configPath
  if (-not $configText.Contains($fragment)) { throw $message }
}
function Assert-TextContains($text, $fragment, $message) {
  if (-not $text.Contains($fragment)) { throw $message }
}
function Assert-GemmaTerminalShim($commandName, $expectsNoConfirmYolo) {
  $command = Get-Command $commandName -CommandType Application -ErrorAction Stop
  $shimPath = $command.Source
  if (-not (Test-Path $shimPath)) { throw "Gemma terminal command $commandName resolved to missing path $shimPath." }
  $shimText = Get-Content -Raw $shimPath
  $launcherPath = Join-Path $root 'gemma-codex.cmd'
  $expectedCall = 'call "' + $launcherPath + '" %*'
  Assert-TextContains $shimText $expectedCall "$commandName terminal shim does not call this Gemma launcher."
  Assert-TextContains $shimText 'GEMMA_CODEX_TARGET_DIR=%CD%' "$commandName terminal shim does not preserve the caller working directory."
  Assert-TextContains $shimText '%*' "$commandName terminal shim does not pass through arguments such as --yolo."
  if ($expectsNoConfirmYolo) {
    Assert-TextContains $shimText 'GEMMA_CODEX_SONION_NO_CONFIRM_YOLO=1' "$commandName terminal shim is missing sonion yolo marker."
  } elseif ($shimText.Contains('GEMMA_CODEX_SONION_NO_CONFIRM_YOLO=1')) {
    throw "$commandName terminal shim unexpectedly sets the sonion yolo marker."
  }
}
function Assert-GemmaSkillBundle($codexHome, $label) {
  foreach ($skillName in $requiredGemmaSkills) {
    $skillPath = Join-Path (Join-Path (Join-Path $codexHome 'skills') $skillName) 'SKILL.md'
    if (-not (Test-Path $skillPath)) { throw "Missing required Gemma skill $skillName in $label." }
  }
}
function Assert-DeepResearchSkill($codexHome, $label) {
  $skillPath = Join-Path $codexHome 'skills\deep-research\SKILL.md'
  if (-not (Test-Path $skillPath)) { throw "Missing deep-research skill in $label. Run npx skillfish add daymade/claude-code-skills deep-research." }
  $skillText = Get-Content -Raw $skillPath
  if ($skillText -notmatch 'Gemma Context7 Runtime Override') { throw "deep-research skill in $label is not configured to use Context7." }
  Assert-TextContains $skillText 'deepsearch' "deep-research skill in $label is missing the deepsearch alias."
  Assert-TextContains $skillText 'deep search' "deep-research skill in $label is missing the deep search alias."
  Assert-TextContains $skillText 'deep-search' "deep-research skill in $label is missing the deep-search alias."
}
function Assert-Context7Skill($codexHome, $label) {
  $skillPath = Join-Path $codexHome 'skills\context7\SKILL.md'
  if (-not (Test-Path $skillPath)) { throw "Missing context7 skill in $label." }
  $skillText = Get-Content -Raw $skillPath
  Assert-TextContains $skillText 'deepsearch requests must use deep-research as the top-level workflow' "context7 skill in $label does not defer deepsearch to deep-research."
}
Assert-PowerShellSyntax (Join-Path $root 'verify-local-setup.ps1')
Assert-PowerShellSyntax (Join-Path $root 'start-gemma-runtime.ps1')
Write-Host "Root: $root"
Write-Host "Local CODEX_HOME: $localHome"
Write-Host "Reasoning CODEX_HOME: $reasoningHome"
Write-Host "Global CODEX_HOME: $globalHome"
Assert-GemmaTerminalShim 'son' $false
Assert-GemmaTerminalShim 'sonion' $true
Assert-GemmaTerminalShim 'operator' $false
if (-not (Test-Path (Join-Path $localHome 'config.toml'))) { throw 'Missing local config.toml' }
if (-not (Test-Path (Join-Path $localHome 'model-catalog.json'))) { throw 'Missing local model-catalog.json' }
if (-not (Test-Path (Join-Path $reasoningHome 'config.toml'))) { throw 'Missing reasoning config.toml' }
if (-not (Test-Path (Join-Path $reasoningHome 'model-catalog.json'))) { throw 'Missing reasoning model-catalog.json' }
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
if (Test-Path $patchedCodex) { $codexBinary = $patchedCodex }
elseif (Test-Path $packagedCodex) { $codexBinary = $packagedCodex }
else { throw 'Missing local Codex binary' }
if (-not (Test-Path $python)) { throw "Missing Python venv at $python" }
Assert-PythonSyntax 'launch_gemma_codex.py'
Assert-PythonSyntax 'gemma_response_proxy.py'
Assert-PythonSyntax 'gemma_reasoning_proxy.py'
Assert-PythonSyntax 'tools\bin\npx_wrapper.py'
& $python -c "import mcp"
if ($LASTEXITCODE -ne 0) { throw 'Missing local MCP dependency. Run .\.venv\Scripts\python.exe -m pip install -r requirements-gemma.txt' }
& $python -c "import gemma_agent_mcp; gemma_agent_mcp.build_server()"
if ($LASTEXITCODE -ne 0) { throw 'Gemma agent MCP server failed to initialize.' }
if (-not (Test-Path (Join-Path $root 'node_modules\.bin\context7-mcp.cmd'))) { throw 'Missing local Context7 MCP binary. Run npm install.' }
& (Join-Path $root 'node_modules\.bin\context7-mcp.cmd') --version
if ($LASTEXITCODE -ne 0) { throw 'Context7 MCP server failed to initialize.' }
if (-not (Test-Path (Join-Path $root 'node_modules\.bin\ctx7.cmd'))) { throw 'Missing local ctx7 CLI. Run npm install.' }
& (Join-Path $root 'node_modules\.bin\ctx7.cmd') --version
if ($LASTEXITCODE -ne 0) { throw 'ctx7 CLI failed to initialize.' }
if (-not (Test-Path (Join-Path $root 'tools\bin\ctx7.cmd'))) { throw 'Missing Gemma ctx7 wrapper at tools\bin\ctx7.cmd.' }
& (Join-Path $root 'tools\bin\ctx7.cmd') --version
if ($LASTEXITCODE -ne 0) { throw 'ctx7 wrapper failed to initialize.' }
if (-not (Test-Path (Join-Path $root 'tools\bin\npx.cmd'))) { throw 'Missing Gemma npx wrapper at tools\bin\npx.cmd.' }
$ctx7SearchOutput = & (Join-Path $root 'tools\bin\ctx7.cmd') search context7 2>&1
if ($LASTEXITCODE -ne 2 -or (($ctx7SearchOutput -join "`n") -notmatch 'no top-level search command')) { throw 'ctx7 wrapper failed stale search guard.' }
$npxCtx7Output = & (Join-Path $root 'tools\bin\npx.cmd') ctx7@latest ctx7 search context7 2>&1
if ($LASTEXITCODE -ne 2 -or (($npxCtx7Output -join "`n") -notmatch 'Do not repeat the ctx7 binary name')) { throw 'npx wrapper failed repeated ctx7 guard.' }
Write-Host "Codex binary: $codexBinary"
$codexVersion = & $codexBinary --version
if ($LASTEXITCODE -ne 0) { throw 'Codex binary failed to initialize.' }
Write-Host "Codex version: $codexVersion"
$codexSha256 = (Get-FileHash -Algorithm SHA256 $codexBinary).Hash.ToLowerInvariant()
Write-Host "Codex sha256: $codexSha256"
Write-Host 'Codex npm package: @openai/codex@0.139.0'
Write-Host 'Local setup files are present.'
