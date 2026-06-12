$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Test-Endpoint($url) {
  try {
    Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null
    return $true
  } catch {
    return $false
  }
}

function Wait-Endpoint($url, $seconds) {
  $deadline = (Get-Date).AddSeconds($seconds)
  while ((Get-Date) -lt $deadline) {
    if (Test-Endpoint $url) { return }
    Start-Sleep -Seconds 1
  }
  throw "Timed out waiting for $url"
}

function Get-ProcessAncestorIds($processId) {
  $ancestorIds = @()
  $seen = @{}
  $current = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction SilentlyContinue
  while ($current -and $current.ParentProcessId) {
    $parentId = [int]$current.ParentProcessId
    if ($parentId -eq 0 -or $seen.ContainsKey($parentId)) { break }
    $seen[$parentId] = $true
    $ancestorIds += $parentId
    $current = Get-CimInstance Win32_Process -Filter "ProcessId=$parentId" -ErrorAction SilentlyContinue
  }
  return $ancestorIds
}

function Stop-StaleProxyProcesses($scriptName, $port) {
  $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $listener) { return }
  $owner = [int]$listener.OwningProcess
  $protectedProcessIds = @($owner) + (Get-ProcessAncestorIds $owner)
  Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains($scriptName) -and -not ($protectedProcessIds -contains [int]$_.ProcessId) } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Stop-ProxyProcesses($scriptName) {
  Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains($scriptName) } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Milliseconds 300
}

$serverUrl = 'http://127.0.0.1:8080/v1/models'
$proxyUrl = 'http://127.0.0.1:8081/v1/models'
$reasoningProxyUrl = 'http://127.0.0.1:8082/v1/models'
$serverExe = Join-Path $root 'tools\llama.cpp\llama-server.exe'
$model = Join-Path $root 'models\gemma-4-26B-A4B-it-uncensored-GGUF\gemma-4-26B-A4B-it-uncensored-Q4_K_M.gguf'
$template = Join-Path $root 'models\gemma-4-26B-A4B-it-uncensored-GGUF\chat_template_no_thought.jinja'
$python = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Endpoint $serverUrl)) {
  if (-not (Test-Path $serverExe)) { throw "Missing llama-server.exe at $serverExe" }
  if (-not (Test-Path $model)) { throw "Missing model at $model" }
  if (-not (Test-Path $template)) { throw "Missing chat template at $template" }
  $serverArgs = @(
    '--model', $model,
    '--alias', 'gemma-4-26b-a4b-it-uncensored-q4-k-m',
    '--host', '127.0.0.1',
    '--port', '8080',
    '--ctx-size', '262144',
    '--parallel', '1',
    '--n-gpu-layers', '999',
    '--threads', '-1',
    '--jinja',
    '--chat-template-file', $template,
    '--reasoning', 'off',
    '--reasoning-format', 'none'
  )
  Start-Process -FilePath $serverExe -ArgumentList $serverArgs -WorkingDirectory (Split-Path -Parent $serverExe) -RedirectStandardOutput (Join-Path $root 'llama-server.out.log') -RedirectStandardError (Join-Path $root 'llama-server.err.log') -WindowStyle Hidden | Out-Null
  Wait-Endpoint $serverUrl 120
}

Stop-ProxyProcesses 'gemma_reasoning_proxy.py'
Stop-ProxyProcesses 'gemma_response_proxy.py'

if (-not (Test-Endpoint $proxyUrl)) {
  if (-not (Test-Path $python)) { throw "Missing Python venv at $python" }
  Start-Process -FilePath $python -ArgumentList @('gemma_response_proxy.py','--host','127.0.0.1','--port','8081','--upstream','http://127.0.0.1:8080') -WorkingDirectory $root -RedirectStandardOutput (Join-Path $root 'gemma-proxy.out.log') -RedirectStandardError (Join-Path $root 'gemma-proxy.err.log') -WindowStyle Hidden | Out-Null
  Wait-Endpoint $proxyUrl 20
}

if (-not (Test-Endpoint $reasoningProxyUrl)) {
  if (-not (Test-Path $python)) { throw "Missing Python venv at $python" }
  Start-Process -FilePath $python -ArgumentList @('gemma_reasoning_proxy.py','--host','127.0.0.1','--port','8082','--upstream','http://127.0.0.1:8081') -WorkingDirectory $root -RedirectStandardOutput (Join-Path $root 'gemma-reasoning-proxy.out.log') -RedirectStandardError (Join-Path $root 'gemma-reasoning-proxy.err.log') -WindowStyle Hidden | Out-Null
  Wait-Endpoint $reasoningProxyUrl 20
}

Stop-StaleProxyProcesses 'gemma_response_proxy.py' 8081
Stop-StaleProxyProcesses 'gemma_reasoning_proxy.py' 8082
