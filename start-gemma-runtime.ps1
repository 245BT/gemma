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

function Test-ProxyProcessCommandLineForPort($commandLine, $scriptName, $port) {
  if (-not ($commandLine -and $commandLine.Contains($scriptName))) { return $false }
  $expectedPort = [string]$port
  $tokens = $commandLine -split '\s+'
  for ($i = 0; $i -lt $tokens.Count; $i++) {
    $token = $tokens[$i] -replace '^[\s"'',]+|[\s"'',]+$', ''
    if ($token -eq '--port' -and ($i + 1) -lt $tokens.Count) {
      $next = $tokens[$i + 1] -replace '^[\s"'',]+|[\s"'',]+$', ''
      if ($next -eq $expectedPort) { return $true }
    }
    if ($token -eq "--port=$expectedPort") { return $true }
  }
  return $false
}

function Get-ProxyListenerFamilyIds($owner, $scriptName, $port) {
  $familyIds = @([int]$owner)
  foreach ($ancestorId in (Get-ProcessAncestorIds $owner)) {
    $ancestor = Get-CimInstance Win32_Process -Filter "ProcessId=$ancestorId" -ErrorAction SilentlyContinue
    if ($ancestor -and (Test-ProxyProcessCommandLineForPort $ancestor.CommandLine $scriptName $port)) {
      $familyIds += [int]$ancestorId
    }
  }
  return $familyIds | Select-Object -Unique
}

function Get-ProtectedProxyListenerProcessIds($scriptName) {
  $protectedProcessIds = @()
  Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
    $listener = $_
    $owner = [int]$listener.OwningProcess
    $ownerProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$owner" -ErrorAction SilentlyContinue
    if ($ownerProcess -and (Test-ProxyProcessCommandLineForPort $ownerProcess.CommandLine $scriptName $listener.LocalPort)) {
      $protectedProcessIds += Get-ProxyListenerFamilyIds $owner $scriptName $listener.LocalPort
    }
  }
  return $protectedProcessIds | Select-Object -Unique
}

function Stop-StaleProxyProcesses($scriptName, $port) {
  $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $listener) { return }
  $protectedProcessIds = @(Get-ProtectedProxyListenerProcessIds $scriptName)
  Get-CimInstance Win32_Process |
    Where-Object { (Test-ProxyProcessCommandLineForPort $_.CommandLine $scriptName $port) -and -not ($protectedProcessIds -contains [int]$_.ProcessId) } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Stop-UnhealthyProxyListener($url, $scriptName, $port) {
  if (Test-Endpoint $url) { return }
  $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $listener) { return }
  $owner = [int]$listener.OwningProcess
  $ownerProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$owner" -ErrorAction SilentlyContinue
  if (-not ($ownerProcess -and (Test-ProxyProcessCommandLineForPort $ownerProcess.CommandLine $scriptName $port))) { return }
  Get-ProxyListenerFamilyIds $owner $scriptName $port |
    ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Milliseconds 300
}

function Start-ProxyIfNeeded($url, $scriptName, $port, $python, $arguments, $root) {
  if (Test-Endpoint $url) { return }
  Stop-UnhealthyProxyListener $url $scriptName $port
  if (Test-Endpoint $url) { return }
  if (-not (Test-Path $python)) { throw "Missing Python venv at $python" }
  Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden | Out-Null
  Wait-Endpoint $url 20
}

function Stop-StaleLlamaServerProcesses($serverExe, $port) {
  $serverExeText = [System.IO.Path]::GetFullPath($serverExe)
  $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  $protectedProcessIds = @()
  if ($listener) {
    $owner = [int]$listener.OwningProcess
    $protectedProcessIds = @($owner) + (Get-ProcessAncestorIds $owner)
  }
  Get-CimInstance Win32_Process |
    Where-Object { $_.Name -eq 'llama-server.exe' -and $_.CommandLine -and $_.CommandLine.Contains($serverExeText) -and -not ($protectedProcessIds -contains [int]$_.ProcessId) } |
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

Stop-StaleLlamaServerProcesses $serverExe 8080

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
  Start-Process -FilePath $serverExe -ArgumentList $serverArgs -WorkingDirectory (Split-Path -Parent $serverExe) -WindowStyle Hidden | Out-Null
  Wait-Endpoint $serverUrl 120
}

$directProxyArgs = @('gemma_response_proxy.py','--host','127.0.0.1','--port','8081','--upstream','http://127.0.0.1:8080')
Start-ProxyIfNeeded $proxyUrl 'gemma_response_proxy.py' 8081 $python $directProxyArgs $root

$reasoningProxyArgs = @('gemma_reasoning_proxy.py','--host','127.0.0.1','--port','8082','--upstream','http://127.0.0.1:8081')
Start-ProxyIfNeeded $reasoningProxyUrl 'gemma_reasoning_proxy.py' 8082 $python $reasoningProxyArgs $root

Stop-StaleProxyProcesses 'gemma_response_proxy.py' 8081
Stop-StaleProxyProcesses 'gemma_reasoning_proxy.py' 8082
