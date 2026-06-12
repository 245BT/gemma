@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Missing Python venv at "%PYTHON%"
  exit /b 1
)
"%PYTHON%" "%~dp0launch_gemma_codex.py" %*
