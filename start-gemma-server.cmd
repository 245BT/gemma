@echo off
setlocal
cd /d "%~dp0"
set "MODEL=%~dp0models\gemma-4-26B-A4B-it-uncensored-GGUF\gemma-4-26B-A4B-it-uncensored-Q4_K_M.gguf"
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
"%LLAMA_SERVER%" --model "%MODEL%" --alias gemma-4-26b-a4b-it-uncensored-q4-k-m --host 127.0.0.1 --port 8080 --ctx-size 262144 --parallel 1 --n-gpu-layers 999 --threads -1 --jinja --chat-template-file "%CHAT_TEMPLATE%" --reasoning off --reasoning-format none
