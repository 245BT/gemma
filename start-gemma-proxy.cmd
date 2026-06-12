@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" gemma_response_proxy.py --host 127.0.0.1 --port 8081 --upstream http://127.0.0.1:8080
