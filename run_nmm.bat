@echo off
setlocal EnableDelayedExpansion
title Nine Men's Morris

set "NMM_DIR=%~dp0"
set "VENV_PY=%NMM_DIR%.venv\Scripts\python.exe"
set "HOST=127.0.0.1"
set "PORT=8000"

if not exist "%VENV_PY%" (
    echo [NMM] ERROR: .venv not found. Run install.bat first.
    pause & exit /b 1
)

rem -- Start Ollama if installed but not yet running --
where ollama >nul 2>&1
if %errorlevel%==0 (
    curl -sf http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo [NMM] Starting Ollama service...
        start /B "" ollama serve
        timeout /t 3 /nobreak >nul
    ) else (
        echo [NMM] Ollama already running.
    )
)

rem -- Fall back to port 8080 if 8000 is busy --
netstat -an | findstr /C:":%PORT% " | findstr /C:"LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo [NMM] Port %PORT% is in use, switching to 8080...
    set "PORT=8080"
)

echo [NMM] Starting Nine Men's Morris at http://%HOST%:%PORT% ...
cd /d "%NMM_DIR%"

rem -- Launch uvicorn via 'python -m uvicorn' so we don't depend on the
rem -- .exe shim location (more robust when install paths contain spaces) --
start /B "" "%VENV_PY%" -m uvicorn web.app:app --host %HOST% --port %PORT%

rem -- Poll /api/ping until the server is ready (up to 60s) --
echo [NMM] Waiting for server to be ready...
set /a _tries=0
:wait_loop
set /a _tries+=1
if %_tries% gtr 60 (
    echo [NMM] Server took too long to respond -- opening browser anyway.
    goto open_browser
)
timeout /t 1 /nobreak >nul
curl -sf "http://%HOST%:%PORT%/api/ping" >nul 2>&1
if errorlevel 1 goto wait_loop

:open_browser
echo [NMM] Opening browser at http://%HOST%:%PORT%
start "" "http://%HOST%:%PORT%"

echo.
echo [NMM] Server is running. Close this window to stop.
echo.
pause >nul
