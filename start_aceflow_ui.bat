@echo off
setlocal

cd /d "%~dp0"

rem Minimal ACEFlow UI launcher for standard ACE-Step uv workflow
rem Optional overrides before launch:
rem   set PORT=7861
rem   set SERVER_NAME=127.0.0.1
rem   set ACEFLOW_CONFIG_PATH=acestep-v15-turbo
rem   set ACEFLOW_LM_MODEL_PATH=acestep-5Hz-lm-4B
rem   set ACEFLOW_DEVICE=auto
rem   set ACEFLOW_RESULTS_DIR=%CD%\aceflow_outputs

if "%PORT%"=="" set "PORT=7861"
if "%SERVER_NAME%"=="" set "SERVER_NAME=127.0.0.1"
if "%ACEFLOW_CONFIG_PATH%"=="" set "ACEFLOW_CONFIG_PATH=acestep-v15-turbo"
if "%ACEFLOW_LM_MODEL_PATH%"=="" set "ACEFLOW_LM_MODEL_PATH=acestep-5Hz-lm-4B"
if "%ACEFLOW_DEVICE%"=="" set "ACEFLOW_DEVICE=auto"
if "%ACEFLOW_RESULTS_DIR%"=="" set "ACEFLOW_RESULTS_DIR=%CD%\aceflow_outputs"

where uv >nul 2>nul
if errorlevel 1 (
    echo [ACEFLOW] uv not found in PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [ACEFLOW] .venv not found, running uv sync...
    uv sync
    if errorlevel 1 (
        echo [ACEFLOW] uv sync failed.
        pause
        exit /b 1
    )
)

set "ACESTEP_REMOTE_CONFIG_PATH=%ACEFLOW_CONFIG_PATH%"
set "ACESTEP_REMOTE_LM_MODEL_PATH=%ACEFLOW_LM_MODEL_PATH%"
set "ACESTEP_REMOTE_DEVICE=%ACEFLOW_DEVICE%"
set "ACESTEP_REMOTE_RESULTS_DIR=%ACEFLOW_RESULTS_DIR%"

echo Starting ACEFlow UI...
echo http://%SERVER_NAME%:%PORT%
echo [ACEFLOW] CFG=%ACESTEP_REMOTE_CONFIG_PATH% ^| LM=%ACESTEP_REMOTE_LM_MODEL_PATH%

call .venv\Scripts\activate.bat
python -m acestep.ui.aceflow.run --host %SERVER_NAME% --port %PORT%

pause
