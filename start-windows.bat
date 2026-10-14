@echo off
rem scView launcher for Windows: double-click to start, then use stop-windows.bat.
rem Starts Docker Desktop if needed, creates .env on first run, builds and starts
rem the production stack, and opens http://localhost:3000.
setlocal EnableExtensions
title scView
cd /d "%~dp0"

echo.
echo   scView - starting up
echo   ====================
echo.

if not exist docker-compose.yml (
    echo   This file has to be run from the unzipped scView folder.
    echo   Right-click the downloaded ZIP, choose "Extract All", open the
    echo   extracted folder, and double-click start-windows.bat there.
    goto :fail
)

where docker >nul 2>&1
if errorlevel 1 (
    echo   Docker Desktop is not installed.
    echo   Install it from https://www.docker.com/products/docker-desktop/
    echo   restart the computer, then double-click this file again.
    goto :fail
)

docker info >nul 2>&1
if not errorlevel 1 goto :docker_ready
echo   Starting Docker Desktop, this can take a minute or two...
if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
set /a tries=0
:wait_docker
timeout /t 5 /nobreak >nul
docker info >nul 2>&1
if not errorlevel 1 goto :docker_ready
set /a tries+=1
if %tries% lss 36 goto :wait_docker
echo.
echo   Docker Desktop did not start. Open it from the Start menu, wait until
echo   it shows "Engine running", then double-click this file again.
goto :fail

:docker_ready
echo   [ok] Docker is running.

if exist .env goto :env_ready
echo.
echo   Optional: paste a DeepInfra API key to turn on the AI features.
echo   Press Enter to skip. scView works without one, and you can add it
echo   later by editing the .env file in this folder.
set "KEY="
set /p "KEY=  DeepInfra API key: "
findstr /v /b /c:"DEEPINFRA_API_KEY=" .env.example > .env
>>.env echo DEEPINFRA_API_KEY=%KEY%
echo   [ok] Settings saved to .env
:env_ready

echo.
echo   Downloading and starting scView. The first run downloads about 1 GB.
echo   If a prebuilt part isn't available it is built here instead, which
echo   can take tens of minutes. Later starts take about a minute.
echo   You can ignore a "must be built from source" warning: this window
echo   does that for you.
echo.
docker compose pull --ignore-pull-failures
docker compose up -d
if errorlevel 1 (
    echo.
    echo   scView did not start; the messages above say why.
    echo   See "Windows troubleshooting" in the README for common fixes.
    goto :fail
)

echo.
echo   Waiting for the app to answer...
set /a tries=0
:wait_app
curl.exe -sf -o nul http://localhost:3000/ >nul 2>&1
if not errorlevel 1 goto :app_ready
set /a tries+=1
if %tries% geq 30 goto :app_ready
timeout /t 2 /nobreak >nul
goto :wait_app

:app_ready
start "" http://localhost:3000
echo.
echo   scView is running at http://localhost:3000
echo.
echo   You can close this window; scView keeps running in Docker Desktop.
echo   To stop it, double-click stop-windows.bat. Your datasets are kept.
echo.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
