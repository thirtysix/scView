@echo off
rem scView launcher for Windows: double-click to start, then use stop-windows.bat.
rem Uses Docker Desktop, or Podman (Podman Desktop) when Docker isn't there;
rem starts it if needed, creates .env on first run, builds and starts the
rem production stack, and opens http://localhost:3000.
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

rem Docker when it is running, else Podman, else Docker (started below).
set "ENGINE="
set "ENGINE_APP=Docker Desktop"
docker info >nul 2>&1
if not errorlevel 1 set "ENGINE=docker"
if not defined ENGINE (
    where podman >nul 2>&1
    if not errorlevel 1 set "ENGINE=podman"
)
if not defined ENGINE (
    where docker >nul 2>&1
    if not errorlevel 1 set "ENGINE=docker"
)
if not defined ENGINE (
    echo   Neither Docker Desktop nor Podman is installed. Install one of them:
    echo     Docker Desktop: https://www.docker.com/products/docker-desktop/
    echo     Podman Desktop: https://podman-desktop.io/  ^(include Compose when it asks^)
    echo   restart the computer, then double-click this file again.
    goto :fail
)
if "%ENGINE%"=="podman" goto :podman

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
goto :engine_ready

:podman
set "ENGINE_APP=Podman"
podman info >nul 2>&1
if not errorlevel 1 goto :podman_ready
podman machine list --format "{{.Name}}" 2>nul | findstr /r "." >nul
if not errorlevel 1 goto :podman_start
echo   Creating the Podman machine, first time only: this takes a few minutes...
podman machine init
if errorlevel 1 (
    echo.
    echo   Could not create the Podman machine. Open Podman Desktop and follow
    echo   its setup, then double-click this file again.
    goto :fail
)
:podman_start
echo   Starting the Podman machine, this can take a minute...
rem Windows shutting down with the machine running leaves its API socket file
rem behind; the next start then has no Docker API pipe for Compose
rem ("win-sshproxy.exe failed to start"). The machine is stopped here, so the
rem file can only be stale.
del /q "%TEMP%\podman\*-api.sock" >nul 2>&1
podman machine start >nul 2>&1
podman info >nul 2>&1
if not errorlevel 1 goto :podman_ready
echo.
echo   Podman did not start. Open Podman Desktop, start the Podman machine
echo   there, then double-click this file again.
goto :fail

:podman_ready
rem Compose reaches Podman through that pipe, podman itself does not. Missing
rem pipe (a machine Podman Desktop started over a stale socket): restart once.
powershell -NoProfile -Command "if ([IO.Directory]::GetFiles('\\.\pipe\') -like '*\podman-*') {exit 0} else {exit 1}" >nul 2>&1
if not errorlevel 1 goto :podman_api_ready
if defined API_RETRY goto :podman_api_ready
set "API_RETRY=1"
echo   Restarting the Podman machine: its connection for Compose did not start...
podman machine stop >nul 2>&1
goto :podman_start
:podman_api_ready
podman compose version >nul 2>&1
if not errorlevel 1 goto :podman_compose_ready
echo   Podman needs Compose to run scView, and it isn't installed. In Podman
echo   Desktop open Settings, then Resources, find Compose and click Setup.
echo   Then double-click this file again.
goto :fail
:podman_compose_ready
rem Drop the ">>>> Executing external compose provider" banner on every command.
set "PODMAN_COMPOSE_WARNING_LOGS=false"
echo   [ok] Podman is running.

:engine_ready

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

rem Memory limit (SCVIEW_MEMORY_GB, docker-compose.yml): a dataset too big for
rem the computer stops scView, not Windows. Suggested once: free memory minus
rem 3 GB to the nearest GB, at least 4 (WANTGB), then at most what Docker/Podman
rem has minus 0.5 GB, rounded down, at least 1 (CAPGB). A limit above what the
rem engine has lets its machine run out of memory before the limit stops scView.
set "MEMGB="
for /f "tokens=2 delims==" %%a in ('findstr /b /c:"SCVIEW_MEMORY_GB=" .env 2^>nul') do set "MEMGB=%%a"
set "ENGINE_BYTES=0"
if "%ENGINE%"=="podman" (
    for /f %%b in ('podman info --format "{{.Host.MemTotal}}" 2^>nul') do set "ENGINE_BYTES=%%b"
) else (
    for /f %%b in ('docker info --format "{{.MemTotal}}" 2^>nul') do set "ENGINE_BYTES=%%b"
)
set "FREEGB=?"
set "ENGGB=?"
set "SUGGEST=4"
set "WANTGB=4"
set "CAPGB=0"
for /f "tokens=1-5" %%a in ('powershell -NoProfile -Command "$c=[cultureinfo]::InvariantCulture; $f=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB; $e=%ENGINE_BYTES%/1GB; $w=[math]::Round($f-3); if ($w -lt 4) {$w=4}; $x=0; if ($e -gt 0) {$x=[math]::Max(1,[math]::Floor($e-0.5))}; $s=$w; if ($x -gt 0 -and $s -gt $x) {$s=$x}; $f.ToString('F1',$c)+' '+$e.ToString('F1',$c)+' '+$s+' '+$w+' '+$x"') do (
    set "FREEGB=%%a"
    set "ENGGB=%%b"
    set "SUGGEST=%%c"
    set "WANTGB=%%d"
    set "CAPGB=%%e"
)
if defined MEMGB goto :memory_ready
echo.
echo   Free memory now: %FREEGB% GB (%ENGINE_APP% has %ENGGB% GB).
if %SUGGEST% geq %WANTGB% goto :memory_ask
echo   %ENGINE_APP% has only %ENGGB% GB of memory, so scView can use at most %SUGGEST% GB.
echo   To give it more, see "Windows troubleshooting" in the README: a .wslconfig
echo   file in your user folder sets it. Then change SCVIEW_MEMORY_GB in the .env
echo   file, or delete that line to be asked again.
:memory_ask
echo   How much memory may scView use? A dataset that needs more stops
echo   scView, not the computer. Press Enter for the suggestion.
set "ANSWER="
set /p "ANSWER=  Memory for scView in GB [%SUGGEST%]: "
set "MEMGB=%SUGGEST%"
if defined ANSWER (
    echo %ANSWER%| findstr /r "^[0-9][0-9]*$" >nul && set "MEMGB=%ANSWER%"
)
findstr /v /b /c:"SCVIEW_MEMORY_GB=" .env > .env.tmp
>>.env.tmp echo SCVIEW_MEMORY_GB=%MEMGB%
move /y .env.tmp .env >nul
:memory_ready
echo   [ok] Memory limit: %MEMGB% GB (SCVIEW_MEMORY_GB in the .env file)
if %CAPGB% equ 0 goto :memory_fits
if %MEMGB% leq %CAPGB% goto :memory_fits
echo   %ENGINE_APP% has only %ENGGB% GB of memory: at %MEMGB% GB a big dataset can run
echo   it out of memory before scView's own limit stops it. Set SCVIEW_MEMORY_GB in
echo   the .env file to %CAPGB% or less. To give %ENGINE_APP% more, see
echo   "Windows troubleshooting" in the README: a .wslconfig file in your user folder.
:memory_fits
if %MEMGB% lss 4 echo   Under 4 GB: processing a 14,000-cell dataset needs about 3.5 GB.

echo.
echo   Downloading and starting scView. The first run downloads about 1 GB.
echo   If a prebuilt part isn't available it is built here instead, which
echo   can take tens of minutes. Later starts take about a minute.
echo   You can ignore a "must be built from source" warning: this window
echo   does that for you.
echo.
if "%ENGINE%"=="docker" (
    docker compose pull --ignore-pull-failures
) else (
    podman compose pull
)
%ENGINE% compose up -d
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
echo   You can close this window; scView keeps running in %ENGINE_APP%.
echo   To stop it, double-click stop-windows.bat. Your datasets are kept.
echo.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
