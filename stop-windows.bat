@echo off
rem Stops scView. Uploaded datasets live in a Docker or Podman volume and are kept.
setlocal EnableExtensions
title scView
cd /d "%~dp0"

echo.
echo   Stopping scView (your datasets are kept)...
echo.
rem The engine start-windows.bat used: Docker when it is running, else Podman.
set "ENGINE=docker"
docker info >nul 2>&1
if errorlevel 1 (
    where podman >nul 2>&1
    if not errorlevel 1 set "ENGINE=podman"
)
%ENGINE% compose down
if errorlevel 1 (
    echo.
    echo   Could not stop scView. Is Docker Desktop or Podman running?
) else (
    echo.
    echo   scView has stopped. Double-click start-windows.bat to start it again.
)
echo.
pause
