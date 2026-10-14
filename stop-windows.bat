@echo off
rem Stops scView. Uploaded datasets live in a Docker volume and are kept.
setlocal EnableExtensions
title scView
cd /d "%~dp0"

echo.
echo   Stopping scView (your datasets are kept)...
echo.
docker compose down
if errorlevel 1 (
    echo.
    echo   Could not stop scView. Is Docker Desktop running?
) else (
    echo.
    echo   scView has stopped. Double-click start-windows.bat to start it again.
)
echo.
pause
