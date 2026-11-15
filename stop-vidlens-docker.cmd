@echo off
setlocal
title Stop VidLens Docker
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop-vidlens-docker.ps1"
if errorlevel 1 pause
endlocal
