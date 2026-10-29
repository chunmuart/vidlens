@echo off
setlocal
title VidLens Launcher
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-vidlens.ps1" %*
if errorlevel 1 (
  echo.
  echo VidLens failed to start. See .runtime\logs for details.
  pause
)
endlocal
