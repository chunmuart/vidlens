@echo off
setlocal
title VidLens Docker
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-vidlens-docker.ps1" %*
if errorlevel 1 (
  echo.
  echo VidLens Docker startup failed. See the message above.
  pause
)
endlocal
