@echo off
REM Double-click to set up stock_predict (AlphaEdge). Runs setup.ps1; extra options are passed on,
REM for example:  setup.bat -CpuOnly   or   setup.bat -AllModels
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
echo.
pause
