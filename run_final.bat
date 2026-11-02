@echo off
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
python Cleaner\FINAL.PY --watch
pause
