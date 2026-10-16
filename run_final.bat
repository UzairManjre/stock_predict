@echo off
cd /d "D:\IMP\NEED MONEY THINGS\stock_predict"
call .\venv\Scripts\activate.bat
python Cleaner\FINAL.PY --watch
pause
