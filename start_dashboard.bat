@echo off
REM Start the database (if it is not running) and open the AlphaEdge dashboard at http://localhost:8501
cd /d "%~dp0"
call "%~dp0db_start.bat"
if errorlevel 1 (
    echo Could not start PostgreSQL. Run setup.bat first.
    pause
    exit /b 1
)
"%~dp0venv\Scripts\python.exe" -m streamlit run "%~dp0dashboard\app.py"
pause
