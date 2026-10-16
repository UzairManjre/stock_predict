@echo off
REM Start the portable PostgreSQL 16 server (port 5433) that lives next to the project folder.
set PGROOT=%~dp0..\postgres
"%PGROOT%\pgsql\bin\pg_ctl.exe" -D "%PGROOT%\data" -l "%PGROOT%\postgres.log" status >nul 2>&1
if %errorlevel%==0 (
    echo PostgreSQL is already running.
    exit /b 0
)
"%PGROOT%\pgsql\bin\pg_ctl.exe" -D "%PGROOT%\data" -l "%PGROOT%\postgres.log" -w start
