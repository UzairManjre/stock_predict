@echo off
REM Start the project's portable PostgreSQL 16 server (port set in postgres\data\postgresql.conf; 5433 by default).
REM setup.ps1 installs it in .\postgres; the original PC keeps it next to the project folder (..\postgres).
set "PGROOT=%~dp0postgres"
if not exist "%PGROOT%\pgsql\bin\pg_ctl.exe" set "PGROOT=%~dp0..\postgres"
if not exist "%PGROOT%\pgsql\bin\pg_ctl.exe" (
    echo PostgreSQL was not found in "%~dp0postgres" or "%~dp0..\postgres". Run setup.ps1 first.
    exit /b 1
)
"%PGROOT%\pgsql\bin\pg_ctl.exe" -D "%PGROOT%\data" -l "%PGROOT%\postgres.log" status >nul 2>&1
if %errorlevel%==0 (
    echo PostgreSQL is already running.
    exit /b 0
)
"%PGROOT%\pgsql\bin\pg_ctl.exe" -D "%PGROOT%\data" -l "%PGROOT%\postgres.log" -w start
