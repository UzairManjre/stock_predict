@echo off
REM Stop the portable PostgreSQL 16 server started by db_start.bat.
set "PGROOT=%~dp0postgres"
if not exist "%PGROOT%\pgsql\bin\pg_ctl.exe" set "PGROOT=%~dp0..\postgres"
"%PGROOT%\pgsql\bin\pg_ctl.exe" -D "%PGROOT%\data" -m fast stop
