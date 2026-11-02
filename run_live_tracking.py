"""
run_live_tracking.py — One unattended live-tracking run, for Windows Task Scheduler.

  1. Starts the portable PostgreSQL server (db_start.bat) and waits until it accepts connections;
     it does not start by itself after a reboot.
  2. Runs run_daily_update.py: fetch prices, repair them, rebuild features, retrain if due,
     forecast with the final model and its confidence flag, then score every forecast whose
     5- or 21-session horizon has passed (Cleaner/accuracy_checker.py).
  3. Appends everything to logs/daily_YYYY-MM-DD.log.

Scheduled on weekdays at 18:30 by setup_live_tracking.ps1. Run by hand with:
    venv\\Scripts\\python.exe run_live_tracking.py
Extra options go to run_daily_update.py, for example to skip retraining:
    venv\\Scripts\\python.exe run_live_tracking.py --retrain-interval 9999
"""

import os
import sys
import time
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(HERE, 'venv', 'Scripts', 'python.exe')
LOG_DIR = os.path.join(HERE, 'logs')

# The portable PostgreSQL: setup.ps1 installs it in .\postgres; the original PC keeps it in ..\postgres
PG_ISREADY = next((p for p in (os.path.join(HERE, 'postgres', 'pgsql', 'bin', 'pg_isready.exe'),
                               os.path.join(HERE, '..', 'postgres', 'pgsql', 'bin', 'pg_isready.exe'))
                   if os.path.exists(p)), 'pg_isready')


def env_setting(key, default):
    """Read one setting from .env (the same file every pipeline script reads), else the default."""
    path = os.path.join(HERE, '.env')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                name, sep, value = line.strip().partition('=')
                if sep and name.strip() == key and not name.startswith('#'):
                    return value.strip()
    return default


DB_PORT = env_setting('DB_PORT', '5433')


def database_ready():
    result = subprocess.run([PG_ISREADY, '-h', 'localhost', '-p', DB_PORT], capture_output=True)
    return result.returncode == 0


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"daily_{datetime.now():%Y-%m-%d}.log")
    with open(log_path, 'a', encoding='utf-8', errors='replace') as log:
        def note(text):
            log.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}\n")
            log.flush()

        note("Live tracking run started")
        if not database_ready():
            note("Starting PostgreSQL (db_start.bat)")
            subprocess.run(['cmd', '/c', os.path.join(HERE, 'db_start.bat')], cwd=HERE, stdout=log, stderr=log)
            # Recovery after an unclean shutdown can take a minute
            for _ in range(60):
                if database_ready():
                    break
                time.sleep(3)
        if not database_ready():
            note("PostgreSQL is not accepting connections; stopping")
            return 1
        note("PostgreSQL ready; running run_daily_update.py")

        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        result = subprocess.run([PYTHON, os.path.join(HERE, 'run_daily_update.py')] + sys.argv[1:], cwd=HERE,
                                stdout=log, stderr=subprocess.STDOUT, env=env)
        note(f"run_daily_update.py finished with exit code {result.returncode}")
        return result.returncode


if __name__ == '__main__':
    sys.exit(main())
