"""
verify_setup.py — Check that this copy of stock_predict (AlphaEdge) is ready to run.

Checks, in order:
  1. the Python packages import (and whether PyTorch sees a GPU);
  2. the database settings in .env and the PostgreSQL tables (row counts, price date range);
  3. the data and model files that come from the GitHub releases;
  4. an end-to-end run: today's top-20 picks and confidence flag from the final model
     (Cleaner/final_model.py --score), which reads the features file and every model artifact.

    venv\\Scripts\\python.exe verify_setup.py           # everything
    venv\\Scripts\\python.exe verify_setup.py --quick   # skip step 4

Exits with code 0 when every check passes. Nothing is written.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
failures = []


def ok(text):
    print(f"  [ok]   {text}")


def fail(text, fix=""):
    failures.append(text)
    print(f"  [FAIL] {text}")
    if fix:
        print(f"         fix: {fix}")


def read_env():
    cfg = {"DB_HOST": "localhost", "DB_PORT": "5433", "DB_USER": "postgres",
           "DB_PASSWORD": "postgres", "DB_NAME": "stock_predict"}
    path = os.path.join(HERE, ".env")
    found = os.path.exists(path)
    if found:
        with open(path, encoding="utf-8") as f:
            for line in f:
                name, sep, value = line.strip().partition("=")
                if sep and not name.startswith("#"):
                    cfg[name.strip()] = value.strip()
    return cfg, found


def check_packages():
    print("\n1. Python packages")
    print(f"  Python {sys.version.split()[0]} ({sys.executable})")
    for name in ["torch", "lightning", "pytorch_forecasting", "lightgbm", "pandas", "pyarrow",
                 "psycopg2", "streamlit", "plotly", "yfinance", "transformers", "vaderSentiment"]:
        try:
            module = importlib.import_module(name)
            ok(f"{name} {getattr(module, '__version__', '')}".rstrip())
        except Exception as e:  # noqa: BLE001 - report any import problem
            fail(f"{name} does not import ({type(e).__name__}: {e})",
                 "run setup.ps1 again, or: venv\\Scripts\\python.exe -m pip install -r requirements.txt")
    try:
        import torch
        if torch.cuda.is_available():
            ok(f"PyTorch sees the GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  [info] PyTorch is running on the CPU: predictions work; retraining the neural network is slow")
    except Exception:  # noqa: BLE001 - already reported above
        pass


def check_database(cfg, env_found):
    print("\n2. Database")
    if env_found:
        ok(f".env found: {cfg['DB_USER']}@{cfg['DB_HOST']}:{cfg['DB_PORT']}/{cfg['DB_NAME']}")
    else:
        print("  [info] no .env file; using the defaults (localhost:5433, user postgres). "
              "Copy .env.example to .env to change them.")
    try:
        import psycopg2
        conn = psycopg2.connect(host=cfg["DB_HOST"], port=cfg["DB_PORT"], dbname=cfg["DB_NAME"],
                                user=cfg["DB_USER"], password=cfg["DB_PASSWORD"], connect_timeout=5)
    except Exception as e:  # noqa: BLE001
        fail(f"cannot connect to PostgreSQL ({str(e).strip().splitlines()[0]})",
             "start it with db_start.bat, or run setup.ps1 to install and restore it")
        return
    # (table, minimum rows expected after restoring the release dump, what it holds)
    tables = [
        ("companies", 200, "stocks with sector and industry"),
        ("market_data_daily", 300_000, "daily prices"),
        ("daily_sentiment_aggregates", 40_000, "daily announcement tone per stock"),
        ("predictions", 0, "forecasts made by the daily pipeline"),
        ("prediction_accuracy", 0, "forecasts scored against real prices"),
        ("headline_hit_rates", 0, "live top-20 hit rates"),
    ]
    with conn, conn.cursor() as cur:
        for table, minimum, what in tables:
            try:
                cur.execute(f"SELECT count(*) FROM {table}")
                n = cur.fetchone()[0]
            except Exception as e:  # noqa: BLE001
                conn.rollback()
                fail(f"table {table} is missing ({str(e).strip().splitlines()[0]})",
                     "restore the database: run setup.ps1 (it restores the db-2026-11-01 release)")
                continue
            if n >= minimum:
                ok(f"{table}: {n:,} rows ({what})")
            else:
                fail(f"{table}: only {n:,} rows, expected at least {minimum:,}", "run setup.ps1 to restore the database")
        try:
            cur.execute("SELECT min(date), max(date), count(DISTINCT ticker) FROM market_data_daily")
            first, last, tickers = cur.fetchone()
            ok(f"prices cover {tickers} tickers from {first} to {last}")
        except Exception:  # noqa: BLE001 - reported above
            conn.rollback()
    conn.close()


def check_files():
    print("\n3. Data and model files")
    files = [
        ("training_data/processed_features.parquet", "training features (every model reads this)"),
        ("artifacts/training_dataset.pkl", "TFT dataset definition (predict.py)"),
        ("artifacts/final_model.json", "final 21-day blend settings"),
        ("artifacts/final_lgbm_q50_21d.txt", "final LightGBM median model"),
        ("artifacts/final_rank_regressor_21d.txt", "final rank-regression model"),
        ("artifacts/final_per_stock_median_21d.csv", "final per-stock history member"),
        ("artifacts/agreement_history.csv", "confidence-flag history"),
        ("data/filings_meta.parquet", "NSE announcements"),
        ("data/filings_summary_sentiment.parquet", "FinBERT tone of announcement summaries"),
    ]
    try:
        with open(os.path.join(HERE, "artifacts", "model_registry.json"), encoding="utf-8") as f:
            ckpt = json.load(f)["tft_checkpoint"].replace("\\", "/")
        files.insert(2, (ckpt, "registered TFT checkpoint (artifacts/model_registry.json)"))
    except Exception as e:  # noqa: BLE001
        fail(f"artifacts/model_registry.json unreadable ({e})", "git checkout artifacts/model_registry.json")
    missing = False
    for rel, what in files:
        path = os.path.join(HERE, *rel.split("/"))
        if os.path.exists(path):
            ok(f"{rel} ({os.path.getsize(path) / 1048576:.1f} MB): {what}")
        else:
            missing = True
            fail(f"missing {rel}: {what}")
    if missing:
        print("         fix: run setup.ps1 (it downloads and unpacks the data-2026-11-01 release)")
    features = os.path.join(HERE, "training_data", "processed_features.parquet")
    if os.path.exists(features):
        try:
            import pyarrow.parquet as pq
            meta = pq.ParquetFile(features).metadata
            dates = pq.read_table(features, columns=["date"]).column("date").to_pandas()
            ok(f"features: {meta.num_rows:,} rows x {meta.num_columns} columns, "
               f"{dates.min():%d %b %Y} to {dates.max():%d %b %Y}")
        except Exception as e:  # noqa: BLE001
            fail(f"processed_features.parquet does not open ({e})", "delete it and run setup.ps1 again")


def check_scoring():
    print("\n4. End-to-end: today's picks from the final model")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        result = subprocess.run([sys.executable, os.path.join(HERE, "Cleaner", "final_model.py"), "--score"],
                                cwd=HERE, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=900, env=env)
    except subprocess.TimeoutExpired:
        fail("final_model.py --score took more than 15 minutes")
        return
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines()[:22]:
            print(f"         {line}")
        ok("final model scored the latest trading day")
    else:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        fail(f"final_model.py --score failed (exit code {result.returncode}):\n{tail}")


def main():
    parser = argparse.ArgumentParser(description="Check that stock_predict is set up and ready to run.")
    parser.add_argument("--quick", action="store_true", help="skip the end-to-end scoring run")
    args = parser.parse_args()

    print("Checking the stock_predict setup in", HERE)
    cfg, env_found = read_env()
    check_packages()
    check_database(cfg, env_found)
    check_files()
    if not args.quick:
        check_scoring()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed. See the 'fix' lines above, or the Troubleshooting section of README.md.")
        return 1
    print("All checks passed. Start the dashboard with:  venv\\Scripts\\streamlit run dashboard\\app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
