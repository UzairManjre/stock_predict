"""
accuracy_checker.py — TFT Prediction Accuracy Evaluator

How it works:
  1. Scans the `predictions` table for batches where enough calendar time
     has elapsed (>= 7 days) for 5 TRADING days of post-prediction prices
     to have arrived in `market_data_daily`.
  2. For each ticker in those batches, computes the ACTUAL 5-trading-day
     forward return using the price on the prediction date as the base.
  3. Writes one row per (prediction_batch, ticker) into `prediction_accuracy`
     with full error metrics.
  4. Also computes and stores per-batch aggregate statistics into
     `prediction_accuracy_summary`.

Metrics computed per ticker:
  - actual_5d_return        : realized 5-day excess log return vs the NIFTY 50
                              (the quantity the model forecasts)
  - abs_error               : |q50 - actual|
  - sq_error                : (q50 - actual)^2
  - direction_correct       : 1 if sign(q50) == sign(actual), else 0
  - in_quantile_band        : 1 if q10 <= actual <= q90 (should be ~80%)
  - q10_miss                : actual < q10  (model too optimistic on downside)
  - q90_miss                : actual > q90  (model too optimistic on upside)

Aggregate metrics per batch:
  - mae, rmse, direction_accuracy, quantile_coverage
  - hit_rate_buy            : accuracy of stocks labeled BUY/STRONG BUY
  - avg_return_if_bullish   : avg actual return for q50 > 0.005
  - sharpe_of_signals       : mean(actual returns for top-signal stocks) / std

Usage:
  python accuracy_checker.py [--min-days 7] [--force-rebatch]
"""

import os
import sys
import argparse
import warnings
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

warnings.filterwarnings("ignore")

# ─── Load .env ────────────────────────────────────────────────────────────────
_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
if os.path.exists(_env):
    with open(_env, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip()

DB_CFG = dict(
    host=os.environ.get("DB_HOST", "localhost"),
    port=os.environ.get("DB_PORT", "5433"),
    database=os.environ.get("DB_NAME", "stock_predict"),
    user=os.environ.get("DB_USER", "postgres"),
    password=os.environ.get("DB_PASSWORD", "postgres"),
)


# ─── DB Setup ─────────────────────────────────────────────────────────────────
SETUP_SQL = """
CREATE TABLE IF NOT EXISTS prediction_accuracy (
    id                  BIGSERIAL PRIMARY KEY,
    batch_date          DATE NOT NULL,
    generated_at        TIMESTAMP NOT NULL,
    ticker              TEXT NOT NULL,
    sector              TEXT,

    -- Predicted quantiles
    q10_predicted       FLOAT NOT NULL,
    q50_predicted       FLOAT NOT NULL,
    q90_predicted       FLOAT NOT NULL,
    spread              FLOAT,
    signal_strength     FLOAT,

    -- Realized prices
    price_at_prediction FLOAT,
    price_5d_later      FLOAT,
    actual_5d_return    FLOAT,

    -- Error metrics
    abs_error           FLOAT,
    sq_error            FLOAT,
    direction_correct   SMALLINT,
    in_quantile_band    SMALLINT,
    q10_miss            SMALLINT,
    q90_miss            SMALLINT,
    pct_error           FLOAT,

    evaluated_at        TIMESTAMP DEFAULT NOW(),
    UNIQUE (generated_at, ticker)
);

CREATE TABLE IF NOT EXISTS prediction_accuracy_summary (
    id                      BIGSERIAL PRIMARY KEY,
    batch_date              DATE NOT NULL UNIQUE,
    generated_at            TIMESTAMP NOT NULL,
    tickers_evaluated       INT,
    tickers_missing_data    INT,

    -- Core accuracy
    mae                     FLOAT,
    rmse                    FLOAT,
    median_abs_error        FLOAT,
    direction_accuracy      FLOAT,
    quantile_coverage       FLOAT,
    q10_miss_rate           FLOAT,
    q90_miss_rate           FLOAT,

    -- Signal-based analysis
    hit_rate_buy            FLOAT,
    hit_rate_strong_buy     FLOAT,
    avg_return_bullish      FLOAT,
    avg_return_bearish      FLOAT,
    avg_actual_top10        FLOAT,

    -- Portfolio simulation
    long_only_return        FLOAT,
    top10_long_return       FLOAT,
    nifty_return_same_period FLOAT,

    evaluated_at            TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pred_acc_batch   ON prediction_accuracy(batch_date);
CREATE INDEX IF NOT EXISTS idx_pred_acc_ticker  ON prediction_accuracy(ticker);
"""


def ensure_tables(conn):
    cur = conn.cursor()
    cur.execute(SETUP_SQL)
    conn.commit()
    cur.close()


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_pending_batches(conn, min_days: int = 7, force_rebatch: bool = False) -> pd.DataFrame:
    """
    Returns prediction rows for batches that are:
      - old enough (>= min_days calendar days ago) that 5 trading days have passed
      - not yet evaluated (unless --force-rebatch)
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    if force_rebatch:
        already_done = set()
    else:
        cur.execute("SELECT DISTINCT generated_at FROM prediction_accuracy")
        already_done = {r['generated_at'] for r in cur.fetchall()}

    cutoff = datetime.utcnow() - timedelta(days=min_days)
    cur.execute("""
        SELECT p.generated_at, p.ticker, c.sector,
               p.q10_return AS q10_5d_return, p.q50_return AS q50_5d_return, p.q90_return AS q90_5d_return,
               p.spread, p.signal_strength
        FROM predictions p
        LEFT JOIN companies c ON p.ticker = c.ticker
        WHERE p.horizon = '5d' AND p.generated_at <= %s
        ORDER BY p.generated_at, p.ticker
    """, (cutoff,))

    rows = cur.fetchall()
    cur.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if not force_rebatch and already_done:
        df = df[~df['generated_at'].isin(already_done)]

    return df


def load_price_around(conn, tickers: list, date_from: date, date_to: date) -> pd.DataFrame:
    """Load adj_close for tickers between two dates."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT ticker, date, adj_close
        FROM market_data_daily
        WHERE ticker = ANY(%s)
          AND date BETWEEN %s AND %s
        ORDER BY ticker, date
    """, (tickers, date_from, date_to))
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows)


# ─── Accuracy computation ─────────────────────────────────────────────────────
def get_nth_trading_day_price(price_pivot: pd.DataFrame, ticker: str, ref_date: date, n: int = 5):
    """
    Get price on the Nth trading day AFTER ref_date.
    price_pivot: MultiIndex DataFrame with (ticker, date) structure —
                 or a flat one filtered for the ticker.
    """
    if ticker not in price_pivot.columns:
        return None, None

    ticker_prices = price_pivot[ticker].dropna()
    # Filter to dates strictly after ref_date
    forward = ticker_prices[ticker_prices.index > pd.Timestamp(ref_date)]
    if len(forward) == 0:
        return None, None

    price_on_pred_date_series = ticker_prices[ticker_prices.index <= pd.Timestamp(ref_date)]
    if price_on_pred_date_series.empty:
        return None, None

    base_price = float(price_on_pred_date_series.iloc[-1])

    # Score only complete horizons; a batch without n sessions yet is retried on a later run
    if len(forward) < n:
        return None, None
    price_n = float(forward.iloc[n - 1])

    return base_price, price_n


def evaluate_batch(batch_df: pd.DataFrame, conn) -> tuple:
    """
    Returns (row_records, summary_record) for a single prediction batch.
    """
    generated_at = batch_df['generated_at'].iloc[0]
    batch_date = generated_at.date() if hasattr(generated_at, 'date') else date.fromisoformat(str(generated_at)[:10])

    tickers = batch_df['ticker'].tolist()
    date_from = batch_date - timedelta(days=3)         # include a few days before for base price
    date_to   = batch_date + timedelta(days=15)        # 15 calendar ≈ 10+ trading days

    # Load price data (with the NIFTY 50, the benchmark the forecasts are measured against)
    price_df = load_price_around(conn, tickers + ['NIFTY50'], date_from, date_to)

    if price_df.empty:
        print(f"  [WARN] No price data found for batch {batch_date}")
        return [], None

    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df['adj_close'] = price_df['adj_close'].astype(float)

    # Pivot: index=date, columns=ticker
    price_pivot = price_df.pivot(index='date', columns='ticker', values='adj_close').sort_index()

    nifty_base, nifty_5d = get_nth_trading_day_price(price_pivot, 'NIFTY50', batch_date, n=5)
    if nifty_base is None or nifty_5d is None:
        print(f"  [INFO] Batch {batch_date}: 5 sessions of NIFTY 50 prices not available yet; will retry later.")
        return [], None
    nifty_ret = float(np.log(nifty_5d / nifty_base))

    row_records = []
    missing_count = 0

    for _, row in batch_df.iterrows():
        ticker = row['ticker']
        q10 = float(row['q10_5d_return'])
        q50 = float(row['q50_5d_return'])
        q90 = float(row['q90_5d_return'])
        spread = float(row['spread']) if row['spread'] is not None else None
        sig_str = float(row['signal_strength']) if row['signal_strength'] is not None else None

        base_price, price_5d = get_nth_trading_day_price(price_pivot, ticker, batch_date, n=5)

        if base_price is None or price_5d is None:
            missing_count += 1
            continue

        # The forecasts are 5-day excess log returns over the NIFTY 50, so score against the same
        actual_ret = float(np.log(price_5d / base_price)) - nifty_ret

        abs_err = abs(q50 - actual_ret)
        sq_err  = (q50 - actual_ret) ** 2
        dir_ok  = 1 if (q50 >= 0) == (actual_ret >= 0) else 0
        in_band = 1 if q10 <= actual_ret <= q90 else 0
        q10_miss = 1 if actual_ret < q10 else 0
        q90_miss = 1 if actual_ret > q90 else 0
        pct_err  = abs_err / (abs(actual_ret) + 1e-8)

        row_records.append({
            'batch_date':          batch_date,
            'generated_at':        generated_at,
            'ticker':              ticker,
            'sector':              row.get('sector'),
            'q10_predicted':       q10,
            'q50_predicted':       q50,
            'q90_predicted':       q90,
            'spread':              spread,
            'signal_strength':     sig_str,
            'price_at_prediction': base_price,
            'price_5d_later':      price_5d,
            'actual_5d_return':    actual_ret,
            'abs_error':           abs_err,
            'sq_error':            sq_err,
            'direction_correct':   dir_ok,
            'in_quantile_band':    in_band,
            'q10_miss':            q10_miss,
            'q90_miss':            q90_miss,
            'pct_error':           pct_err,
        })

    if not row_records:
        print(f"  [WARN] No tickers had resolvable prices for batch {batch_date}")
        return [], None

    acc_df = pd.DataFrame(row_records)

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    mae    = acc_df['abs_error'].mean()
    rmse   = np.sqrt(acc_df['sq_error'].mean())
    median_ae = acc_df['abs_error'].median()
    dir_acc = acc_df['direction_correct'].mean()
    q_cov   = acc_df['in_quantile_band'].mean()
    q10_miss_rate = acc_df['q10_miss'].mean()
    q90_miss_rate = acc_df['q90_miss'].mean()

    # Signal-based
    buy_mask       = acc_df['q50_predicted'] > 0.005
    strong_buy_mask = (acc_df['q50_predicted'] > 0.02) & (acc_df['signal_strength'].fillna(0) > 1.5)
    bearish_mask   = acc_df['q50_predicted'] < -0.005

    hit_rate_buy = None
    if buy_mask.sum() > 0:
        hit_rate_buy = acc_df.loc[buy_mask, 'direction_correct'].mean()

    hit_rate_strong_buy = None
    if strong_buy_mask.sum() > 0:
        hit_rate_strong_buy = acc_df.loc[strong_buy_mask, 'direction_correct'].mean()

    avg_return_bullish = float(acc_df.loc[buy_mask, 'actual_5d_return'].mean()) if buy_mask.sum() > 0 else None
    avg_return_bearish = float(acc_df.loc[bearish_mask, 'actual_5d_return'].mean()) if bearish_mask.sum() > 0 else None

    # Top-10 picks actual return
    top10 = acc_df.nlargest(10, 'q50_predicted')
    avg_actual_top10 = float(top10['actual_5d_return'].mean())

    # Long-only portfolio: buy all positive q50 signals
    long_only_return = float(acc_df.loc[buy_mask, 'actual_5d_return'].mean()) if buy_mask.sum() > 0 else None
    top10_long_return = float(top10['actual_5d_return'].mean())

    summary = {
        'batch_date':              batch_date,
        'generated_at':            generated_at,
        'tickers_evaluated':       len(acc_df),
        'tickers_missing_data':    missing_count,
        'mae':                     mae,
        'rmse':                    rmse,
        'median_abs_error':        median_ae,
        'direction_accuracy':      dir_acc,
        'quantile_coverage':       q_cov,
        'q10_miss_rate':           q10_miss_rate,
        'q90_miss_rate':           q90_miss_rate,
        'hit_rate_buy':            hit_rate_buy,
        'hit_rate_strong_buy':     hit_rate_strong_buy,
        'avg_return_bullish':      avg_return_bullish,
        'avg_return_bearish':      avg_return_bearish,
        'avg_actual_top10':        avg_actual_top10,
        'long_only_return':        long_only_return,
        'top10_long_return':       top10_long_return,
        'nifty_return_same_period': nifty_ret,
    }

    return row_records, summary


# ─── DB Write ─────────────────────────────────────────────────────────────────
def write_accuracy_rows(conn, rows: list):
    if not rows:
        return
    cur = conn.cursor()
    psycopg2.extras.execute_values(cur, """
        INSERT INTO prediction_accuracy (
            batch_date, generated_at, ticker, sector,
            q10_predicted, q50_predicted, q90_predicted, spread, signal_strength,
            price_at_prediction, price_5d_later, actual_5d_return,
            abs_error, sq_error, direction_correct, in_quantile_band,
            q10_miss, q90_miss, pct_error
        ) VALUES %s
        ON CONFLICT (generated_at, ticker) DO UPDATE
            SET actual_5d_return  = EXCLUDED.actual_5d_return,
                price_5d_later    = EXCLUDED.price_5d_later,
                abs_error         = EXCLUDED.abs_error,
                sq_error          = EXCLUDED.sq_error,
                direction_correct = EXCLUDED.direction_correct,
                in_quantile_band  = EXCLUDED.in_quantile_band,
                q10_miss          = EXCLUDED.q10_miss,
                q90_miss          = EXCLUDED.q90_miss,
                pct_error         = EXCLUDED.pct_error,
                evaluated_at      = NOW()
    """, [(
        r['batch_date'], r['generated_at'], r['ticker'], r['sector'],
        r['q10_predicted'], r['q50_predicted'], r['q90_predicted'], r['spread'], r['signal_strength'],
        r['price_at_prediction'], r['price_5d_later'], r['actual_5d_return'],
        r['abs_error'], r['sq_error'], r['direction_correct'], r['in_quantile_band'],
        r['q10_miss'], r['q90_miss'], r['pct_error'],
    ) for r in rows])
    conn.commit()
    cur.close()


def write_summary(conn, summary: dict):
    # psycopg2 cannot adapt numpy scalars; convert them to plain Python values
    summary = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in summary.items()}
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO prediction_accuracy_summary (
            batch_date, generated_at, tickers_evaluated, tickers_missing_data,
            mae, rmse, median_abs_error, direction_accuracy, quantile_coverage,
            q10_miss_rate, q90_miss_rate,
            hit_rate_buy, hit_rate_strong_buy,
            avg_return_bullish, avg_return_bearish, avg_actual_top10,
            long_only_return, top10_long_return, nifty_return_same_period
        ) VALUES (
            %(batch_date)s, %(generated_at)s, %(tickers_evaluated)s, %(tickers_missing_data)s,
            %(mae)s, %(rmse)s, %(median_abs_error)s, %(direction_accuracy)s, %(quantile_coverage)s,
            %(q10_miss_rate)s, %(q90_miss_rate)s,
            %(hit_rate_buy)s, %(hit_rate_strong_buy)s,
            %(avg_return_bullish)s, %(avg_return_bearish)s, %(avg_actual_top10)s,
            %(long_only_return)s, %(top10_long_return)s, %(nifty_return_same_period)s
        )
        ON CONFLICT (batch_date) DO UPDATE
            SET mae                       = EXCLUDED.mae,
                rmse                      = EXCLUDED.rmse,
                direction_accuracy        = EXCLUDED.direction_accuracy,
                quantile_coverage         = EXCLUDED.quantile_coverage,
                hit_rate_buy              = EXCLUDED.hit_rate_buy,
                hit_rate_strong_buy       = EXCLUDED.hit_rate_strong_buy,
                avg_return_bullish        = EXCLUDED.avg_return_bullish,
                top10_long_return         = EXCLUDED.top10_long_return,
                nifty_return_same_period  = EXCLUDED.nifty_return_same_period,
                evaluated_at              = NOW()
    """, summary)
    conn.commit()
    cur.close()


# ─── Backtest mode — uses train_tft.py output artifacts ───────────────────────
def run_backtest_from_csv(conn, csv_path: str):
    """
    If you have historical prediction CSVs from train_tft.py backtest,
    import them here to seed the accuracy table with historical data.
    The CSV must have columns: ticker, generated_at / date, q10_5d_return,
    q50_5d_return, q90_5d_return, spread, signal_strength.
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # Normalize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # predict.py's own CSVs hold every horizon with generic column names; keep the 5-day rows
    if 'horizon' in df.columns:
        df = df[df['horizon'] == '5d'].rename(columns={
            'q10_return': 'q10_5d_return', 'q50_return': 'q50_5d_return', 'q90_return': 'q90_5d_return'})

    required = {'ticker', 'q50_5d_return'}
    if not required.issubset(df.columns):
        print(f"[ERROR] CSV missing required columns. Found: {df.columns.tolist()}")
        return

    # Try to parse generated_at / date
    for col in ['generated_at', 'date', 'prediction_date']:
        if col in df.columns:
            df['generated_at'] = pd.to_datetime(df[col])
            break
    else:
        print("[ERROR] No date column found in CSV.")
        return

    if 'q10_5d_return' not in df.columns:
        df['q10_5d_return'] = df['q50_5d_return'] - 0.01
    if 'q90_5d_return' not in df.columns:
        df['q90_5d_return'] = df['q50_5d_return'] + 0.01
    if 'spread' not in df.columns:
        df['spread'] = df['q90_5d_return'] - df['q10_5d_return']
    if 'signal_strength' not in df.columns:
        df['signal_strength'] = df['q50_5d_return'] / (df['spread'] + 1e-8)
    if 'sector' not in df.columns:
        df['sector'] = None

    print(f"Loaded {len(df)} rows from CSV: {csv_path}")

    for gen_at, batch in df.groupby('generated_at'):
        print(f"Processing batch {gen_at} ({len(batch)} tickers)...")
        rows, summary = evaluate_batch(batch.rename(columns={
            'q10_5d_return': 'q10_5d_return',
            'q50_5d_return': 'q50_5d_return',
            'q90_5d_return': 'q90_5d_return',
        }), conn)
        if rows:
            write_accuracy_rows(conn, rows)
            if summary:
                write_summary(conn, summary)
            print(f"  Wrote {len(rows)} rows | MAE={summary['mae']:.5f} | Dir={summary['direction_accuracy']:.1%}")


# ─── Print Report ─────────────────────────────────────────────────────────────
def print_accuracy_report(conn):
    """Print a human-readable accuracy report across all evaluated batches."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT batch_date, tickers_evaluated, mae, rmse, direction_accuracy,
               quantile_coverage, hit_rate_buy, hit_rate_strong_buy,
               top10_long_return, nifty_return_same_period
        FROM prediction_accuracy_summary
        ORDER BY batch_date DESC
    """)
    rows = cur.fetchall()

    # Overall aggregate
    cur.execute("""
        SELECT
            COUNT(*)                    AS total_evaluated,
            AVG(abs_error)              AS overall_mae,
            SQRT(AVG(sq_error))         AS overall_rmse,
            AVG(direction_correct::float) AS overall_dir_acc,
            AVG(in_quantile_band::float)  AS overall_q_coverage,
            SUM(CASE WHEN q50_predicted > 0.005 AND direction_correct = 1 THEN 1 ELSE 0 END)::float /
                NULLIF(SUM(CASE WHEN q50_predicted > 0.005 THEN 1 ELSE 0 END), 0) AS buy_hit_rate
        FROM prediction_accuracy
    """)
    agg = cur.fetchone()
    cur.close()

    sep = "=" * 70
    print(f"\n{sep}")
    print("  TFT PREDICTION ACCURACY REPORT")
    print(sep)
    if agg and agg['total_evaluated']:
        print(f"  Total Evaluated Predictions  : {int(agg['total_evaluated'])}")
        print(f"  Overall MAE (return units)   : {agg['overall_mae']:.5f}  ({agg['overall_mae']*100:.3f}%)")
        print(f"  Overall RMSE                 : {agg['overall_rmse']:.5f}")
        print(f"  Direction Accuracy           : {agg['overall_dir_acc']:.1%}  (random baseline = 50%)")
        print(f"  Quantile Band Coverage       : {agg['overall_q_coverage']:.1%}  (Q10–Q90, ideal ≈ 80%)")
        print(f"  Buy Signal Hit Rate          : {agg['buy_hit_rate']:.1%}" if agg['buy_hit_rate'] else "  Buy Signal Hit Rate          : N/A")
    else:
        print("  No evaluated predictions yet.")

    print(f"\n{'-' * 70}")
    print(f"  {'Batch Date':<12} {'N':>5} {'MAE':>8} {'RMSE':>8} {'Dir%':>7} {'Q-Cov':>7} {'Top10':>8} {'NIFTY':>8}")
    print(f"  {'-' * 12} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")
    for r in rows:
        nifty = f"{r['nifty_return_same_period']*100:+.2f}%" if r['nifty_return_same_period'] is not None else "N/A"
        top10 = f"{r['top10_long_return']*100:+.2f}%" if r['top10_long_return'] is not None else "N/A"
        print(f"  {str(r['batch_date']):<12} {r['tickers_evaluated']:>5} "
              f"{r['mae']*100:>7.3f}% {r['rmse']*100:>7.3f}% "
              f"{r['direction_accuracy']:>6.1%} {r['quantile_coverage']:>6.1%} "
              f"{top10:>8} {nifty:>8}")
    print(sep)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TFT Prediction Accuracy Checker")
    parser.add_argument('--min-days', type=int, default=7,
                        help='Minimum calendar days after prediction before evaluating (default: 7)')
    parser.add_argument('--force-rebatch', action='store_true',
                        help='Re-evaluate already-computed batches (overwrites existing rows)')
    parser.add_argument('--backtest-csv', type=str, default=None,
                        help='Path to a historical prediction CSV to import and evaluate')
    parser.add_argument('--report-only', action='store_true',
                        help='Just print the accuracy report, do not run evaluation')
    args = parser.parse_args()

    print(f"Connecting to {DB_CFG['database']}@{DB_CFG['host']}:{DB_CFG['port']}...")
    conn = psycopg2.connect(**DB_CFG)
    ensure_tables(conn)

    if args.report_only:
        print_accuracy_report(conn)
        conn.close()
        return

    if args.backtest_csv:
        run_backtest_from_csv(conn, args.backtest_csv)
        print_accuracy_report(conn)
        conn.close()
        return

    # ── Live evaluation: check pending batches ────────────────────────────────
    print(f"Scanning for prediction batches older than {args.min_days} days...")
    pending_df = load_pending_batches(conn, min_days=args.min_days, force_rebatch=args.force_rebatch)

    if pending_df.empty:
        print("No pending batches to evaluate.")
        print("(Predictions need at least 7 calendar days to pass before 5 trading days of prices arrive.)")
        print_accuracy_report(conn)
        conn.close()
        return

    batches = pending_df.groupby('generated_at')
    print(f"Found {len(batches)} batch(es) to evaluate ({len(pending_df)} prediction rows total).")

    total_rows = 0
    for gen_at, batch_df in batches:
        batch_date = gen_at.date() if hasattr(gen_at, 'date') else gen_at
        print(f"\nEvaluating batch: {batch_date} ({len(batch_df)} tickers)...")

        rows, summary = evaluate_batch(batch_df, conn)

        if rows:
            write_accuracy_rows(conn, rows)
            if summary:
                write_summary(conn, summary)
            print(f"  Stored {len(rows)} accuracy rows")
            print(f"  MAE          : {summary['mae']*100:.4f}%")
            print(f"  Direction Acc: {summary['direction_accuracy']:.1%}")
            print(f"  Quantile Cov : {summary['quantile_coverage']:.1%}")
            if summary.get('hit_rate_buy') is not None:
                print(f"  Buy Hit Rate : {summary['hit_rate_buy']:.1%}")
            if summary.get('top10_long_return') is not None:
                print(f"  Top-10 Return: {summary['top10_long_return']*100:+.3f}%")
            total_rows += len(rows)
        else:
            print("  No usable data found for this batch.")

    print(f"\nEvaluation complete. Processed {total_rows} ticker-predictions.")
    print_accuracy_report(conn)
    conn.close()


if __name__ == '__main__':
    main()
