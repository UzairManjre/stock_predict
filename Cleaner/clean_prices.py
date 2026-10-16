"""
clean_prices.py — Repair known Yahoo Finance data errors in market_data_daily.

Run after daily_fetch.py. Safe to run repeatedly: every repair checks that the error is still
present before applying it.

Repairs:
  1. Bad prints: a single row that jumps >50% and reverts the next day (Yahoo left one row
     unadjusted for a later split, e.g. KOTAKBANK on 2025-08-27). The row is deleted.
  2. Zero-volume stock rows: stale placeholders and whole exchange holidays that Yahoo filled
     with fake rows (e.g. 2025-08-27). The rows are deleted, and so is the NIFTY50 row on any
     date where most stocks are zero-volume (index volume is always 0 on Yahoo, so it can't be
     judged on its own).
  3. Unadjusted corporate actions (demergers, mis-dated bonus adjustments). All prices before
     the action date are multiplied by the jump ratio, so the series has no fake return. This
     assumes no genuine price move on that day.
Finally it lists every remaining one-day move over 20% for manual review.

Usage:
    python Cleaner/clean_prices.py [--dry-run]
"""

import os
import sys
import argparse
import warnings
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Scaper.utils.db_connector import get_db_connection

# (ticker, first date at the new price level, reason)
CORPORATE_ACTIONS = [
    ("CONCOR", "2025-01-01", "1:4 bonus applied on the wrong date by Yahoo"),
    ("TATAMOTORS", "2025-10-14", "demerger of the commercial-vehicle business"),
    ("TRENT", "2026-01-01", "1:2 bonus applied on the wrong date by Yahoo"),
    ("VEDL", "2026-04-30", "demerger"),
]
ACTION_MIN_JUMP = 0.15   # skip an action whose jump is already gone (keeps the script idempotent)
REVIEW_THRESHOLD = 0.20


def load_prices(conn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # pandas prefers SQLAlchemy; psycopg2 works fine
        df = pd.read_sql_query(
            "SELECT date, ticker, close_price, adj_close, volume FROM market_data_daily ORDER BY ticker, date",
            conn, parse_dates=['date'])
    df['ret'] = df.groupby('ticker')['adj_close'].pct_change()
    df['next_ret'] = df.groupby('ticker')['ret'].shift(-1)
    return df


def find_bad_prints(df):
    spike = (df['ret'] > 0.5) & (df['next_ret'] < -0.33)
    reverts = ((1 + df['ret']) * (1 + df['next_ret']) - 1).abs() < 0.15
    return df[spike & reverts]


def main():
    parser = argparse.ArgumentParser(description="Repair known price data errors.")
    parser.add_argument('--dry-run', action='store_true', help="Report what would change without writing.")
    args = parser.parse_args()

    conn = get_db_connection()
    cur = conn.cursor()
    df = load_prices(conn)

    # 1. Bad prints
    bad = find_bad_prints(df)
    print(f"Bad prints: {len(bad)}")
    for r in bad.itertuples():
        print(f"  {r.date.date()} {r.ticker}: +{r.ret:.0%} then {r.next_ret:.0%}")
        if not args.dry_run:
            cur.execute("DELETE FROM market_data_daily WHERE ticker = %s AND date = %s", (r.ticker, r.date.date()))

    # 2. Zero-volume stock rows (stale placeholders), plus the NIFTY50 row on dates where most
    #    stocks are zero-volume (exchange holidays that Yahoo filled with fake rows)
    stocks = df[df['ticker'] != 'NIFTY50']
    zero = stocks[stocks['volume'] == 0]
    zero_share = zero.groupby('date').size() / stocks.groupby('date').size()
    holidays = sorted(zero_share[zero_share > 0.5].index)
    print(f"Zero-volume stock rows: {len(zero)} on {zero['date'].nunique()} dates; "
          f"holiday dates: {[d.date().isoformat() for d in holidays]}")
    if not args.dry_run:
        cur.executemany("DELETE FROM market_data_daily WHERE ticker = %s AND date = %s",
                        [(r.ticker, r.date.date()) for r in zero.itertuples()])
        cur.executemany("DELETE FROM market_data_daily WHERE ticker = 'NIFTY50' AND date = %s",
                        [(d.date(),) for d in holidays])

    if not args.dry_run:
        conn.commit()
        df = load_prices(conn)

    # 3. Unadjusted corporate actions
    print("Corporate actions:")
    for ticker, action_date, reason in CORPORATE_ACTIONS:
        t = df[df['ticker'] == ticker].set_index('date')
        d = pd.Timestamp(action_date)
        if d not in t.index:
            print(f"  {ticker} {action_date}: date not found, skipped")
            continue
        jump = t.loc[d, 'ret']
        if pd.isna(jump) or abs(jump) < ACTION_MIN_JUMP:
            print(f"  {ticker} {action_date}: already adjusted (move {jump:+.1%}), skipped")
            continue
        factor = float(1 + jump)  # psycopg2 cannot adapt numpy floats
        print(f"  {ticker} {action_date}: {jump:+.1%} ({reason}); prices before scaled by {factor:.4f}")
        if not args.dry_run:
            cur.execute("""
                UPDATE market_data_daily
                SET open_price = open_price * %s, high_price = high_price * %s, low_price = low_price * %s,
                    close_price = close_price * %s, adj_close = adj_close * %s
                WHERE ticker = %s AND date < %s
            """, (factor, factor, factor, factor, factor, ticker, action_date))

    if not args.dry_run:
        conn.commit()
        df = load_prices(conn)

    # 4. What is left for a human to judge
    big = df[df['ret'].abs() > REVIEW_THRESHOLD].sort_values('ret')
    print(f"Remaining one-day moves over {REVIEW_THRESHOLD:.0%} (check these are real events): {len(big)}")
    for r in big.itertuples():
        print(f"  {r.date.date()} {r.ticker:11s} {r.ret:+.1%}")

    cur.close()
    conn.close()
    print("Dry run, nothing written." if args.dry_run else "Price cleaning complete.")


if __name__ == '__main__':
    main()
