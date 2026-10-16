"""
hit_rate_history.py — How often momentum's top 20 beat the average stock and the NIFTY 50.

12-1 month momentum is a fixed rule with nothing fitted, so it can be scored over the whole
history (from Sept 2021, once 12 months of prices exist), not just the walk-forward test windows.
For each holding period it reports the hit rate over every start day and over non-overlapping
periods, plus the 21-day hit rate year by year. Survivorship bias applies: the universe is
today's stocks, which flatters momentum.

Outputs results/hit_rate_history/. Usage:
    python Cleaner/hit_rate_history.py [--start 2021-09-01]
"""

import os
import argparse
import pandas as pd

from evaluate import top_n_hits, non_overlapping

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEATURES = os.path.join(ROOT, 'training_data', 'processed_features.parquet')
OUT_DIR = os.path.join(ROOT, 'results', 'hit_rate_history')
HORIZONS = [5, 10, 21, 126]


def main():
    parser = argparse.ArgumentParser(description="Momentum top-20 hit rates over the full history.")
    parser.add_argument('--start', default='2021-09-01')
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(FEATURES, columns=['date', 'ticker', 'momentum_12_1'] + [f'target_{h}d' for h in HORIZONS])
    df = df[df['date'] >= args.start].sort_values(['date', 'ticker'])

    rows, yearly = [], None
    for h in HORIZONS:
        valid = df.dropna(subset=['momentum_12_1', f'target_{h}d'])
        hits = top_n_hits(valid['date'], valid['momentum_12_1'], valid[f'target_{h}d']).sort_index()
        nov = non_overlapping(hits, h)
        rows.append({'horizon_days': h, 'start_days': len(hits), 'hit_vs_avg': hits['hit_vs_avg'].mean(),
                     'hit_vs_nifty': hits['hit_vs_nifty'].mean(), 'periods': len(nov),
                     'hit_vs_avg_periods': nov['hit_vs_avg'].mean(), 'hit_vs_nifty_periods': nov['hit_vs_nifty'].mean()})
        if h == 21:
            by_year = hits.groupby(pd.DatetimeIndex(hits.index).year)
            yearly = pd.DataFrame({'start_days': by_year.size(), 'hit_vs_avg': by_year['hit_vs_avg'].mean(),
                                   'hit_vs_nifty': by_year['hit_vs_nifty'].mean()})
            yearly.index.name = 'year'
            hits.to_csv(os.path.join(OUT_DIR, 'daily_hits_21d.csv'))

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, 'hit_rate_by_horizon.csv'), index=False)
    yearly.to_csv(os.path.join(OUT_DIR, 'hit_rate_21d_by_year.csv'))
    with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 200):
        print(summary.to_string(index=False))
        print()
        print(yearly.to_string())
    print(f"\nResults written to {OUT_DIR}")


if __name__ == '__main__':
    main()
