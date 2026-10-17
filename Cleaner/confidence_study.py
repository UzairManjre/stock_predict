"""
confidence_study.py — How often the final model's top 20 beat the market, on all days and on
high-confidence days, using exactly the rules in final_model.py.

Input: the walk-forward test scores saved by evaluate.py (results/horizon_21d/test_predictions.parquet).
For each test day: the top 20 by the smoothed four-model blend; did they beat the NIFTY 50, the
average stock and the bottom 20 over the next 21 sessions? Days are split by the confidence flag
(member agreement above the median of the previous 126 days). Uncertainty comes from a moving-block
bootstrap with 21-session blocks, since neighbouring days share most of their 21-day window.

Outputs results/confidence_study/. Usage:
    python Cleaner/confidence_study.py
"""

import os
import numpy as np
import pandas as pd

from final_model import (MEMBERS, TOP_N, AGREEMENT_LOOKBACK, AGREEMENT_MIN_HISTORY, blend_scores,
                         daily_agreement, pick_order, WALK_FORWARD_PREDICTIONS, ROOT)

OUT_DIR = os.path.join(ROOT, 'results', 'confidence_study')
HORIZON = 21
MEASURES = ['beat_nifty', 'beat_average', 'beat_bottom']


def daily_outcomes(scored):
    rows = []
    for dt, day in scored.groupby('date'):
        day = day.dropna(subset=['target'])
        if len(day) < 2 * TOP_N:
            continue
        order = pick_order(day)
        top, bottom = order.head(TOP_N)['target'].mean(), order.tail(TOP_N)['target'].mean()
        rows.append({'date': dt, 'window': int(day['window'].iloc[0]), 'top20_excess': top,
                     'beat_nifty': top > 0, 'beat_average': top > day['target'].mean(), 'beat_bottom': top > bottom})
    return pd.DataFrame(rows).set_index('date').sort_index()


def block_bootstrap(flags, block=HORIZON, n=5000, seed=0):
    """90% interval for a hit rate when neighbouring days overlap."""
    values = flags.to_numpy(dtype=float)
    if len(values) < block:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    starts = np.arange(len(values) - block + 1)
    n_blocks = int(np.ceil(len(values) / block))
    means = [np.concatenate([values[s:s + block] for s in rng.choice(starts, n_blocks)])[:len(values)].mean()
             for _ in range(n)]
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    preds = pd.read_parquet(WALK_FORWARD_PREDICTIONS)
    preds = preds[preds['horizon'] == HORIZON].copy()
    preds['date'] = pd.to_datetime(preds['date'])
    scored = blend_scores(preds[['date', 'ticker', 'window', 'target'] + MEMBERS])

    out = daily_outcomes(scored)
    agreement = daily_agreement(scored)
    out = out.join(agreement)
    # Same rule as live: compare with the median of the previous 126 days (at least 21)
    out['threshold'] = out['agreement'].shift(1).rolling(AGREEMENT_LOOKBACK, min_periods=AGREEMENT_MIN_HISTORY).median()
    out = out.dropna(subset=['threshold'])
    out['high_confidence'] = out['agreement'] > out['threshold']
    out.to_csv(os.path.join(OUT_DIR, 'daily_outcomes.csv'))

    rows = []
    for label, sel in [('All days', out.index == out.index), ('High-confidence days', out['high_confidence']),
                       ('Other days', ~out['high_confidence'])]:
        part = out[sel]
        row = {'days': label, 'n_days': len(part), 'share_of_days': len(part) / len(out),
               'independent_months': round(len(part) / HORIZON, 1), 'avg_top20_excess': part['top20_excess'].mean()}
        for m in MEASURES:
            lo, hi = block_bootstrap(part[m])
            row[m], row[f'{m}_lo90'], row[f'{m}_hi90'] = part[m].mean(), lo, hi
        rows.append(row)
    summary = pd.DataFrame(rows)
    by_window = out.groupby(['window', 'high_confidence'])[MEASURES].mean().unstack('high_confidence')
    by_window['days_called'] = out.groupby('window')['high_confidence'].mean()
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)
    by_window.to_csv(os.path.join(OUT_DIR, 'by_window.csv'))

    with pd.option_context('display.width', 220, 'display.float_format', '{:.3f}'.format, 'display.max_columns', 30):
        print(summary.to_string(index=False))
        print()
        print(by_window.to_string())
    print(f"\nResults written to {OUT_DIR}")


if __name__ == '__main__':
    main()
