"""
ensemble_study.py — Out-of-sample ensembles of the walk-forward models, and turnover control.

Reads the test-window scores saved by `evaluate.py --save-predictions` and compares:
  - single models;
  - equal-weight blends of daily percentile ranks (no fitting);
  - IC-weighted and ridge-stacked blends, fitted for each window on earlier test windows only
    (dropping the last 5 sessions, whose targets overlap the window being scored);
  - top-20 portfolios with a holding buffer (keep a name while it stays in the top K), and a
    momentum-filtered portfolio (top 40 by momentum, best 20 of those by the blend).

Every model and blend also gets the headline hit rates: how often its top 20 beat the average
stock and the NIFTY 50 over the horizon.

Usage:
    python Cleaner/evaluate.py --horizons 5 --save-predictions --out results/ensemble_study
    python Cleaner/ensemble_study.py [--dir results/ensemble_study] [--horizon 5]
    (21-day: evaluate.py --horizons 21 --save-predictions --out results/horizon_21d, then --horizon 21)
    --extra adds score columns from other scripts (e.g. tune_ranker.py's "Tuned LightGBM" and "CatBoost")
    as single models and as members of extra equal-weight blends.
"""

import os
import argparse
import numpy as np
import pandas as pd

from evaluate import (daily_rank_ic, decile_spread, non_overlapping, top_n_hits, backtest, summarize_backtest,
                      ROUND_TRIP_COST)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE = ['Momentum 12-1', 'Rank regression', 'LightGBM', 'Per-stock history', 'Factor composite',
        'Reversal 1-month', 'Rank regression, factors only', 'LambdaRank']
POOL = ['Momentum 12-1', 'Rank regression', 'LightGBM', 'Per-stock history']
EQUAL_BLENDS = {
    'Momentum + rank regression': ['Momentum 12-1', 'Rank regression'],
    'Momentum + rank reg + per-stock': ['Momentum 12-1', 'Rank regression', 'Per-stock history'],
    'Equal weight, 4 models': POOL,
}
TOP_N = 20
SMOOTH_DAYS = 5
SMOOTHED = ['Equal weight, 4 models', 'Momentum + rank regression']


def blend(frame, weights):
    w = pd.Series(weights, dtype=float)
    w = w / w.abs().sum()
    return sum(frame['p:' + k] * v for k, v in w.items())


def fitted_weights(prior):
    """IC weights (negative ICs floored at zero) and non-negative ridge weights from earlier windows."""
    if len(prior) == 0:
        equal = {m: 1.0 for m in POOL}
        return equal, equal
    ic_w = {m: max(daily_rank_ic(prior['date'], prior[m], prior['target']).mean(), 0.0) for m in POOL}
    if sum(ic_w.values()) == 0:
        ic_w = {m: 1.0 for m in POOL}
    # Ridge on centred percentile ranks; heavy penalty because the signal is tiny
    X = prior[['p:' + m for m in POOL]].values - 0.5
    y = prior['p:target'].values - 0.5
    coef = np.clip(np.linalg.solve(X.T @ X + 0.1 * len(X) * np.eye(X.shape[1]), X.T @ y), 0, None)
    ridge_w = dict(zip(POOL, coef)) if coef.sum() > 0 else {m: 1.0 for m in POOL}
    return ic_w, ridge_w


def buffered_backtest(d, score, keep, horizon, prefilter=None):
    """Top-20 rebalanced every `horizon` sessions, but a held name stays while it ranks within the top `keep`."""
    ret = f'fwd_raw_{horizon}d'
    held, out = set(), []
    for dt in sorted(d['date'].unique())[::horizon]:
        day = d[d['date'] == dt].dropna(subset=[ret])
        if prefilter:
            day = day[day[prefilter].rank(ascending=False) <= 2 * TOP_N]
        order = list(day.dropna(subset=[score]).sort_values(score, ascending=False)['ticker'])
        stay = [t for t in order[:keep] if t in held]
        picks = stay + [t for t in order if t not in stay][:TOP_N - len(stay)]
        turnover = len(set(picks) - held) / TOP_N if held else 1.0
        held = set(picks)
        r = np.expm1(day.set_index('ticker').loc[picks, ret]).mean()
        out.append({'date': dt, 'x': r - turnover * ROUND_TRIP_COST, 'x turnover': turnover})
    return summarize_backtest(pd.DataFrame(out), periods_per_year=252 / horizon).iloc[0]


def extra_blends(extras):
    """Equal-weight blends that add each extra model, or swap it in for the untuned rank regression."""
    blends = dict(EQUAL_BLENDS)
    for m in extras:
        blends[f'Momentum + {m}'] = ['Momentum 12-1', m]
        blends[f'4 models + {m}'] = POOL + [m]
        blends[f'4 models, {m} for rank regression'] = [x if x != 'Rank regression' else m for x in POOL]
    if len(extras) > 1:
        blends['4 models + ' + ' + '.join(extras)] = POOL + list(extras)
    return blends


def main():
    parser = argparse.ArgumentParser(description="Out-of-sample ensembles and turnover control on saved test scores.")
    parser.add_argument('--dir', default=os.path.join(ROOT, 'results', 'ensemble_study'))
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--extra', nargs='*', default=[], help='Extra score columns to evaluate and blend')
    args = parser.parse_args()
    h = args.horizon
    base = BASE + list(args.extra)
    blends = extra_blends(args.extra)

    d = pd.read_parquet(os.path.join(args.dir, 'test_predictions.parquet'))
    d = d[d['horizon'] == h].copy()
    d['date'] = pd.to_datetime(d['date'])

    # Percentile rank of every score within its day; a missing score sits in the middle
    pct = d.groupby('date')[base].rank(pct=True).fillna(0.5)
    pct.columns = ['p:' + c for c in base]
    d = pd.concat([d, pct], axis=1)
    d['p:target'] = d.groupby('date')['target'].rank(pct=True)

    for name, members in blends.items():
        d[name] = blend(d, {m: 1 for m in members})
    # Gate 1: average each stock's blended score over its last SMOOTH_DAYS sessions to cut turnover
    d = d.sort_values(['ticker', 'date'])
    smoothed = []
    for name in SMOOTHED:
        d[f'{name}, smoothed'] = d.groupby('ticker')[name].transform(lambda x: x.rolling(SMOOTH_DAYS, min_periods=1).mean())
        smoothed.append(f'{name}, smoothed')

    d['IC-weighted (expanding)'] = np.nan
    d['Ridge stack (expanding)'] = np.nan
    weights_log = []
    windows = sorted(d['window'].unique())
    for k in windows:
        test = d['window'] == k
        prior = d[d['window'] < k]
        if len(prior):
            prior = prior[~prior['date'].isin(sorted(prior['date'].unique())[-h:])]
        ic_w, ridge_w = fitted_weights(prior)
        d.loc[test, 'IC-weighted (expanding)'] = blend(d[test], ic_w)
        d.loc[test, 'Ridge stack (expanding)'] = blend(d[test], ridge_w)
        weights_log.append({'window': int(k), **{f'ic_w:{m}': ic_w[m] / sum(ic_w.values()) for m in POOL},
                            **{f'ridge_w:{m}': ridge_w[m] / sum(ridge_w.values()) for m in POOL}})

    models = base + list(blends) + smoothed + ['IC-weighted (expanding)', 'Ridge stack (expanding)']
    rows, ic_series, per_window = [], {}, []
    for m in models:
        ic = daily_rank_ic(d['date'], d[m], d['target'])
        ic_series[m] = ic
        nov = non_overlapping(ic, h)
        spread = decile_spread(d['date'], d[m], d['target'])
        hits = top_n_hits(d['date'], d[m], d['target'])
        rows.append({'model': m, 'mean_ic': ic.mean(), 'ic_tstat': nov.mean() / nov.std() * np.sqrt(len(nov)),
                     'ic_ir_daily': ic.mean() / ic.std(), 'ic_positive_days': (ic > 0).mean(),
                     'decile_spread': non_overlapping(spread, h).mean(),
                     'hit_vs_avg': hits['hit_vs_avg'].mean(), 'hit_vs_nifty': hits['hit_vs_nifty'].mean(),
                     'hit_periods': len(non_overlapping(hits, h))})
        for k in windows:
            sel = d['window'] == k
            per_window.append({'model': m, 'window': int(k),
                               'mean_ic': daily_rank_ic(d.loc[sel, 'date'], d.loc[sel, m], d.loc[sel, 'target']).mean()})

    ret, nifty = f'fwd_raw_{h}d', f'fwd_nifty_{h}d'
    scores = d[['date', 'ticker', ret, nifty]].copy()
    for m in models:
        scores['score:' + m] = d[m].values
    bt = backtest(scores.sort_values('date'), step=h, ret=ret, nifty=nifty)
    bts = summarize_backtest(bt, periods_per_year=252 / h)
    summary = pd.DataFrame(rows).merge(bts.rename(columns={'strategy': 'model'}), on='model', how='left')

    buffer_rows = []
    for score in ['Momentum 12-1', 'Rank regression'] + list(args.extra) + list(blends) + smoothed:
        for keep in [20, 40, 60]:
            s = buffered_backtest(d, score, keep, h)
            buffer_rows.append({'portfolio': score, 'keep_while_in_top': keep, **s.drop('strategy').to_dict()})
    for keep in [20, 40]:
        s = buffered_backtest(d, 'Equal weight, 4 models', keep, h, prefilter='Momentum 12-1')
        buffer_rows.append({'portfolio': 'Momentum top 40, ensemble picks 20', 'keep_while_in_top': keep,
                            **s.drop('strategy').to_dict()})
    buffers = pd.DataFrame(buffer_rows)

    summary.to_csv(os.path.join(args.dir, 'ensemble_summary.csv'), index=False)
    pd.DataFrame(ic_series)[models].corr().to_csv(os.path.join(args.dir, 'ic_correlation.csv'))
    pd.DataFrame(per_window).pivot(index='model', columns='window', values='mean_ic').loc[models] \
        .to_csv(os.path.join(args.dir, 'ensemble_ic_by_window.csv'))
    pd.DataFrame(weights_log).to_csv(os.path.join(args.dir, 'ensemble_weights.csv'), index=False)
    bts.to_csv(os.path.join(args.dir, 'ensemble_backtest.csv'), index=False)
    buffers.to_csv(os.path.join(args.dir, 'buffer_study.csv'), index=False)

    with pd.option_context('display.width', 250, 'display.float_format', '{:.4f}'.format, 'display.max_columns', 20):
        print(summary.to_string(index=False))
        print()
        print(buffers.to_string(index=False))
    print(f"\nResults written to {args.dir}")


if __name__ == '__main__':
    main()
