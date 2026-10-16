"""
evaluate.py — Walk-forward evaluation of the forecasting models against simple baselines.

For each of N expanding windows (default 6 windows of 63 trading days, i.e. ~3 months each,
covering the most recent ~18 months):

    | train (everything earlier) | emb | early-stop slice (42 sessions) | emb | test window |

  - emb = an embargo of `horizon` sessions, so no target window straddles two blocks
  - LightGBM quantile models (q10/q50/q90, same features and settings as train_lgbm.py) are
    refit on the train block and early-stopped on the slice; the test window is never seen
  - Baselines: predict zero; each stock's historical quantiles from the train block;
    12-1 month momentum and 1-month reversal (ranking scores only, for IC and backtests)

Metrics per window, horizon and model: MAE of the median, pinball loss, q10-q90 coverage,
direction accuracy, daily cross-sectional rank IC, and the top-minus-bottom decile spread.
IC t-stats use one session every `horizon` sessions so overlapping targets don't inflate them.

Backtest (5-day horizon): every 5th session buy the top 20 stocks by score, equal weight,
hold 5 sessions, pay 0.18% round trip on the names replaced. Compared with the NIFTY 50 and an
equal-weight portfolio of all stocks.

Outputs go to results/ (CSV + PNG). Usage:
    python Cleaner/evaluate.py [--horizons 5 10 21] [--windows 6] [--window-size 63]
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)

from ranking import (has_factor_features, add_cross_sectional_ranks, fit_rank_regressor,
                     fit_lambdarank, predict_scores, factor_composite, FACTOR_FEATURES)

ROOT =os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_PARQUET = os.path.join(ROOT, 'training_data', 'processed_features.parquet')
DEFAULT_OUT = os.path.join(ROOT, 'results')

CATEGORICAL_COLS = ["ticker", "sector", "industry", "day_of_week", "month", "is_earnings_week",
                    "has_sentiment_history", "quarter_end"]
REAL_COLS = [
    'adj_close', 'volume', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'volume_spike', 'nifty_50_return',
    'decayed_net_sentiment_score', 'decayed_bullish_intensity', 'decayed_bearish_intensity',
    'log_return_1d', 'log_return_3d', 'log_return_5d',
    'realized_vol_20d', 'atr_14', 'sector_relative_return_1d', 'sentiment_volume_7d',
    'stoch_rsi_14', 'bb_width', 'price_vs_sma50', 'price_vs_52w_high', 'momentum_10d', 'obv_normalized',
    'volatility_regime', 'pvt_normalized', 'rsi_divergence', 'nifty_volatility_20d', 'sector_volatility_20d'
]
QUANTILES = (0.1, 0.5, 0.9)
EARLY_STOP_SESSIONS = 42      # ~2 months held out of each train block for early stopping
TOP_N = 20                    # stocks held by the backtest portfolio
ROUND_TRIP_COST = 0.0018      # brokerage + STT + slippage, as in backtest_engine.py
LGBM_PARAMS = {
    'objective': 'quantile', 'metric': 'quantile', 'boosting_type': 'gbdt',
    'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': -1,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'verbose': -1, 'num_threads': os.cpu_count() or 4,
}


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_data(path, horizons):
    df = pd.read_parquet(path)
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).astype('category')   # one category set for every window
    for col in REAL_COLS:
        df[col] = df[col].astype('float32')
    g = df.groupby('ticker', observed=True)
    # Ranking-only baselines (known at t): 12-1 month momentum and 1-month reversal
    df['score_momentum_12_1'] = np.log(g['adj_close'].shift(21) / g['adj_close'].shift(252))
    df['score_reversal_1m'] = -df['log_return_21d']
    # Realized forward 5-day raw return, for the backtest
    df['fwd_raw_5d'] = g['log_return_5d'].shift(-5)
    df['fwd_nifty_5d'] = df.groupby('ticker', observed=True)['nifty_5d_return'].shift(-5)
    if has_factor_features(df):
        df = add_cross_sectional_ranks(df, horizons)
    return df


def make_windows(sessions, n_windows, window_size, horizon):
    """Expanding windows ending at the last session that has a target for `horizon`."""
    usable = sessions[:len(sessions) - horizon]
    windows = []
    for k in range(n_windows):
        test_end = len(usable) - (n_windows - 1 - k) * window_size     # exclusive index
        test_start = test_end - window_size
        es_end = test_start - horizon
        es_start = es_end - EARLY_STOP_SESSIONS
        train_end = es_start - horizon
        windows.append({
            'window': k + 1,
            'train': (usable[0], usable[train_end - 1]),
            'early_stop': (usable[es_start], usable[es_end - 1]),
            'test': (usable[test_start], usable[test_end - 1]),
        })
    return windows


def between(df, span):
    return df[(df['date'] >= span[0]) & (df['date'] <= span[1])]


# ─── Models ───────────────────────────────────────────────────────────────────

def fit_lgbm_quantiles(train, early_stop, target):
    feats = CATEGORICAL_COLS + REAL_COLS
    models = {}
    for q in QUANTILES:
        dtrain = lgb.Dataset(train[feats], label=train[target], categorical_feature=CATEGORICAL_COLS)
        dvalid = lgb.Dataset(early_stop[feats], label=early_stop[target],
                             categorical_feature=CATEGORICAL_COLS, reference=dtrain)
        models[q] = lgb.train({**LGBM_PARAMS, 'alpha': q}, dtrain, num_boost_round=1000, valid_sets=[dvalid],
                              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return models


def predict_lgbm(models, test):
    feats = CATEGORICAL_COLS + REAL_COLS
    p = {q: models[q].predict(test[feats], num_iteration=models[q].best_iteration) for q in QUANTILES}
    # Quantile models are fit separately and can cross; sort so q10 <= q50 <= q90
    stacked = np.sort(np.vstack([p[0.1], p[0.5], p[0.9]]), axis=0)
    return stacked[0], stacked[1], stacked[2]


def per_stock_quantiles(train, test, target):
    q = train.groupby('ticker', observed=True)[target].quantile(list(QUANTILES)).unstack()
    fallback = train[target].quantile(list(QUANTILES))
    out = test[['ticker']].join(q, on='ticker')
    return tuple(out[a].fillna(fallback[a]).to_numpy() for a in QUANTILES)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def daily_rank_ic(dates, score, actual, min_stocks=20):
    frame = pd.DataFrame({'date': dates, 's': score, 'y': actual}).dropna()
    frame = frame[frame.groupby('date')['s'].transform('size') >= min_stocks]
    frame['rs'] = frame.groupby('date')['s'].rank()
    frame['ry'] = frame.groupby('date')['y'].rank()
    return frame.groupby('date').apply(lambda g: g['rs'].corr(g['ry']), include_groups=False).dropna()


def decile_spread(dates, score, actual, min_stocks=20):
    frame = pd.DataFrame({'date': dates, 's': score, 'y': actual}).dropna()
    frame = frame[frame.groupby('date')['s'].transform('size') >= min_stocks]
    pct = frame.groupby('date')['s'].rank(pct=True)
    top = frame[pct > 0.9].groupby('date')['y'].mean()
    bottom = frame[pct <= 0.1].groupby('date')['y'].mean()
    return (top - bottom).dropna()


def non_overlapping(series, step):
    """Every `step`-th session, so samples don't share target days."""
    return series.iloc[::step]


def point_metrics(y, q10, q50, q90):
    nz = y != 0
    pinball = np.mean([np.mean(np.maximum(a * (y - q), (a - 1) * (y - q))) for a, q in zip(QUANTILES, (q10, q50, q90))])
    return {
        'mae': float(np.mean(np.abs(y - q50))),
        'pinball': float(pinball),
        'coverage_q10_q90': float(np.mean((y >= q10) & (y <= q90))),
        # A constant-zero forecast has no direction, so the metric does not apply to it
        'direction_acc': float(np.mean(np.sign(q50[nz]) == np.sign(y[nz]))) if np.any(q50 != 0) else np.nan,
    }


def ranking_metrics(dates, score, y, step):
    ic = daily_rank_ic(dates, score, y)
    spread = decile_spread(dates, score, y)
    return {'mean_ic': float(ic.mean()) if len(ic) else np.nan,
            'decile_spread': float(non_overlapping(spread, step).mean()) if len(spread) else np.nan}, ic, spread


# ─── Backtest ─────────────────────────────────────────────────────────────────

def backtest(scores, step=5):
    """scores: DataFrame [date, ticker, fwd_raw_5d, fwd_nifty_5d, <score columns>]."""
    rebalance_dates = sorted(scores['date'].unique())[::step]
    strategies = [c for c in scores.columns if c.startswith('score:')]
    rows, held = [], {s: set() for s in strategies}
    for d in rebalance_dates:
        day = scores[scores['date'] == d].dropna(subset=['fwd_raw_5d'])
        if len(day) < TOP_N * 2:
            continue
        row = {'date': d,
               'NIFTY 50': float(np.expm1(day['fwd_nifty_5d'].iloc[0])),
               'Equal weight, all stocks': float(np.expm1(day['fwd_raw_5d']).mean())}
        for s in strategies:
            picks = day.dropna(subset=[s]).nlargest(TOP_N, s)
            new = set(picks['ticker'])
            turnover = len(new - held[s]) / TOP_N if held[s] else 1.0
            held[s] = new
            row[s.replace('score:', '')] = float(np.expm1(picks['fwd_raw_5d']).mean() - turnover * ROUND_TRIP_COST)
            row[s.replace('score:', '') + ' turnover'] = turnover
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_backtest(bt, periods_per_year=252 / 5):
    out = []
    for col in [c for c in bt.columns if c != 'date' and not c.endswith('turnover')]:
        r = bt[col]
        equity = (1 + r).cumprod()
        years = len(r) / periods_per_year
        out.append({
            'strategy': col,
            'total_return': float(equity.iloc[-1] - 1),
            'annual_return': float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
            'sharpe': float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else np.nan,
            'max_drawdown': float((equity / equity.cummax() - 1).min()),
            'avg_turnover': float(bt[col + ' turnover'].mean()) if col + ' turnover' in bt else np.nan,
        })
    return pd.DataFrame(out)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-forward evaluation with baselines and a cost-aware backtest.")
    parser.add_argument('--parquet', default=DEFAULT_PARQUET)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--horizons', type=int, nargs='+', default=[5, 10, 21])
    parser.add_argument('--windows', type=int, default=6)
    parser.add_argument('--window-size', type=int, default=63)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading {args.parquet}...")
    df = load_data(args.parquet, args.horizons)
    use_rankers = has_factor_features(df)
    if not use_rankers:
        print("Factor features not found in the parquet; skipping the ranking models.")
    sessions = np.array(sorted(df['date'].unique()))
    print(f"{len(df)} rows, {df['ticker'].nunique()} stocks, {len(sessions)} sessions "
          f"({pd.Timestamp(sessions[0]).date()} to {pd.Timestamp(sessions[-1]).date()})")

    metric_rows, ic_rows, backtest_frames, window_rows = [], [], [], []
    for h in args.horizons:
        target = f'target_{h}d'
        for w in make_windows(sessions, args.windows, args.window_size, h):
            train = between(df, w['train']).dropna(subset=[target])
            early_stop = between(df, w['early_stop']).dropna(subset=[target])
            test = between(df, w['test']).dropna(subset=[target])
            window_rows.append({'horizon': h, 'window': w['window'],
                                **{f'{k}_{i}': str(pd.Timestamp(v[i]).date()) for k, v in w.items() if k != 'window' for i in (0, 1)},
                                'train_rows': len(train), 'test_rows': len(test)})
            print(f"[{h}d] window {w['window']}: train to {pd.Timestamp(w['train'][1]).date()}, "
                  f"test {pd.Timestamp(w['test'][0]).date()} to {pd.Timestamp(w['test'][1]).date()} "
                  f"({len(test)} rows)")

            y = test[target].to_numpy()
            dates = test['date'].to_numpy()
            models = fit_lgbm_quantiles(train, early_stop, target)
            lq10, lq50, lq90 = predict_lgbm(models, test)
            sq10, sq50, sq90 = per_stock_quantiles(train, test, target)
            zeros = np.zeros_like(y)

            candidates = {
                'LightGBM': (lq10, lq50, lq90, lq50),
                'Per-stock history': (sq10, sq50, sq90, sq50),
                'Predict zero': (sq10, zeros, sq90, None),
                'Momentum 12-1': (None, None, None, test['score_momentum_12_1'].to_numpy()),
                'Reversal 1-month': (None, None, None, test['score_reversal_1m'].to_numpy()),
            }
            if use_rankers:
                rank_reg = predict_scores(fit_rank_regressor(train, early_stop, h), test)
                lambda_rank = predict_scores(fit_lambdarank(train, early_stop, h), test)
                rank_reg_factors = predict_scores(fit_rank_regressor(train, early_stop, h, FACTOR_FEATURES), test)
                candidates['Rank regression'] = (None, None, None, rank_reg)
                candidates['Rank regression, factors only'] = (None, None, None, rank_reg_factors)
                candidates['LambdaRank'] = (None, None, None, lambda_rank)
                candidates['Factor composite'] = (None, None, None, factor_composite(test))
            for name, (q10, q50, q90, score) in candidates.items():
                row = {'horizon': h, 'window': w['window'], 'model': name, 'n': len(y)}
                if q50 is not None:
                    row.update(point_metrics(y, q10, q50, q90))
                if score is not None:
                    rank, ic, _ = ranking_metrics(dates, score, y, h)
                    row.update(rank)
                    ic_rows.append(pd.DataFrame({'horizon': h, 'window': w['window'], 'model': name,
                                                 'date': ic.index, 'ic': ic.values}))
                metric_rows.append(row)

            if h == 5:
                frame = pd.DataFrame({
                    'date': test['date'].values, 'ticker': test['ticker'].astype(str).values,
                    'fwd_raw_5d': test['fwd_raw_5d'].values, 'fwd_nifty_5d': test['fwd_nifty_5d'].values,
                    'score:LightGBM top 20': lq50,
                    'score:Momentum 12-1 top 20': test['score_momentum_12_1'].values,
                })
                if use_rankers:
                    frame['score:Rank regression top 20'] = candidates['Rank regression'][3]
                    frame['score:Rank regression, factors only top 20'] = candidates['Rank regression, factors only'][3]
                    frame['score:LambdaRank top 20'] = candidates['LambdaRank'][3]
                    frame['score:Factor composite top 20'] = candidates['Factor composite'][3]
                backtest_frames.append(frame)

    metrics = pd.DataFrame(metric_rows)
    ics = pd.concat(ic_rows, ignore_index=True) if ic_rows else pd.DataFrame()
    metrics.to_csv(os.path.join(args.out, 'metrics_by_window.csv'), index=False)
    ics.to_csv(os.path.join(args.out, 'daily_ic.csv'), index=False)
    pd.DataFrame(window_rows).to_csv(os.path.join(args.out, 'windows.csv'), index=False)

    # Pooled summary: averages over windows; IC t-stat from non-overlapping sessions across all windows
    summary = metrics.groupby(['horizon', 'model'], sort=False).mean(numeric_only=True).drop(columns=['window', 'n'])
    tstats = {}
    for (h, m), g in ics.groupby(['horizon', 'model'], sort=False):
        s = non_overlapping(g.sort_values('date')['ic'], h)
        tstats[(h, m)] = (float(s.mean() / s.std() * np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else np.nan, len(s))
    summary['ic_tstat'] = [tstats.get(k, (np.nan, 0))[0] for k in summary.index]
    summary['ic_samples'] = [tstats.get(k, (np.nan, 0))[1] for k in summary.index]
    summary['windows_ic_positive'] = metrics.assign(pos=metrics['mean_ic'] > 0).groupby(['horizon', 'model'], sort=False)['pos'].sum()
    summary = summary.reset_index()
    summary.to_csv(os.path.join(args.out, 'metrics_summary.csv'), index=False)

    print("\n=========== WALK-FORWARD SUMMARY (mean over windows) ===========")
    with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.float_format', '{:.4f}'.format):
        print(summary.to_string(index=False))

    if backtest_frames:
        bt = backtest(pd.concat(backtest_frames, ignore_index=True))
        bt.to_csv(os.path.join(args.out, 'backtest_weekly_returns.csv'), index=False)
        bt_summary = summarize_backtest(bt)
        bt_summary.to_csv(os.path.join(args.out, 'backtest_summary.csv'), index=False)
        print("\n=========== BACKTEST (weekly, top 20, after 0.18% round-trip costs) ===========")
        with pd.option_context('display.width', 200, 'display.float_format', '{:.4f}'.format):
            print(bt_summary.to_string(index=False))
        plot_results(summary, ics, bt, args.out)

    with open(os.path.join(args.out, 'run_info.json'), 'w') as f:
        json.dump({'parquet': os.path.relpath(args.parquet, ROOT), 'horizons': args.horizons,
                   'windows': args.windows, 'window_size': args.window_size,
                   'generated_at': pd.Timestamp.now().isoformat(timespec='seconds')}, f, indent=2)
    print(f"\nResults written to {args.out}")


def plot_results(summary, ics, bt, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Equity curves
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in [c for c in bt.columns if c != 'date' and not c.endswith('turnover')]:
        ax.plot(pd.to_datetime(bt['date']), (1 + bt[col]).cumprod(), label=col,
                linewidth=2.2 if col == 'LightGBM top 20' else 1.4)
    ax.axhline(1, color='#999', linewidth=0.8)
    ax.set_title('Growth of 1 rupee, weekly rebalanced, after costs')
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'backtest_equity.png'), dpi=140)
    plt.close(fig)

    # Mean rank IC by window and model, 5-day horizon
    five = ics[ics['horizon'] == 5]
    if len(five):
        by_window = five.groupby(['window', 'model'])['ic'].mean().unstack()
        ax = by_window.plot(kind='bar', figsize=(10, 5), width=0.8)
        ax.axhline(0, color='#444', linewidth=0.8)
        ax.set_title('Mean daily rank IC by test window, 5-day horizon')
        ax.set_xlabel('Walk-forward window')
        ax.set_ylabel('Rank IC')
        ax.grid(alpha=0.25, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'ic_by_window_5d.png'), dpi=140)
        plt.close()


if __name__ == '__main__':
    main()
