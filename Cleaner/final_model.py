"""
final_model.py — The frozen 21-day stock ranking and its daily confidence flag.

The model (chosen on the walk-forward test, frozen 17 Oct 2026):
  - four members: 12-1 month momentum, a rank regression (LightGBM on daily ranks), a LightGBM
    median forecast, and each stock's historical median 21-day excess return;
  - each member's score becomes a percentile rank within the day (a missing score sits at 0.5);
    the blend is their equal-weight average, smoothed over each stock's last 5 sessions;
  - the top 20 by the smoothed blend are the picks.

Confidence flag: on each day, `agreement` is the mean pairwise Spearman correlation between the
four members' rankings. A day is HIGH CONFIDENCE when agreement is above the median of the
previous 126 recorded days (at least 21 needed). This rule was fixed before it was measured; on
the walk-forward test (Mar 2025 to Sep 2026) high-confidence days were 46% of days, and on them the
top 20 beat the NIFTY 50 over 21 sessions 74% of the time (68% on all days). See
Cleaner/confidence_study.py.

The agreement history starts from the walk-forward test (out-of-sample) and grows by one live
row per scored day (artifacts/agreement_history.csv).

Usage:
    python Cleaner/final_model.py --train        # fit members, write artifacts (run on retrain days)
    python Cleaner/final_model.py --score        # print today's picks and confidence, no writes
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

from evaluate import load_data, CATEGORICAL_COLS, REAL_COLS, LGBM_PARAMS, DEFAULT_PARQUET, ROOT
from ranking import fit_rank_regressor, predict_scores

HORIZON = 21
MEMBERS = ['Momentum 12-1', 'Rank regression', 'LightGBM', 'Per-stock history']
SMOOTH_DAYS = 5
AGREEMENT_LOOKBACK, AGREEMENT_MIN_HISTORY = 126, 21
TOP_N = 20
EARLY_STOP_SESSIONS = 42
ARTIFACTS = os.path.join(ROOT, 'artifacts')
RANKER_PATH = os.path.join(ARTIFACTS, 'final_rank_regressor_21d.txt')
LGBM_PATH = os.path.join(ARTIFACTS, 'final_lgbm_q50_21d.txt')
PER_STOCK_PATH = os.path.join(ARTIFACTS, 'final_per_stock_median_21d.csv')
HISTORY_PATH = os.path.join(ARTIFACTS, 'agreement_history.csv')
INFO_PATH = os.path.join(ARTIFACTS, 'final_model.json')
WALK_FORWARD_PREDICTIONS = os.path.join(ROOT, 'results', 'horizon_21d', 'test_predictions.parquet')


# ─── Shared rules (used live and by confidence_study.py) ──────────────────────

def member_ranks(frame, members=MEMBERS):
    """Percentile rank of each member's score within its day; missing scores sit at 0.5."""
    ranks = frame.groupby('date')[members].rank(pct=True).fillna(0.5)
    ranks.columns = ['p:' + m for m in members]
    return ranks


def blend_scores(frame, members=MEMBERS):
    """Adds 'blend' (equal-weight average of member ranks) and 'blend_smoothed' (mean of each stock's
    last SMOOTH_DAYS blends). Needs columns date, ticker and the member scores."""
    out = pd.concat([frame, member_ranks(frame, members)], axis=1)
    out['blend'] = out[['p:' + m for m in members]].mean(axis=1)
    out = out.sort_values(['ticker', 'date'])
    out['blend_smoothed'] = out.groupby('ticker')['blend'].transform(
        lambda x: x.rolling(SMOOTH_DAYS, min_periods=1).mean())
    return out


def daily_agreement(frame, members=MEMBERS):
    """Mean pairwise Spearman correlation between the members' rankings, per date."""
    cols = ['p:' + m for m in members]
    pairs = np.triu_indices(len(members), 1)
    return frame.groupby('date')[cols].apply(
        lambda g: g.corr(method='spearman').values[pairs].mean()).rename('agreement')


def confidence_threshold(history, before_date):
    """Median agreement over the last AGREEMENT_LOOKBACK recorded days before `before_date`."""
    past = history[history['date'] < pd.Timestamp(before_date)].sort_values('date').tail(AGREEMENT_LOOKBACK)
    return float(past['agreement'].median()) if len(past) >= AGREEMENT_MIN_HISTORY else np.nan


def pick_order(day):
    """Picks for one day: highest smoothed blend first, ties broken by ticker."""
    return day.sort_values(['blend_smoothed', 'ticker'], ascending=[False, True])


# ─── Training ─────────────────────────────────────────────────────────────────

def split_for_training(df):
    """As in evaluate.py: train | embargo | early-stop slice, ending where 21-day targets end."""
    target = f'target_{HORIZON}d'
    sessions = np.array(sorted(df.dropna(subset=[target])['date'].unique()))
    es_dates = sessions[-EARLY_STOP_SESSIONS:]
    train_end = sessions[-EARLY_STOP_SESSIONS - HORIZON - 1]
    train = df[(df['date'] <= train_end)].dropna(subset=[target])
    early_stop = df[df['date'].isin(es_dates)].dropna(subset=[target])
    return train, early_stop


def bootstrap_history():
    """Agreement on every walk-forward test day (out of sample), as the start of the history."""
    preds = pd.read_parquet(WALK_FORWARD_PREDICTIONS)
    preds = preds[preds['horizon'] == HORIZON].copy()
    preds['date'] = pd.to_datetime(preds['date'])
    ranked = pd.concat([preds[['date', 'ticker']], member_ranks(preds)], axis=1)
    hist = daily_agreement(ranked).reset_index()
    hist['source'] = 'walk-forward'
    return hist


def train(parquet):
    target = f'target_{HORIZON}d'
    df = load_data(parquet, [HORIZON])
    tr, es = split_for_training(df)
    print(f"Training on {tr['date'].min().date()} to {tr['date'].max().date()}, early stopping on "
          f"{es['date'].min().date()} to {es['date'].max().date()}")

    ranker = fit_rank_regressor(tr, es, HORIZON)
    ranker.save_model(RANKER_PATH, num_iteration=ranker.best_iteration)

    feats = CATEGORICAL_COLS + REAL_COLS
    dtrain = lgb.Dataset(tr[feats], label=tr[target], categorical_feature=CATEGORICAL_COLS)
    dvalid = lgb.Dataset(es[feats], label=es[target], categorical_feature=CATEGORICAL_COLS, reference=dtrain)
    median_model = lgb.train({**LGBM_PARAMS, 'alpha': 0.5}, dtrain, num_boost_round=1000, valid_sets=[dvalid],
                             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    median_model.save_model(LGBM_PATH, num_iteration=median_model.best_iteration)

    per_stock = df.dropna(subset=[target]).groupby('ticker', observed=True)[target].median().rename('median')
    per_stock.to_csv(PER_STOCK_PATH)

    # Keep live rows already recorded; refresh the walk-forward start
    hist = bootstrap_history()
    if os.path.exists(HISTORY_PATH):
        old = pd.read_csv(HISTORY_PATH, parse_dates=['date'])
        hist = pd.concat([hist, old[old['source'] == 'live']], ignore_index=True)
    hist.drop_duplicates('date', keep='last').sort_values('date').to_csv(HISTORY_PATH, index=False)

    info = {'members': MEMBERS, 'horizon': HORIZON, 'smooth_days': SMOOTH_DAYS, 'top_n': TOP_N,
            'agreement_lookback': AGREEMENT_LOOKBACK, 'trained_to': str(tr['date'].max().date()),
            'ranker_rounds': ranker.best_iteration, 'lgbm_rounds': median_model.best_iteration,
            'trained_at': pd.Timestamp.now().isoformat(timespec='seconds')}
    json.dump(info, open(INFO_PATH, 'w'), indent=2)
    print(f"Saved final-model artifacts to {ARTIFACTS} (rank regression {ranker.best_iteration} rounds, "
          f"LightGBM median {median_model.best_iteration} rounds, {len(hist)} agreement history rows)")


# ─── Live scoring ─────────────────────────────────────────────────────────────

def artifacts_ready():
    return all(os.path.exists(p) for p in (RANKER_PATH, LGBM_PATH, PER_STOCK_PATH, HISTORY_PATH))


def score_latest(parquet=DEFAULT_PARQUET, record=False):
    """Scores the latest session. Returns (per-stock DataFrame, day summary dict).
    With record=True, appends today's agreement to the history file."""
    df = load_data(parquet, [HORIZON])
    sessions = sorted(df['date'].unique())
    recent = df[df['date'].isin(sessions[-SMOOTH_DAYS:])].copy()
    latest = pd.Timestamp(sessions[-1])

    ranker = lgb.Booster(model_file=RANKER_PATH)
    median_model = lgb.Booster(model_file=LGBM_PATH)
    per_stock = pd.read_csv(PER_STOCK_PATH, index_col=0)['median']
    recent['Momentum 12-1'] = recent['score_momentum_12_1']
    recent['Rank regression'] = predict_scores(ranker, recent)
    recent['LightGBM'] = median_model.predict(recent[CATEGORICAL_COLS + REAL_COLS])
    recent['Per-stock history'] = recent['ticker'].astype(str).map(per_stock).fillna(per_stock.median()).to_numpy()
    frame = recent[['date', 'ticker'] + MEMBERS].copy()
    frame['ticker'] = frame['ticker'].astype(str)

    scored = blend_scores(frame)
    agreement = float(daily_agreement(scored[scored['date'] == latest]).iloc[0])
    history = pd.read_csv(HISTORY_PATH, parse_dates=['date'])
    threshold = confidence_threshold(history, latest)
    high = bool(agreement > threshold) if not np.isnan(threshold) else False

    today = pick_order(scored[scored['date'] == latest]).reset_index(drop=True)
    today['blend_rank'] = np.arange(1, len(today) + 1)
    if record:
        history = pd.concat([history[history['date'] != latest],
                             pd.DataFrame([{'date': latest, 'agreement': agreement, 'source': 'live'}])])
        history.sort_values('date').to_csv(HISTORY_PATH, index=False)
    summary = {'date': str(latest.date()), 'agreement': agreement, 'threshold': threshold,
               'high_confidence': high, 'stocks': len(today)}
    return today[['ticker', 'blend_smoothed', 'blend_rank']].rename(columns={'blend_smoothed': 'blend_score'}), summary


def main():
    parser = argparse.ArgumentParser(description="Train or score the frozen 21-day blend and its confidence flag.")
    parser.add_argument('--parquet', default=DEFAULT_PARQUET)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--score', action='store_true')
    args = parser.parse_args()
    if args.train:
        train(args.parquet)
    if args.score or not args.train:
        picks, day = score_latest(args.parquet)
        label = 'HIGH CONFIDENCE' if day['high_confidence'] else 'normal'
        print(f"{day['date']}: model agreement {day['agreement']:.3f} vs threshold {day['threshold']:.3f} -> {label}")
        print(picks.head(TOP_N).to_string(index=False))


if __name__ == '__main__':
    main()
