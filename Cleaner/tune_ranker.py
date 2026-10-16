"""
tune_ranker.py — Tune 21-day cross-sectional rankers without touching the test period.

Protocol (so the walk-forward test stays an honest, one-shot check):
  1. Tuning folds: 4 blocks of 126 sessions that END `horizon` sessions before the first
     walk-forward test window starts. Each fold trains on everything up to `horizon` sessions
     before its block, so no target overlaps the block or the test period.
  2. Optuna (TPE sampler, median pruning) searches LightGBM rank-regression settings and the
     feature set. The score is the mean daily rank IC over the folds minus half its spread,
     so a stable setting beats one that is lucky in one fold.
  3. Baselines on the same folds: the untuned rank regression (as in evaluate.py), 12-1 momentum,
     and a few CatBoost settings on the GPU.
  4. `--evaluate` then refits the best LightGBM and the best CatBoost setting in each of the
     6 walk-forward windows (training on everything up to the embargo before the test window,
     with a fixed number of rounds) and adds their test scores to the saved predictions, so
     ensemble_study.py can blend them.

Outputs results/tuning_21d/. Usage:
    python Cleaner/tune_ranker.py [--trials 60] [--horizon 21]
    python Cleaner/tune_ranker.py --evaluate          # after tuning
    python Cleaner/ensemble_study.py --dir results/tuning_21d --horizon 21 --extra "Tuned LightGBM" "CatBoost"
"""

import os
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

from evaluate import load_data, make_windows, between, daily_rank_ic, top_n_hits, DEFAULT_PARQUET, ROOT
from ranking import RANK_FEATURES, FACTOR_FEATURES, fit_rank_regressor, predict_scores

warnings.filterwarnings('ignore')
OUT_DIR = os.path.join(ROOT, 'results', 'tuning_21d')
WALK_FORWARD_DIR = os.path.join(ROOT, 'results', 'horizon_21d')
MARKET_FEATURES = ['nifty_volatility_20d', 'nifty_21d_return', 'nifty_126d_return']
FEATURE_SETS = {
    'all ranks': RANK_FEATURES,
    'factors only': FACTOR_FEATURES,
    'all ranks + market': RANK_FEATURES + MARKET_FEATURES,
}
N_FOLDS, FOLD_SIZE = 4, 126
EARLY_STOP_SESSIONS = 42
THREADS = os.cpu_count() or 4
CATBOOST_GRID = [
    {'depth': 4, 'learning_rate': 0.03, 'l2_leaf_reg': 10, 'iterations': 800},
    {'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 10, 'iterations': 800},
    {'depth': 6, 'learning_rate': 0.01, 'l2_leaf_reg': 30, 'iterations': 2000},
    {'depth': 8, 'learning_rate': 0.02, 'l2_leaf_reg': 30, 'iterations': 1000},
]


# ─── Folds ────────────────────────────────────────────────────────────────────

def make_folds(sessions, horizon):
    """Tuning blocks that end before the first walk-forward test window, with embargoes."""
    first_test = make_windows(sessions, 6, 63, horizon)[0]['test'][0]
    test_idx = int(np.searchsorted(sessions, first_test))
    last_val_end = test_idx - horizon          # exclusive: val targets end before the test starts
    folds = []
    for k in range(N_FOLDS):
        val_end = last_val_end - (N_FOLDS - 1 - k) * FOLD_SIZE
        val_start = val_end - FOLD_SIZE
        train_end = val_start - horizon
        folds.append({'fold': k + 1, 'train_end': sessions[train_end - 1],
                      'val': (sessions[val_start], sessions[val_end - 1])})
    return folds


def fold_frames(df, fold, target):
    train = df[df['date'] <= fold['train_end']].dropna(subset=[target])
    val = between(df, fold['val']).dropna(subset=[target])
    return train, val


def score(val, pred, target_raw):
    ic = daily_rank_ic(val['date'].to_numpy(), pred, val[target_raw].to_numpy())
    hits = top_n_hits(val['date'].to_numpy(), pred, val[target_raw].to_numpy())
    return float(ic.mean()), float(hits['hit_vs_avg'].mean())


# ─── Models ───────────────────────────────────────────────────────────────────

def lgb_params(p):
    return {'objective': 'regression', 'verbose': -1, 'num_threads': THREADS, 'bagging_freq': 1,
            **{k: v for k, v in p.items() if k not in ('feature_set', 'num_rounds')}}


def fit_lgb(p, train, target):
    features = FEATURE_SETS[p['feature_set']]
    data = lgb.Dataset(train[features], label=train[target], categorical_feature=['sector'])
    return lgb.train(lgb_params(p), data, num_boost_round=p['num_rounds']), features


def fit_catboost(cfg, train, target):
    from catboost import CatBoostRegressor
    features = RANK_FEATURES
    X = train[features].copy()
    X['sector'] = X['sector'].astype(str)
    model = CatBoostRegressor(loss_function='RMSE', task_type='GPU', verbose=0, random_seed=0,
                              cat_features=['sector'], **cfg)
    model.fit(X, train[target].to_numpy())
    return model, features


def predict_catboost(model, features, frame):
    X = frame[features].copy()
    X['sector'] = X['sector'].astype(str)
    return model.predict(X)


# ─── Search ───────────────────────────────────────────────────────────────────

def suggest(trial):
    return {
        'feature_set': trial.suggest_categorical('feature_set', list(FEATURE_SETS)),
        'num_rounds': trial.suggest_int('num_rounds', 50, 1500, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 7, 63, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 5000, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.9),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-2, 100, log=True),
        'max_bin': trial.suggest_categorical('max_bin', [63, 255]),
        'extra_trees': trial.suggest_categorical('extra_trees', [False, True]),
    }


def run_search(df, folds, h, n_trials):
    target, target_raw = f'xs_target_{h}d', f'target_{h}d'
    frames = [fold_frames(df, f, target) for f in folds]

    def objective(trial):
        p = suggest(trial)
        ics, hits = [], []
        for k, (train, val) in enumerate(frames):
            model, features = fit_lgb(p, train, target)
            ic, hit = score(val, model.predict(val[features]), target_raw)
            ics.append(ic)
            hits.append(hit)
            trial.report(float(np.mean(ics)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr('fold_ic', ics)
        trial.set_user_attr('fold_hit', hits)
        return float(np.mean(ics) - 0.5 * np.std(ics))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    started = time.time()
    study.optimize(objective, n_trials=n_trials,
                   callbacks=[lambda s, t: print(f"  trial {t.number}: {t.state.name} "
                                                 f"{'' if t.value is None else f'{t.value:.4f}'} "
                                                 f"(best {s.best_value:.4f}, {time.time() - started:.0f}s)", flush=True)])
    return study


def baselines(df, folds, h):
    target, target_raw = f'xs_target_{h}d', f'target_{h}d'
    rows = []
    for f in folds:
        train, val = fold_frames(df, f, target)
        # As in evaluate.py: an early-stopping slice at the end of the train block, with an embargo
        dates = np.array(sorted(train['date'].unique()))
        es = train[train['date'] >= dates[-EARLY_STOP_SESSIONS]]
        tr = train[train['date'] < dates[-EARLY_STOP_SESSIONS - h]]
        untuned = fit_rank_regressor(tr, es, h)
        row = {'fold': f['fold'], 'val_start': str(pd.Timestamp(f['val'][0]).date()),
               'val_end': str(pd.Timestamp(f['val'][1]).date())}
        row['untuned_ic'], row['untuned_hit'] = score(val, predict_scores(untuned, val), target_raw)
        row['momentum_ic'], row['momentum_hit'] = score(val, val['xs_momentum_12_1'].to_numpy(), target_raw)
        for i, cfg in enumerate(CATBOOST_GRID):
            model, features = fit_catboost(cfg, train, target)
            row[f'catboost{i}_ic'], row[f'catboost{i}_hit'] = score(val, predict_catboost(model, features, val), target_raw)
        print(f"  fold {f['fold']}: " + ', '.join(f'{k} {v:.3f}' for k, v in row.items() if k.endswith('_ic')), flush=True)
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Walk-forward check of the chosen settings ────────────────────────────────

def evaluate_chosen(df, sessions, h):
    target = f'xs_target_{h}d'
    best = json.load(open(os.path.join(OUT_DIR, 'best_params.json')))
    preds = pd.read_parquet(os.path.join(WALK_FORWARD_DIR, 'test_predictions.parquet'))
    preds = preds[preds['horizon'] == h].copy()
    preds['date'] = pd.to_datetime(preds['date'])
    frames = []
    for w in make_windows(sessions, 6, 63, h):
        # Fixed rounds need no early-stopping slice: train on everything up to the embargo
        train = df[df['date'] <= w['early_stop'][1]].dropna(subset=[target])
        test = between(df, w['test']).dropna(subset=[f'target_{h}d'])
        model, features = fit_lgb(best['lightgbm'], train, target)
        cat_model, cat_features = fit_catboost(best['catboost'], train, target)
        frames.append(pd.DataFrame({'date': pd.to_datetime(test['date'].to_numpy()),
                                    'ticker': test['ticker'].astype(str).to_numpy(),
                                    'Tuned LightGBM': model.predict(test[features]),
                                    'CatBoost': predict_catboost(cat_model, cat_features, test)}))
        print(f"  window {w['window']}: trained to {pd.Timestamp(w['early_stop'][1]).date()}, "
              f"scored {len(test)} rows", flush=True)
    new = pd.concat(frames, ignore_index=True)
    merged = preds.merge(new, on=['date', 'ticker'], how='left')
    merged.to_parquet(os.path.join(OUT_DIR, 'test_predictions.parquet'), index=False)
    print(f"Saved {os.path.join(OUT_DIR, 'test_predictions.parquet')} ({merged['Tuned LightGBM'].notna().sum()} tuned scores)")


def main():
    parser = argparse.ArgumentParser(description="Tune 21-day rankers on pre-test folds only.")
    parser.add_argument('--parquet', default=DEFAULT_PARQUET)
    parser.add_argument('--horizon', type=int, default=21)
    parser.add_argument('--trials', type=int, default=60)
    parser.add_argument('--evaluate', action='store_true', help='Score the chosen settings on the walk-forward windows')
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    h = args.horizon

    df = load_data(args.parquet, [h])
    sessions = np.array(sorted(df['date'].unique()))
    if args.evaluate:
        evaluate_chosen(df, sessions, h)
        return

    folds = make_folds(sessions, h)
    for f in folds:
        print(f"Fold {f['fold']}: train to {pd.Timestamp(f['train_end']).date()}, "
              f"validate {pd.Timestamp(f['val'][0]).date()} to {pd.Timestamp(f['val'][1]).date()}")

    print("\nBaselines on the tuning folds (untuned rank regression, momentum, CatBoost grid)...")
    base = baselines(df, folds, h)
    base.to_csv(os.path.join(OUT_DIR, 'baselines.csv'), index=False)

    print(f"\nOptuna search, {args.trials} trials...")
    study = run_search(df, folds, h, args.trials)
    trials = study.trials_dataframe()
    trials.to_csv(os.path.join(OUT_DIR, 'trials.csv'), index=False)

    best_trial = study.best_trial
    lgb_best = {**best_trial.params}
    cat_scores = {i: base[f'catboost{i}_ic'].mean() - 0.5 * base[f'catboost{i}_ic'].std(ddof=0)
                  for i in range(len(CATBOOST_GRID))}
    cat_best = CATBOOST_GRID[max(cat_scores, key=cat_scores.get)]
    untuned = base['untuned_ic'].mean() - 0.5 * base['untuned_ic'].std(ddof=0)
    summary = {
        'lightgbm': lgb_best, 'catboost': cat_best,
        'scores': {'tuned_lightgbm': study.best_value, 'best_catboost': max(cat_scores.values()),
                   'untuned_rank_regression': float(untuned),
                   'momentum': float(base['momentum_ic'].mean() - 0.5 * base['momentum_ic'].std(ddof=0))},
        'tuned_fold_ic': best_trial.user_attrs['fold_ic'], 'tuned_fold_hit': best_trial.user_attrs['fold_hit'],
        'folds': [{'fold': f['fold'], 'train_end': str(pd.Timestamp(f['train_end']).date()),
                   'val': [str(pd.Timestamp(v).date()) for v in f['val']]} for f in folds],
        'trials': len(study.trials),
    }
    json.dump(summary, open(os.path.join(OUT_DIR, 'best_params.json'), 'w'), indent=2)
    print("\nScore = mean fold IC - half its spread (higher is better):")
    for k, v in summary['scores'].items():
        print(f"  {k:26s} {v:.4f}")
    print(f"Best LightGBM settings: {lgb_best}")
    print(f"Best CatBoost settings: {cat_best}")
    print(f"\nResults written to {OUT_DIR}")


if __name__ == '__main__':
    main()
