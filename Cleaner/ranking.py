"""
ranking.py — Cross-sectional ranking models shared by evaluate.py and train_ranker.py.

Stock picking only needs the ORDER of stocks on a given day to be right, so these models learn
the daily rank of each stock's future excess return instead of its exact value:

  - every stock-specific feature is converted to its percentile rank across stocks on that day
    (market-wide inputs such as the NIFTY return are identical for all stocks and are dropped)
  - the stock ID, calendar month and raw price/volume levels are left out, so the models can't
    memorise each stock's past average return
  - two learners: a rank regression (L2 on the percentile-rank target) and LambdaRank, which
    optimises the ordering within each day directly
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb

STOCK_FEATURES = [
    'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'volume_spike',
    'decayed_net_sentiment_score', 'decayed_bullish_intensity', 'decayed_bearish_intensity',
    'log_return_1d', 'log_return_3d', 'log_return_5d', 'log_return_10d', 'log_return_21d', 'log_return_126d',
    'realized_vol_20d', 'atr_14', 'sector_relative_return_1d', 'sentiment_volume_7d',
    'stoch_rsi_14', 'bb_width', 'price_vs_sma50', 'price_vs_52w_high', 'momentum_10d', 'obv_normalized',
    'volatility_regime', 'pvt_normalized', 'rsi_divergence', 'sector_volatility_20d',
    # factor inputs from build_features.py
    'momentum_12_1', 'volatility_60d', 'amihud_illiquidity_21d', 'size_proxy', 'beta_60d',
]
RANK_FEATURES = ['xs_' + c for c in STOCK_FEATURES] + ['sector']
# Well-known cross-sectional factors only; tests whether the technical indicators add or dilute signal
FACTOR_FEATURES = ['xs_' + c for c in [
    'momentum_12_1', 'log_return_21d', 'volatility_60d', 'amihud_illiquidity_21d', 'size_proxy',
    'beta_60d', 'price_vs_52w_high', 'price_vs_sma50', 'sector_relative_return_1d',
]] + ['sector']
N_RELEVANCE_LEVELS = 10   # LambdaRank labels: the target's decile within the day (0 = worst)

BASE_PARAMS = {
    'learning_rate': 0.03, 'num_leaves': 31, 'min_data_in_leaf': 200,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'lambda_l2': 1.0,
    'verbose': -1, 'num_threads': os.cpu_count() or 4,
}


def has_factor_features(df):
    return all(c in df.columns for c in STOCK_FEATURES)


def add_cross_sectional_ranks(df, horizons):
    """Adds xs_<feature> and xs_target_<h>d columns: percentile ranks within each date."""
    by_date = df.groupby('date')
    for c in STOCK_FEATURES:
        df['xs_' + c] = by_date[c].rank(pct=True).astype('float32')
    for h in horizons:
        df[f'xs_target_{h}d'] = by_date[f'target_{h}d'].rank(pct=True).astype('float32')
    return df


def _sorted_by_date(frame):
    return frame.sort_values(['date', 'ticker'], kind='mergesort')


def _relevance(frame, h):
    return np.clip((frame[f'xs_target_{h}d'] * N_RELEVANCE_LEVELS).astype(int), 0, N_RELEVANCE_LEVELS - 1)


def factor_composite(frame):
    """No-training baseline: average daily rank of momentum, low volatility and nearness to the 52-week high."""
    return ((frame['xs_momentum_12_1'] + (1 - frame['xs_volatility_60d']) + frame['xs_price_vs_52w_high']) / 3).to_numpy()


def fit_rank_regressor(train, early_stop, h, features=RANK_FEATURES):
    dtrain = lgb.Dataset(train[features], label=train[f'xs_target_{h}d'], categorical_feature=['sector'])
    dvalid = lgb.Dataset(early_stop[features], label=early_stop[f'xs_target_{h}d'],
                         categorical_feature=['sector'], reference=dtrain)
    return lgb.train({**BASE_PARAMS, 'objective': 'regression', 'metric': 'l2'}, dtrain,
                     num_boost_round=2000, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])


def fit_lambdarank(train, early_stop, h):
    train, early_stop = _sorted_by_date(train), _sorted_by_date(early_stop)
    dtrain = lgb.Dataset(train[RANK_FEATURES], label=_relevance(train, h),
                         group=train.groupby('date', sort=True).size().to_numpy(), categorical_feature=['sector'])
    dvalid = lgb.Dataset(early_stop[RANK_FEATURES], label=_relevance(early_stop, h),
                         group=early_stop.groupby('date', sort=True).size().to_numpy(),
                         categorical_feature=['sector'], reference=dtrain)
    params = {**BASE_PARAMS, 'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [20],
              'label_gain': list(range(N_RELEVANCE_LEVELS))}
    return lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])


def predict_scores(model, frame):
    """Higher score = expected to rank higher. Order matches `frame`."""
    return model.predict(frame[model.feature_name()], num_iteration=model.best_iteration)
