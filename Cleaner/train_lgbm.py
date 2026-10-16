import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'training_data', 'processed_features.parquet')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def main():
    print("Loading preprocessed features for LightGBM...")
    df = pd.read_parquet(DATA_PATH)
    df = df[df['target'].notna()].copy()
    
    # Sort chronologically
    df.sort_values('time_idx', inplace=True)
    
    categorical_cols = ["ticker", "sector", "industry", "day_of_week", "month", "is_earnings_week", "has_sentiment_history", "quarter_end"]
    
    real_cols = [
        'adj_close', 'volume', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'volume_spike', 'nifty_50_return',
        'decayed_net_sentiment_score', 'decayed_bullish_intensity', 'decayed_bearish_intensity',
        'log_return_1d', 'log_return_3d', 'log_return_5d',
        'realized_vol_20d', 'atr_14', 'sector_relative_return_1d', 'sentiment_volume_7d',
        'stoch_rsi_14', 'bb_width', 'price_vs_sma50', 'price_vs_52w_high', 'momentum_10d', 'obv_normalized', 'volatility_regime',
        'pvt_normalized', 'rsi_divergence', 'nifty_volatility_20d', 'sector_volatility_20d'
    ]
    
    features = categorical_cols + real_cols
    
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        
    for col in real_cols:
        df[col] = df[col].astype('float32')

    horizons = ['5d', '10d', '21d', '126d']
    
    for horizon in horizons:
        target_col = f'target_{horizon}'
        print(f"\n==================================================")
        print(f" TRAINING HORIZON: {horizon.upper()}")
        print(f"==================================================")
        
        # Prepare data for this horizon
        df_horizon = df[df[target_col].notna()].copy()
        
        # Sort chronologically
        df_horizon.sort_values('time_idx', inplace=True)
        
        X = df_horizon[features]
        y = df_horizon[target_col]
        time_idx = df_horizon['time_idx']
        
        # Train-test split (walk-forward chronological)
        train_end_idx = df_horizon['time_idx'].quantile(0.8)
        
        train_mask = df_horizon['time_idx'] <= train_end_idx
        val_mask = df_horizon['time_idx'] > train_end_idx
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
        
        quantiles = [0.1, 0.5, 0.9]
        models = {}
        
        for q in quantiles:
            q_label = f'q{int(q*100)}'
            print(f"\n--- Training LightGBM for {horizon} | Quantile {q} ---")
            params = {
                'objective': 'quantile',
                'alpha': q,
                'metric': 'quantile',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 63,
                'max_depth': -1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_jobs': 4
            }
            
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
            val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols, reference=train_data)
            
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[train_data, val_data],
                callbacks=callbacks
            )
            
            models[q_label] = model
            
            # Save model with horizon in filename
            model_path = os.path.join(ARTIFACTS_DIR, f'lgbm_model_{horizon}_{q_label}.txt')
            model.save_model(model_path)
            print(f"Saved LGBM model to {model_path}")
            
            if q == 0.5:
                # Log best score for Q50
                best_iter = model.best_iteration
                best_score = model.best_score['valid_1']['quantile']
                print(f"[{horizon}] Q50 Best Iteration: {best_iter}, Val MAE: {best_score:.4f}")

        # Feature Importance for Q50 of this horizon
        print(f"\n--- Feature Importance ({horizon} Q50) ---")
        importance = models['q50'].feature_importance(importance_type='gain')
        imp_df = pd.DataFrame({'feature': features, 'gain': importance}).sort_values('gain', ascending=False)
        print(imp_df.head(10))
        
        imp_path = os.path.join(ARTIFACTS_DIR, f'lgbm_feature_importance_{horizon}.csv')
        imp_df.to_csv(imp_path, index=False)
    
if __name__ == "__main__":
    main()
