import os
import json
import warnings
import pandas as pd
import pytorch_lightning as pl
import optuna
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss

warnings.filterwarnings("ignore")

# Force float32 precision for matmul
import torch
torch.set_float32_matmul_precision('medium')

def objective(trial):
    # Hyperparameters to tune
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
    attention_head_size = trial.suggest_categorical("attention_head_size", [1, 2, 4])
    
    hidden_continuous_size = max(16, hidden_size // 2)

    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'training_data', 'processed_features.parquet')
    df = pd.read_parquet(data_path)
    df = df[df['target'].notna()].copy()

    # Data types
    for col in ["ticker", "sector", "industry", "day_of_week", "month", "is_earnings_week", "has_sentiment_history", "quarter_end"]:
        df[col] = df[col].astype(str)

    real_cols = [
        'adj_close', 'volume', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'volume_spike', 'nifty_50_return',
        'decayed_net_sentiment_score', 'decayed_bullish_intensity', 'decayed_bearish_intensity',
        'log_return_1d', 'log_return_3d', 'log_return_5d',
        'realized_vol_20d', 'atr_14', 'sector_relative_return_1d', 'sentiment_volume_7d',
        'stoch_rsi_14', 'bb_width', 'price_vs_sma50', 'price_vs_52w_high', 'momentum_10d', 'obv_normalized', 'volatility_regime',
        'pvt_normalized', 'rsi_divergence', 'nifty_volatility_20d', 'sector_volatility_20d'
    ]
    for col in real_cols:
        df[col] = df[col].astype('float32')

    df['time_idx'] = df['time_idx'].astype(int)
    df['target'] = df['target'].astype('float32')

    # Reduce dataset size significantly for fast tuning
    # Sort by date and take the last 30% of data to tune faster
    min_time_idx = df['time_idx'].quantile(0.7)
    df = df[df['time_idx'] >= min_time_idx].copy()

    train_end_idx = df['time_idx'].max() - 20
    val_end_idx = df['time_idx'].max() - 10
    
    train_df = df[df['time_idx'] <= train_end_idx]
    val_df = df[(df['time_idx'] > train_end_idx) & (df['time_idx'] <= val_end_idx)]
    
    max_encoder_length = 60
    max_prediction_length = 5
    
    val_df_with_history = pd.concat([
        train_df.groupby('ticker').tail(max_encoder_length),
        val_df
    ]).sort_values(['ticker', 'time_idx'])

    from pytorch_forecasting.data import GroupNormalizer
    target_normalizer = GroupNormalizer(groups=["ticker"], center=True)

    training_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target",
        group_ids=["ticker"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["ticker", "sector", "industry", "has_sentiment_history"],
        time_varying_known_categoricals=["day_of_week", "month", "is_earnings_week", "quarter_end"],
        time_varying_unknown_reals=real_cols,
        target_normalizer=target_normalizer,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_df_with_history, stop_randomization=True)

    batch_size = 256
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=4)

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=lr,
        hidden_size=hidden_size,
        lstm_layers=2,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=2,
    )
    
    trainer = pl.Trainer(
        max_epochs=3, # Fast tuning epochs
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False
    )
    
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    print("Starting Optuna Hyperparameter Optimization for TFT...")
    
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=10, timeout=3600)
    
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    out_path = os.path.join(os.path.dirname(__file__), '..', 'training_data', 'best_hparams.json')
    with open(out_path, 'w') as f:
        json.dump(trial.params, f, indent=4)
        
    print(f"Saved optimized hyperparameters to {out_path}")
