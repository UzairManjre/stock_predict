import os
import gc
import json
import warnings
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

def patch_mask_bias_for_fp16(model):
    patched = 0
    for name, module in model.named_modules():
        if hasattr(module, 'mask_bias') and hasattr(module.mask_bias, 'data'):
            module.mask_bias.data = torch.tensor(-65500.0,
                dtype=module.mask_bias.dtype, device=module.mask_bias.device)
            patched += 1
    print(f"Patched {patched} multihead attention mask_bias parameters for float16.")

def run_walk_forward(df, real_cols, hparams, num_windows=3):
    max_time_idx = df['time_idx'].max()
    min_time_idx = df['time_idx'].min()
    
    # Calculate window size based on total time span
    total_span = max_time_idx - min_time_idx
    # Reserve last 30% for out-of-sample walk-forward
    test_span = int(total_span * 0.3)
    window_size = test_span // num_windows
    
    dead_zone = 10  # 10 days dead zone to prevent leakage
    max_encoder_length = 60
    max_prediction_length = 5
    
    cv_results = []
    out_of_sample_preds = []

    for i in range(num_windows):
        print(f"\\n{'='*50}")
        print(f" WALK-FORWARD WINDOW {i+1} / {num_windows}")
        print(f"{'='*50}")
        
        # Define expanding window boundaries
        train_end = max_time_idx - test_span + (i * window_size)
        val_start = train_end + dead_zone
        val_end = val_start + window_size
        
        print(f"  Train: idx <= {train_end}")
        print(f"  Val:   idx {val_start} to {val_end}")
        
        train_df = df[df['time_idx'] <= train_end]
        val_df = df[(df['time_idx'] >= val_start - max_encoder_length) & (df['time_idx'] <= val_end)]
        
        # Prevent unknown category errors
        val_df = val_df[val_df['ticker'].isin(train_df['ticker'].unique())]
        
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
        
        validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_df, stop_randomization=True)
        
        batch_size = 256
        train_loader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=4, pin_memory=True)
        val_loader = validation_dataset.to_dataloader(train=False, batch_size=batch_size*2, num_workers=4, pin_memory=True)
        
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=hparams.get('lr', 0.001),
            hidden_size=hparams.get('hidden_size', 128),
            lstm_layers=2,
            attention_head_size=hparams.get('attention_head_size', 4),
            dropout=hparams.get('dropout', 0.25),
            hidden_continuous_size=max(16, hparams.get('hidden_size', 128) // 2),
            loss=QuantileLoss(),
            log_interval=50,
            reduce_on_plateau_patience=3,
            weight_decay=1e-3  # AdamW style weight decay
        )
        
        patch_mask_bias_for_fp16(tft)
        
        early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, mode="min", verbose=True)
        
        trainer = Trainer(
            max_epochs=15,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            gradient_clip_val=0.1,
            callbacks=[early_stop],
            logger=False,
            enable_checkpointing=False,
            precision="16-mixed" if torch.cuda.is_available() else 32
        )
        
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float('inf'))).item()
        print(f"Window {i+1} Final Val Loss: {val_loss:.4f}")
        cv_results.append(val_loss)
        
        # Predict on out-of-sample block to save for backtester
        with torch.no_grad():
            tft.eval()
            
            # Extract and log Variable Selection Network (VSN) weights for explainability
            print(f"\\n--- Top VSN Encoder Weights (Window {i+1}) ---")
            # We take a small sample batch to run through the model to get attention/VSN weights
            for batch in val_loader:
                x, _ = batch
                out, info = tft(x)
                # v_s = encoder_variables (features)
                # The shape of encoder_variables is typically [batch_size, time_steps, num_features]
                if 'encoder_variables' in info:
                    weights = info['encoder_variables'].mean(dim=(0, 1)).cpu().numpy()
                    features = tft.encoder_variables
                    vsn_df = pd.DataFrame({'feature': features, 'weight': weights}).sort_values('weight', ascending=False)
                    print(vsn_df.head(10))
                break # Just need one batch to get the weights structure
            
            out = tft.predict(val_loader, mode="quantiles", return_index=True)
            preds = out.output.numpy()
            
            for j, ticker in enumerate(out.index['ticker'].values):
                # We store just the median of the horizon for simplicity in backtest tracking
                q50 = preds[j, :, 1].mean()
                out_of_sample_preds.append({
                    'window': i+1,
                    'ticker': ticker,
                    'q50_pred': q50,
                    # We would map time_idx back to real dates here in production
                })
        
        # Free memory
        del tft, trainer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\\n{'='*50}")
    print(" WALK-FORWARD CROSS VALIDATION COMPLETE")
    print(f" Losses: {cv_results}")
    print(f" Mean Loss: {np.mean(cv_results):.4f} +/- {np.std(cv_results):.4f}")
    print(f"{'='*50}")

def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'training_data', 'processed_features.parquet')
    print("Loading features...")
    df = pd.read_parquet(data_path)
    df = df[df['target'].notna()].copy()
    
    categorical_cols = ["ticker", "sector", "industry", "day_of_week", "month", "is_earnings_week", "has_sentiment_history", "quarter_end"]
    for col in categorical_cols:
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
    
    # Add Gaussian Noise Augmentation to numeric features
    noise_factor = 0.01 # 1% noise
    print(f"Applying {noise_factor*100}% Gaussian Noise augmentation to numeric features...")
    for col in real_cols:
        noise = np.random.normal(0, noise_factor, size=len(df))
        df[col] = df[col] * (1 + noise)

    hparams = {'hidden_size': 128, 'dropout': 0.25, 'lr': 0.001, 'attention_head_size': 4}
    
    run_walk_forward(df, real_cols, hparams, num_windows=3)

if __name__ == "__main__":
    main()
