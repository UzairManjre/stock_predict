import os
import sys
import json
import pickle
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import torch

# Suppress noisy sklearn warnings from PyTorch Forecasting internals
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
# Suppress Lightning model summary warning about 16-mixed precision size estimation
warnings.filterwarnings("ignore", message="Precision 16-mixed is not supported by the model summary")
# Suppress Lightning warning about saving nn.Module in hyperparameters
warnings.filterwarnings("ignore", message=".*is an instance of `nn.Module` and is already saved during checkpointing.*")

# Load env variables from .env
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()

# Fallback imports for PyTorch Lightning
try:
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    print("Using 'lightning.pytorch' for training.")
except ImportError:
    try:
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger
        print("Using legacy 'pytorch_lightning' for training.")
    except ImportError:
        print("[ERROR] PyTorch Lightning/Lightning not found. Please install dependencies.")
        sys.exit(1)

# Try setting up MLflow
try:
    import mlflow
    import mlflow.pytorch
    mlflow.pytorch.autolog(log_every_n_epoch=1)
    MLFLOW_AVAILABLE = True
    print("MLflow experiment autologging enabled.")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not installed. Experiment tracking will log locally via CSVLogger only.")

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import EncoderNormalizer, TorchNormalizer

def patch_mask_bias_for_fp16(model):
    """
    Finds all 'mask_bias' attributes in MultiheadAttention and forces them
    to -65500.0. This prevents float16 overflow crashes when running in mixed-precision.
    """
    count = 0
    # Determine the actual device the model is currently on
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    
    for name, module in model.named_modules():
        if hasattr(module, "mask_bias"):
            if isinstance(module.mask_bias, (float, int)):
                module.mask_bias = -65500.0
                count += 1
            elif hasattr(module.mask_bias, "data"):
                # Force the mask_bias onto the model's actual device, not its previous device
                module.mask_bias = torch.tensor(-65500.0, dtype=module.mask_bias.dtype, device=device)
                count += 1
    print(f"Patched {count} multihead attention mask_bias parameters for float16 mixed-precision on {device}.")

def run_simple_backtest(df, preds, index_df, transaction_cost=0.0012):
    """
    Performs a transaction-cost adjusted, volatility-scaled long-only portfolio backtest.
    Position sizing is scaled inversely to quantile spread (q90-q10): tighter spread
    (higher model confidence) → larger position, wider spread → smaller position.
    Transaction cost default is 0.12% round-trip (STT + brokerage + SEBI fees on NSE).
    """
    print(f"\nStarting backtest on Test Fold [friction={transaction_cost:.4%} round-trip]...")
    
    # preds shape: [num_samples, prediction_length, num_quantiles]
    #   quantile idx: 0=q10, 1=q50 (median), 2=q90
    q10 = preds[..., 0].mean(axis=1).cpu().numpy()
    q50 = preds[..., 1].mean(axis=1).cpu().numpy()   # median return forecast
    q90 = preds[..., 2].mean(axis=1).cpu().numpy()

    spread = q90 - q10  # uncertainty band; used for position sizing

    # Map back to group indices
    map_df = index_df.copy()
    if 'time_idx' not in map_df.columns:
        # decoded_index names the predicted (decoder) step time_idx_first_prediction
        map_df['time_idx'] = map_df['time_idx_first_prediction']
    map_df['q10']   = q10
    map_df['q50']   = q50
    map_df['q90']   = q90
    map_df['spread'] = spread

    # Merge with original data to get 1d returns and nifty returns
    aligned = pd.merge(
        map_df,
        df[['ticker', 'time_idx', 'date', 'log_return_1d', 'nifty_50_return']],
        on=['ticker', 'time_idx'],
        how='inner'
    )

    # ── Volatility-scaled position sizing ────────────────────────────────────
    # Go long if q50 > 0; size inversely proportional to spread.
    # A tight quantile band (high confidence) → full position.
    # A wide band (uncertainty) → fraction of full position.
    # Normalize so the median spread maps to a weight of 1.0.
    median_spread = aligned['spread'].median()
    aligned['pos_size'] = np.where(
        aligned['q50'] > 0,
        np.clip(median_spread / (aligned['spread'] + 1e-8), 0.1, 1.0),
        0.0
    )
    aligned['signal'] = (aligned['q50'] > 0).astype(int)
    
    # Execution: signal at close of t → position held overnight, booked at t+1 close
    aligned['prev_signal']   = aligned.groupby('ticker')['signal'].shift(1).fillna(0)
    aligned['prev_pos_size'] = aligned.groupby('ticker')['pos_size'].shift(1).fillna(0.0)

    # Friction: charged on position change (entry + exit)  
    aligned['trades'] = (aligned['signal'] != aligned['prev_signal']).astype(int)
    aligned['strat_return'] = (
        aligned['prev_pos_size'] * aligned['log_return_1d']
        - aligned['trades'] * transaction_cost
    )
    
    # Aggregate portfolio daily returns
    daily_perf = aligned.groupby('date').agg(
        strat_return=('strat_return', 'mean'),
        nifty_return=('nifty_50_return', 'first')
    ).reset_index()
    
    # Performance metrics
    daily_perf['cum_strat'] = (1 + daily_perf['strat_return']).cumprod() - 1
    daily_perf['cum_nifty'] = (1 + daily_perf['nifty_return']).cumprod() - 1
    
    strat_total = daily_perf['cum_strat'].iloc[-1]
    nifty_total = daily_perf['cum_nifty'].iloc[-1]
    
    # Sharpe Ratio (annualized, assuming 252 trading days)
    strat_std = daily_perf['strat_return'].std()
    strat_mean = daily_perf['strat_return'].mean()
    sharpe = (strat_mean / (strat_std + 1e-8)) * np.sqrt(252) if strat_std > 0 else 0
    
    # Drawdown
    strat_cum = daily_perf['cum_strat'] + 1
    peaks = strat_cum.cummax()
    drawdowns = (strat_cum - peaks) / peaks
    max_dd = drawdowns.min()
    
    actual = aligned['target_actual'] if 'target_actual' in aligned.columns else aligned['q50'] * 0.0 # Placeholder
    
    # Calculate advanced metrics if target is present
    has_target = 'target' in df.columns
    if has_target:
        # Re-merge to get the actual target to compare with q50 (q50 is now Alpha target)
        aligned = pd.merge(aligned, df[['ticker', 'time_idx', 'target']], on=['ticker', 'time_idx'], how='left')
        actual = aligned['target']
        mae = np.abs(aligned['q50'] - actual).mean()
        dir_acc = np.mean(np.sign(aligned['q50']) == np.sign(actual))
        q_cov = np.mean((actual >= aligned['q10']) & (actual <= aligned['q90']))
    
    print("================ BACKTEST PERFORMANCE SUMMARY ================")
    print(f"  Backtest Period:      {daily_perf['date'].min().strftime('%Y-%m-%d')} to {daily_perf['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Strategy Return:      {strat_total:.2%}")
    print(f"  NIFTY 50 Return:      {nifty_total:.2%}")
    print(f"  Annualized Sharpe:    {sharpe:.2f}")
    print(f"  Max Peak Drawdown:    {max_dd:.2%}")
    if has_target:
        print(f"  Alpha MAE:            {mae:.4f}")
        print(f"  Direction Accuracy:   {dir_acc:.2%}")
        print(f"  Q-Band Coverage:      {q_cov:.2%}")
    print("==============================================================")
    
    # Save backtest results
    perf_file = os.path.join(os.path.dirname(__file__), '..', 'training_data', 'backtest_perf.csv')
    daily_perf.to_csv(perf_file, index=False)
    print(f"Saved daily equity curve to {perf_file}")

def main():
    # Check and display GPU status
    cuda_avail = torch.cuda.is_available()
    print("====================================================")
    print(f"  CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"  GPU Device:     {torch.cuda.get_device_name(0)}")
        print("  Status:         TFT will execute on the GPU.")
    else:
        print("  [WARNING] CUDA not detected. Running on CPU.")
    print("====================================================")

    # Command line argument parsing for retraining automation
    import argparse
    parser = argparse.ArgumentParser(description="Train Temporal Fusion Transformer on stock data.")
    parser.add_argument("--train-end", type=str, default="2024-07-01", help="Train end date (YYYY-MM-DD)")
    parser.add_argument("--val-end", type=str, default="2025-07-01", help="Validation end date (YYYY-MM-DD)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--gradient-clip", type=float, default=0.1, help="Gradient norm clipping value")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--experiment", action="store_true",
                        help="Do not register the checkpoint for predict.py (trial runs)")
    args = parser.parse_args()

    # 1. Load preprocessed features
    processed_file = os.path.join(os.path.dirname(__file__), '..', 'training_data', 'processed_features.parquet')
    if not os.path.exists(processed_file):
        print(f"[ERROR] Preprocessed features file not found at {processed_file}. Please run build_features.py first.")
        sys.exit(1)
        
    print(f"Loading preprocessed features from {processed_file}...")
    df = pd.read_parquet(processed_file)
    print(f"Loaded dataset of shape {df.shape}")

    # The TFT decoder step is the session after the encoder ends, so row t+1 carries the excess
    # return over (t, t+5]: the same quantity LightGBM predicts from the features at t.
    df = df.sort_values(['ticker', 'time_idx'])
    df['target'] = df.groupby('ticker')['target_5d'].shift(1)
    # Drop rows whose target is unknown (first row per stock, last sessions without 5 days of future)
    df = df[df['target'].notna()]
    print(f"Dataset shape after dropping NaN targets: {df.shape}")

    # Ensure correct data types
    df['ticker'] = df['ticker'].astype(str)
    df['sector'] = df['sector'].astype(str)
    df['industry'] = df['industry'].astype(str)
    df['day_of_week'] = df['day_of_week'].astype(str)
    df['month'] = df['month'].astype(str)
    df['is_earnings_week'] = df['is_earnings_week'].astype(str)
    df['has_sentiment_history'] = df['has_sentiment_history'].astype(str)
    df['quarter_end'] = df['quarter_end'].astype(str)
    
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

    # 2. Chronological Splits with 5-day Purged Embargo Gap
    print("Splitting dataset chronologically with 5-day purged boundaries...")
    max_encoder_length = 60    # ~3 trading months lookback
    max_prediction_length = 1  # one decoder step; it carries the 5-day excess return
    TARGET_HORIZON = 5         # sessions covered by the target, and the embargo between splits

    train_end_idx = df[df['date'] < args.train_end]['time_idx'].max()
    val_end_idx = df[df['date'] < args.val_end]['time_idx'].max()

    # Embargo: no target window may straddle two splits
    val_start_idx = train_end_idx + TARGET_HORIZON + 1
    test_start_idx = val_end_idx + TARGET_HORIZON + 1
    
    train_df = df[df['time_idx'] <= train_end_idx].copy()
    val_df = df[(df['time_idx'] >= val_start_idx) & (df['time_idx'] <= val_end_idx)].copy()
    test_df = df[df['time_idx'] >= test_start_idx].copy()
    
    print(f"  Train records: {len(train_df)} (Up to idx {train_end_idx})")
    print(f"  Val records:   {len(val_df)} (Idx {val_start_idx} to {val_end_idx})")
    print(f"  Test records:  {len(test_df)} (From idx {test_start_idx} onwards)")
    
    # Include lookback history for validation and test sequence initialization
    val_df_with_history = pd.concat([
        train_df.groupby('ticker').tail(max_encoder_length),
        val_df
    ]).sort_values(['ticker', 'time_idx'])
    
    test_df_with_history = pd.concat([
        val_df_with_history.groupby('ticker').tail(max_encoder_length),
        test_df
    ]).sort_values(['ticker', 'time_idx'])

    # 3. Define TimeSeriesDataSet
    print("Building TimeSeriesDataSet with robust PyTorch-native scalers...")

    # GroupNormalizer: normalizes the target per-ticker using mean+std of that ticker's train returns.
    # This is leakage-free and far superior to a global StandardScaler which would let high-volatility
    # stocks dominate the scale for all other stocks.
    from pytorch_forecasting.data import GroupNormalizer
    target_normalizer = GroupNormalizer(
        groups=["ticker"],
        center=True,
    )

    # Feature scaling: using default StandardScaler (sklearn warnings are suppressed above).
    # TorchNormalizer cannot be used here as TimeSeriesDataSet does not call .fit() on scalers dict entries.

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
        time_varying_known_categoricals=["day_of_week", "month", "quarter_end"],
        # Derived from actual filings, so it is not known in advance (was a look-ahead leak as "known")
        time_varying_unknown_categoricals=["is_earnings_week"],
        time_varying_unknown_reals=real_cols,

        # GroupNormalizer for target: normalizes per-ticker (leakage-free, per-stock scale)
        # Feature scalers: default StandardScaler (warnings suppressed at top of file)
        target_normalizer=target_normalizer,

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_df_with_history, stop_randomization=True)
    test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_df_with_history, stop_randomization=True)

    # Convert to dataloaders with GPU-optimized settings
    print("Creating PyTorch Dataloaders (num_workers=4, pin_memory=True)...")
    batch_size = 256
    train_dataloader = training_dataset.to_dataloader(
        batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_dataloader = validation_dataset.to_dataloader(
        batch_size=batch_size * 2, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    test_dataloader = test_dataset.to_dataloader(
        batch_size=batch_size * 2, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    # Load optimal hyperparameters from optuna if available
    hparams_path = os.path.join(os.path.dirname(__file__), '..', 'training_data', 'best_hparams.json')
    hparams = {}
    if os.path.exists(hparams_path):
        print(f"Loading optimized hyperparameters from {hparams_path}...")
        try:
            with open(hparams_path, 'r') as f:
                hparams = json.load(f)
            print(f"Loaded params: {hparams}")
        except Exception as e:
            print(f"Failed to read tuning file, falling back to default hyperparameters: {e}")

    # Set parameters with fallback to VRAM-friendly values
    # hidden_size=256, attention_head_size=4, lstm_layers=2 gives ~6-7M params
    # GTX 1660 SUPER uses only ~1.9GB at hidden_size=128 so 256 has plenty of headroom
    hidden_size = hparams.get("hidden_size", 256)
    dropout = hparams.get("dropout", 0.1)
    lr = hparams.get("lr", 1e-3)
    attention_head_size = hparams.get("attention_head_size", 4)
    hidden_continuous_size = max(32, hidden_size // 2)

    # 4. Instantiate TFT Model
    print("Instantiating Temporal Fusion Transformer (TFT)...")
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=lr,
        hidden_size=hidden_size,
        lstm_layers=2,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]), # Predict 10th, 50th, 90th percentiles
        log_interval=50,
        reduce_on_plateau_patience=5,
        weight_decay=args.weight_decay
    )
    print(f"Number of parameters in TFT: {tft.size() / 1e3:.1f}k")

    # Apply float16 mixed precision patch to allow training without overflow crashes
    patch_mask_bias_for_fp16(tft)

    # 5. Training Setup
    logger = CSVLogger("csv_logs", name="tft_stock_predictor")
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-4, 
        patience=5, 
        verbose=True, 
        mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
        mode="min",
        filename="best-tft-{epoch:02d}-{val_loss:.4f}"
    )

    print("Configuring Trainer...")
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=args.gradient_clip,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        precision="16-mixed" if torch.cuda.is_available() else 32
    )

    # Auto-detect latest checkpoint to resume training
    import glob
    last_ckpt_path = None
    # last_ckpts = glob.glob(os.path.join("csv_logs", "tft_stock_predictor", "**", "last.ckpt"), recursive=True)
    # if last_ckpts:
    #     last_ckpt_path = max(last_ckpts, key=os.path.getmtime)
    #     print(f"Resuming training from latest checkpoint: {last_ckpt_path}")
    print("Training from scratch (checkpoint resume disabled due to architecture changes).")

    # Run with MLflow tracking if available
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="tft_quantile_pipeline"):
            print("Starting training loop (with MLflow tracking)...")
            trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=last_ckpt_path)
    else:
        print("Starting training loop...")
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=last_ckpt_path)

    print("Training finished! Loading best model weights...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Best model saved at: {best_model_path}")

        # Load checkpoint and explicitly move to GPU before patching
        device = torch.device("cuda" if cuda_avail else "cpu")
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        best_tft = best_tft.to(device)
        best_tft.eval()

        # Apply float16 mask_bias patch AFTER moving to the final device
        patch_mask_bias_for_fp16(best_tft)

        # ── 6. Evaluation ─────────────────────────────────────────────────────
        # We call model.forward() directly inside a plain torch.no_grad() loop.
        # This completely bypasses Lightning trainers, MLflow autolog hooks, and
        # pytorch_forecasting's predict_step callback machinery — all of which
        # have proven unreliable in this environment.
        print("Evaluating model performance on Validation and Out-Of-Sample Test Sets...")

        # Build fresh single-worker dataloaders for eval — multi-worker loaders
        # with pin_memory can silently drop the last partial batch, causing
        # mismatches between predictions and index lengths.
        eval_batch_size = 512
        val_eval_loader  = validation_dataset.to_dataloader(batch_size=eval_batch_size, shuffle=False, num_workers=0, drop_last=False)
        test_eval_loader = test_dataset.to_dataloader(     batch_size=eval_batch_size, shuffle=False, num_workers=0, drop_last=False)

        def _manual_eval(tft_model, dataloader, dev):
            """
            Direct forward-pass evaluation. No Lightning, no callbacks, no MLflow.
            Returns (preds_median, targets) both as float32 CPU tensors.
              preds_median : [N, horizon]   — median quantile (idx 1 of 3)
              targets      : [N, horizon]
            """
            preds_list, targets_list = [], []
            with torch.no_grad():
                for x_batch, y_batch in dataloader:
                    # Move every tensor in the input dict to the model device
                    x_dev = {
                        k: v.to(dev) if isinstance(v, torch.Tensor) else v
                        for k, v in x_batch.items()
                    }
                    out = tft_model(x_dev)  # returns NamedTuple / dict
                    # Support both dict-style and attribute-style access
                    pred_tensor = out['prediction'] if isinstance(out, dict) else out.prediction
                    preds_list.append(pred_tensor.cpu())
                    targets_list.append(y_batch[0].cpu())
            preds   = torch.cat(preds_list,   dim=0).float()  # [N, horizon, Q]
            targets = torch.cat(targets_list, dim=0).float()  # [N, horizon]
            return preds, targets

        val_preds,  val_target  = _manual_eval(best_tft, val_eval_loader,  device)
        test_preds, test_target = _manual_eval(best_tft, test_eval_loader, device)
        val_preds_median  = val_preds[..., 1]   # quantile idx: 0=q10, 1=q50, 2=q90
        test_preds_median = test_preds[..., 1]

        # Guard against drop_last length mismatch
        n_val  = min(len(val_preds_median),  len(val_target))
        n_test = min(len(test_preds_median), len(test_target))
        val_mae  = (val_preds_median[:n_val]   - val_target[:n_val]).abs().mean().item()
        test_mae = (test_preds_median[:n_test] - test_target[:n_test]).abs().mean().item()

        print("\n================ EVALUATION SUMMARY ================")
        print(f"  Best Model Path: {best_model_path}")
        print(f"  Validation MAE (Median):  {val_mae:.5f}")
        print(f"  Test (OOS) MAE (Median):  {test_mae:.5f}")
        print("====================================================")

        # ── 7. Attention Interpretation ───────────────────────────────────────
        print("\nRunning attention interpretation on validation set...")
        try:
            # collect raw forward-pass outputs (with extra keys like attention weights)
            raw_list = []
            with torch.no_grad():
                for x_batch, _ in val_eval_loader:
                    x_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in x_batch.items()}
                    raw_list.append({k: v.cpu() for k, v in best_tft(x_dev).items()})
            # Concatenate across batches
            raw_out = {k: torch.cat([b[k] for b in raw_list], dim=0) for k in raw_list[0]}
            interpretation = best_tft.interpret_output(raw_out, reduction='sum')
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            figs = best_tft.plot_interpretation(interpretation)
            interp_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'interpretation')
            if args.experiment:
                # Experiments must not overwrite the plots of the registered model
                run_name = os.path.basename(os.path.dirname(os.path.dirname(best_model_path)))
                interp_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'tft_experiments', run_name)
            os.makedirs(interp_dir, exist_ok=True)
            for name, fig in figs.items():
                out_path = os.path.join(interp_dir, f'{name}.png')
                fig.savefig(out_path, bbox_inches='tight', dpi=120)
                plt.close(fig)
                print(f"  Saved: {out_path}")
            print("Attention interpretation complete.")
        except Exception as e:
            print(f"[WARN] Attention interpretation failed: {e}")

        # ── 8. Backtest ───────────────────────────────────────────────────────
        # Use all three quantiles from _manual_eval so the spread-based position sizing works,
        # aligned with the dataset's stored index.
        try:
            test_preds_bt = test_preds
            # Trim index to match actual prediction count (multi-worker dataloaders
            # may silently drop the last partial batch, producing fewer predictions
            # than dataset.index rows).
            n_preds = len(test_preds_median)
            test_index_aligned = test_dataset.decoded_index.iloc[:n_preds].copy()
            run_simple_backtest(df, test_preds_bt, test_index_aligned)
        except Exception as e:
            print(f"[WARN] Backtest failed: {e}")

        # 9. Save artifacts needed for inference pipeline
        print("\nSaving inference artifacts...")
        artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)

        # 9a. Pickle the training TimeSeriesDataSet (contains fitted scaler params)
        dataset_path = os.path.join(artifacts_dir, 'training_dataset.pkl')
        with open(dataset_path, 'wb') as f:
            pickle.dump(training_dataset, f)
        print(f"  Saved training_dataset.pkl → {dataset_path}")

        # 9b. Save date-to-time_idx mapping for live data alignment
        date_to_idx = {str(d): int(i) for d, i in zip(
            df['date'].unique(), df.groupby('date')['time_idx'].first()
        )}
        date_map_path = os.path.join(artifacts_dir, 'date_to_idx.json')
        with open(date_map_path, 'w') as f:
            json.dump(date_to_idx, f)
        print(f"  Saved date_to_idx.json → {date_map_path}")

        # 9c. Save feature distribution baseline for KS drift detection
        NUMERIC_FEATURES = [
            'adj_close', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'volume_spike', 'nifty_50_return',
            'decayed_net_sentiment_score', 'log_return_1d', 'realized_vol_20d',
            'atr_14', 'sector_relative_return_1d', 'sentiment_volume_7d',
            'stoch_rsi_14', 'bb_width', 'price_vs_sma50', 'price_vs_52w_high', 'momentum_10d', 'obv_normalized', 'volatility_regime',
            'pvt_normalized', 'rsi_divergence', 'nifty_volatility_20d', 'sector_volatility_20d'
        ]
        feature_stats = {}
        for col in NUMERIC_FEATURES:
            if col in train_df.columns:
                vals = train_df[col].dropna().astype(float)
                # Store a random 2000-sample reservoir to keep file small
                sample = vals.sample(min(2000, len(vals)), random_state=42).tolist()
                feature_stats[col] = {
                    'mean':    float(vals.mean()),
                    'std':     float(vals.std()),
                    'p25':     float(vals.quantile(0.25)),
                    'p75':     float(vals.quantile(0.75)),
                    'samples': sample
                }
        stats_path = os.path.join(artifacts_dir, 'feature_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(feature_stats, f)
        print(f"  Saved feature_stats.json → {stats_path}")

        # 9d. Record which checkpoint predict.py should load
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        best_score = checkpoint_callback.best_model_score
        registry = {
            'tft_checkpoint': os.path.relpath(os.path.abspath(best_model_path), project_root),
            'target': 'excess log return vs NIFTY 50 over (t, t+5], on the decoder step t+1',
            'val_loss': float(best_score) if best_score is not None else None,
            'train_end': args.train_end,
            'val_end': args.val_end,
            'trained_at': datetime.now().isoformat(timespec='seconds'),
        }
        registry_path = os.path.join(artifacts_dir, 'model_registry.json')
        if args.experiment:
            print(f"  Experiment run: checkpoint NOT registered ({registry['tft_checkpoint']})")
        else:
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            print(f"  Saved model_registry.json → {registry_path}")
        print("All inference artifacts saved. ✓")
        
    else:
        print("[WARNING] No checkpoint found. Evaluation and Backtest skipped.")

if __name__ == '__main__':
    main()
