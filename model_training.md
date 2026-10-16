# Software Design Document (SDD)

**Project:** Hybrid Temporal Fusion Transformer for Indian Equity Markets  
**Core Stack:** Python, PostgreSQL, PyTorch Forecasting, FinBERT  

---

## 1. Executive Summary

This document outlines the architecture of a quantitative trading pipeline designed to predict stock price movements (5 days ahead) across 209 Indian equities. 

The pipeline trains a single, global **Temporal Fusion Transformer (TFT)**. To handle the high sparsity of corporate filings (96.68% of market days have no filings) and hardware constraints (6GB VRAM GPU), the system implements a Stage-1 Data Engine to model announcement decay and a memory-efficient Stage-2 training profile.

---

## 2. System Architecture: The Two-Stage Pipeline

```
+-----------------------------------+     +-----------------------------------+
|     PostgreSQL (market_data)      |     |    PostgreSQL (sentiment_aggr)    |
+-----------------+-----------------+     +-----------------+-----------------+
                  |                                         |
                  |                                         | Shift by +1 day (no lookahead)
                  |                                         v
                  |                               +---------+---------+
                  |                               | Exponential Decay |
                  |                               +---------+---------+
                  |                                         |
                  +--------------------+--------------------+
                                       |
                                       v  (Join on Date & Ticker)
                        +--------------+--------------+
                        |    Stage 1: Feature Engine  | -> Materializes Technical Indicators,
                        +--------------+--------------+    NIFTY50 Returns & Static Metadata.
                                       |
                                       v  (Output: processed_features.parquet)
                        +--------------+--------------+
                        |   Stage 2: Training Engine  | -> Walk-Forward Validation & Lightweight
                        +-----------------------------+    TFT Model Fit (VRAM optimized).
```

### Stage 1: Data Engine (`Cleaner/build_features.py`)
Responsible for data cleanup, feature engineering, and alignment. It joins `market_data_daily` and `daily_sentiment_aggregates` into a dense, lookahead-bias-free feature matrix, then writes to Parquet.

### Stage 2: Model Engine (`Cleaner/train_tft.py`)
Consumes the Parquet feature matrix, constructs PyTorch Forecasting `TimeSeriesDataSet` instances, and trains a lightweight TFT model tailored to run efficiently under 6GB VRAM.

---

## 3. Stage 1: Data Engineering Implementation

### 3.1. Lookahead Bias Mitigation (The 1-Day Shift)
Historically, corporate reports are stored with a midnight placeholder (`00:00:00`) for `received_time`. Because we cannot determine the exact intraday filing hour, we prevent lookahead bias by shifting all sentiment indicators by **+1 trading day**. 
* **Rule**: News sentiment from announcements filed on Date $D$ is only available to inform trading from Date $D+1$ onwards.

### 3.2. Dynamic Per-Ticker Sentiment Decay
Corporate announcements are episodic. High-cap liquid stocks (e.g. RELIANCE, TCS) absorb sentiment quickly, while mid-caps and small-caps take longer.
Instead of a global decay, we assign decay constants dynamically based on the stock's median daily dollar volume:
* **Large-cap (Top 20% by dollar volume)**: half-life of 1.5 trading days.
* **Small-cap (Bottom 20% by dollar volume)**: half-life of 6.0 trading days.
* **Mid-cap (Middle 60%)**: half-life of 4.0 trading days.

The decay formula remains:
$$S_t = S_{t-1} \times e^{-\lambda}$$
where $\lambda = \ln(2) / \text{half-life}$. Tickers with no announcement history receive a static indicator `has_sentiment_history = False`.

### 3.3. Technical, Corporate Action & Microstructure Indicators
Calculated per stock prior to feeding the TFT. All indicators are calculated on split/dividend adjusted price fields (`adj_close`, `adj_open`, `adj_high`, `adj_low`) to prevent artificial returns spikes from corporate actions:
* **Log Returns**: 1-day, 3-day, and 5-day log returns on adjusted prices.
* **Momentum**: 14-day RSI (Wilder's EMA method), MACD on adjusted prices.
* **Volume Spike**: Daily volume relative to its 30-day moving average.
* **Index Reference**: Daily return of the NIFTY 50 index (`NIFTY50`), capturing market-wide beta.
* **Realized Volatility (20d)**: 20-day rolling standard deviation of 1d log returns.
* **Average True Range (ATR_14)**: Average True Range computed over adjusted price ranges.
* **Sector-Relative Return**: stock return minus sector average return (isolating idiosyncratic alpha).
* **Sentiment Volume (7d)**: 7-day rolling sum of chunk counts.

---

## 4. Stage 2: Temporal Fusion Transformer Implementation

### 4.1. Input Variable Routing
The TFT routes features into specialized encoders:

1. **Static Covariates (Assets metadata)**:
   * `ticker`: Categorical ID (209 equities).
   * `sector`: Categorical NSE sector.
   * `industry`: Categorical NSE industry.
   * `has_sentiment_history`: Binary flag.
2. **Time-Varying Known Inputs (Deterministic future indices)**:
   * `day_of_week`: Categorical (0-6).
   * `month`: Categorical (1-12).
   * `quarter_end`: Categorical flag (1/0) for quarter-end sessions.
   * `is_earnings_week`: Binary indicator (inferred from decayed announcement volume).
3. **Time-Varying Unknown Inputs (Past observed data)**:
   * `adj_close` (normalized scale)
   * `volume`
   * `rsi_14`, `macd`, `volume_spike`
   * `nifty_50_return` (Index context)
   * `decayed_net_sentiment_score`, `decayed_bullish_intensity`, `decayed_bearish_intensity`
   * `realized_vol_20d`, `atr_14`, `sector_relative_return_1d`, `sentiment_volume_7d`
4. **Target**:
   * `target` ($5$-day ahead log return on adjusted prices: $\ln(\text{adj\_close}_{t+5} / \text{adj\_close}_t)$)

### 4.2. VRAM Optimization & Training Blueprint
To fit the model on a **GTX 1660 SUPER (6GB VRAM)**:
* `hidden_size = 32` (scales memory footprint down)
* `attention_head_size = 2`
* `batch_size = 32`
* Precision: **32-bit single precision** (`precision=32` to avoid CUDA mixed precision float16 mask bias overflow)
* Dataloaders: `num_workers = 0` (avoids Windows spawn multiprocessing bottlenecks)

---

## 5. Walk-Forward Validation Strategy

To prevent temporal data leakage, we split data chronologically:
1. **Train Fold**: 2020-09-21 to 2024-06-30
2. **Validation Fold**: 2024-07-01 to 2025-06-30
3. **Out-of-Sample Test Fold**: 2025-07-01 to 2025-09-19

For validation and test datasets, we prepend `max_encoder_length` (21 trading days) of history from the previous split to allow sequence initialization.

---

## 6. Execution Instructions

1. **Fetch NIFTY 50 Index prices**:
   ```bash
   python Cleaner/fetch_nifty50.py
   ```
2. **Materialize Feature Matrix**:
   ```bash
   python Cleaner/build_features.py
   ```
3. **Run TFT Training & Evaluation**:
   ```bash
   python Cleaner/train_tft.py
   ```