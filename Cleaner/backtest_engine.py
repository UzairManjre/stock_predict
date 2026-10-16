import pandas as pd
import numpy as np

def run_backtest(predictions_df, market_df, nifty_df):
    """
    predictions_df: DataFrame with columns [date, ticker, q50_pred, q10_pred, q90_pred]
    market_df: Raw market data with [date, ticker, open_price, close_price, volume]
    nifty_df: Nifty 50 market data with [date, close_price]
    """
    print("Initializing Institutional Backtest Engine...")
    
    # 1. Merge predictions with forward looking market data
    # We enter at OPEN the day after prediction, exit at CLOSE 5 days later.
    
    # Sort and align
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    market_df['date'] = pd.to_datetime(market_df['date'])
    nifty_df['date'] = pd.to_datetime(nifty_df['date'])
    
    # Calculate Nifty returns for benchmarking
    nifty_df['nifty_ret_1d'] = nifty_df['close_price'].pct_change()
    nifty_daily_returns = nifty_df.set_index('date')['nifty_ret_1d']
    
    # Simulate portfolio
    portfolio_value = 1000000.0  # 1M starting capital
    positions = {}
    
    # Cost modeling
    BROKERAGE_BPS = 3      # 0.03%
    STT_BPS = 10           # 0.1%
    SLIPPAGE_BPS = 5       # 0.05%
    TOTAL_FRICTION = (BROKERAGE_BPS + STT_BPS + SLIPPAGE_BPS) / 10000.0
    
    dates = sorted(predictions_df['date'].unique())
    daily_pnl = []
    
    for i, current_date in enumerate(dates):
        if i >= len(dates) - 5:
            break
            
        entry_date = dates[i+1] # Enter next day
        exit_date = dates[i+5]  # Exit 5 days later (Horizon)
        
        day_preds = predictions_df[predictions_df['date'] == current_date]
        
        # Portfolio logic: Buy top 10 stocks predicted to have Alpha > 1%
        long_candidates = day_preds[(day_preds['q50_pred'] > 0.01)].sort_values('q50_pred', ascending=False).head(10)
        
        capital_per_trade = (portfolio_value * 0.95) / 10.0 # 10% per position
        
        day_pnl = 0
        for _, row in long_candidates.iterrows():
            ticker = row['ticker']
            
            # Fetch execution prices
            entry_row = market_df[(market_df['ticker'] == ticker) & (market_df['date'] == entry_date)]
            exit_row = market_df[(market_df['ticker'] == ticker) & (market_df['date'] == exit_date)]
            
            if entry_row.empty or exit_row.empty:
                continue
                
            entry_price = entry_row['open_price'].values[0]
            exit_price = exit_row['close_price'].values[0]
            
            # Apply friction
            entry_price *= (1 + TOTAL_FRICTION)
            exit_price *= (1 - TOTAL_FRICTION)
            
            trade_return = (exit_price - entry_price) / entry_price
            trade_pnl = capital_per_trade * trade_return
            day_pnl += trade_pnl
            
        portfolio_value += day_pnl
        daily_pnl.append({
            'date': exit_date,
            'portfolio_value': portfolio_value,
            'pnl': day_pnl,
            'pct_return': day_pnl / portfolio_value
        })
        
    res_df = pd.DataFrame(daily_pnl)
    res_df.set_index('date', inplace=True)
    
    # 2. Compute Risk Metrics
    mean_daily_ret = res_df['pct_return'].mean()
    std_daily_ret = res_df['pct_return'].std()
    
    annualized_return = mean_daily_ret * 252
    annualized_vol = std_daily_ret * np.sqrt(252)
    
    risk_free_rate = 0.065 # 6.5% Indian risk-free rate
    sharpe_ratio = (annualized_return - risk_free_rate) / (annualized_vol + 1e-8)
    
    # Max Drawdown
    cum_returns = (1 + res_df['pct_return']).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
    
    # Information Ratio (vs Nifty 50)
    res_df = res_df.join(nifty_daily_returns, how='left')
    active_returns = res_df['pct_return'] - res_df['nifty_ret_1d'].fillna(0)
    tracking_error = active_returns.std() * np.sqrt(252)
    info_ratio = (annualized_return - (res_df['nifty_ret_1d'].mean()*252)) / (tracking_error + 1e-8)
    
    print("\\n" + "="*40)
    print(" INSTITUTIONAL BACKTEST REPORT")
    print("="*40)
    print(f" Total Return:      {((portfolio_value/1000000.0) - 1)*100:.2f}%")
    print(f" Annualized Return: {annualized_return*100:.2f}%")
    print(f" Annualized Vol:    {annualized_vol*100:.2f}%")
    print(f" Max Drawdown:      {max_drawdown*100:.2f}%")
    print("-" * 40)
    print(f" Sharpe Ratio:      {sharpe_ratio:.2f}")
    print(f" Calmar Ratio:      {calmar_ratio:.2f}")
    print(f" Information Ratio: {info_ratio:.2f}")
    print("="*40)

if __name__ == "__main__":
    print("Backtest engine initialized. Ready to process walk-forward predictions.")
