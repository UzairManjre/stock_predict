"""
event_study.py — Which kinds of corporate filing move prices, and when?

Uses data/filings_meta.parquet (Scaper/fetch_filings_metadata.py) and the daily returns in
training_data/processed_features.parquet.

  1. Event type: NSE's own filing category (`desc`) mapped to a handful of groups.
  2. Event day (day 0): the filing's own session if it was disseminated before the 15:30 IST
     close on a trading day, otherwise the next session.
  3. Abnormal return = stock log return - the average log return of all stocks that day, from
     day -1 to day +5.
  4. Per type: mean abnormal return per day, CAR(0,1) (the immediate reaction) and CAR(2,5)
     (drift after the reaction, which a strategy acting the next day could still capture),
     with t-stats. Events are treated as independent, so t-stats are somewhat optimistic.

Outputs go to results/event_study/. Usage:
    python Cleaner/event_study.py
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILINGS = os.path.join(ROOT, 'data', 'filings_meta.parquet')
FEATURES = os.path.join(ROOT, 'training_data', 'processed_features.parquet')
OUT_DIR = os.path.join(ROOT, 'results', 'event_study')
MARKET_CLOSE = pd.Timedelta(hours=15, minutes=30)
WINDOW = range(-1, 6)

# (event type, substrings of NSE's category that select it), checked in order
EVENT_TYPES = [
    ('Results', ['financial result']),
    ('Order win or contract', ['order', 'contract', 'bagging']),
    ('Credit rating', ['credit rating']),
    ('Acquisition or merger', ['acquisition', 'amalgamation', 'merger', 'scheme of arrangement', 'disinvestment', 'takeover']),
    ('Management change', ['change in director', 'key managerial', 'resignation', 'appointment', 'cessation']),
    ('Dividend, buyback or bonus', ['dividend', 'buy back', 'buyback', 'bonus', 'split']),
    ('Fund raising', ['allotment', 'raising', 'preferential', 'qualified institution', 'issue of securities', 'debentures']),
    ('Investor meet or presentation', ['analysts', 'investor meet', 'investor presentation', 'con. call', 'conference call']),
]


def classify(desc):
    text = str(desc).lower()
    for name, keys in EVENT_TYPES:
        if any(k in text for k in keys):
            return name
    return 'Routine or other'


def assign_event_sessions(filings, sessions):
    """Day 0 = the session the market could first react in (same day if before the close)."""
    ts = filings['exchdisstime'].fillna(filings['an_dt'])
    day = ts.dt.normalize()
    after_close = (ts - day) >= MARKET_CLOSE
    session_values = sessions.values
    # First session on or after the filing day...
    pos = np.searchsorted(session_values, day.values, side='left')
    # ...but if it is that same day and the filing came after the close, move one session on
    same_day = (pos < len(sessions)) & (session_values[np.minimum(pos, len(sessions) - 1)] == day.values)
    pos = pos + (same_day & after_close.values).astype(int)
    valid = pos < len(sessions)
    out = filings[valid].copy()
    out['session_idx'] = pos[valid]
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    filings = pd.read_parquet(FILINGS)
    filings['event_type'] = filings['desc'].map(classify)
    prices = pd.read_parquet(FEATURES, columns=['date', 'ticker', 'log_return_1d', 'nifty_50_return'])
    # Benchmark = the average of all stocks that day, not the NIFTY: in this sample the average stock
    # beat the cap-weighted index, which would make every event type look like it had positive drift
    prices['abnormal'] = prices['log_return_1d'] - prices.groupby('date')['log_return_1d'].transform('mean')
    sessions = pd.DatetimeIndex(sorted(prices['date'].unique()))
    session_pos = {d: i for i, d in enumerate(sessions)}
    prices['session_idx'] = prices['date'].map(session_pos)
    ar = prices.set_index(['ticker', 'session_idx'])['abnormal']

    events = assign_event_sessions(filings, sessions)
    # Several filings of one type by one stock on one session count as one event
    events = events.drop_duplicates(subset=['ticker', 'session_idx', 'event_type'])
    print(f"{len(filings)} filings -> {len(events)} events; by type:")
    print(events['event_type'].value_counts().to_string())

    rows = []
    for k in WINDOW:
        idx = pd.MultiIndex.from_arrays([events['ticker'], events['session_idx'] + k])
        events[f'ar_{k}'] = ar.reindex(idx).to_numpy()
    events['car_0_1'] = events['ar_0'] + events['ar_1']
    events['car_2_5'] = events[[f'ar_{k}' for k in range(2, 6)]].sum(axis=1, min_count=4)

    def tstat(s):
        s = s.dropna()
        return s.mean() / (s.std() / np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else np.nan

    for event_type, g in events.groupby('event_type'):
        row = {'event_type': event_type, 'events': len(g)}
        for k in WINDOW:
            row[f'mean_ar_day_{k}'] = g[f'ar_{k}'].mean()
        for col in ['car_0_1', 'car_2_5']:
            row[f'mean_{col}'] = g[col].mean()
            row[f'tstat_{col}'] = tstat(g[col])
            row[f'abs_mean_{col}'] = g[col].abs().mean()   # size of moves, regardless of direction
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values('abs_mean_car_0_1', ascending=False)
    summary.to_csv(os.path.join(OUT_DIR, 'event_study_summary.csv'), index=False)
    events.drop(columns=['attchmntText'], errors='ignore').to_parquet(os.path.join(OUT_DIR, 'events.parquet'), index=False)

    # Baseline: average absolute abnormal return on any stock-day, to judge whether events move prices
    base_abs_2d = (prices.groupby('ticker')['abnormal'].transform(lambda s: s + s.shift(-1))).abs().mean()
    print(f"\nAverage |2-day abnormal return| on an ordinary stock-day: {base_abs_2d:.4f}")
    with pd.option_context('display.width', 220, 'display.float_format', '{:.4f}'.format):
        print(summary[['event_type', 'events', 'mean_car_0_1', 'tstat_car_0_1', 'abs_mean_car_0_1',
                       'mean_car_2_5', 'tstat_car_2_5']].to_string(index=False))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for _, r in summary.iterrows():
        path = np.cumsum([r[f'mean_ar_day_{k}'] for k in WINDOW])
        ax.plot(list(WINDOW), path * 100, marker='o', label=f"{r['event_type']} (n={r['events']})")
    ax.axhline(0, color='#444', linewidth=0.8)
    ax.axvline(0, color='#999', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Trading days relative to the filing (day 0 = first session that could react)')
    ax.set_ylabel('Cumulative abnormal return, %')
    ax.set_title('Average price reaction by filing type (stock minus the average stock)')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'event_study_car.png'), dpi=140)
    print(f"\nResults written to {OUT_DIR}")


if __name__ == '__main__':
    main()
