"""
score_filing_summaries.py — Does the tone of NSE's filing summaries predict the price reaction?

1. Scores every unique summary in data/filings_meta.parquet with the same FinancialBERT model
   the original PDF pipeline used (FINAL.PY), giving P(positive), P(negative), P(neutral).
2. Joins the scores to the events from Cleaner/event_study.py (results/event_study/events.parquet)
   and tests, overall and per filing type, whether net tone (positive - negative) predicts the
   abnormal return on days 0-1 (the reaction) and days 2-5 (drift a next-day strategy could catch).

Run event_study.py first. Outputs: data/filings_summary_sentiment.parquet and
results/filing_tone/filing_tone_summary.csv. Usage:
    python Cleaner/score_filing_summaries.py [--batch-size 64]
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch

# Same compatibility patches as FINAL.PY: transformers 5 expects torch >= 2.6 to load .bin weights
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)
try:
    import transformers.utils.import_utils as import_utils
    import transformers.utils as tf_utils
    import transformers.modeling_utils as modeling_utils
    for module in (import_utils, tf_utils, modeling_utils):
        module.check_torch_load_is_safe = lambda: None
except Exception:
    pass
from transformers import pipeline
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILINGS = os.path.join(ROOT, 'data', 'filings_meta.parquet')
EVENTS = os.path.join(ROOT, 'results', 'event_study', 'events.parquet')
SCORES = os.path.join(ROOT, 'data', 'filings_summary_sentiment.parquet')
OUT_DIR = os.path.join(ROOT, 'results', 'filing_tone')
MODEL = "ahmedrachid/FinancialBERT-Sentiment-Analysis"


def score_texts(texts, batch_size):
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("sentiment-analysis", model=MODEL, device=device, top_k=None, truncation=True, max_length=256)
    results = clf(texts, batch_size=batch_size)
    rows = []
    for text, res in zip(texts, results):
        probs = {r['label'].lower(): r['score'] for r in res}
        rows.append({'attchmntText': text, 'pos': probs.get('positive', 0.0),
                     'neg': probs.get('negative', 0.0), 'neu': probs.get('neutral', 0.0)})
    out = pd.DataFrame(rows)
    out['net'] = out['pos'] - out['neg']
    return out


def tstat(s):
    s = s.dropna()
    return s.mean() / (s.std() / np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else np.nan


def main():
    parser = argparse.ArgumentParser(description="Score NSE filing summaries and test their tone against price reactions.")
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    filings = pd.read_parquet(FILINGS)
    filings['attchmntText'] = filings['attchmntText'].fillna('').str.strip()
    texts = sorted(t for t in filings['attchmntText'].unique() if t)
    print(f"Scoring {len(texts)} unique summaries from {len(filings)} filings with {MODEL}...")
    scored = score_texts(texts, args.batch_size)
    per_filing = filings[['ticker', 'seq_id', 'attchmntText']].merge(scored, on='attchmntText', how='left')
    per_filing.drop(columns=['attchmntText']).to_parquet(SCORES, index=False)
    print(f"Saved {SCORES}")

    events = pd.read_parquet(EVENTS)
    events = events.merge(per_filing[['ticker', 'seq_id', 'net', 'pos', 'neg']], on=['ticker', 'seq_id'], how='left')
    events = events.dropna(subset=['net'])

    rows = []
    for event_type, g in [('All filings', events)] + list(events.groupby('event_type')):
        row = {'event_type': event_type, 'events': len(g), 'share_not_neutral': float((g['net'].abs() > 0.5).mean())}
        for col in ['car_0_1', 'car_2_5']:
            valid = g[['net', col]].dropna()
            row[f'spearman_{col}'] = spearmanr(valid['net'], valid[col]).statistic if len(valid) > 10 else np.nan
            # Clearly positive vs clearly negative summaries
            positive, negative = valid[valid['net'] > 0.5][col], valid[valid['net'] < -0.5][col]
            row[f'n_pos_{col}'], row[f'n_neg_{col}'] = len(positive), len(negative)
            row[f'pos_minus_neg_{col}'] = positive.mean() - negative.mean() if len(positive) and len(negative) else np.nan
            row[f'tstat_pos_{col}'] = tstat(positive)
            row[f'tstat_neg_{col}'] = tstat(negative)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, 'filing_tone_summary.csv'), index=False)
    with pd.option_context('display.width', 240, 'display.float_format', '{:.4f}'.format):
        print(summary[['event_type', 'events', 'share_not_neutral', 'spearman_car_0_1', 'pos_minus_neg_car_0_1',
                       'spearman_car_2_5', 'pos_minus_neg_car_2_5', 'n_pos_car_0_1', 'n_neg_car_0_1']].to_string(index=False))
    print(f"\nResults written to {OUT_DIR}")


if __name__ == '__main__':
    main()
