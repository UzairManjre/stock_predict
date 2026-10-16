"""
download_filing_pdfs.py — Download the original filing PDFs listed in data/filings_meta.parquet.

By default only results filings are fetched: NSE's "Financial Result" categories, plus
"Outcome of Board Meeting" filings whose summary mentions financial results (how NSE files
most results since 2025). Files land in data/filing_pdfs/<ticker>/<seq_id>.<ext>, and every
attempt is logged to data/filing_pdfs/manifest.csv, so a rerun skips what is already done.

Run Scaper/fetch_filings_metadata.py first. Usage:
    python Scaper/download_filing_pdfs.py [--all] [--from-date 2024-01-01] [--limit 100] [--max-gb 20] [--delay 0.5]

Newest filings come first; --max-gb stops the run once this many GB are on disk (results PDFs average ~8 MB).
"""

import os
import time
import argparse
import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
META = os.path.join(ROOT, 'data', 'filings_meta.parquet')
OUT_DIR = os.path.join(ROOT, 'data', 'filing_pdfs')
MANIFEST = os.path.join(OUT_DIR, 'manifest.csv')
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/pdf,application/octet-stream,*/*',
    'Referer': 'https://www.nseindia.com/',
}


def select_filings(meta, include_all, from_date):
    meta = meta[meta['attchmntFile'].fillna('').str.startswith('http')]
    if from_date:
        meta = meta[meta['an_dt'] >= pd.Timestamp(from_date)]
    if include_all:
        return meta
    desc = meta['desc'].fillna('').str.lower()
    text = meta['attchmntText'].fillna('').str.lower()
    results = desc.str.contains('financial result') | \
        (desc.str.contains('outcome of board meeting') & text.str.contains('financial result'))
    return meta[results]


def download(session, url, path, retries=3):
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=120)
            if r.status_code == 200 and r.content:
                tmp = path + '.part'
                with open(tmp, 'wb') as fh:
                    fh.write(r.content)
                os.replace(tmp, path)
                return 'ok', len(r.content)
            if r.status_code == 404:
                return 'missing', 0
            status = f'http {r.status_code}'
        except requests.RequestException as e:
            status = type(e).__name__
        time.sleep(5 * (attempt + 1))
    return status, 0


def main():
    parser = argparse.ArgumentParser(description="Download filing PDFs from NSE's archive.")
    parser.add_argument('--all', action='store_true', help='Every filing, not just results (very large)')
    parser.add_argument('--from-date', help='Only filings on or after this date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, help='Stop after this many new downloads')
    parser.add_argument('--max-gb', type=float, default=20.0, help='Stop once the folder holds this many GB')
    parser.add_argument('--delay', type=float, default=0.5, help='Seconds between requests')
    args = parser.parse_args()

    meta = pd.read_parquet(META)
    todo = select_filings(meta, args.all, args.from_date).sort_values('an_dt', ascending=False)
    done = set()
    if os.path.exists(MANIFEST):
        log = pd.read_csv(MANIFEST, dtype={'seq_id': str})
        done = set(log.loc[log['status'].isin(['ok', 'missing']), 'seq_id'])
    todo = todo[~todo['seq_id'].astype(str).isin(done)]
    print(f"{len(todo)} filings to download ({len(done)} already done)")

    os.makedirs(OUT_DIR, exist_ok=True)
    session = requests.Session()
    on_disk = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT_DIR) for f in fs)
    new_rows, fetched, total_bytes = [], 0, 0
    for i, row in enumerate(todo.itertuples(index=False), 1):
        ext = os.path.splitext(row.attchmntFile.split('?')[0])[1].lower() or '.pdf'
        folder = os.path.join(OUT_DIR, row.ticker)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f'{row.seq_id}{ext}')
        status, size = download(session, row.attchmntFile, path)
        fetched += status == 'ok'
        total_bytes += size
        new_rows.append({'ticker': row.ticker, 'seq_id': str(row.seq_id), 'an_dt': row.an_dt, 'desc': row.desc,
                         'url': row.attchmntFile, 'path': os.path.relpath(path, ROOT) if status == 'ok' else '',
                         'bytes': size, 'status': status})
        if i % 50 == 0 or i == len(todo):
            pd.DataFrame(new_rows).to_csv(MANIFEST, mode='a', header=not os.path.exists(MANIFEST), index=False)
            new_rows = []
            print(f"  {i}/{len(todo)}: {fetched} downloaded, {total_bytes / 1e9:.2f} GB", flush=True)
        if args.limit and fetched >= args.limit:
            break
        if (on_disk + total_bytes) / 1e9 >= args.max_gb:
            print(f"Reached the {args.max_gb:g} GB cap; rerun with a higher --max-gb to continue")
            break
        time.sleep(args.delay)
    if new_rows:
        pd.DataFrame(new_rows).to_csv(MANIFEST, mode='a', header=not os.path.exists(MANIFEST), index=False)
    print(f"Done: {fetched} new files, {total_bytes / 1e9:.2f} GB, in {OUT_DIR}")


if __name__ == '__main__':
    main()
