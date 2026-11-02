import os
import sys
import time
import re
import shutil
import subprocess
from datetime import datetime
import psycopg2
import psycopg2.extras

# Ensure stdout/stderr use UTF-8 on Windows to prevent UnicodeEncodeError
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import fitz  # PyMuPDF
import torch

# Monkey-patch missing torch float8 attribute to prevent transformers import crashes
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

try:
    import transformers.utils.import_utils as import_utils
    import_utils.check_torch_load_is_safe = lambda: None
    import transformers.utils as utils
    utils.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as modeling_utils
    modeling_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from transformers import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(HERE, '..'))
WATCH_DIR = os.path.join(BASE_DIR, 'uploaded_documents')

# Load env variables from .env
env_path = os.path.abspath(os.path.join(BASE_DIR, '.env'))
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5433")
DB_NAME = os.environ.get("DB_NAME", "stock_predict")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

FINBERT_MODEL = "ahmedrachid/FinancialBERT-Sentiment-Analysis"
CHUNK_SIZE = 256
BATCH_SIZE = 64

# ================== DB CONNECTIONS ==================
def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

# ================== HELPERS ==================
def get_latest_market_data_dir(base_dir):
    data_dir = os.path.join(base_dir, 'data')
    if not os.path.exists(data_dir):
        return None
    sessions = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if d.startswith("market_data_")]
    if not sessions:
        return None
    return max(sessions, key=os.path.getmtime)

def is_valid_pdf(pdf_path):
    try:
        if not os.path.exists(pdf_path):
            return False
        if os.path.getsize(pdf_path) < 4:
            return False
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False

def extract_text(pdf_path):
    if not is_valid_pdf(pdf_path):
        return ""
    try:
        with fitz.open(pdf_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Warning: Error reading {pdf_path}: {e}")
        return ""

def parse_uploaded_filename(filename, valid_tickers):
    """
    Parses an uploaded PDF filename to extract:
    1. Ticker
    2. Date
    3. Headline
    """
    basename = os.path.splitext(filename)[0]
    basename_upper = basename.upper()
    
    # 1. Match ticker (sort descending by length to match LTIMINDTREE before LTI)
    matched_ticker = None
    for ticker in sorted(valid_tickers, key=len, reverse=True):
        pattern = r'(^|[^A-Z0-9])' + re.escape(ticker) + r'([^A-Z0-9]|$)'
        if re.search(pattern, basename_upper):
            matched_ticker = ticker
            break
            
    if not matched_ticker:
        return None, None, None

    # 2. Extract Date (support YYYY-MM-DD or DD-MM-YYYY or similar)
    filing_date = None
    # YYYY-MM-DD
    match_iso = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', basename)
    if match_iso:
        try:
            filing_date = datetime.strptime(f"{match_iso.group(1)}-{match_iso.group(2)}-{match_iso.group(3)}", '%Y-%m-%d').date()
        except ValueError:
            pass
            
    # DD-MM-YYYY
    if not filing_date:
        match_dd = re.search(r'(\d{1,2})[-_](\d{2})[-_](\d{4})', basename)
        if match_dd:
            try:
                filing_date = datetime.strptime(f"{match_dd.group(1)}-{match_dd.group(2)}-{match_dd.group(3)}", '%d-%m-%Y').date()
            except ValueError:
                pass
                
    if not filing_date:
        # Default to today's date
        filing_date = datetime.now().date()

    # 3. Extract Headline
    # Remove ticker and date matches from the filename, then clean up
    cleaned = basename_upper.replace(matched_ticker, '')
    if match_iso:
        cleaned = cleaned.replace(match_iso.group(0), '')
    elif 'match_dd' in locals() and match_dd:
        cleaned = cleaned.replace(match_dd.group(0), '')
        
    cleaned = cleaned.replace('_', ' ').replace('-', ' ').strip()
    # Collapse double spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    headline = cleaned.strip().title() if cleaned.strip() else "Announcement"
    
    return matched_ticker, filing_date, headline

# ================== AI PROCESSING ==================
def process_report_sentiment(report_id, ticker, filing_date, pdf_path, financial_sentiment_pipeline, cursor, conn):
    """Processes sentiment for a registered document immediately."""
    print(f"   [AI Engine] Extracting text from {os.path.basename(pdf_path)}...")
    text = extract_text(pdf_path)
    
    if not text.strip():
        print("   [AI Engine] No selectable text found. Registering neutral placeholder.")
        cursor.execute("""
            INSERT INTO sentiment_chunks (report_id, chunk_index, chunk_text, score_positive, score_negative, score_neutral, dominant_label)
            VALUES (%s, 0, '[Scanned/Image PDF - No selectable text]', 0.0, 0.0, 1.0, 'neutral')
            ON CONFLICT DO NOTHING
        """, (report_id,))
        
        cursor.execute("""
            INSERT INTO daily_sentiment_aggregates (date, ticker, net_sentiment_score, bullish_intensity, bearish_intensity, total_chunks_analyzed)
            VALUES (%s, %s, 0.0, 0.0, 0.0, 1)
            ON CONFLICT (date, ticker) DO UPDATE SET
                total_chunks_analyzed = daily_sentiment_aggregates.total_chunks_analyzed + 1
        """, (filing_date, ticker))
        conn.commit()
        return

    text_clean = text.replace('\n', ' ').replace('\x00', '').replace('\u0000', '').strip()
    chunks = [text_clean[i:i+CHUNK_SIZE] for i in range(0, len(text_clean), CHUNK_SIZE)]
    chunks = [c.strip() for c in chunks if c.strip()]
    
    if not chunks:
        return

    print(f"   [AI Engine] Run FinBERT on {len(chunks)} text chunks...")
    finbert_results = financial_sentiment_pipeline(chunks, batch_size=BATCH_SIZE)
    
    # Save chunks
    for idx, (chunk_txt, res_list) in enumerate(zip(chunks, finbert_results)):
        scores = {r['label']: r['score'] for r in res_list}
        score_pos = scores.get('positive', 0.0)
        score_neg = scores.get('negative', 0.0)
        score_neu = scores.get('neutral', 0.0)
        
        dominant = max(res_list, key=lambda x: x['score'])['label']
        
        cursor.execute("""
            INSERT INTO sentiment_chunks (report_id, chunk_index, chunk_text, score_positive, score_negative, score_neutral, dominant_label)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (report_id, idx, chunk_txt, score_pos, score_neg, score_neu, dominant))

    # Recalculate daily averages
    cursor.execute("""
        SELECT score_positive, score_negative, dominant_label
        FROM sentiment_chunks
        WHERE report_id IN (
            SELECT report_id FROM financial_reports WHERE ticker = %s AND filing_date = %s
        )
    """, (ticker, filing_date))
    
    day_chunks = cursor.fetchall()
    if day_chunks:
        total_ch = len(day_chunks)
        net_sentiment = sum(c[0] - c[1] for c in day_chunks) / total_ch
        
        pos_ch = [c for c in day_chunks if c[2] == 'positive']
        neg_ch = [c for c in day_chunks if c[2] == 'negative']
        
        bullish_int = sum(c[0] for c in pos_ch) / len(pos_ch) if pos_ch else 0.0
        bearish_int = sum(c[1] for c in neg_ch) / len(neg_ch) if neg_ch else 0.0
        
        cursor.execute("""
            INSERT INTO daily_sentiment_aggregates (date, ticker, net_sentiment_score, bullish_intensity, bearish_intensity, total_chunks_analyzed)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, ticker) DO UPDATE SET
                net_sentiment_score = EXCLUDED.net_sentiment_score,
                bullish_intensity = EXCLUDED.bullish_intensity,
                bearish_intensity = EXCLUDED.bearish_intensity,
                total_chunks_analyzed = EXCLUDED.total_chunks_analyzed
        """, (filing_date, ticker, net_sentiment, bullish_int, bearish_int, total_ch))
        
    conn.commit()
    print(f"   [AI Engine] Saved sentiment. Net score: {net_sentiment:.4f} | Total Chunks: {total_ch}")

# ================== PIPELINE TRIGGER ==================
def trigger_pipeline():
    print("[PIPELINE] Triggering downstream features and predictions...")
    try:
        cmd = [sys.executable, os.path.join(HERE, 'pipeline.py'), '--stages', 'features', 'predict', 'drift']
        subprocess.run(cmd, cwd=HERE, check=True)
        print("[PIPELINE] Downstream pipeline successfully completed!")
    except Exception as e:
        print(f"[PIPELINE] Error running downstream pipeline: {e}")

# ================== WATCHER MAIN LOOP ==================
def main():
    os.makedirs(WATCH_DIR, exist_ok=True)
    
    # Establish valid tickers list
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT ticker FROM companies WHERE is_active = TRUE")
    valid_tickers = {row[0] for row in cursor.fetchall()}
    cursor.close()
    conn.close()
    
    print(f"[INFO] Watch folder initialized at: {WATCH_DIR}")
    print(f"[INFO] Watching for new PDFs. Valid tickers loaded: {len(valid_tickers)}")
    
    # Establish hardware device
    device = 0 if torch.cuda.is_available() else -1
    print(f"[INFO] Hardware acceleration: Using {'CUDA' if device == 0 else 'CPU'}")
    
    print("[INFO] Initializing FinBERT pipeline...")
    financial_sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=FINBERT_MODEL,
        device=device,
        top_k=None
    )
    print("[INFO] FinBERT model loaded. Ready to watch and process files.")

    # Start watching
    try:
        while True:
            files = [f for f in os.listdir(WATCH_DIR) if f.lower().endswith('.pdf')]
            
            if files:
                print(f"\n[WATCHER] Detected {len(files)} new files to process.")
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                any_processed = False
                
                for filename in files:
                    file_path = os.path.join(WATCH_DIR, filename)
                    
                    # 1. Parse details
                    ticker, filing_date, headline = parse_uploaded_filename(filename, valid_tickers)
                    
                    if not ticker:
                        print(f"   [SKIP] File '{filename}' does not match any valid ticker. Rename to contain ticker (e.g. TCS_report.pdf).")
                        # Move to a skipped directory so we don't scan it forever
                        skipped_dir = os.path.join(WATCH_DIR, 'skipped')
                        os.makedirs(skipped_dir, exist_ok=True)
                        shutil.move(file_path, os.path.join(skipped_dir, filename))
                        continue
                        
                    print(f"\n   [PROCESS] Found match: Ticker={ticker} | Date={filing_date} | Headline={headline}")
                    
                    # 2. Resolve destination path in latest session directory
                    session_dir = get_latest_market_data_dir(BASE_DIR)
                    if not session_dir:
                        # Fallback to default market_data directory
                        session_dir = os.path.join(BASE_DIR, 'data', 'market_data_live')
                        
                    dest_pdf_dir = os.path.join(session_dir, 'individual_data', ticker, 'PDFs')
                    os.makedirs(dest_pdf_dir, exist_ok=True)
                    
                    # Target filename in target directory
                    dest_file_path = os.path.join(dest_pdf_dir, filename)
                    
                    # Check if file already exists in dest, append timestamp to make unique if it does
                    if os.path.exists(dest_file_path):
                        base_n, ext_n = os.path.splitext(filename)
                        dest_file_path = os.path.join(dest_pdf_dir, f"{base_n}_{int(time.time())}{ext_n}")
                    
                    # Move PDF to its sorted project folder
                    shutil.move(file_path, dest_file_path)
                    print(f"   [SORTED] Moved file to: {dest_file_path}")
                    
                    # 3. Register in financial_reports
                    try:
                        received_time = datetime.combine(filing_date, datetime.now().time())
                        
                        cursor.execute("""
                            INSERT INTO financial_reports (ticker, filing_date, report_type, source_url, headline, received_time)
                            VALUES (%s, %s, 'Announcement', %s, %s, %s)
                            RETURNING report_id
                        """, (ticker, filing_date, os.path.basename(dest_file_path), headline, received_time))
                        
                        report_id = cursor.fetchone()[0]
                        conn.commit()
                        
                        print(f"   [REGISTERED] Added report ID {report_id} to DB.")
                        
                        # 4. Perform AI sentiment analysis immediately
                        process_report_sentiment(
                            report_id=report_id,
                            ticker=ticker,
                            filing_date=filing_date,
                            pdf_path=dest_file_path,
                            financial_sentiment_pipeline=financial_sentiment_pipeline,
                            cursor=cursor,
                            conn=conn
                        )
                        any_processed = True
                        
                    except Exception as ins_err:
                        conn.rollback()
                        print(f"   [ERROR] Failed to save database entries for {filename}: {ins_err}")
                        
                cursor.close()
                conn.close()
                
                # 5. Run prediction updates if we processed anything
                if any_processed:
                    trigger_pipeline()
                    
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nWatcher stopped by user.")
    except Exception as e:
        print(f"\nWatcher encountered critical error: {e}")

if __name__ == '__main__':
    main()
