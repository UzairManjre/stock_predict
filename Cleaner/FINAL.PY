import os
import sys
import time

# Ensure stdout/stderr use UTF-8 on Windows to prevent UnicodeEncodeError
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr.encoding is None or sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import fitz  # PyMuPDF
import torch
# Monkey-patch missing torch float8 attribute to prevent transformers import crashes on this PyTorch version
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)
# Monkey patch check_torch_load_is_safe in transformers to allow loading models with PyTorch 2.5
try:
    import transformers.utils.import_utils as import_utils
    import_utils.check_torch_load_is_safe = lambda: None
    
    import transformers.utils as utils
    utils.check_torch_load_is_safe = lambda: None
    
    import transformers.modeling_utils as modeling_utils
    modeling_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

import pandas as pd
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline

# Add parent dir to path for db_connector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Scaper.utils.db_connector import get_db_connection

# ================== CONFIG ==================
CHECKPOINT_FILE = "sentiment_checkpoint_local.csv"
FINBERT_MODEL = "ahmedrachid/FinancialBERT-Sentiment-Analysis"
CHUNK_SIZE = 256   # smaller chunks = more batches, less truncation
BATCH_SIZE = 64   # push bigger batches to GPU
MAX_WORKERS = 8    # threads for PDF extraction

# ================== DEVICE ==================
if torch.cuda.is_available():
    device = 0
    device_name = torch.cuda.get_device_name(0)
    print(f"[INFO] CUDA is available. Device set to GPU: {device_name}")
else:
    device = -1
    print("[INFO] CUDA is NOT available. Device set to CPU")

# ================== MODELS ==================
print(f"[INFO] Loading FinBERT model '{FINBERT_MODEL}' on {'GPU' if device == 0 else 'CPU'}...")
start_model_load = time.time()
# Configure pipeline to return all scores so we get positive, negative, neutral
try:
    print("[INFO] Initializing Hugging Face pipeline...")
    financial_sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=FINBERT_MODEL,
        device=device,
        top_k=None  # returns all labels and scores
    )
    print(f"[INFO] FinBERT model loaded successfully in {time.time() - start_model_load:.2f} seconds.")
except Exception as init_err:
    print(f"[CRITICAL ERROR] Failed to initialize pipeline: {init_err}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ================== HELPERS ==================
def get_latest_market_data_dir(base_dir):
    """Finds the most recent market data directory inside data/"""
    data_dir = os.path.join(base_dir, 'data')
    if not os.path.exists(data_dir):
        return None
    sessions = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if d.startswith("market_data_")]
    if not sessions:
        return None
    return max(sessions, key=os.path.getmtime)

def is_valid_pdf(pdf_path):
    """Check if the file exists, is non-empty, and starts with the %PDF signature."""
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
    """Extract all text from a PDF file using PyMuPDF, with safety checks."""
    if not is_valid_pdf(pdf_path):
        return ""
    try:
        with fitz.open(pdf_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"[WARNING] Error reading {pdf_path}: {e}")
        return ""

# ================== MAIN ==================
def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    watch_mode = "--watch" in sys.argv
    poll_interval = 15 # check every 15 seconds
    
    print(f"[INFO] Running in {'CONTINUOUS WATCH' if watch_mode else 'ONE-SHOT'} mode.")

    while True:
        # Resolve dynamic target directory inside the loop to support updates to the scraping session folder
        session_dir = get_latest_market_data_dir(base_dir)
        if not session_dir:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] No market data session directory found in data/.")
            if not watch_mode:
                return
            time.sleep(poll_interval)
            continue
            
        individual_data_dir = os.path.join(session_dir, 'individual_data')
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Resolved active session directory: {session_dir}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Scanning individual data directory: {individual_data_dir}")

        # 1. Fetch reports from PostgreSQL that do not have sentiment chunks processed yet
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Failed to connect to DB: {e}")
            if not watch_mode:
                return
            time.sleep(poll_interval)
            continue

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Querying database for unprocessed reports...")
        cursor.execute("""
            SELECT report_id, ticker, source_url, filing_date 
            FROM financial_reports 
            WHERE report_id NOT IN (SELECT DISTINCT report_id FROM sentiment_chunks)
              AND source_url IS NOT NULL 
              AND source_url != '' 
              AND source_url != '-'
            LIMIT 60
        """)
        unprocessed_reports = cursor.fetchall()
        report_info = {r[0]: (r[1], r[3]) for r in unprocessed_reports}

        if not unprocessed_reports:
            cursor.close()
            conn.close()
            if not watch_mode:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] All reports are already processed! Exiting.")
                return
            next_check_str = datetime.fromtimestamp(time.time() + poll_interval).strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] No new unprocessed reports found. Waiting {poll_interval}s. Next check at {next_check_str}...")
            time.sleep(poll_interval)
            continue

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Found {len(unprocessed_reports)} unprocessed report entries in DB.")

        # 2. Match database entries with local PDF files
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Matching database entries with local files in {individual_data_dir}...")
        reports_to_process = []
        pdf_paths = {}
        matched_count = 0
        missing_count = 0
        
        for report in unprocessed_reports:
            report_id = report[0]
            ticker = report[1]
            source_url = report[2]
            filing_date = report[3]
            
            pdf_filename = os.path.basename(source_url)
            # Handle cases where source_url is not a clean link
            if not pdf_filename.lower().endswith('.pdf'):
                # Try fuzzy matching in company's PDFs folder
                pdf_filename = f"{filing_date}_*.pdf"
                
            local_pdf_path = os.path.join(individual_data_dir, ticker, 'PDFs', pdf_filename)
            search_reason = f"direct path: {local_pdf_path}"
            
            # Check if file exists, if not, do a directory search for wildcard
            if not os.path.exists(local_pdf_path):
                pdf_dir = os.path.join(individual_data_dir, ticker, 'PDFs')
                if os.path.exists(pdf_dir):
                    files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
                    # Find matching PDF by date prefix or similar
                    date_prefix = filing_date.strftime('%Y-%m-%d')
                    matching = [f for f in files if os.path.basename(f).startswith(date_prefix)]
                    if matching:
                        local_pdf_path = matching[0]
                        search_reason = f"fuzzy date match: {os.path.basename(local_pdf_path)}"
                    elif files:
                        # Fallback to first PDF if only one exists
                        local_pdf_path = files[0]
                        search_reason = f"first available file fallback: {os.path.basename(local_pdf_path)}"
                        
            if os.path.exists(local_pdf_path):
                reports_to_process.append(report)
                pdf_paths[report_id] = local_pdf_path
                matched_count += 1
                if matched_count <= 20:
                    print(f"   [MATCHED] {ticker} (Filing Date: {filing_date}) -> {os.path.basename(local_pdf_path)} (via {search_reason})")
                elif matched_count == 21:
                    print(f"   [INFO] ... matched count exceeds 20. Hiding subsequent matches to prevent stdout overflow.")
            else:
                missing_count += 1
                if missing_count <= 20:
                    print(f"   [MISSING] Local PDF file not found for {ticker} (Filing Date: {filing_date}) at expected path: {local_pdf_path}")
                elif missing_count == 21:
                    print(f"   [INFO] ... missing count exceeds 20. Hiding subsequent missing logs to prevent stdout overflow.")

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Match Summary: Total Database Entries={len(unprocessed_reports)}, Matched PDFs={matched_count}, Missing PDFs={missing_count}")

        if not reports_to_process:
            cursor.close()
            conn.close()
            if not watch_mode:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] No matching local PDF files found on disk. Exiting.")
                return
            next_check_str = datetime.fromtimestamp(time.time() + poll_interval).strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] No local PDFs found for the {len(unprocessed_reports)} database records. Waiting {poll_interval}s. Next check at {next_check_str}...")
            time.sleep(poll_interval)
            continue

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Successfully matched {len(reports_to_process)} database entries with local PDFs. Starting text extraction...")

        # 3. Parallel text extraction
        texts = {}
        extract_start = time.time()
        extracted_ok = 0
        extracted_err = 0
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Starting extraction of {len(reports_to_process)} matched PDFs...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_report = {
                executor.submit(extract_text, pdf_paths[r[0]]): r 
                for r in reports_to_process
            }
            for future in tqdm(future_to_report, total=len(future_to_report), desc="Extracting PDFs"):
                report = future_to_report[future]
                text = future.result()
                texts[report[0]] = text if text else ""
                char_count = len(text) if text else 0
                if char_count > 0:
                    extracted_ok += 1
                    if extracted_ok <= 10:
                        print(f"   [EXTRACTION] {report[1]} ({report[3]}) -> Extracted {char_count} characters of text.")
                    elif extracted_ok == 11:
                        print(f"   [INFO] ... extraction success count exceeds 10. Hiding subsequent extraction logs.")
                else:
                    extracted_err += 1
                    if extracted_err <= 10:
                        print(f"   [EXTRACTION] {report[1]} ({report[3]}) -> ERROR or Empty/Scanned PDF (0 characters extracted).")
                    elif extracted_err == 11:
                        print(f"   [INFO] ... extraction failure count exceeds 10. Hiding subsequent extraction failure logs.")
        extract_duration = time.time() - extract_start
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] PDF extraction finished. Total time: {extract_duration:.2f} seconds. Success={extracted_ok}, Failed/Scanned={extracted_err}.")

        # 4. Prepare chunks for FinBERT
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Segmenting text into {CHUNK_SIZE}-character chunks...")
        all_chunks = []
        chunk_map = {} # report_id -> list of indices in all_chunks
        empty_reports = [] # reports with empty or non-selectable text
        chunk_count_printed = 0
        
        for report in reports_to_process:
            report_id = report[0]
            ticker = report[1]
            filing_date = report[3]
            text = texts.get(report_id, "")
            
            if not text.strip():
                empty_reports.append(report_id)
                continue
                
            # Clean text slightly, removing NUL bytes to prevent PostgreSQL/psycopg2 crash
            text_clean = text.replace('\n', ' ').replace('\x00', '').replace('\u0000', '').strip()
            chunks = [text_clean[i:i+CHUNK_SIZE] for i in range(0, len(text_clean), CHUNK_SIZE)]
            # Filter empty chunks
            chunks = [c.strip() for c in chunks if c.strip()]
            if chunks:
                chunk_map[report_id] = list(range(len(all_chunks), len(all_chunks)+len(chunks)))
                all_chunks.extend(chunks)
                chunk_count_printed += 1
                if chunk_count_printed <= 10:
                    print(f"   [CHUNKING] {ticker} ({filing_date}) -> Generated {len(chunks)} chunks.")
                elif chunk_count_printed == 11:
                    print(f"   [INFO] ... chunking logs exceed 10. Hiding subsequent chunking logs.")
            else:
                empty_reports.append(report_id)

        # 4b. Write empty/scanned PDFs to DB as neutral to prevent endless watch loops
        if empty_reports:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Processing {len(empty_reports)} scanned/empty PDFs as neutral...")
            skipped_printed = 0
            for r_id in empty_reports:
                ticker, filing_date = report_info.get(r_id, ("UNKNOWN", datetime.now().date()))
                
                skipped_printed += 1
                if skipped_printed <= 10:
                    print(f"   [SKIPPED/SCANNED] {ticker} ({filing_date}) -> No selectable text in PDF. Writing placeholder neutral entry.")
                elif skipped_printed == 11:
                    print(f"   [INFO] ... skipped count exceeds 10. Hiding subsequent skipped logs to prevent stdout overflow.")
                
                cursor.execute("""
                    INSERT INTO sentiment_chunks (report_id, chunk_index, chunk_text, score_positive, score_negative, score_neutral, dominant_label)
                    VALUES (%s, 0, '[Scanned/Image PDF - No selectable text]', 0.0, 0.0, 1.0, 'neutral')
                    ON CONFLICT DO NOTHING
                """, (r_id,))
                
                cursor.execute("""
                    INSERT INTO daily_sentiment_aggregates (date, ticker, net_sentiment_score, bullish_intensity, bearish_intensity, total_chunks_analyzed)
                    VALUES (%s, %s, 0.0, 0.0, 0.0, 1)
                    ON CONFLICT (date, ticker) DO UPDATE SET
                        total_chunks_analyzed = daily_sentiment_aggregates.total_chunks_analyzed + 1
                """, (filing_date, ticker))
            conn.commit()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Committed scanned/empty PDF neutral placeholders to DB.")

        if not all_chunks and not empty_reports:
            cursor.close()
            conn.close()
            if not watch_mode:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] No text chunks extracted from any reports. Exiting.")
                return
            next_check_str = datetime.fromtimestamp(time.time() + poll_interval).strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] No text chunks extracted. Waiting {poll_interval}s. Next check at {next_check_str}...")
            time.sleep(poll_interval)
            continue

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Total chunks to process with FinBERT: {len(all_chunks)}")

        # 5. Run FinBERT in batch (GPU heavy)
        finbert_results = []
        if all_chunks:
            pipeline_device = f"GPU (cuda:0)" if device == 0 else "CPU"
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Starting FinBERT inference on {len(all_chunks)} chunks using {pipeline_device} with batch_size={BATCH_SIZE}...")
            start_inference = time.time()
            finbert_results = financial_sentiment_pipeline(all_chunks, batch_size=BATCH_SIZE)
            inference_duration = time.time() - start_inference
            chunks_per_sec = len(all_chunks) / inference_duration if inference_duration > 0 else 0
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [SUCCESS] FinBERT inference completed. Total Time: {inference_duration:.2f}s ({chunks_per_sec:.2f} chunks/sec).")

        # 6. Aggregate + insert into PostgreSQL
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Writing sentiment chunks and daily aggregates to database...")
        db_start = time.time()
        saved_reports_count = 0
        
        for report in tqdm(reports_to_process, desc="Saving Sentiments"):
            report_id = report[0]
            ticker = report[1]
            filing_date = report[3]
            
            # Skip if report was empty
            if report_id in empty_reports:
                continue
                
            chunk_indices = chunk_map.get(report_id, [])
            if not chunk_indices:
                continue
                    
            # Write chunks to sentiment_chunks
            for idx, ci in enumerate(chunk_indices):
                res_list = finbert_results[ci]
                # Parse positive, negative, neutral scores
                scores = {r['label']: r['score'] for r in res_list}
                score_pos = scores.get('positive', 0.0)
                score_neg = scores.get('negative', 0.0)
                score_neu = scores.get('neutral', 0.0)
                
                # Find dominant label
                dominant = max(res_list, key=lambda x: x['score'])['label']
                chunk_txt = all_chunks[ci]
                
                cursor.execute("""
                    INSERT INTO sentiment_chunks (report_id, chunk_index, chunk_text, score_positive, score_negative, score_neutral, dominant_label)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (report_id, idx, chunk_txt, score_pos, score_neg, score_neu, dominant))
                
            # 7. Calculate and update daily_sentiment_aggregates for this ticker and date
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
                
                if saved_reports_count <= 20:
                    print(f"   [DATABASE SAVED] {ticker} ({filing_date}) | Chunks Inserted: {len(chunk_indices)} | Day Chunks Count: {total_ch} | Net Score: {net_sentiment:.4f} | Bullish Intensity: {bullish_int:.4f} | Bearish Intensity: {bearish_int:.4f}")
                elif saved_reports_count == 21:
                    print(f"   [INFO] ... database save count exceeds 20. Hiding subsequent database logs.")
                
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
            saved_reports_count += 1

        cursor.close()
        conn.close()
        db_duration = time.time() - db_start
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [SUCCESS] All sentiment results successfully saved to database. Reports saved: {saved_reports_count}. Database time: {db_duration:.2f}s.")
        
        if not watch_mode:
            break
            
        next_check_str = datetime.fromtimestamp(time.time() + poll_interval).strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Waiting {poll_interval} seconds for new reports. Next check will start at {next_check_str}...")
        time.sleep(poll_interval)

if __name__ == "__main__":
    main()
