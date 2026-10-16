import os
import sys
import time
import re
import csv
from datetime import datetime

# Add parent dir to path for db_connector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Scaper.utils.db_connector import get_db_connection

def get_latest_market_data_dir(base_dir):
    """Finds the most recent market data directory inside data/"""
    data_dir = os.path.join(base_dir, 'data')
    if not os.path.exists(data_dir):
        return None
    sessions = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if d.startswith("market_data_")]
    if not sessions:
        return None
    return max(sessions, key=os.path.getmtime)

def parse_filename(filename):
    """
    Parses a PDF filename to extract filing date and headline.
    Supports YYYY-MM-DD_Subject.pdf and DD-Mmm-YYYY_Subject.pdf
    """
    basename = os.path.splitext(filename)[0]
    
    # 1. Match YYYY-MM-DD prefix
    match_iso = re.match(r'^(\d{4}-\d{2}-\d{2})_(.*)$', basename)
    if match_iso:
        date_str = match_iso.group(1)
        headline_str = match_iso.group(2).replace('_', ' ')
        try:
            filing_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            return filing_date, headline_str
        except Exception:
            pass
            
    # 2. Match DD-Mmm-YYYY prefix (e.g. 22-Sep-2023)
    match_mmm = re.match(r'^(\d{1,2})-([A-Za-z]{3})-(\d{4})_(.*)$', basename)
    if match_mmm:
        date_str = f"{match_mmm.group(1)}-{match_mmm.group(2)}-{match_mmm.group(3)}"
        headline_str = match_mmm.group(4).replace('_', ' ')
        try:
            filing_date = datetime.strptime(date_str, '%d-%b-%Y').date()
            return filing_date, headline_str
        except Exception:
            pass
            
    return None, None


def scan_and_populate():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    session_dir = get_latest_market_data_dir(base_dir)
    if not session_dir:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] No market data session directory found in data/.")
        return
        
    individual_data_dir = os.path.join(session_dir, 'individual_data')
    if not os.path.exists(individual_data_dir):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Individual data directory not found: {individual_data_dir}")
        return

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Scanning folders in {individual_data_dir}...")
    
    tickers = [t for t in os.listdir(individual_data_dir) if os.path.isdir(os.path.join(individual_data_dir, t))]
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Failed to connect to DB: {e}")
        return

    new_records_count = 0
    scanned_files_count = 0

    for ticker in tickers:
        ticker_dir = os.path.join(individual_data_dir, ticker)
        pdf_dir = os.path.join(ticker_dir, 'PDFs')
        csv_path = os.path.join(ticker_dir, 'corporate_announcements.csv')

        if not os.path.exists(pdf_dir):
            continue

        # Load CSV if it exists to map detailed fields
        csv_data = []
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    csv_data = list(reader)
            except Exception as e:
                print(f"   [WARNING] Error reading CSV for {ticker}: {e}")

        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        
        for pdf_file in pdf_files:
            scanned_files_count += 1
            # 1. Parse date and subject/headline from filename
            filing_date, parsed_headline = parse_filename(pdf_file)
            if not filing_date:
                # Fallback: use file creation/modified date
                mtime = os.path.getmtime(os.path.join(pdf_dir, pdf_file))
                filing_date = datetime.fromtimestamp(mtime).date()
                parsed_headline = os.path.splitext(pdf_file)[0].replace('_', ' ')

            # 2. Try to match the PDF with a row in the corporate_announcements.csv
            matched_row = None
            for row in csv_data:
                # Match by checking if the PDF link contains the filename or if the subject/headline is similar
                csv_pdf = row.get('PDF_Link', '')
                if csv_pdf and os.path.basename(csv_pdf) == pdf_file:
                    matched_row = row
                    break
                # Try fuzzy matching using subject or headline matching our parsed subject
                csv_subject = row.get('Subject', '')
                if csv_subject and (csv_subject.replace(' ', '_').lower() in pdf_file.lower() or pdf_file.lower() in csv_subject.replace(' ', '_').lower()):
                    # Verify filing date is close (same day)
                    csv_rec = row.get('Received_Time', '')
                    if csv_rec:
                        try:
                            # Try parsing 'dd-mm-yyyy hh:mm:ss' or 'yyyy-mm-dd'
                            if '-' in csv_rec:
                                if len(csv_rec.split(' ')[0].split('-')[0]) == 4:
                                    csv_date = datetime.strptime(csv_rec.split(' ')[0], '%Y-%m-%d').date()
                                else:
                                    csv_date = datetime.strptime(csv_rec.split(' ')[0], '%d-%m-%Y').date()
                            else:
                                csv_date = datetime.strptime(csv_rec.split(' ')[0], '%Y/%m/%d').date()
                                
                            if csv_date == filing_date:
                                matched_row = row
                                break
                        except Exception:
                            pass

            # Extract fields
            if matched_row:
                headline = matched_row.get('Headline', parsed_headline)
                source_url = matched_row.get('PDF_Link', pdf_file)
                received_time_str = matched_row.get('Received_Time', '')
                disseminated_time_str = matched_row.get('Disseminated_Time', '')
                
                # Parse timestamps
                received_time = None
                if received_time_str and received_time_str != 'N/A':
                    try:
                        received_time = datetime.strptime(received_time_str, '%d-%m-%Y %H:%M:%S')
                    except Exception:
                        try:
                            received_time = datetime.strptime(received_time_str.split(' ')[0], '%Y-%m-%d')
                        except Exception:
                            pass
                            
                disseminated_time = None
                if disseminated_time_str and disseminated_time_str != 'N/A':
                    try:
                        disseminated_time = datetime.strptime(disseminated_time_str, '%d-%m-%Y %H:%M:%S')
                    except Exception:
                        pass
            else:
                headline = parsed_headline
                source_url = pdf_file
                received_time = datetime.combine(filing_date, datetime.min.time())
                disseminated_time = None

            # 3. Check if it already exists in the database
            cursor.execute("""
                SELECT report_id FROM financial_reports 
                WHERE ticker = %s AND (source_url = %s OR (filing_date = %s AND headline = %s))
            """, (ticker, source_url, filing_date, headline))
            exists = cursor.fetchone()

            if not exists:
                # Ensure company exists in companies table
                cursor.execute("""
                    INSERT INTO companies (ticker, company_name, is_active)
                    VALUES (%s, %s, TRUE)
                    ON CONFLICT (ticker) DO NOTHING
                """, (ticker, ticker))
                
                # Insert report
                cursor.execute("""
                    INSERT INTO financial_reports (ticker, filing_date, report_type, source_url, headline, received_time, disseminated_time)
                    VALUES (%s, %s, 'Announcement', %s, %s, %s, %s)
                """, (ticker, filing_date, source_url, headline, received_time, disseminated_time))
                
                print(f"   [REGISTERED] Ticker: {ticker} | Date: {filing_date} | File: {pdf_file} | Headline: {headline[:60]}...")
                new_records_count += 1

    if new_records_count > 0:
        conn.commit()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [SUCCESS] Scanned {scanned_files_count} files. Newly registered {new_records_count} announcements in PostgreSQL.")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Scanned {scanned_files_count} files. No new files needed registration.")

    cursor.close()
    conn.close()

def main():
    watch_mode = "--watch" in sys.argv
    poll_interval = 20 # check every 20 seconds
    
    print(f"[INFO] Directory DB Populator starting in {'WATCH' if watch_mode else 'ONE-SHOT'} mode...")
    
    while True:
        try:
            scan_and_populate()
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CRITICAL] Error during scan: {e}")
            
        if not watch_mode:
            break
            
        next_check_str = datetime.fromtimestamp(time.time() + poll_interval).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Waiting {poll_interval}s before next folder scan. Next scan at {next_check_str}...")
        time.sleep(poll_interval)

if __name__ == '__main__':
    main()
