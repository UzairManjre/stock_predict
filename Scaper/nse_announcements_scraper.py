import os
import csv
import json
import time
import urllib3
import requests
import sys
import asyncio
from datetime import datetime, timedelta
from playwright.async_api import async_playwright
import concurrent.futures

# Disable insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Safety Limit: Maximum PDF files to download per company to prevent IP blocking
MAX_PDF_DOWNLOADS_PER_COMPANY = 15

# Thread pool for downloading PDFs to avoid blocking the event loop
pdf_download_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)

def get_latest_market_data_dir(base_dir):
    """Finds the most recent market data directory inside data/"""
    data_dir = os.path.join(base_dir, 'data')
    if not os.path.exists(data_dir):
        return None
    sessions = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if d.startswith("market_data_")]
    if not sessions:
        return None
    return max(sessions, key=os.path.getmtime)

def download_pdf(pdf_url, save_path):
    """Downloads a PDF announcement file safely"""
    if os.path.exists(save_path):
        return True
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': 'https://www.nseindia.com/'
    }
    
    try:
        response = requests.get(pdf_url, headers=headers, timeout=20, verify=False)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"      📥 Downloaded PDF: {os.path.basename(save_path)}")
        time.sleep(0.5)
        return True
    except Exception as e:
        print(f"      ⚠️ Failed to download PDF {pdf_url}: {e}")
        return False

async def process_symbol(symbol, idx, total, individual_data_dir, page, semaphore):
    async with semaphore:
        print(f"\n⚡ [{idx}/{total}] Fetching announcements for: {symbol}...")
        api_url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={symbol}"
        
        # Setup PostgreSQL connection for this task/symbol
        pg_conn = None
        pg_cursor = None
        try:
            from Scaper.utils.db_connector import get_db_connection
            pg_conn = get_db_connection()
            pg_cursor = pg_conn.cursor()
            pg_cursor.execute("""
                INSERT INTO companies (ticker, company_name, is_active)
                VALUES (%s, %s, TRUE)
                ON CONFLICT (ticker) DO NOTHING
            """, (symbol, symbol))
            pg_conn.commit()
        except Exception as db_init_err:
            print(f"   ⚠️ Database connection setup failed for {symbol}: {db_init_err}")
            pg_conn = None
            
        try:
            # Fetch announcements via browser evaluation
            response_text = await page.evaluate("""
                async (url) => {
                    const res = await fetch(url);
                    return await res.text();
                }
            """, api_url)
            
            if not response_text or "Missing index" in response_text:
                print(f"   ⚠️ No valid announcements found for {symbol}.")
                if pg_conn:
                    pg_cursor.close()
                    pg_conn.close()
                await asyncio.sleep(1)
                return
                
            announcements = json.loads(response_text)
            if not isinstance(announcements, list):
                print(f"   ⚠️ Unexpected API response type for {symbol}.")
                if pg_conn:
                    pg_cursor.close()
                    pg_conn.close()
                await asyncio.sleep(1)
                return
                
            print(f"   ✅ Found {len(announcements)} total announcements for {symbol}.")
            
            stock_folder = os.path.join(individual_data_dir, symbol)
            csv_path = os.path.join(stock_folder, 'corporate_announcements.csv')
            
            # Load existing CSV data to avoid duplicates
            existing_rows = []
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        existing_rows = list(reader)
                except Exception:
                    pass
            
            # Filter out duplicates and format new announcements
            new_announcements_added = 0
            cutoff_date = datetime.now() - timedelta(days=5*365)
            headers_csv = ['Subject', 'Category', 'PDF_Link', 'Headline', 'Received_Time', 'Disseminated_Time', 'Time_Taken']
            
            pdf_download_tasks = []
            loop = asyncio.get_running_loop()
            
            for item in announcements:
                # Date check: Skip announcements older than 5 years
                sort_date_str = item.get('sort_date')
                try:
                    if sort_date_str:
                        ann_date = datetime.strptime(sort_date_str, '%Y-%m-%d %H:%M:%S')
                        if ann_date < cutoff_date:
                            continue
                except Exception:
                    pass

                subject = item.get('desc', 'N/A')
                category = item.get('desc', 'N/A')
                pdf_link = item.get('attchmntFile', '')
                headline = item.get('attchmntText', 'N/A')
                received_time = item.get('an_dt', 'N/A')
                disseminated_time = item.get('exchdisstime', 'N/A')
                time_taken = item.get('difference', 'N/A')
                
                announcement_row = {
                    'Subject': subject,
                    'Category': category,
                    'PDF_Link': pdf_link,
                    'Headline': headline,
                    'Received_Time': received_time,
                    'Disseminated_Time': disseminated_time,
                    'Time_Taken': time_taken
                }
                
                # Deduplicate based on Received_Time and Headline
                is_duplicate = any(
                    row.get('Received_Time') == received_time and row.get('Headline') == headline
                    for row in existing_rows
                )
                
                if not is_duplicate:
                    existing_rows.append(announcement_row)
                    new_announcements_added += 1
                    
                    # Store in PostgreSQL
                    if pg_conn:
                        try:
                            # Parse dates
                            rec_dt = None
                            try:
                                rec_dt = datetime.strptime(received_time, '%d-%m-%Y %H:%M:%S')
                            except Exception:
                                try:
                                    rec_dt = datetime.strptime(received_time, '%Y-%m-%d')
                                except Exception:
                                    pass
                            
                            diss_dt = None
                            try:
                                diss_dt = datetime.strptime(disseminated_time, '%d-%m-%Y %H:%M:%S')
                            except Exception:
                                pass
                                
                            filing_date = rec_dt.date() if rec_dt else datetime.now().date()
                            
                            pg_cursor.execute("""
                                INSERT INTO financial_reports (ticker, filing_date, report_type, source_url, headline, received_time, disseminated_time, raw_text)
                                VALUES (%s, %s, 'Announcement', %s, %s, %s, %s, %s)
                                ON CONFLICT DO NOTHING
                            """, (symbol, filing_date, pdf_link, headline, rec_dt, diss_dt, json.dumps(item)))
                        except Exception as pg_ins_err:
                            print(f"      ⚠️ Failed to save announcement to DB: {pg_ins_err}")
                    
                    # Queue PDF download
                    if pdf_link and pdf_link != '-' and pdf_link.lower().startswith('http'):
                        pdf_dir = os.path.join(stock_folder, 'PDFs')
                        os.makedirs(pdf_dir, exist_ok=True)
                        
                        try:
                            date_obj = datetime.strptime(received_time, '%d-%m-%Y %H:%M:%S')
                            date_str = date_obj.strftime('%Y-%m-%d')
                        except ValueError:
                            date_str = received_time.split(' ')[0].replace('/', '-')
                            
                        clean_subject = "".join([c for c in subject if c.isalpha() or c.isdigit() or c==' ']).rstrip()
                        clean_subject = clean_subject.replace(' ', '_')[:50]
                        
                        pdf_filename = f"{date_str}_{clean_subject}.pdf"
                        pdf_save_path = os.path.join(pdf_dir, pdf_filename)
                        
                        pdf_download_tasks.append(
                            loop.run_in_executor(pdf_download_pool, download_pdf, pdf_link, pdf_save_path)
                        )
            
            if pg_conn:
                pg_conn.commit()
                
            if pdf_download_tasks:
                await asyncio.gather(*pdf_download_tasks)
            
            # Save updated rows back to CSV (sorted latest first)
            if new_announcements_added > 0:
                # Sort by Received_Time descending if date format allows
                try:
                    existing_rows.sort(
                        key=lambda r: datetime.strptime(r['Received_Time'], '%d-%m-%Y %H:%M:%S') if r['Received_Time'] != 'N/A' else datetime.min,
                        reverse=True
                    )
                except Exception:
                    pass
                    
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers_csv)
                    writer.writeheader()
                    writer.writerows(existing_rows)
                print(f"   💾 Saved {new_announcements_added} new announcements to CSV and database for {symbol}.")
            else:
                print(f"   ℹ️ No new announcements to add for {symbol}.")
                
        except Exception as e:
            print(f"   ❌ Error processing symbol {symbol}: {e}")
        finally:
            if pg_conn:
                pg_cursor.close()
                pg_conn.close()
            
        # Respectful delay between symbols to avoid rate limits
        await asyncio.sleep(1)

async def async_main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
    
    # 1. Resolve target directory
    session_dir = get_latest_market_data_dir(base_dir)
    if not session_dir:
        print("❌ No market data session directory found in data/. Exiting.")
        return
        
    individual_data_dir = os.path.join(session_dir, 'individual_data')
    if not os.path.exists(individual_data_dir):
        print(f"❌ Individual data directory not found at {individual_data_dir}. Exiting.")
        return
        
    # Get all company symbols from folders
    symbols = [d for d in os.listdir(individual_data_dir) if os.path.isdir(os.path.join(individual_data_dir, d))]
    print(f"📂 Found {len(symbols)} companies to process in {individual_data_dir}")
    
    # 2. Start Playwright session
    print("🌐 Launching browser in headful mode to bypass NSE firewall...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-http2"
            ]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            locale="en-US"
        )
        page = await context.new_page()
        await page.add_init_script("delete navigator.__proto__.webdriver;")

        try:
            print("⏳ Accessing NSE Corporate Filings page to establish session cookies...")
            await page.goto("https://www.nseindia.com/companies-listing/corporate-filings-announcements", timeout=30000)
            await page.wait_for_timeout(3000)
        except Exception as e:
            print(f"❌ Failed to load NSE page: {e}")
            await browser.close()
            return

        # 3. Process symbols concurrently
        semaphore = asyncio.Semaphore(5) # Allow up to 5 concurrent symbol fetches
        tasks = []
        total = len(symbols)
        for idx, symbol in enumerate(symbols, 1):
            tasks.append(process_symbol(symbol, idx, total, individual_data_dir, page, semaphore))
        
        await asyncio.gather(*tasks)
            
        print("\n🎉 Done processing all companies!")
        await browser.close()

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")

if __name__ == "__main__":
    main()

