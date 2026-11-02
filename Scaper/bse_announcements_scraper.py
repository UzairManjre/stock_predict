import os
import csv
import re
import requests
from datetime import datetime, timedelta
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from bs4 import BeautifulSoup

def get_stock_map(file_path):
    """Reads the stock map from a CSV file."""
    stock_map = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stock_map[row['FolderName']] = {
                    'BseNameSlug': row['BseNameSlug'],
                    'BseTicker': row['BseTicker'],
                    'BseSecurityCode': row['BseSecurityCode']
                }
    except FileNotFoundError:
        print(f"Error: stock_map.csv not found at {file_path}")
    except Exception as e:
        print(f"Error reading stock_map.csv: {e}")
    return stock_map

def navigate_to_announcements(driver, stock_info, screenshot_path):
    """Navigates to the corporate announcements page for a given stock."""
    url = f"https://www.bseindia.com/stock-share-price/{stock_info['BseNameSlug']}/{stock_info['BseTicker']}/{stock_info['BseSecurityCode']}/"
    print(f"Navigating to {url}")
    try:
        driver.get(url)
        driver.save_screenshot(screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        corp_announcements_link = WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Corp Announcements') or contains(text(), 'Corporate Announcements')]"))
        )
        
        driver.execute_script("arguments[0].click();", corp_announcements_link)
        print(f"Successfully clicked 'Corp Announcements' for {stock_info['BseTicker']}")
        return True
    except TimeoutException:
        print(f"Timeout: Could not find or click 'Corp Announcements' link for {stock_info['BseTicker']}")
        return False
    except WebDriverException as e:
        print(f"WebDriver error navigating for {stock_info['BseTicker']}: {e}")
        return False

def set_date_range(driver):
    """Sets the date range for the last 11 months and submits the form."""
    to_date = datetime.now()
    # Approximately 11 months. Using 335 days is a safe approximation.
    from_date = to_date - timedelta(days=335)
    
    to_date_str = to_date.strftime('%d/%m/%Y')
    from_date_str = from_date.strftime('%d/%m/%Y')

    print(f"Setting date range from {from_date_str} to {to_date_str}")
    try:
        from_date_element = WebDriverWait(driver, 20).until(
            EC.visibility_of_element_located((By.ID, 'txtFromDt'))
        )
        to_date_element = WebDriverWait(driver, 20).until(
            EC.visibility_of_element_located((By.ID, 'txtToDt'))
        )
        submit_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.ID, 'btnSubmit'))
        )
        print("Date input fields and submit button found correctly.")

        driver.execute_script(f"arguments[0].value = '{from_date_str}'; arguments[0].dispatchEvent(new Event('change'))", from_date_element)
        driver.execute_script(f"arguments[0].value = '{to_date_str}'; arguments[0].dispatchEvent(new Event('change'))", to_date_element)
        
        submit_button.click()
        print("Successfully set date range and clicked submit.")
        return True
        
    except (TimeoutException, NoSuchElementException) as e:
        print(f"Error: Could not find form elements or timed out. {e}")
        return False

def scrape_page(driver):
    """Scrapes all announcements from the currently visible page."""
    announcements = []
    try:
        WebDriverWait(driver, 40).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "table[ng-repeat='cann in CorpannData.Table']"))
        )
        print("Announcements table loaded.")
    except TimeoutException:
        print("Timeout: No announcements table found. This might mean no data for the selected range.")
        return [], True 

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    announcement_tables = soup.select("table[ng-repeat='cann in CorpannData.Table']")
    
    if not announcement_tables:
        print("No announcement entries found on this page.")
        return [], True

    for table in announcement_tables:
        subject_tag = table.select_one('span[ng-bind-html="cann.NEWSSUB"]')
        subject = subject_tag.text.strip() if subject_tag else "N/A"

        category_tags = table.select('td.tdcolumngrey')
        category = category_tags[1].text.strip() if len(category_tags) > 1 else "N/A"
        
        pdf_link_tag = table.select_one('a[href*=".pdf"]')
        pdf_link = ""
        if pdf_link_tag and pdf_link_tag.has_attr('href'):
            relative_link = pdf_link_tag['href']
            pdf_link = f"https://www.bseindia.com{relative_link}" if not relative_link.startswith('http') else relative_link

        headline_tag = table.select_one('span[ng-bind-html="cann.HEADLINE"]')
        headline = headline_tag.text.strip() if headline_tag else "N/A"
        
        time_diff_row = table.select_one('tr[ng-if="cann.TimeDiff"]')
        received_time, disseminated_time, time_taken = "N/A", "N/A", "N/A"
        if time_diff_row:
            b_tags = time_diff_row.select('b.ng-binding')
            if len(b_tags) >= 3:
                received_time = b_tags[0].text.strip()
                disseminated_time = b_tags[1].text.strip()
                time_taken = b_tags[2].text.strip()

        announcements.append({
            'Subject': subject,
            'Category': category,
            'PDF_Link': pdf_link,
            'Headline': headline,
            'Received_Time': received_time,
            'Disseminated_Time': disseminated_time,
            'Time_Taken': time_taken
        })

    is_last_page = True
    try:
        pagination_elements = driver.find_elements(By.CSS_SELECTOR, 'b.ng-binding')
        for element in pagination_elements:
            if "Current Page Number" in element.text:
                pagination_text = element.text
                parts = pagination_text.split()
                current_page = int(parts[3])
                total_pages = int(parts[5])
                if current_page < total_pages:
                    is_last_page = False
                break
    except Exception:
        pass

    return announcements, is_last_page

def save_to_csv(data, file_path):
    """Saves the scraped data to a CSV file."""
    headers = ['Subject', 'Category', 'PDF_Link', 'Headline', 'Received_Time', 'Disseminated_Time', 'Time_Taken']
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        print(f"Successfully saved {len(data)} announcements to {file_path}")
    except IOError as e:
        print(f"Error saving to CSV {file_path}: {e}")

def sanitize_filename(filename):
    """Removes illegal characters from a filename."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def download_pdfs(announcements_data, stock_folder_path):
    """Downloads all PDFs from a list of announcements."""
    pdf_dir = os.path.join(stock_folder_path, 'PDFs')
    os.makedirs(pdf_dir, exist_ok=True)
    print(f"Downloading PDFs to: {pdf_dir}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Referer': 'https://www.bseindia.com/'
    }

    for i, announcement in enumerate(announcements_data):
        pdf_url = announcement.get('PDF_Link')
        if pdf_url:
            try:
                date_str = "NA"
                if announcement.get('Received_Time') != "N/A":
                    try:
                       date_obj = datetime.strptime(announcement['Received_Time'].split(' ')[0], '%d-%m-%Y')
                       date_str = date_obj.strftime('%Y-%m-%d')
                    except ValueError:
                       date_str = announcement['Received_Time'].split(' ')[0].replace('/', '-')

                sanitized_subject = sanitize_filename(announcement['Subject'])
                filename = f"{date_str}_{sanitized_subject[:100]}.pdf"
                file_path = os.path.join(pdf_dir, filename)

                print(f"Downloading ({i+1}/{len(announcements_data)}): {filename}")
                
                response = requests.get(pdf_url, stream=True, timeout=30, headers=headers)
                response.raise_for_status() 

                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                time.sleep(1) 

            except requests.exceptions.RequestException as e:
                print(f"  -> Failed to download {pdf_url}. Error: {e}")
            except Exception as e:
                print(f"  -> An unexpected error occurred during download of {pdf_url}: {e}")

def save_to_postgres(ticker, announcements):
    """Saves the scraped data to PostgreSQL database."""
    try:
        from utils.db_connector import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Ensure company exists
        cursor.execute("""
            INSERT INTO companies (ticker, company_name, is_active)
            VALUES (%s, %s, TRUE)
            ON CONFLICT (ticker) DO NOTHING
        """, (ticker, ticker))
        conn.commit()
        
        # 2. Insert announcements
        for ann in announcements:
            subject = ann.get('Subject', 'N/A')
            category = ann.get('Category', 'N/A')
            pdf_link = ann.get('PDF_Link', '')
            headline = ann.get('Headline', 'N/A')
            received_time = ann.get('Received_Time', 'N/A')
            disseminated_time = ann.get('Disseminated_Time', 'N/A')
            
            # Parse dates
            rec_dt = None
            try:
                rec_dt = datetime.strptime(received_time, '%d-%m-%Y %H:%M:%S')
            except Exception:
                try:
                    rec_dt = datetime.strptime(received_time.split(' ')[0], '%d-%m-%Y')
                except Exception:
                    pass
            
            diss_dt = None
            try:
                diss_dt = datetime.strptime(disseminated_time, '%d-%m-%Y %H:%M:%S')
            except Exception:
                try:
                    diss_dt = datetime.strptime(disseminated_time.split(' ')[0], '%d-%m-%Y')
                except Exception:
                    pass
                    
            filing_date = rec_dt.date() if rec_dt else datetime.now().date()
            
            cursor.execute("""
                INSERT INTO financial_reports (ticker, filing_date, report_type, source_url, headline, received_time, disseminated_time, raw_text)
                VALUES (%s, %s, 'Announcement', %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                ticker, 
                filing_date, 
                pdf_link, 
                headline, 
                rec_dt, 
                diss_dt, 
                None
            ))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Successfully saved {len(announcements)} announcements to PostgreSQL.")
    except Exception as e:
        print(f"⚠️ Warning: Could not save announcements to PostgreSQL: {e}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
    stock_map_file = os.path.join(base_dir, 'stock_map.csv')
    data_dir = os.path.join(base_dir, 'data', 'market_data_20250920_225539', 'individual_data')
    debug_dir = os.path.join(base_dir, 'debug_output')
    os.makedirs(debug_dir, exist_ok=True)

    stock_map = get_stock_map(stock_map_file)
    if not stock_map:
        print("Exiting: No stock mapping found. Please create 'stock_map.csv'.")
        return
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return
        
    options = webdriver.ChromeOptions()
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36")
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    for folder_name in os.listdir(data_dir):
        stock_folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(stock_folder_path) and folder_name in stock_map:
            print(f"\n--- Processing: {folder_name} ---")
            
            stock_info = stock_map[folder_name]
            all_announcements = []
            driver = None
            
            try:
                driver = webdriver.Chrome(options=options)
                driver.set_page_load_timeout(60)

                screenshot_path = os.path.join(debug_dir, f"{folder_name}_1_before_click.png")
                if not navigate_to_announcements(driver, stock_info, screenshot_path):
                    continue
                
                try:
                    WebDriverWait(driver, 15).until(EC.number_of_windows_to_be(2))
                    driver.switch_to.window(driver.window_handles[-1])
                    print("Successfully switched to the new announcements tab.")
                except TimeoutException:
                    print("Error: A new tab did not open as expected.")
                    continue

                if not set_date_range(driver):
                    continue
                
                time.sleep(5)

                page_num = 1
                while True:
                    print(f"Scraping page {page_num} for {folder_name}")
                    
                    announcements, is_last_page = scrape_page(driver)
                    all_announcements.extend(announcements)
                    
                    if is_last_page:
                        print(f"Finished scraping all pages for {folder_name}.")
                        break
                    else:
                        try:
                            # **FIX: Scroll to the button and use JavaScript to click**
                            next_button = WebDriverWait(driver, 20).until(
                                EC.presence_of_element_located((By.XPATH, "//a[contains(text(),'Next')]" ))
                            )
                            # Scroll the element into the viewport
                            driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                            time.sleep(1) # Wait a moment for scrolling to finish
                            # Click via JavaScript
                            driver.execute_script("arguments[0].click();", next_button)
                            
                            print("Clicked 'Next' button. Waiting for next page...")
                            time.sleep(5)
                            page_num += 1
                        except (NoSuchElementException, TimeoutException):
                            print("Could not find a clickable 'Next' button. Ending process.")
                            break
                
                if all_announcements:
                    output_csv_path = os.path.join(stock_folder_path, 'corporate_announcements.csv')
                    save_to_csv(all_announcements, output_csv_path)
                    save_to_postgres(folder_name, all_announcements)
                    
                    download_pdfs(all_announcements, stock_folder_path)
                else:
                    print(f"No announcements were found for {folder_name}.")

            except WebDriverException as e:
                print(f"A critical WebDriver error occurred for {folder_name}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred for {folder_name}: {e}")
            finally:
                if driver:
                    driver.quit()
                    print(f"Closed browser for {folder_name}.")
        elif os.path.isdir(stock_folder_path):
            print(f"Skipping folder '{folder_name}': Not found in stock_map.csv")

if __name__ == '__main__':
    main()

