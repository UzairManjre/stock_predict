import os
import csv
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

def create_bhavcopy_lookup(bhavcopy_path):
    """Reads the BSE Bhavcopy CSV and creates a lookup dictionary for validation."""
    print(f"Reading Bhavcopy file from: {bhavcopy_path}")
    try:
        df = pd.read_csv(bhavcopy_path)
        df.columns = [col.strip().upper() for col in df.columns]
        
        lookup = {}
        ticker_col = 'SC_ID' if 'SC_ID' in df.columns else 'TCKRSYMB'
        
        for _, row in df.iterrows():
            ticker = str(row[ticker_col]).strip()
            lookup[ticker] = True # We just need to know if the ticker exists
            
        print(f"Successfully created lookup table with {len(lookup)} stocks.")
        return lookup
    except FileNotFoundError:
        print(f"FATAL ERROR: Bhavcopy file not found at '{bhavcopy_path}'.")
        return None
    except Exception as e:
        print(f"Error reading Bhavcopy file: {e}")
        return None

def get_details_from_url(url):
    """Extracts BseNameSlug, BseTicker, and BseSecurityCode from a URL."""
    match = re.search(r'/stock-share-price/([^/]+)/([^/]+)/([^/]+)/', url)
    return match.groups() if match else (None, None, None)

def generate_stock_map_from_api(data_dir, bhavcopy_lookup, output_csv_path):
    """
    Generates the stock_map.csv by calling the BSE search API directly.
    """
    stock_data_list = []
    folder_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    
    print(f"\nFound {len(folder_names)} stock folders to process.")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Referer': 'https://www.bseindia.com/about.html' # A simple, static page
    }
    
    with requests.Session() as session:
        session.headers.update(headers)

        for i, folder_name in enumerate(folder_names):
            print(f"Processing ({i+1}/{len(folder_names)}): {folder_name}")
            
            ticker_from_folder = folder_name.upper()
            if ticker_from_folder not in bhavcopy_lookup:
                print(f"  -> Warning: Ticker '{ticker_from_folder}' not found in Bhavcopy file. Skipping.")
                continue

            try:
                # Use the API endpoint you discovered
                api_url = f"https://api.bseindia.com/Msource/1D/getQouteSearch.aspx?Type=EQ&text={folder_name}&flag=site"
                response = session.get(api_url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find the first search result link
                first_result = soup.select_one("li.quotemenu a")
                
                if first_result and first_result.has_attr('href'):
                    href = first_result['href']
                    bse_name_slug, bse_ticker, bse_security_code = get_details_from_url(href)
                    
                    if all([bse_name_slug, bse_ticker, bse_security_code]):
                        stock_data_list.append({
                            'FolderName': folder_name,
                            'BseNameSlug': bse_name_slug,
                            'BseTicker': bse_ticker,
                            'BseSecurityCode': bse_security_code
                        })
                        print(f"  -> Success: Found details for {bse_ticker}")
                    else:
                        print(f"  -> Warning: Could not parse details from URL for {folder_name}.")
                else:
                    print(f"  -> Warning: No valid link found in API response for {folder_name}.")

                time.sleep(0.5) # Be polite to the server

            except requests.exceptions.RequestException as e:
                print(f"  -> Error: API request failed for {folder_name}. {e}")

    # Write the collected data to the CSV file
    if stock_data_list:
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['FolderName', 'BseNameSlug', 'BseTicker', 'BseSecurityCode'])
                writer.writeheader()
                writer.writerows(stock_data_list)
            print(f"\nSuccessfully created stock map with {len(stock_data_list)} entries at: {output_csv_path}")
        except IOError as e:
            print(f"\nError writing to CSV file: {e}")
    else:
        print("\nNo stock data was found to write to the CSV file.")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
    data_dir_to_scan = os.path.join(base_dir, 'data', 'market_data_20250920_225539', 'individual_data')
    
    # Using the path you specified
    bhavcopy_csv_filename = os.path.join('Scaper', 'BhavCopy.CSV')
    bhavcopy_full_path = os.path.join(base_dir, bhavcopy_csv_filename)
    
    output_csv_file = os.path.join(base_dir, 'stock_map.csv')
    
    bhavcopy_data = create_bhavcopy_lookup(bhavcopy_full_path)
    
    if bhavcopy_data:
        generate_stock_map_from_api(data_dir_to_scan, bhavcopy_data, output_csv_file)

