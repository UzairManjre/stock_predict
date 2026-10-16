import requests
import time
from requests.exceptions import ReadTimeout, RequestException
from .cookie_manager import CookieManager

class NSEDataFetcher:
    def __init__(self):
        self.cookie_manager = CookieManager()

    def get_all_tickers(self, force_refresh_cookies=False):
        """
        Fetches all NSE tickers with proper authentication
        """
        print("Fetching the latest list of all NSE tickers...")
        
        # Get headers and cookies
        headers, cookies = self.cookie_manager.get_auth_data(force_refresh_cookies)
        
        try:
            # The URL from NSE that lists all securities
            url = 'https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O'
            
            for attempt in range(2):
                try:
                    response = requests.get(url, headers=headers, cookies=cookies, timeout=5)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract symbols from the response
                    tickers = [item['symbol'] for item in data.get('data', [])]
                    
                    # Add the '.NS' suffix required by yfinance
                    nse_tickers = [ticker + ".NS" for ticker in tickers]
                    
                    print(f"  > Found {len(nse_tickers)} total tickers on NSE.")
                    return nse_tickers
                    
                except (ReadTimeout, RequestException) as e:
                    print(f"  > Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(1)
                    
                    # On last attempt failure, try to refresh cookies
                    if attempt == 1:
                        print("  > Refreshing cookies and trying one last time...")
                        headers, cookies = self.cookie_manager.fetch_from_web()
                    
            raise Exception("All attempts to fetch tickers failed")
            
        except Exception as e:
            print(f"  > ERROR: Could not fetch ticker list from NSE. Reason: {e}")
            return []
