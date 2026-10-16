import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

def get_nse_cookies():
    print("Launching headed Chrome browser to bypass Cloudflare...")
    chrome_options = Options()
    # We purposefully do NOT use --headless so Cloudflare sees a real browser
    # and the user can pass CAPTCHA if strictly required.
    chrome_options.add_argument("--window-size=1280,720")
    chrome_options.add_argument("--disable-infobars")
    
    # Setup Chrome
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    print("Navigating to NSE India homepage...")
    driver.get("https://www.nseindia.com/")
    
    print("Waiting 10 seconds to establish session and pass Cloudflare checks...")
    time.sleep(10)
    
    # Extract cookies
    selenium_cookies = driver.get_cookies()
    driver.quit()
    
    print("Session established. Cookies extracted.")
    
    # Format cookies for requests
    cookie_dict = {}
    for cookie in selenium_cookies:
        cookie_dict[cookie['name']] = cookie['value']
        
    return cookie_dict

def fetch_fii_dii_flow():
    cookies = get_nse_cookies()
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/reports/fii-dii"
    }
    
    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    
    print("Fetching FII/DII API endpoint...")
    response = requests.get(url, headers=headers, cookies=cookies)
    
    if response.status_code != 200:
        print(f"Failed to fetch FII/DII data. HTTP Status: {response.status_code}")
        return None, None
        
    try:
        data = response.json()
        print("Raw JSON Data:", data)
        fii_net = None
        dii_net = None
        
        for record in data:
            if record.get('category') == 'FII/FPI':
                fii_net = float(record.get('netValue', 0))
            elif record.get('category') == 'DII':
                dii_net = float(record.get('netValue', 0))
                
        print(f"FII Net Flow: {fii_net} Cr")
        print(f"DII Net Flow: {dii_net} Cr")
        
        return fii_net, dii_net
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(response.text[:500])
        return None, None

if __name__ == "__main__":
    fetch_fii_dii_flow()
