import json
import os
from playwright.sync_api import sync_playwright

class CookieManager:
    def __init__(self, cache_file="nse_cookies.json"):
        self.cache_file = cache_file

    def fetch_from_web(self):
        """Fetches fresh cookies and headers using Playwright"""
        print("🌐 Launching browser to fetch NSE cookies...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            print("⏳ Visiting NSE website...")
            page.goto("https://www.nseindia.com/get-quotes/equity", timeout=60000)
            page.wait_for_timeout(5000)

            cookies_list = context.cookies()
            cookies = {cookie['name']: cookie['value'] for cookie in cookies_list}

            headers = {
                "user-agent": page.evaluate("() => navigator.userAgent"),
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, br",
                "accept-language": "en-US,en;q=0.9",
                "referer": "https://www.nseindia.com",
            }

            browser.close()
            print("✅ Cookies and headers fetched successfully")

            self._save_to_cache(headers, cookies)
            return headers, cookies

    def get_auth_data(self, force_refresh=False):
        """Gets headers either from cache or fresh from web"""
        if not force_refresh and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    print("📦 Loaded headers + cookies from cache")
                    return data["headers"], data["cookies"]
            except Exception as e:
                print(f"⚠️ Failed to load cache: {e}, falling back to fresh fetch.")

        return self.fetch_from_web()

    def _save_to_cache(self, headers, cookies):
        """Saves the headers and cookies to cache file"""
        with open(self.cache_file, "w") as f:
            json.dump({"headers": headers, "cookies": cookies}, f, indent=2)
