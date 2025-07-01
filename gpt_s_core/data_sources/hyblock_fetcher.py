import os
import requests
from dotenv import load_dotenv

load_dotenv()
HYBLOCK_API_KEY = os.getenv("HYBLOCK_API_KEY")

def get_liquidation_levels(symbol="SOLUSDT", chain="binance-futures"):
    url = "https://api.hyblockcapital.com/v1/liquidation-levels"
    headers = {"Authorization": f"Bearer {HYBLOCK_API_KEY}"}
    params = {"symbol": symbol, "chain": chain}

    print(f"[DEBUG] Fetching {symbol} liquidation levels from {chain}...")

    try:
        response = requests.get(url, headers=headers, params=params)
        print(f"[DEBUG] Status: {response.status_code}")
        print(f"[DEBUG] Sample Data: {response.text[:200]}...")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[HyblockFetcher] ❌ Error: {e}")
        return None
