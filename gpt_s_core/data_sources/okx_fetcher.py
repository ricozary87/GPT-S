import os
import requests
from dotenv import load_dotenv

load_dotenv()
OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_API_SECRET = os.getenv("OKX_API_SECRET")
OKX_API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE")

BASE_URL = "https://www.okx.com"

def get_candlesticks(inst_id="SOL-USDT", bar="1m", limit=100):
    url = f"{BASE_URL}/api/v5/market/candles"
    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": limit
    }

    try:
        response = requests.get(url, params=params)
        print(f"[DEBUG] Candles Status: {response.status_code}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[OKX Fetcher] ❌ Error get_candlesticks: {e}")
        return None

def get_orderbook(inst_id="SOL-USDT", depth=10):
    url = f"{BASE_URL}/api/v5/market/books"
    params = {
        "instId": inst_id,
        "sz": depth
    }

    try:
        response = requests.get(url, params=params)
        print(f"[DEBUG] Orderbook Status: {response.status_code}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[OKX Fetcher] ❌ Error get_orderbook: {e}")
        return None
