import os
import time
import hmac
import hashlib
import logging
from threading import Lock, Timer
from datetime import datetime, timedelta
from typing import Dict, Optional, Union

import requests
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from requests.exceptions import RequestException

# ✅ Load environment variables
load_dotenv()

# ✅ Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OKXAPIManager:
    def __init__(self):
        self.api_key = os.getenv('OKX_API_KEY')
        self.secret_key = os.getenv('OKX_SECRET_KEY')
        self.passphrase = os.getenv('OKX_PASSPHRASE')
        self.base_url = "https://www.okx.com/api/v5"

        # Rate limiting
        self.rate_limit = 20
        self.request_count = 0
        self.last_request = datetime.now()
        self.lock = Lock()

        # Caching
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_ttl = timedelta(minutes=5)

        # Health
        self.consecutive_failures = 0
        self.max_retries = 3
        self.backoff_factor = 2

        # TA Strategy
        self.ta_strategy = ta.Strategy(
            name="SMC Strategy",
            description="Smart Money Concept Technical Indicators",
            ta=[
                {"kind": "rsi", "length": 14},
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                {"kind": "bbands", "length": 20, "std": 2},
                {"kind": "obv"},
                {"kind": "vwap"},
                {"kind": "supertrend", "period": 10, "multiplier": 3},
                {"kind": "ema", "length": 50},
                {"kind": "ema", "length": 200},
            ]
        )

    def _sign_request(self, method: str, path: str, body: str = '') -> Dict[str, str]:
        timestamp = datetime.utcnow().isoformat()[:-3] + 'Z'
        message = timestamp + method.upper() + path + str(body)
        signature = hmac.new(
            bytes(self.secret_key, 'utf-8'),
            bytes(message, 'utf-8'),
            hashlib.sha256
        ).hexdigest()

        return {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': signature,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }

    def _rate_limiter(self) -> None:
        with self.lock:
            now = datetime.now()
            elapsed = (now - self.last_request).total_seconds()
            effective_rate = self.rate_limit * (0.9 ** self.consecutive_failures)
            min_delay = 1 / effective_rate

            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)

            self.last_request = datetime.now()
            self.request_count += 1

            if self.request_count % 100 == 0:
                self._monitor_health()

    def _monitor_health(self) -> None:
        if self.consecutive_failures > 5:
            logger.warning(f"High failure rate: {self.consecutive_failures} failures")

    def _request_with_retry(self, method: str, path: str, params: Optional[Dict] = None) -> Dict:
        for attempt in range(self.max_retries):
            try:
                self._rate_limiter()
                headers = self._sign_request(method, path)

                if method.upper() == 'GET':
                    response = requests.get(f"{self.base_url}{path}", headers=headers, params=params, timeout=10)
                else:
                    response = requests.post(f"{self.base_url}{path}", headers=headers, json=params, timeout=10)

                response.raise_for_status()
                self.consecutive_failures = 0
                return response.json()

            except RequestException as e:
                self.consecutive_failures += 1
                wait_time = self.backoff_factor ** attempt
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Final attempt failed for {path}")
                    raise ConnectionError(f"API request failed: {str(e)}")

    def get_candles(self, symbol: str, timeframe: str = '1H', limit: int = 300,
                    use_cache: bool = True, refresh_ta: bool = True) -> pd.DataFrame:
        cache_key = f"{symbol}_{timeframe}_{limit}"
        if use_cache and cache_key in self.cache:
            if datetime.now() - self.cache[cache_key]['timestamp'] < self.cache_ttl:
                return self.cache[cache_key]['data'].copy()

        try:
            path = f"/market/candles?instId={symbol}&bar={timeframe}&limit={limit}"
            data = self._request_with_retry('GET', path)
            df = self._parse_candles(data)

            if refresh_ta:
                df = self._add_technical_indicators(df)

            self.cache[cache_key] = {
                'data': df.copy(),
                'timestamp': datetime.now()
            }

            return df

        except Exception as e:
            logger.error(f"Failed to get candles for {symbol}: {str(e)}")
            raise

    def _parse_candles(self, data: Dict) -> pd.DataFrame:
        df = pd.DataFrame(
            data['data'],
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy']
        )
        df[['open', 'high', 'low', 'close', 'volume', 'volCcy']] = df[['open', 'high', 'low', 'close', 'volume', 'volCcy']].apply(pd.to_numeric)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df.ta.strategy(self.ta_strategy)
            df['50_200_cross'] = df['EMA_50'] > df['EMA_200']
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_spike'] = df['volume'] > 2 * df['volume_ma']
            return df
        except Exception as e:
            logger.error(f"Technical analysis failed: {str(e)}")
            return df

    def get_orderbook(self, symbol: str, depth: int = 25) -> Dict:
        path = f"/market/books?instId={symbol}&sz={depth}"
        return self._request_with_retry('GET', path)

    def get_account_balance(self) -> Dict:
        path = "/account/balance"
        return self._request_with_retry('GET', path)

    def clear_cache(self) -> None:
        self.cache.clear()
        logger.info("Cache cleared")


# =========================
# ✅ ALIAS for external usage (unit test)
# =========================
_api_instance = OKXAPIManager()

def get_candlesticks(symbol="BTC-USDT", timeframe="1H", limit=100) -> pd.DataFrame:
    return _api_instance.get_candles(symbol, timeframe, limit)

def get_orderbook(symbol="BTC-USDT") -> Dict:
    return _api_instance.get_orderbook(symbol)
