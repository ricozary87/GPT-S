import os
import time
import hmac
import hashlib
import logging
from threading import Lock
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Union, List
from collections import deque
import requests
import pandas as pd
from dotenv import load_dotenv
from requests.exceptions import RequestException, Timeout, ConnectionError

# ✅ Load environment variables
load_dotenv()

# ✅ Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OKXAPIManager:
    def __init__(self):
        # ✅ Load and validate environment variables
        self.api_key = os.getenv('OKX_API_KEY')
        self.secret_key = os.getenv('OKX_SECRET_KEY')
        self.passphrase = os.getenv('OKX_PASSPHRASE')

        if not all([self.api_key, self.secret_key, self.passphrase]):
            raise EnvironmentError("Missing OKX API credentials in environment variables. Please set OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE.")

        self.base_url = "https://www.okx.com/api/v5"

        # Rate limiting: Using a deque to track request timestamps for per-second limiting
        self.rate_limit_per_second = 20  # OKX public API limit for market data
        self.request_timestamps = deque()
        self.lock = Lock()

        # Caching
        self.cache: Dict[str, Dict[str, Union[pd.DataFrame, datetime]]] = {}
        self.cache_ttl = timedelta(minutes=5)

        # Health monitoring and retry
        self.consecutive_failures = 0
        self.max_retries = 3
        self.backoff_factor = 2  # Exponential backoff base

        logger.info("OKXAPIManager initialized.")

    def _sign_request(self, method: str, path: str, params: Optional[Dict] = None) -> Dict[str, str]:
        """
        Generate authentication headers for OKX API requests.
        Handles GET (query params in path) and POST (JSON body).
        """
        timestamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

        if method.upper() == 'GET':
            body_str = ''
        elif method.upper() == 'POST' and params is not None:
            import json
            body_str = json.dumps(params)
        else:
            body_str = ''

        message = timestamp + method.upper() + path + body_str
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
        """
        Implement rate limiting to prevent API bans.
        Ensures no more than `rate_limit_per_second` requests in a 1-second window.
        """
        with self.lock:
            now = datetime.now(timezone.utc)

            while self.request_timestamps and (now - self.request_timestamps[0]).total_seconds() >= 1:
                self.request_timestamps.popleft()

            if len(self.request_timestamps) >= self.rate_limit_per_second:
                wait_time = 1 - (now - self.request_timestamps[0]).total_seconds()
                if wait_time > 0:
                    logger.debug(f"Rate limit hit. Waiting for {wait_time:.4f}s.")
                    time.sleep(wait_time)
                now = datetime.now(timezone.utc)
                while self.request_timestamps and (now - self.request_timestamps[0]).total_seconds() >= 1:
                    self.request_timestamps.popleft()

            self.request_timestamps.append(now)

    def _request_with_retry(self, method: str, path: str, params: Optional[Dict] = None, timeout: Optional[int] = 10) -> Dict:
        """
        Make API requests with retry logic and error handling.
        """
        full_url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries):
            try:
                self._rate_limiter()

                headers = self._sign_request(method, path, params if method.upper() == 'POST' else None)

                if method.upper() == 'GET':
                    response = requests.get(full_url, headers=headers, params=params, timeout=timeout)
                else:
                    response = requests.post(full_url, headers=headers, json=params, timeout=timeout)

                response.raise_for_status()
                self.consecutive_failures = 0
                return response.json()

            except (RequestException, Timeout, ConnectionError) as e:
                self.consecutive_failures += 1
                wait_time = self.backoff_factor ** attempt
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {full_url}. Retrying in {wait_time:.2f}s. Error: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Final attempt {attempt+1}/{self.max_retries} failed for {full_url}. Error: {e}", exc_info=True)
                    raise ConnectionError(f"API request failed after {self.max_retries} attempts for {full_url}: {str(e)}")
            except Exception as e:
                logger.error(f"An unexpected error occurred during request for {full_url}: {e}", exc_info=True)
                raise

    def get_candles(self, symbol: str, timeframe: str = '1H', limit: int = 300, use_cache: bool = True, add_ta: bool = False, timeout: Optional[int] = 10) -> pd.DataFrame:
        """
        Get candle data with caching and optional technical analysis.
        """
        cache_key = f"{symbol}_{timeframe}_{limit}_{add_ta}"

        if use_cache and cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if datetime.now(timezone.utc) - cache_entry['timestamp'] < self.cache_ttl:
                logger.debug(f"Returning cached data for {symbol} {timeframe}.")
                return cache_entry['data'].copy()

        try:
            path = f"/market/candles?instId={symbol}&bar={timeframe}&limit={limit}"
            data = self._request_with_retry('GET', path, timeout=timeout)
            df = self._parse_candles(data)

            if add_ta:
                df = self._add_technical_indicators(df)

            self.cache[cache_key] = {
                'data': df.copy(),
                'timestamp': datetime.now(timezone.utc)
            }
            logger.info(f"Fetched and cached {len(df)} candles for {symbol} {timeframe}.")
            return df

        except Exception as e:
            logger.error(f"Failed to get candles for {symbol} {timeframe}: {str(e)}", exc_info=True)
            raise

    def _parse_candles(self, data: Dict) -> pd.DataFrame:
        """
        Parse candle data from API response into DataFrame with proper validation.
        """
        if not data or 'data' not in data or not isinstance(data['data'], list) or not data['data']:
            logger.error(f"Invalid or empty candle data structure received: {data}")
            raise ValueError("No valid candle data received from API or data structure is incorrect.")

        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm']
        try:
            df = pd.DataFrame(data['data'], columns=columns)
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=numeric_cols, inplace=True)
            if df.empty:
                raise ValueError("No valid numeric candle data remaining after parsing and sanitization.")

            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp'], errors='coerce'), unit='ms')
            df.dropna(subset=['timestamp'], inplace=True)
            if df.empty:
                raise ValueError("No valid timestamped candle data remaining after parsing and sanitization.")

            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']]

        except KeyError as ke:
            logger.error(f"Missing expected column in API response: {ke}. Raw data sample: {data['data'][:1]}", exc_info=True)
            raise ValueError(f"Invalid candle data format: Missing column {ke}")
        except Exception as e:
            logger.error(f"Failed to parse candle data into DataFrame: {str(e)}", exc_info=True)
            raise ValueError("Invalid candle data format or unexpected parsing error.")

# =========================
# ✅ Public API Interface (Singleton pattern)
# =========================
_api_instance = OKXAPIManager()

def get_candlesticks(symbol: str = "BTC-USDT", timeframe: str = "1H", limit: int = 100, add_ta: bool = False, use_cache: bool = True, timeout: Optional[int] = None) -> pd.DataFrame:
    """Public interface for getting candle data."""
    return _api_instance.get_candles(symbol, timeframe, limit, use_cache=use_cache, add_ta=add_ta, timeout=timeout)