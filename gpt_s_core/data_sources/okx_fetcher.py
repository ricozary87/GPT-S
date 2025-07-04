import os
import time
import hmac
import hashlib
import logging
from threading import Lock
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Union, List, Any
from collections import deque # Added for rate limiting

import requests
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from requests.exceptions import RequestException, Timeout, ConnectionError # Import specific exceptions

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
        self.request_timestamps = deque() # Stores timestamps of recent requests
        self.lock = Lock() # For thread-safety

        # Caching
        self.cache: Dict[str, Dict[str, Union[pd.DataFrame, datetime]]] = {}
        self.cache_ttl = timedelta(minutes=5)

        # Health monitoring and retry
        self.consecutive_failures = 0
        self.max_retries = 3
        self.backoff_factor = 2 # Exponential backoff base

        logger.info("OKXAPIManager initialized.")

    def _sign_request(self, method: str, path: str, params: Optional[Dict] = None) -> Dict[str, str]:
        """
        Generate authentication headers for OKX API requests.
        Handles GET (query params in path) and POST (JSON body).
        """
        timestamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

        if method.upper() == 'GET':
            # For GET, the path already contains query parameters, so body is empty
            body_str = ''
        elif method.upper() == 'POST' and params is not None:
            # For POST, the body is the JSON string of params
            import json # Import json locally for this method
            body_str = json.dumps(params)
        else:
            body_str = '' # No body for other methods or empty POST

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
            'Content-Type': 'application/json' # Always specify for consistency
        }

    def _rate_limiter(self) -> None:
        """
        Implement rate limiting to prevent API bans.
        Ensures no more than `rate_limit_per_second` requests in a 1-second window.
        """
        with self.lock:
            now = datetime.now(timezone.utc)
            
            # Remove timestamps older than 1 second
            while self.request_timestamps and (now - self.request_timestamps[0]).total_seconds() >= 1:
                self.request_timestamps.popleft()

            # If we've hit the limit for the current second, wait
            if len(self.request_timestamps) >= self.rate_limit_per_second:
                # Calculate time until the oldest request in the window expires
                wait_time = 1 - (now - self.request_timestamps[0]).total_seconds()
                if wait_time > 0:
                    logger.debug(f"Rate limit hit. Waiting for {wait_time:.4f}s.")
                    time.sleep(wait_time)
                # After waiting, re-check and remove old timestamps again
                now = datetime.now(timezone.utc)
                while self.request_timestamps and (now - self.request_timestamps[0]).total_seconds() >= 1:
                    self.request_timestamps.popleft()

            # Add current request timestamp
            self.request_timestamps.append(now)
            self.last_request = now # Keep track of last request for health monitoring (less critical now)

            # Health check based on consecutive failures, not request count
            # self._monitor_health() # Removed from here, called after request attempt

    def _monitor_health(self) -> None:
        """Monitor API health and log warnings if failures are high."""
        if self.consecutive_failures > 5:
            logger.warning(f"High failure rate: {self.consecutive_failures} consecutive failures. Consider reducing request rate or checking network.")

    def _request_with_retry(self, method: str, path: str, params: Optional[Dict] = None) -> Dict:
        """Make API requests with retry logic and error handling."""
        full_url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries):
            try:
                self._rate_limiter() # Apply rate limit before each attempt

                # For GET requests, params are part of the URL path for signing
                # For POST requests, params are the JSON body
                headers = self._sign_request(method, path, params if method.upper() == 'POST' else None)

                if method.upper() == 'GET':
                    response = requests.get(
                        full_url,
                        headers=headers,
                        params=params, # requests handles params encoding for GET
                        timeout=10
                    )
                else: # POST
                    response = requests.post(
                        full_url,
                        headers=headers,
                        json=params, # requests handles JSON serialization for POST
                        timeout=10
                    )

                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                self.consecutive_failures = 0 # Reset on success
                return response.json()

            except (RequestException, Timeout, ConnectionError) as e:
                self.consecutive_failures += 1
                wait_time = self.backoff_factor ** attempt # Exponential backoff
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {full_url}. Retrying in {wait_time:.2f}s. Error: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Final attempt {attempt+1}/{self.max_retries} failed for {full_url}. Error: {e}", exc_info=True)
                    # After all retries, re-raise a custom exception for orchestrator to handle
                    raise ConnectionError(f"API request failed after {self.max_retries} attempts for {full_url}: {str(e)}")
            except Exception as e:
                # Catch any other unexpected errors (e.g., JSON parsing issues)
                logger.error(f"An unexpected error occurred during request for {full_url}: {e}", exc_info=True)
                raise # Re-raise unexpected errors immediately

    def get_candles(
        self,
        symbol: str,
        timeframe: str = '1H',
        limit: int = 300,
        use_cache: bool = True,
        add_ta: bool = False,
    ) -> pd.DataFrame:
        """Get candle data with caching and optional technical analysis."""
        cache_key = f"{symbol}_{timeframe}_{limit}_{add_ta}"
        
        if use_cache and cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if datetime.now(timezone.utc) - cache_entry['timestamp'] < self.cache_ttl:
                logger.debug(f"Returning cached data for {symbol} {timeframe}.")
                return cache_entry['data'].copy() # Return a copy to prevent external modification

        try:
            # OKX API path for candles. Query parameters are part of the path for signing.
            path = f"/market/candles?instId={symbol}&bar={timeframe}&limit={limit}"
            # No 'params' dict for GET request, as they are part of the path
            data = self._request_with_retry('GET', path)
            df = self._parse_candles(data)

            if add_ta:
                df = self._add_technical_indicators(df)

            # Store in cache
            self.cache[cache_key] = {
                'data': df.copy(),
                'timestamp': datetime.now(timezone.utc)
            }
            logger.info(f"Fetched and cached {len(df)} candles for {symbol} {timeframe}.")
            return df

        except Exception as e:
            logger.error(f"Failed to get candles for {symbol} {timeframe}: {str(e)}", exc_info=True)
            # Re-raise to allow higher-level error handling (e.g., in orchestrator)
            raise

    def _parse_candles(self, data: Dict) -> pd.DataFrame:
        """Parse candle data from API response into DataFrame with proper validation."""
        if not data or 'data' not in data or not isinstance(data['data'], list) or not data['data']:
            logger.error(f"Invalid or empty candle data structure received: {data}")
            raise ValueError("No valid candle data received from API or data structure is incorrect.")

        # OKX API returns 9 columns
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm']
        
        try:
            df = pd.DataFrame(data['data'], columns=columns)
            
            # Convert to numeric types, coercing errors to NaN
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Drop rows with NaN values in critical numeric columns after coercion
            initial_rows = len(df)
            df.dropna(subset=numeric_cols, inplace=True)
            if len(df) < initial_rows:
                logger.warning(f"Dropped {initial_rows - len(df)} rows due to non-numeric price/volume data during parsing.")

            if df.empty:
                raise ValueError("No valid numeric candle data remaining after parsing and sanitization.")

            # Convert timestamp safely, coercing errors
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp'], errors='coerce'), unit='ms')
            df.dropna(subset=['timestamp'], inplace=True) # Drop rows with invalid timestamps

            if df.empty:
                raise ValueError("No valid timestamped candle data remaining after parsing and sanitization.")

            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True) # Ensure chronological order by timestamp index
            
            # Return only the essential OHLCV columns
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except KeyError as ke:
            logger.error(f"Missing expected column in API response: {ke}. Raw data sample: {data['data'][:1]}", exc_info=True)
            raise ValueError(f"Invalid candle data format: Missing column {ke}")
        except Exception as e:
            logger.error(f"Failed to parse candle data into DataFrame: {str(e)}", exc_info=True)
            raise ValueError("Invalid candle data format or unexpected parsing error.")

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators with proper error handling."""
        if df.empty:
            logger.warning("DataFrame is empty, cannot add technical indicators.")
            return df

        # Make a copy to avoid modifying original and potential SettingWithCopyWarning
        df_ta = df.copy()
        
        try:
            self._calculate_rsi(df_ta)
            self._calculate_macd(df_ta)
            self._calculate_emas(df_ta)
            self._calculate_volume_indicators(df_ta)
            
            return df_ta
            
        except Exception as e:
            logger.error(f"Technical analysis failed for some indicators: {str(e)}", exc_info=True)
            # Return df_ta even if some indicators failed, it will contain successfully calculated ones
            return df_ta

    def _calculate_rsi(self, df: pd.DataFrame) -> None:
        """Calculate RSI indicator."""
        try:
            # RSI needs at least 14 periods
            if len(df) >= 14:
                df['RSI_14'] = ta.rsi(df['close'], length=14)
            else:
                logger.debug("Not enough data for RSI_14 calculation.")
        except Exception as e:
            logger.warning(f"RSI calculation failed: {str(e)}", exc_info=True)

    def _calculate_macd(self, df: pd.DataFrame) -> None:
        """Calculate MACD indicator."""
        try:
            # MACD needs at least 26 periods for default settings
            if len(df) >= 26:
                macd_results = ta.macd(df['close'], fast=12, slow=26, signal=9)
                # Check if macd_results is not None/empty before assigning
                if macd_results is not None and not macd_results.empty:
                    df['MACD'] = macd_results['MACD_12_26_9']
                    df['MACD_Signal'] = macd_results['MACDs_12_26_9']
                    df['MACD_Hist'] = macd_results['MACDh_12_26_9']
                else:
                    logger.warning("MACD calculation returned empty or None results.")
            else:
                logger.debug("Not enough data for MACD calculation.")
        except Exception as e:
            logger.warning(f"MACD calculation failed: {str(e)}", exc_info=True)

    def _calculate_emas(self, df: pd.DataFrame) -> None:
        """Calculate EMA indicators."""
        try:
            # EMA_50 needs 50 periods, EMA_200 needs 200
            if len(df) >= 200: # Ensure enough data for the longest EMA
                df['EMA_50'] = ta.ema(df['close'], length=50)
                df['EMA_200'] = ta.ema(df['close'], length=200)
                # Ensure EMAs are not NaN before comparison
                if not df['EMA_50'].iloc[-1] is np.nan and not df['EMA_200'].iloc[-1] is np.nan:
                    df['EMA_CROSS'] = df['EMA_50'] > df['EMA_200']
                else:
                    logger.debug("EMA_50 or EMA_200 resulted in NaN, EMA_CROSS not calculated.")
            else:
                logger.debug("Not enough data for EMA_50 or EMA_200 calculation.")
        except Exception as e:
            logger.warning(f"EMA calculation failed: {str(e)}", exc_info=True)

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> None:
        """Calculate volume-based indicators."""
        try:
            if 'volume' in df.columns and len(df) >= 20:
                df['VOLUME_MA_20'] = df['volume'].rolling(20).mean()
                # Ensure VOLUME_MA_20 is not NaN before comparison
                if not df['VOLUME_MA_20'].iloc[-1] is np.nan:
                    df['VOLUME_SPIKE'] = df['volume'] > 2 * df['VOLUME_MA_20']
                else:
                    logger.debug("VOLUME_MA_20 resulted in NaN, VOLUME_SPIKE not calculated.")
            else:
                logger.debug("Not enough data or 'volume' column missing for volume indicators.")
        except Exception as e:
            logger.warning(f"Volume indicators calculation failed: {str(e)}", exc_info=True)

    def get_orderbook(self, symbol: str, depth: int = 25) -> Dict:
        """Get order book data for a symbol."""
        path = f"/market/books?instId={symbol}&sz={depth}"
        return self._request_with_retry('GET', path)

    def get_account_balance(self) -> Dict:
        """Get account balance information."""
        path = "/account/balance"
        return self._request_with_retry('GET', path)

    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a trading pair.
        Uses cached 1m candles if available and fresh, otherwise fetches.
        """
        try:
            # Try to get from cache first with a very short TTL for "latest" price
            # Using get_candles with use_cache=True and a very short TTL for this specific call
            # This requires a separate cache_ttl for get_latest_price if you want it different from main candles.
            # For simplicity, we'll just use the main cache_ttl for now.
            # Or, for truly latest, always fetch without cache.
            # For this function, it's better to bypass cache or have a very short specific cache.
            # Let's fetch without cache for "latest" price.
            candles = self.get_candles(symbol, timeframe='1m', limit=1, use_cache=False) # Always fetch fresh
            if not candles.empty:
                return float(candles['close'].iloc[-1])
            else:
                logger.warning(f"No latest price data returned for {symbol}.")
                return 0.0
        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {str(e)}", exc_info=True)
            return 0.0

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        logger.info("Cache cleared.")


# =========================
# ✅ Public API Interface (Singleton pattern)
# =========================
_api_instance = OKXAPIManager()

# Expose the get_candles method as get_candlesticks for external use
def get_candlesticks(
    symbol: str = "BTC-USDT",
    timeframe: str = "1H",
    limit: int = 100,
    add_ta: bool = False,
    use_cache: bool = True,
    timeout: Optional[int] = None
) -> pd.DataFrame:
    """Public interface for getting candle data."""
    return _api_instance.get_candles(symbol, timeframe, limit, use_cache=use_cache, add_ta=add_ta, timeout=timeout)

# Expose other methods as needed
def get_orderbook(symbol: str = "BTC-USDT", depth: int = 25) -> Dict:
    """Public interface for getting order book data."""
    return _api_instance.get_orderbook(symbol, depth)

def get_latest_price(symbol: str = "BTC-USDT") -> float:
    """Public interface for getting latest price."""
    return _api_instance.get_latest_price(symbol)
