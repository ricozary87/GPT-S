import os
import time
import hmac
import hashlib
import logging
from threading import Lock
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Union, List, Any

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
        # ✅ Load and validate environment variables
        self.api_key = os.getenv('OKX_API_KEY')
        self.secret_key = os.getenv('OKX_SECRET_KEY')
        self.passphrase = os.getenv('OKX_PASSPHRASE')
        
        if not all([self.api_key, self.secret_key, self.passphrase]):
            raise EnvironmentError("Missing OKX API credentials in environment variables.")

        self.base_url = "https://www.okx.com/api/v5"

        # Rate limiting
        self.rate_limit = 20  # Requests per second
        self.request_count = 0
        self.last_request = datetime.now(timezone.utc)
        self.lock = Lock()

        # Caching
        self.cache: Dict[str, Dict[str, Union[pd.DataFrame, datetime]]] = {}
        self.cache_ttl = timedelta(minutes=5)

        # Health monitoring
        self.consecutive_failures = 0
        self.max_retries = 3
        self.backoff_factor = 2

    def _sign_request(self, method: str, path: str, body: str = '') -> Dict[str, str]:
        """Generate authentication headers for OKX API requests."""
        timestamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
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
        """Implement rate limiting to prevent API bans."""
        with self.lock:
            now = datetime.now(timezone.utc)
            elapsed = (now - self.last_request).total_seconds()
            effective_rate = self.rate_limit * (0.9 ** self.consecutive_failures)
            min_delay = 1 / effective_rate

            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)

            self.last_request = datetime.now(timezone.utc)
            self.request_count += 1

            # Health check every 100 requests
            if self.request_count % 100 == 0:
                self._monitor_health()

    def _monitor_health(self) -> None:
        """Monitor API health and log warnings if failures are high."""
        if self.consecutive_failures > 5:
            logger.warning(f"High failure rate: {self.consecutive_failures} consecutive failures")

    def _request_with_retry(self, method: str, path: str, params: Optional[Dict] = None) -> Dict:
        """Make API requests with retry logic and error handling."""
        for attempt in range(self.max_retries):
            try:
                self._rate_limiter()
                headers = self._sign_request(method, path)

                if method.upper() == 'GET':
                    response = requests.get(
                        f"{self.base_url}{path}",
                        headers=headers,
                        params=params,
                        timeout=10
                    )
                else:
                    response = requests.post(
                        f"{self.base_url}{path}",
                        headers=headers,
                        json=params,
                        timeout=10
                    )

                response.raise_for_status()
                self.consecutive_failures = 0
                return response.json()

            except RequestException as e:
                self.consecutive_failures += 1
                wait_time = self.backoff_factor ** attempt
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed for {path}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Final attempt failed for {path}")
                    raise ConnectionError(f"API request failed after {self.max_retries} attempts: {str(e)}")

    def get_candles(
        self,
        symbol: str,
        timeframe: str = '1H',
        limit: int = 300,
        use_cache: bool = True,
        add_ta: bool = False
    ) -> pd.DataFrame:
        """Get candle data with caching and optional technical analysis."""
        cache_key = f"{symbol}_{timeframe}_{limit}_{add_ta}"
        
        if use_cache and cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if datetime.now(timezone.utc) - cache_entry['timestamp'] < self.cache_ttl:
                return cache_entry['data'].copy()

        try:
            path = f"/market/candles?instId={symbol}&bar={timeframe}&limit={limit}"
            data = self._request_with_retry('GET', path)
            df = self._parse_candles(data)

            if add_ta:
                df = self._add_technical_indicators(df)

            self.cache[cache_key] = {
                'data': df.copy(),
                'timestamp': datetime.now(timezone.utc)
            }

            return df

        except Exception as e:
            logger.error(f"Failed to get candles for {symbol}: {str(e)}")
            raise

    def _parse_candles(self, data: Dict) -> pd.DataFrame:
        """Parse candle data from API response into DataFrame with proper validation."""
        if not data or 'data' not in data or not data['data']:
            raise ValueError("No candle data received from API")

        # OKX API returns 9 columns
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm']
        
        try:
            df = pd.DataFrame(data['data'], columns=columns)
            
            # Convert to numeric types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            
            # Convert timestamp safely
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Failed to parse candle data: {str(e)}")
            raise ValueError("Invalid candle data format")

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators with proper error handling."""
        try:
            # Make a copy to avoid modifying original
            df_ta = df.copy()
            
            # Calculate indicators
            self._calculate_rsi(df_ta)
            self._calculate_macd(df_ta)
            self._calculate_emas(df_ta)
            self._calculate_volume_indicators(df_ta)
            
            return df_ta
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {str(e)}")
            return df

    def _calculate_rsi(self, df: pd.DataFrame) -> None:
        """Calculate RSI indicator."""
        try:
            if len(df) >= 14:
                df['RSI_14'] = df.ta.rsi(length=14)
        except Exception as e:
            logger.warning(f"RSI calculation failed: {str(e)}")

    def _calculate_macd(self, df: pd.DataFrame) -> None:
        """Calculate MACD indicator."""
        try:
            if len(df) >= 26:
                macd = df.ta.macd(fast=12, slow=26, signal=9)
                df['MACD'] = macd['MACD_12_26_9']
                df['MACD_Signal'] = macd['MACDs_12_26_9']
                df['MACD_Hist'] = macd['MACDh_12_26_9']
        except Exception as e:
            logger.warning(f"MACD calculation failed: {str(e)}")

    def _calculate_emas(self, df: pd.DataFrame) -> None:
        """Calculate EMA indicators."""
        try:
            if len(df) >= 200:
                df['EMA_50'] = df.ta.ema(length=50)
                df['EMA_200'] = df.ta.ema(length=200)
                df['EMA_CROSS'] = df['EMA_50'] > df['EMA_200']
        except Exception as e:
            logger.warning(f"EMA calculation failed: {str(e)}")

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> None:
        """Calculate volume-based indicators."""
        try:
            if 'volume' in df.columns:
                df['VOLUME_MA_20'] = df['volume'].rolling(20).mean()
                df['VOLUME_SPIKE'] = df['volume'] > 2 * df['VOLUME_MA_20']
        except Exception as e:
            logger.warning(f"Volume indicators calculation failed: {str(e)}")

    def get_orderbook(self, symbol: str, depth: int = 25) -> Dict:
        """Get order book data for a symbol."""
        path = f"/market/books?instId={symbol}&sz={depth}"
        return self._request_with_retry('GET', path)

    def get_account_balance(self) -> Dict:
        """Get account balance information."""
        path = "/account/balance"
        return self._request_with_retry('GET', path)

    def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a trading pair."""
        try:
            candles = self.get_candles(symbol, timeframe='1m', limit=1, use_cache=False)
            return float(candles.iloc[-1]['close'])
        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {str(e)}")
        return 0.0  # atau None jika ingin bisa dikontrol dari luar

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        logger.info("Cache cleared")


# =========================
# ✅ Public API Interface
# =========================
_api_instance = OKXAPIManager()

def get_candlesticks(
    symbol: str = "BTC-USDT",
    timeframe: str = "1H",
    limit: int = 100,
    add_ta: bool = False,
    use_cache: bool = True
) -> pd.DataFrame:
    """Public interface for getting candle data."""
    return _api_instance.get_candles(symbol, timeframe, limit, use_cache=use_cache, add_ta=add_ta)

def get_orderbook(symbol: str = "BTC-USDT", depth: int = 25) -> Dict:
    """Public interface for getting order book data."""
    return _api_instance.get_orderbook(symbol, depth)

def get_latest_price(symbol: str = "BTC-USDT") -> float:
    """Public interface for getting latest price."""
    return _api_instance.get_latest_price(symbol)
