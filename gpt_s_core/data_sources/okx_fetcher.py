import os
import time
import hmac
import hashlib
import requests
import pandas as pd
import pandas_ta as ta
from threading import Lock, Timer
from datetime import datetime, timedelta
from typing import Dict, Optional, Union
import logging
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OKXAPIManager:
    def __init__(self):
        """
        Enhanced OKX API manager with built-in technical analysis using pandas-ta.
        
        Features:
        - Rate limiting with dynamic adjustment
        - Automatic retry with exponential backoff
        - Comprehensive technical indicators
        - Caching mechanism
        - Health monitoring
        """
        self.api_key = os.getenv('OKX_API_KEY')
        self.secret_key = os.getenv('OKX_SECRET_KEY')
        self.passphrase = os.getenv('OKX_PASSPHRASE')
        self.base_url = "https://www.okx.com/api/v5"
        
        # Rate limiting configuration
        self.rate_limit = 20  # Requests/second
        self.request_count = 0
        self.last_request = datetime.now()
        self.lock = Lock()
        
        # Caching
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Health monitoring
        self.consecutive_failures = 0
        self.max_retries = 3
        self.backoff_factor = 2
        
        # Technical analysis configuration
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
        """Generate authenticated request headers."""
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
        """Enforce rate limiting with dynamic adjustment."""
        with self.lock:
            now = datetime.now()
            elapsed = (now - self.last_request).total_seconds()
            
            # Dynamic rate adjustment based on recent failures
            effective_rate = self.rate_limit * (0.9 ** self.consecutive_failures)
            min_delay = 1 / effective_rate
            
            if elapsed < min_delay:
                sleep_time = min_delay - elapsed
                time.sleep(sleep_time)
            
            self.last_request = datetime.now()
            self.request_count += 1
            
            # Periodically reset request count
            if self.request_count % 100 == 0:
                self._monitor_health()

    def _monitor_health(self) -> None:
        """Monitor and adjust API health parameters."""
        if self.consecutive_failures > 5:
            logger.warning(f"High failure rate ({self.consecutive_failures} consecutive failures)")
            # Implement circuit breaker if needed
            # time.sleep(60 * self.consecutive_failures)

    def _request_with_retry(self, method: str, path: str, params: Optional[Dict] = None) -> Dict:
        """Execute API request with exponential backoff retry."""
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
                    logger.warning(f"Attempt {attempt + 1} failed for {path}. Retrying in {wait_time}s...")
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
        refresh_ta: bool = True
    ) -> pd.DataFrame:
        """
        Get OHLCV candles with technical indicators.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USDT')
            timeframe: Candle timeframe (e.g., '1m', '5m', '1H', '4H', '1D')
            limit: Number of candles to retrieve (max 300)
            use_cache: Use cached data if available
            refresh_ta: Recalculate technical indicators
        
        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        if use_cache and cache_key in self.cache:
            if datetime.now() - self.cache[cache_key]['timestamp'] < self.cache_ttl:
                logger.debug(f"Returning cached data for {cache_key}")
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
        """Parse API response into DataFrame."""
        df = pd.DataFrame(
            data['data'],
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy']
        )
        
        # Convert types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'volCcy']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set index and sort
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame."""
        try:
            # Run the predefined strategy
            df.ta.strategy(self.ta_strategy)
            
            # Additional custom indicators
            df['50_200_cross'] = df['EMA_50'] > df['EMA_200']
            
            # Volume-based features
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_spike'] = df['volume'] > 2 * df['volume_ma']
            
            return df
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {str(e)}")
            return df

    def get_orderbook(self, symbol: str, depth: int = 25) -> Dict:
        """Get order book data."""
        path = f"/market/books?instId={symbol}&sz={depth}"
        return self._request_with_retry('GET', path)

    def get_account_balance(self) -> Dict:
        """Get account balance information."""
        path = "/account/balance"
        return self._request_with_retry('GET', path)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        logger.info("Cache cleared")

# Example usage
if __name__ == "__main__":
    api = OKXAPIManager()
    
    try:
        # Get candles with technical indicators
        btc_data = api.get_candles("BTC-USDT", "4H", 200)
        print(btc_data.tail())
        
        # Get order book
        orderbook = api.get_orderbook("BTC-USDT")
        print(orderbook)
        
    except Exception as e:
        logger.error(f"API operation failed: {str(e)}")
