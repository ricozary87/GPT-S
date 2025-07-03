import pandas as pd
import pandas_ta as ta
from typing import List, Dict, Optional, Tuple, Deque
from dataclasses import dataclass, field
import logging
import numpy as np
from collections import deque
import dask.dataframe as dd
import unittest
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smc_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MarketStructure:
    recent_high: float
    recent_low: float
    current_close: float
    bos: bool = False
    choch: bool = False
    fvg: Optional[Dict] = None
    order_blocks: List[Dict] = field(default_factory=list)
    liquidity: Dict = field(default_factory=lambda: {
        "equal_highs": [], "equal_lows": [], "volume_clusters": []
    })
    volume_profile: Dict = field(default_factory=lambda: {
        "high_volume_nodes": [], "low_volume_nodes": []
    })
    trend_structure: str = "neutral"
    swing_highs_lows: Dict = field(default_factory=lambda: {
        "highs": [], "lows": []
    })
    imbalance_strength: float = 0.0

class SMAnalyzer:
    def __init__(self, 
                 config_path: str = "smc_config.json",
                 min_candles: int = 50,
                 volume_multiplier: float = 2.5,
                 fvg_min_strength: float = 0.5,
                 liquidity_lookback: int = 20,
                 swing_window: int = 5,
                 real_time_mode: bool = False):
        """
        Enhanced Smart Money Concept analyzer with optimizations for performance and scalability.
        
        Args:
            config_path: Path to configuration file
            min_candles: Minimum candles needed for reliable analysis
            volume_multiplier: Volume threshold multiplier for order blocks
            fvg_min_strength: Minimum percentage strength for valid FVG
            liquidity_lookback: Number of candles to analyze for liquidity zones
            swing_window: Window size for swing point detection
            real_time_mode: Enable streaming data optimizations
        """
        # Load configuration
        self.config = self._load_config(config_path, {
            "min_candles": min_candles,
            "volume_multiplier": volume_multiplier,
            "fvg_min_strength": fvg_min_strength,
            "liquidity_lookback": liquidity_lookback,
            "swing_window": swing_window
        })
        
        self.real_time_mode = real_time_mode
        self.recent_signals = deque(maxlen=100)
        self.state = {}  # For maintaining state in real-time processing

    def _load_config(self, path: str, defaults: Dict) -> Dict:
        """Load configuration from file or use defaults"""
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
            return defaults
        except Exception as e:
            logger.error(f"Config load failed: {str(e)}, using defaults")
            return defaults

    def _validate_data(self, candles: List) -> Optional[pd.DataFrame]:
        """Validate and prepare candle data with enhanced checks and sanitization"""
        if len(candles) < self.config["min_candles"]:
            logger.warning(f"Insufficient data: {len(candles)} candles")
            return None

        try:
            required_columns = ["ts", "open", "high", "low", "close", "vol"]
            df = pd.DataFrame(candles, columns=required_columns)
            
            # Convert and validate numeric columns
            numeric_cols = ["open", "high", "low", "close", "vol"]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Data sanitization
            df = df[df["high"] > df["low"]]  # Remove invalid candles
            df = df[df["vol"] > 0]           # Remove zero volume
            
            if df.empty or df[numeric_cols].isnull().any().any():
                logger.error("Invalid data after sanitization")
                return None
                
            # Validate timestamps
            df["ts"] = pd.to_datetime(df["ts"], unit='ms', errors='coerce')
            if df["ts"].isnull().any():
                logger.error("Invalid timestamp data")
                return None
                
            return df.sort_values('ts').reset_index(drop=True)  # Ensure chronological order
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}", exc_info=True)
            return None

    def detect_structure(self, candles: List) -> Optional[MarketStructure]:
        """
        Enhanced market structure detection with optimizations for large datasets.
        
        Usage:
        1. Initialize SMAnalyzer with desired parameters
        2. Pass candle data as list of [timestamp, open, high, low, close, volume]
        3. Interpret results:
           - bos: Break of Structure (bullish signal)
           - choch: Change of Character (bearish signal)
           - fvg: Fair Value Gap (price imbalance zone)
           - order_blocks: Significant institutional order areas
           - liquidity: Key liquidity zones (equal highs/lows, volume clusters)
           - volume_profile: Significant volume concentration areas
           - trend_structure: Overall market trend (uptrend/downtrend/range)
        """
        try:
            if self.real_time_mode:
                return self._process_incremental(candles)
                
            df = self._validate_data(candles)
            if df is None:
                return None

            # Use Dask for large datasets
            if len(df) > 10000:
                ddf = dd.from_pandas(df, npartitions=4)
                structure = self._process_with_dask(ddf)
            else:
                structure = self._process_dataframe(df)
                
            # Store for pattern recognition
            self.recent_signals.append(structure)
            return structure
            
        except ValueError as e:
            logger.error(f"Value error: {e}")
            # Fallback: Try processing with reduced features
            return self._fallback_processing(candles)
        except Exception as e:
            logger.error(f"Unhandled exception: {e}")
            # Retry mechanism
            return self._retry_processing(candles)

    def _process_incremental(self, new_candles: List) -> Optional[MarketStructure]:
        """Process data incrementally for real-time streaming"""
        # Initialize state if needed
        if not self.state:
            self.state = {
                "df": pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol"]),
                "last_structure": None
            }
            
        # Validate and sanitize new candles
        new_df = self._validate_data(new_candles)
        if new_df is None:
            return self.state["last_structure"]
            
        # Append to existing data
        self.state["df"] = pd.concat([self.state["df"], new_df]).drop_duplicates("ts")
        
        # Process only if we have enough data
        if len(self.state["df"]) < self.config["min_candles"]:
            return self.state["last_structure"]
            
        # Process the last N candles for efficiency
        lookback = min(200, len(self.state["df"]))
        recent_df = self.state["df"].iloc[-lookback:]
        structure = self._process_dataframe(recent_df)
        self.state["last_structure"] = structure
        return structure

    def _process_with_dask(self, ddf: dd.DataFrame) -> MarketStructure:
        """Process large datasets using Dask parallel processing"""
        # Compute basic structure in parallel
        recent_high = ddf["high"].max().compute()
        recent_low = ddf["low"].min().compute()
        current_close = ddf["close"].iloc[-1].compute()
        
        structure = MarketStructure(
            recent_high=recent_high,
            recent_low=recent_low,
            current_close=current_close
        )
        
        # Compute other features in parallel
        structure.trend_structure = self._detect_trend_structure(ddf)
        structure.bos = current_close > recent_high
        structure.choch = (not structure.bos) and (current_close < recent_low)
        
        # Note: Some complex features are simplified for large datasets
        structure.order_blocks = self._detect_order_blocks(ddf)
        return structure

    def _process_dataframe(self, df: pd.DataFrame) -> MarketStructure:
        """Process standard-sized DataFrame"""
        structure = MarketStructure(
            recent_high=df["high"].iloc[-2],
            recent_low=df["low"].iloc[-2],
            current_close=df["close"].iloc[-1]
        )

        structure.trend_structure = self._detect_trend_structure(df)
        structure.bos = structure.current_close > structure.recent_high
        structure.choch = (not structure.bos) and (structure.current_close < structure.recent_low)
        structure.fvg = self._detect_strong_fvg(df)
        structure.order_blocks = self._detect_order_blocks(df)
        structure.liquidity = self._detect_liquidity_zones(df)
        structure.volume_profile = self._analyze_volume_profile(df)
        structure.swing_highs_lows = self._identify_swing_points(df)
        
        return structure

    def _fallback_processing(self, candles: List) -> Optional[MarketStructure]:
        """Simplified processing when main method fails"""
        try:
            if len(candles) < 3:
                return None
                
            return MarketStructure(
                recent_high=max(c[2] for c in candles[-5:]),
                recent_low=min(c[3] for c in candles[-5:]),
                current_close=candles[-1][4],
                bos=candles[-1][4] > candles[-2][2],
                choch=candles[-1][4] < candles[-2][3]
            )
        except Exception:
            return None

    def _retry_processing(self, candles: List, retries: int = 2) -> Optional[MarketStructure]:
        """Retry mechanism for processing"""
        for i in range(retries):
            try:
                logger.info(f"Retry attempt {i+1}")
                return self.detect_structure(candles)
            except Exception:
                pass
        return None

    def _detect_trend_structure(self, df: pd.DataFrame) -> str:
        """Determine trend using EMA cross and price structure"""
        if len(df) < 20:
            return "neutral"
            
        try:
            # Calculate EMA trend
            ema_trend = self._calculate_ema_trend(df)
            
            # Check price structure
            return self._assess_price_structure(df, ema_trend)
        except Exception as e:
            logger.error(f"Trend detection failed: {str(e)}")
            return "neutral"

    def _calculate_ema_trend(self, df: pd.DataFrame) -> str:
        """Calculate EMA-based trend direction"""
        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)
        return "up" if df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1] else "down"

    def _assess_price_structure(self, df: pd.DataFrame, ema_trend: str) -> str:
        """Assess price structure for trend confirmation"""
        lookback = min(10, len(df) // 3)
        
        # Check for higher highs/lows
        if ema_trend == "up":
            for i in range(1, lookback):
                if (df['high'].iloc[-i] > df['high'].iloc[-i-1] or 
                    df['low'].iloc[-i] > df['low'].iloc[-i-1]):
                    return "uptrend"
                    
        # Check for lower highs/lows
        elif ema_trend == "down":
            for i in range(1, lookback):
                if (df['high'].iloc[-i] < df['high'].iloc[-i-1] or 
                    df['low'].iloc[-i] < df['low'].iloc[-i-1]):
                    return "downtrend"
        return "range"

    def _detect_strong_fvg(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect significant FVGs with multi-candle check"""
        if len(df) < 3:
            return None

        try:
            # Look for FVGs in recent 5 candles
            for i in range(1, min(6, len(df)-1):
                idx = -i  # Current candle index
                
                # Bullish FVG: Current low > previous high
                if df["low"].iloc[idx] > df["high"].iloc[idx-1]:
                    strength = (df["low"].iloc[idx] - df["high"].iloc[idx-1]) / df["high"].iloc[idx-1] * 100
                    if strength >= self.config["fvg_min_strength"]:
                        return {
                            "type": "bullish",
                            "range": (df["high"].iloc[idx-1], df["low"].iloc[idx]),
                            "strength": round(strength, 4),
                            "age": i
                        }
                
                # Bearish FVG: Current high < previous low
                elif df["high"].iloc[idx] < df["low"].iloc[idx-1]:
                    strength = (df["low"].iloc[idx-1] - df["high"].iloc[idx]) / df["low"].iloc[idx-1] * 100
                    if strength >= self.config["fvg_min_strength"]:
                        return {
                            "type": "bearish",
                            "range": (df["high"].iloc[idx], df["low"].iloc[idx-1]),
                            "strength": round(strength, 4),
                            "age": i
                        }
            return None
        except Exception as e:
            logger.error(f"FVG detection failed: {str(e)}")
            return None

    def _detect_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Detect significant order blocks with volume confirmation"""
        blocks = []
        if len(df) < 20:
            return blocks
            
        try:
            # Calculate volume thresholds
            df['vol_ma'] = df['vol'].rolling(20).mean()
            df['vol_upper'] = self.config["volume_multiplier"] * df['vol_ma']
            
            # Find significant volume spikes with price confirmation
            vol_mask = df['vol'] > df['vol_upper']
            price_mask = (df['close'] > df['open']) | (df['close'] < df['open'])
            ob_candles = df[vol_mask & price_mask]
            
            for idx, row in ob_candles.iterrows():
                # Focus on recent blocks
                if idx < len(df) - 50:  
                    continue
                    
                blocks.append({
                    "type": "bullish" if row['close'] > row['open'] else "bearish",
                    "price": row['close'],
                    "range": (row['low'], row['high']),
                    "volume": row['vol'],
                    "strength": round(row['vol'] / row['vol_ma'], 2),
                    "timestamp": row['ts']
                })
                
            return blocks
        except Exception as e:
            logger.error(f"Order block detection failed: {str(e)}")
            return []

    def _detect_liquidity_zones(self, df: pd.DataFrame) -> Dict:
        """Liquidity zone detection with tolerance-based matching"""
        zones = {
            "equal_highs": [],
            "equal_lows": [],
            "volume_clusters": []
        }
        
        if len(df) < self.config["liquidity_lookback"]:
            return zones
            
        try:
            # Lookback period for liquidity zones
            lookback = self.config["liquidity_lookback"]
            recent_df = df.iloc[-lookback:]
            
            # Find equal highs and lows with tolerance
            tolerance = recent_df['high'].mean() * 0.0005  # 0.05% tolerance
            
            # Use vectorized operations for better performance
            high_values = recent_df['high'].values
            low_values = recent_df['low'].values
            
            # Find clusters using vectorized operations
            high_clusters = self._find_price_clusters(high_values, tolerance)
            low_clusters = self._find_price_clusters(low_values, tolerance)
            
            # Process high clusters
            for price, count in high_clusters.items():
                if count > 1:
                    zones["equal_highs"].append({
                        "price": price,
                        "occurrences": count
                    })
                    
            # Process low clusters
            for price, count in low_clusters.items():
                if count > 1:
                    zones["equal_lows"].append({
                        "price": price,
                        "occurrences": count
                    })
                    
            # Volume-based liquidity zones (POC)
            if not recent_df.empty:
                max_vol_idx = recent_df['vol'].idxmax()
                zones["volume_clusters"] = [{
                    'price': recent_df.loc[max_vol_idx, 'close'],
                    'volume': recent_df.loc[max_vol_idx, 'vol'],
                    'timestamp': recent_df.loc[max_vol_idx, 'ts']
                }]
                
            return zones
        except Exception as e:
            logger.error(f"Liquidity zone detection failed: {str(e)}")
            return zones

    def _find_price_clusters(self, prices: np.ndarray, tolerance: float) -> Dict[float, int]:
        """Find price clusters using efficient binning"""
        # Create bins based on tolerance
        min_price, max_price = np.min(prices), np.max(prices)
        num_bins = int((max_price - min_price) / tolerance) + 1
        bins = np.linspace(min_price, max_price, num_bins)
        
        # Digitize prices to bins
        indices = np.digitize(prices, bins)
        
        # Find most common bins
        unique, counts = np.unique(indices, return_counts=True)
        clusters = {}
        for i, count in zip(unique, counts):
            if count > 1:
                cluster_price = bins[i-1] + tolerance/2  # Center of bin
                clusters[cluster_price] = count
        return clusters

    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Volume profile analysis using efficient binning"""
        profile = {
            'high_volume_nodes': [],
            'low_volume_nodes': []
        }
        
        if len(df) < 20:
            return profile
            
        try:
            # Efficient volume profile analysis
            bins = 20
            min_low, max_high = df['low'].min(), df['high'].max()
            bin_size = (max_high - min_low) / bins
            
            # Pre-calculate quantiles
            vol_q25 = df['vol'].quantile(0.25)
            vol_q75 = df['vol'].quantile(0.75)
            
            # Process each bin
            for i in range(bins):
                price_low = min_low + i * bin_size
                price_high = price_low + bin_size
                
                # Vectorized filtering
                in_bin = (df['low'] <= price_high) & (df['high'] >= price_low)
                vol_sum = df.loc[in_bin, 'vol'].sum()
                
                if vol_sum > vol_q75:
                    profile['high_volume_nodes'].append({
                        'price_level': (price_low + price_high) / 2,
                        'volume': vol_sum
                    })
                elif vol_sum < vol_q25:
                    profile['low_volume_nodes'].append({
                        'price_level': (price_low + price_high) / 2,
                        'volume': vol_sum
                    })
                    
            return profile
        except Exception as e:
            logger.error(f"Volume profile analysis failed: {str(e)}")
            return profile

    def _identify_swing_points(self, df: pd.DataFrame) -> Dict:
        """Identify swing points using efficient windowing"""
        swings = {'highs': [], 'lows': []}
        window = self.config["swing_window"]
        
        if len(df) < 2 * window:
            return swings
            
        try:
            # Use rolling windows for efficiency
            roll_high = df['high'].rolling(window*2+1, center=True)
            roll_low = df['low'].rolling(window*2+1, center=True)
            
            # Find local maxima/minima
            highs = (df['high'] == roll_high.max()) & (roll_high.count() == window*2+1)
            lows = (df['low'] == roll_low.min()) & (roll_low.count() == window*2+1)
            
            # Process highs
            for idx in highs[highs].index:
                swings['highs'].append({
                    'price': df.loc[idx, 'high'],
                    'strength': self._calculate_swing_strength(df, idx, 'high'),
                    'timestamp': df.loc[idx, 'ts']
                })
                
            # Process lows
            for idx in lows[lows].index:
                swings['lows'].append({
                    'price': df.loc[idx, 'low'],
                    'strength': self._calculate_swing_strength(df, idx, 'low'),
                    'timestamp': df.loc[idx, 'ts']
                })
                
            # Return only recent swings
            recent_threshold = df['ts'].iloc[-20]
            return {
                'highs': [s for s in swings['highs'] if s['timestamp'] > recent_threshold],
                'lows': [s for s in swings['lows'] if s['timestamp'] > recent_threshold]
            }
        except Exception as e:
            logger.error(f"Swing point detection failed: {str(e)}")
            return swings

    def _calculate_swing_strength(self, df: pd.DataFrame, idx: int, swing_type: str) -> float:
        """Calculate swing strength with volume confirmation"""
        base_price = df['high'].iloc[idx] if swing_type == 'high' else df['low'].iloc[idx]
        
        # Price deviation strength
        if swing_type == 'high':
            prev_lows = df['low'].iloc[max(0, idx-5):idx].mean()
            strength = (base_price - prev_lows) / prev_lows
        else:
            prev_highs = df['high'].iloc[max(0, idx-5):idx].mean()
            strength = (prev_highs - base_price) / prev_highs
            
        # Volume confirmation
        vol_strength = df['vol'].iloc[idx] / df['vol'].iloc[max(0, idx-5):idx].mean()
        return round(strength * vol_strength, 4)

class TestSMAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SMAnalyzer(min_candles=10)
        self.test_candles = [
            [1625097600000, 100.0, 105.0, 95.0, 102.0, 1000],
            [1625098200000, 102.0, 107.0, 98.0, 104.0, 1200],
            [1625098800000, 104.0, 110.0, 100.0, 108.0, 1300],
            [1625099400000, 108.0, 112.0, 106.0, 110.0, 900],
            [1625100000000, 110.0, 115.0, 108.0, 112.0, 1500]
        ]
    
    def test_validate_data(self):
        # Valid data
        result = self.analyzer._validate_data(self.test_candles)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(self.test_candles))
        
        # Invalid data (insufficient candles)
        result = self.analyzer._validate_data(self.test_candles[:2])
        self.assertIsNone(result)
        
        # Invalid data (corrupted candles)
        corrupted = self.test_candles.copy()
        corrupted.append([1625100600000, "invalid", 120.0, 110.0, 115.0, 800])
        result = self.analyzer._validate_data(corrupted)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(self.test_candles))  # Corrupted row removed
        
        # Test high < low
        invalid_candle = [[1625100600000, 100.0, 90.0, 110.0, 105.0, 800]]
        result = self.analyzer._validate_data(self.test_candles + invalid_candle)
        self.assertEqual(len(result), len(self.test_candles))  # Invalid candle removed
    
    def test_detect_structure(self):
        result = self.analyzer.detect_structure(self.test_candles)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, MarketStructure)
        self.assertTrue(result.bos or result.choch)
        
        # Test with insufficient data
        result = self.analyzer.detect_structure(self.test_candles[:2])
        self.assertIsNone(result)
    
    def test_detect_order_blocks(self):
        # Create volume spike
        high_vol_candles = self.test_candles.copy()
        high_vol_candles.append([1625100600000, 112.0, 118.0, 110.0, 116.0, 3000])
        df = self.analyzer._validate_data(high_vol_candles)
        blocks = self.analyzer._detect_order_blocks(df)
        self.assertGreater(len(blocks), 0)
        self.assertEqual(blocks[0]["type"], "bullish")
    
    def test_detect_fvg(self):
        # Create FVG pattern
        fvg_candles = [
            [1625097600000, 100.0, 105.0, 99.0, 102.0, 1000],
            [1625098200000, 102.0, 103.0, 98.0, 100.0, 1200],
            [1625098800000, 101.0, 107.0, 101.0, 106.0, 1300]  # FVG between 103 and 101
        ]
        df = self.analyzer._validate_data(fvg_candles)
        fvg = self.analyzer._detect_strong_fvg(df)
        self.assertIsNotNone(fvg)
        self.assertEqual(fvg["type"], "bullish")
        
    def test_real_time_processing(self):
        analyzer = SMAnalyzer(min_candles=3, real_time_mode=True)
        
        # First chunk
        result = analyzer.detect_structure(self.test_candles[:2])
        self.assertIsNone(result)  # Not enough data
        
        # Second chunk
        result = analyzer.detect_structure(self.test_candles[2:4])
        self.assertIsNotNone(result)
        
        # Third chunk
        result = analyzer.detect_structure(self.test_candles[4:])
        self.assertIsNotNone(result)

if __name__ == "__main__":
    """
    Smart Money Concept Analyzer
    ----------------------------
    
    How to use:
    1. Initialize the analyzer:
        analyzer = SMAnalyzer(
            min_candles=50, 
            volume_multiplier=2.5,
            real_time_mode=False
        )
        
    2. Prepare candle data in the format:
        [
            [timestamp, open, high, low, close, volume],
            [timestamp, open, high, low, close, volume],
            ...
        ]
        
    3. Detect market structure:
        structure = analyzer.detect_structure(candles)
        
    4. Interpret results:
        - bos (Break of Structure): Bullish momentum signal
        - choch (Change of Character): Bearish reversal signal
        - fvg (Fair Value Gap): Price imbalance zone (trade opportunity)
        - order_blocks: Significant institutional order areas (support/resistance)
        - liquidity: Key liquidity zones (target areas for price)
        - volume_profile: High volume nodes (significant price levels)
        - trend_structure: Overall market direction (uptrend/downtrend/range)
        
    5. For real-time processing:
        analyzer = SMAnalyzer(real_time_mode=True)
        # Continuously feed new candles:
        structure = analyzer.detect_structure(new_candles)
        
    Example Analysis Interpretation:
        - BOS + Uptrend + Bullish FVG = Strong buy signal
        - CHoCH + Bearish Order Block = Potential reversal short
        - Price at Volume Cluster + FVG = High probability trade setup
    """
    
    # Run unit tests
    unittest.main(argv=[''], exit=False)
    
    # Example analysis
    analyzer = SMAnalyzer(min_candles=30)
    
    # Generate realistic test data
    np.random.seed(42)
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1H').astype(int) // 10**6
    base_prices = np.cumsum(np.random.normal(0, 0.5, 100)) + 100
    spreads = np.random.uniform(0.1, 0.5, 100)
    highs = base_prices + spreads
    lows = base_prices - spreads
    closes = np.random.uniform(lows, highs)
    opens = np.roll(closes, 1)
    opens[0] = closes[0] - 0.1
    volumes = np.random.randint(50, 1000, 100) + np.abs(np.random.normal(0, 200, 100))
    
    test_candles = list(zip(
        timestamps,
        opens,
        highs,
        lows,
        closes,
        volumes
    ))
    
    # Add a clear FVG
    test_candles[50] = (test_candles[50][0], test_candles[50][1], 102.0, 100.0, 101.5, 800)
    test_candles[51] = (test_candles[51][0], 101.5, 103.0, 101.0, 102.5, 900)
    
    # Add an order block
    test_candles[70] = (test_candles[70][0], 108.0, 110.0, 107.5, 109.5, 2500)
    
    # Analyze structure
    result = analyzer.detect_structure(test_candles)
    
    print("\n=== Enhanced Market Structure Analysis ===")
    print(f"Trend Structure: {result.trend_structure}")
    print(f"BOS: {result.bos} | CHoCH: {result.choch}")
    
    if result.fvg:
        print(f"Significant FVG: {result.fvg['type']} (Strength: {result.fvg['strength']:.2f}%)")
    else:
        print("No significant FVG detected")
        
    print(f"Order Blocks Found: {len(result.order_blocks)}")
    print(f"Liquidity Zones: {len(result.liquidity['equal_highs'])} equal highs, "
          f"{len(result.liquidity['equal_lows'])} equal lows, "
          f"{len(result.liquidity['volume_clusters'])} volume clusters")
    print(f"Volume Profile: {len(result.volume_profile['high_volume_nodes'])} high volume nodes")
    print(f"Swing Points: {len(result.swing_highs_lows['highs'])} highs, "
          f"{len(result.swing_highs_lows['lows'])} lows")
    
    # Real-time processing example
    print("\n=== Real-time Processing Simulation ===")
    rt_analyzer = SMAnalyzer(min_candles=5, real_time_mode=True)
    
    for i in range(0, len(test_candles), 5):
        chunk = test_candles[i:i+5]
        result = rt_analyzer.detect_structure(chunk)
        if result:
            print(f"Processed {i+5} candles | Trend: {result.trend_structure} | BOS: {result.bos}")