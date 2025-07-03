import pandas as pd
import pandas_ta as ta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np
from collections import deque

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
    order_blocks: List[Dict] = None
    liquidity: Optional[Dict] = None
    volume_profile: Optional[Dict] = None
    trend_structure: Optional[str] = None
    swing_highs_lows: Optional[Dict] = None

class SMAnalyzer:
    def __init__(self, 
                 min_candles: int = 50,
                 volume_multiplier: float = 2.5,
                 fvg_min_strength: float = 0.5,
                 liquidity_lookback: int = 20):
        """
        Enhanced Smart Money Concept analyzer with additional features.
        
        Args:
            min_candles: Minimum candles needed for reliable analysis
            volume_multiplier: Volume threshold multiplier for order blocks
            fvg_min_strength: Minimum percentage strength for valid FVG
            liquidity_lookback: Number of candles to analyze for liquidity zones
        """
        self.min_candles = min_candles
        self.volume_multiplier = volume_multiplier
        self.fvg_min_strength = fvg_min_strength
        self.liquidity_lookback = liquidity_lookback
        self.recent_signals = deque(maxlen=100)  # Stores recent signals for pattern recognition

    def _validate_data(self, candles: List) -> Optional[pd.DataFrame]:
        """Validate and prepare candle data with enhanced checks."""
        if len(candles) < self.min_candles:
            logger.warning(f"Insufficient data: {len(candles)} candles (minimum {self.min_candles} required)")
            return None

        try:
            df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "vol"])
            
            # Convert and validate numeric columns
            numeric_cols = ["open", "high", "low", "close", "vol"]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            if df[numeric_cols].isnull().any().any():
                logger.error("Invalid numeric data in candles")
                return None
                
            # Validate timestamps
            df["ts"] = pd.to_datetime(df["ts"], unit='ms', errors='coerce')
            if df["ts"].isnull().any():
                logger.error("Invalid timestamp data")
                return None
                
            return df.iloc[::-1].reset_index(drop=True)  # Sort oldest to newest
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}", exc_info=True)
            return None

    def detect_structure(self, candles: List) -> Optional[MarketStructure]:
        """
        Enhanced market structure detection with:
        - More robust trend analysis
        - Multiple order block detection
        - Volume profile analysis
        - Swing point identification
        """
        df = self._validate_data(candles)
        if df is None:
            return None

        # Basic structure
        structure = MarketStructure(
            recent_high=df["high"].iloc[-2],
            recent_low=df["low"].iloc[-2],
            current_close=df["close"].iloc[-1],
            order_blocks=[]
        )

        # Enhanced trend analysis
        structure.trend_structure = self._detect_trend_structure(df)
        
        # Core SMC features
        structure.bos = structure.current_close > structure.recent_high
        structure.choch = (not structure.bos) and (structure.current_close < structure.recent_low)
        
        # Advanced features
        structure.fvg = self._detect_strong_fvg(df)
        structure.order_blocks = self._detect_multiple_order_blocks(df)
        structure.liquidity = self._detect_liquidity_zones(df)
        structure.volume_profile = self._analyze_volume_profile(df)
        structure.swing_highs_lows = self._identify_swing_points(df)
        
        # Store this analysis for pattern recognition
        self.recent_signals.append(structure)
        
        return structure

    def _detect_trend_structure(self, df: pd.DataFrame) -> str:
        """Determine the overall trend structure with confirmation."""
        if len(df) < 20:  # Need enough data for trend analysis
            return "neutral"
            
        # Use EMA cross for trend confirmation
        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)
        
        current_ema_trend = "up" if df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1] else "down"
        
        # Check higher timeframe structure
        swing_highs = df['high'].rolling(5, center=True).max() == df['high']
        swing_lows = df['low'].rolling(5, center=True).min() == df['low']
        
        if current_ema_trend == "up" and any(swing_highs[-5:]):
            return "uptrend"
        elif current_ema_trend == "down" and any(swing_lows[-5:]):
            return "downtrend"
        return "range"

    def _detect_strong_fvg(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect only significant FVGs that meet strength criteria."""
        if len(df) < 3:
            return None

        # Look for FVGs in recent candles
        for i in range(1, min(5, len(df)-1)):
            current_low = df["low"].iloc[-i]
            prev_high = df["high"].iloc[-i-1]
            
            # Bullish FVG
            if current_low > prev_high:
                strength = (current_low - prev_high) / prev_high * 100
                if strength >= self.fvg_min_strength:
                    return {
                        "type": "bullish",
                        "range": (prev_high, current_low),
                        "strength": strength,
                        "age": i
                    }

            current_high = df["high"].iloc[-i]
            prev_low = df["low"].iloc[-i-1]
            
            # Bearish FVG
            if current_high < prev_low:
                strength = (prev_low - current_high) / prev_low * 100
                if strength >= self.fvg_min_strength:
                    return {
                        "type": "bearish",
                        "range": (current_high, prev_low),
                        "strength": strength,
                        "age": i
                    }
        return None

    def _detect_multiple_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Detect multiple significant order blocks."""
        blocks = []
        if len(df) < 10:  # Need sufficient history
            return blocks
            
        # Calculate volume thresholds
        df['vol_ma'] = df['vol'].rolling(20).mean()
        df['vol_upper'] = self.volume_multiplier * df['vol_ma']
        
        # Find all significant volume spikes
        ob_candles = df[df['vol'] > df['vol_upper']].copy()
        
        for _, row in ob_candles.iterrows():
            blocks.append({
                "type": "bullish" if row['close'] > row['open'] else "bearish",
                "range": (row['low'], row['high']),
                "volume": row['vol'],
                "timestamp": row['ts'],
                "strength": row['vol'] / row['vol_ma']
            })
            
        return blocks

    def _detect_liquidity_zones(self, df: pd.DataFrame) -> Dict:
        """Enhanced liquidity zone detection with volume confirmation."""
        zones = {
            "equal_highs": [],
            "equal_lows": [],
            "volume_clusters": []
        }
        
        if len(df) < self.liquidity_lookback:
            return zones
            
        # Find equal highs and lows
        high_counts = df['high'].value_counts()
        low_counts = df['low'].value_counts()
        
        zones['equal_highs'] = [level for level, count in high_counts.items() 
                              if count > 1 and level >= df['close'].iloc[-1]]
        zones['equal_lows'] = [level for level, count in low_counts.items() 
                             if count > 1 and level <= df['close'].iloc[-1]]
        
        # Volume-based liquidity zones
        df['range'] = df['high'] - df['low']
        df['poc'] = df['vol'] / df['range']  # Point of Control calculation
        
        # Get significant volume clusters
        volume_threshold = df['poc'].quantile(0.7)
        significant = df[df['poc'] > volume_threshold]
        
        for _, row in significant.iterrows():
            zones['volume_clusters'].append({
                'price_range': (row['low'], row['high']),
                'volume': row['vol'],
                'poc_strength': row['poc']
            })
            
        return zones

    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Analyze volume profile for significant levels."""
        profile = {
            'high_volume_nodes': [],
            'low_volume_nodes': []
        }
        
        if len(df) < 20:
            return profile
            
        # Simple volume profile analysis
        bins = 20
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / bins
        
        for i in range(bins):
            price_level = df['low'].min() + i * bin_size
            in_bin = (df['low'] <= price_level) & (df['high'] >= price_level)
            vol_sum = df[in_bin]['vol'].sum()
            
            if vol_sum > df['vol'].quantile(0.75):
                profile['high_volume_nodes'].append({
                    'price_level': price_level,
                    'volume': vol_sum
                })
            elif vol_sum < df['vol'].quantile(0.25):
                profile['low_volume_nodes'].append({
                    'price_level': price_level,
                    'volume': vol_sum
                })
                
        return profile

    def _identify_swing_points(self, df: pd.DataFrame) -> Dict:
        """Identify recent swing highs and lows."""
        swings = {'highs': [], 'lows': []}
        
        if len(df) < 10:
            return swings
            
        # Find swing highs (higher than neighbors)
        highs = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        # Find swing lows (lower than neighbors)
        lows = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        
        for i, is_high in enumerate(highs):
            if is_high:
                swings['highs'].append({
                    'price': df['high'].iloc[i],
                    'timestamp': df['ts'].iloc[i],
                    'strength': self._calculate_swing_strength(df, i, 'high')
                })
                
        for i, is_low in enumerate(lows):
            if is_low:
                swings['lows'].append({
                    'price': df['low'].iloc[i],
                    'timestamp': df['ts'].iloc[i],
                    'strength': self._calculate_swing_strength(df, i, 'low')
                })
                
        return swings

    def _calculate_swing_strength(self, df: pd.DataFrame, idx: int, swing_type: str) -> float:
        """Calculate how strong a swing point is."""
        if swing_type == 'high':
            left = max(0, idx-5)
            right = min(len(df)-1, idx+5)
            return df['high'].iloc[idx] / df['high'].iloc[left:right].mean() - 1
        else:
            left = max(0, idx-5)
            right = min(len(df)-1, idx+5)
            return 1 - df['low'].iloc[idx] / df['low'].iloc[left:right].mean()

# Example Usage
if __name__ == "__main__":
    analyzer = SMAnalyzer(min_candles=30)
    
    # Generate test data
    np.random.seed(42)
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1H').astype(int) // 10**9 * 1000
    closes = np.cumsum(np.random.randn(100) * 0.5 + 0.1) + 100
    highs = closes + np.random.rand(100) * 2
    lows = closes - np.random.rand(100) * 2
    volumes = np.random.randint(100, 1000, size=100)
    
    test_candles = list(zip(
        timestamps,
        closes - 0.5,  # open
        highs,
        lows,
        closes,
        volumes
    ))
    
    # Analyze structure
    result = analyzer.detect_structure(test_candles)
    
    print("\n=== Enhanced Market Structure Analysis ===")
    print(f"Trend Structure: {result.trend_structure}")
    print(f"BOS: {result.bos} | CHoCH: {result.choch}")
    print(f"Significant FVG: {result.fvg}")
    print(f"Order Blocks Found: {len(result.order_blocks)}")
    print(f"Liquidity Zones: {result.liquidity}")
    print(f"Volume Profile Nodes: {len(result.volume_profile['high_volume_nodes'])} high, {len(result.volume_profile['low_volume_nodes'])} low")
    print(f"Swing Points: {len(result.swing_highs_lows['highs'])} highs, {len(result.swing_highs_lows['lows'])} lows")
