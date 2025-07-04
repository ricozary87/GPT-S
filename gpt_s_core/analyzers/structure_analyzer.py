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
# Logging configuration should ideally be done once at the application's entry point.
# Here, it's configured to output to a file and console.
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
    imbalance_strength: float = 0.0 # Placeholder for future use

class SMAnalyzer:
    def __init__(self,
                 config_path: str = "smc_config.json",
                 min_candles: int = 50,
                 volume_multiplier: float = 2.5,
                 fvg_min_strength: float = 0.5, # As a percentage, e.g., 0.5 for 0.5%
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
            swing_window: Window size for swing point detection (e.g., 5 for 5 candles left/right)
            real_time_mode: Enable streaming data optimizations
        """
        # Load configuration (merging defaults with loaded config)
        initial_defaults = {
            "min_candles": min_candles,
            "volume_multiplier": volume_multiplier,
            "fvg_min_strength": fvg_min_strength,
            "liquidity_lookback": liquidity_lookback,
            "swing_window": swing_window
        }
        self.config = self._load_config(config_path, initial_defaults)

        # Update instance attributes from config for easy access
        self.min_candles = self.config["min_candles"]
        self.volume_multiplier = self.config["volume_multiplier"]
        self.fvg_min_strength = self.config["fvg_min_strength"]
        self.liquidity_lookback = self.config["liquidity_lookback"]
        self.swing_window = self.config["swing_window"]

        self.real_time_mode = real_time_mode
        self.recent_signals = deque(maxlen=100) # Stores recent MarketStructure objects
        self.state = {}  # For maintaining state in real-time processing (e.g., historical DF)

    def _load_config(self, path: str, defaults: Dict) -> Dict:
        """Load configuration from file or use defaults."""
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge loaded config with defaults, preferring loaded values
                    defaults.update(loaded_config)
            return defaults
        except Exception as e:
            logger.error(f"Config load failed from '{path}': {str(e)}, using defaults", exc_info=True)
            return defaults

    def _validate_data(self, candles: List) -> Optional[pd.DataFrame]:
        """
        Validate and prepare candle data with enhanced checks and sanitization.
        Converts string prices to numeric.
        """
        if len(candles) < self.min_candles: # Use self.min_candles from config
            logger.warning(f"Insufficient data: {len(candles)} candles. Required: {self.min_candles}")
            return None

        try:
            # Ensure column names match the input candle format
            # [timestamp, open, high, low, close, volume]
            required_columns = ["ts", "open", "high", "low", "close", "vol"]
            df = pd.DataFrame(candles, columns=required_columns)

            # Convert and validate numeric columns
            numeric_cols = ["open", "high", "low", "close", "vol"]
            # Use errors='coerce' to turn unparseable values into NaN
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

            # Drop rows with any NaN in numeric columns after conversion
            initial_rows = len(df)
            df.dropna(subset=numeric_cols, inplace=True)
            if len(df) < initial_rows:
                logger.warning(f"Dropped {initial_rows - len(df)} rows due to non-numeric price/volume data.")

            # Data sanitization
            # Candles where high is not greater than low are invalid
            df = df[df["high"] >= df["low"]] # Changed > to >= for edge cases where H==L
            initial_rows_after_high_low = len(df)
            if len(df) < initial_rows_after_high_low:
                logger.warning(f"Dropped {initial_rows_after_high_low - len(df)} rows where high was not >= low.")

            df = df[df["vol"] > 0] # Remove zero volume
            initial_rows_after_vol = len(df)
            if len(df) < initial_rows_after_vol:
                logger.warning(f"Dropped {initial_rows_after_vol - len(df)} rows with zero volume.")

            if df.empty:
                logger.error("No valid data remaining after sanitization.")
                return None

            # Validate timestamps
            df["ts"] = pd.to_datetime(df["ts"], unit='ms', errors='coerce')
            if df["ts"].isnull().any():
                logger.error("Invalid timestamp data detected and coerced to NaT. Dropping rows with NaT timestamps.")
                df.dropna(subset=['ts'], inplace=True)
                if df.empty:
                    logger.error("No valid data remaining after timestamp sanitization.")
                    return None

            # Ensure chronological order
            df = df.sort_values('ts').reset_index(drop=True)

            # Re-check minimum candles after all sanitization
            if len(df) < self.min_candles:
                logger.warning(f"Insufficient valid data after sanitization: {len(df)} candles.")
                return None

            return df

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
                logger.info(f"Processing {len(df)} candles with Dask.")
                ddf = dd.from_pandas(df, npartitions=os.cpu_count() or 4) # Use CPU count for partitions
                structure = self._process_with_dask(ddf)
            else:
                logger.info(f"Processing {len(df)} candles with Pandas.")
                structure = self._process_dataframe(df)

            # Store for pattern recognition
            self.recent_signals.append(structure)
            return structure

        except ValueError as e:
            logger.error(f"Value error during structure detection: {e}", exc_info=True)
            # Fallback: Try processing with reduced features. Max retries are handled within _retry_processing.
            return self._retry_processing(candles, is_fallback=True)
        except Exception as e:
            logger.error(f"Unhandled exception during structure detection: {e}", exc_info=True)
            # Retry mechanism. Max retries are handled within _retry_processing.
            return self._retry_processing(candles)

    def _process_incremental(self, new_candles: List) -> Optional[MarketStructure]:
        """Process data incrementally for real-time streaming"""
        # Initialize state if needed. Ensure 'df' is always a DataFrame.
        if "df" not in self.state or not isinstance(self.state["df"], pd.DataFrame):
            self.state["df"] = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol"])
            self.state["last_structure"] = None

        new_df = self._validate_data(new_candles)
        if new_df is None:
            logger.warning("No valid new candles to process incrementally.")
            return self.state["last_structure"] # Return last known structure if new data is invalid

        # Append to existing data and remove duplicates based on timestamp
        self.state["df"] = pd.concat([self.state["df"], new_df]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

        # Keep only a reasonable lookback window in real-time mode to prevent memory issues
        max_rt_lookback_candles = 500 # Keep last 500 candles in RT mode
        if len(self.state["df"]) > max_rt_lookback_candles:
            self.state["df"] = self.state["df"].iloc[-max_rt_lookback_candles:]

        # Process only if we have enough data after concatenation
        if len(self.state["df"]) < self.min_candles: # Use self.min_candles from config
            logger.info(f"Not enough historical data in real-time buffer ({len(self.state['df'])}/{self.min_candles}). Waiting for more candles.")
            return self.state["last_structure"]

        # Process the full relevant history in state["df"]
        # Using the full self.state["df"] for comprehensive analysis in real-time mode
        structure = self._process_dataframe(self.state["df"])
        self.state["last_structure"] = structure
        logger.info(f"Real-time processing completed for {len(self.state['df'])} candles. Trend: {structure.trend_structure}")
        return structure

    def _process_with_dask(self, ddf: dd.DataFrame) -> MarketStructure:
        """Process large datasets using Dask parallel processing"""
        # Compute basic structure in parallel (compute all at once for efficiency)
        recent_high_dask = ddf["high"].max()
        recent_low_dask = ddf["low"].min()
        current_close_dask = ddf["close"].iloc[-1]

        # Use dask.compute() to execute the graph
        recent_high, recent_low, current_close = dd.compute(recent_high_dask, recent_low_dask, current_close_dask)

        structure = MarketStructure(
            recent_high=recent_high, # Note: This is the max/min of the entire Dask chunk, not necessarily a 'swing'
            recent_low=recent_low,
            current_close=current_close
        )

        # Compute other features. Note: Some complex features like detailed swing points
        # or volume profile might be too complex or computationally expensive for simple Dask aggregation.
        # This implementation simplifies them.
        structure.trend_structure = self._detect_trend_structure(ddf.compute()) # Dask DataFrame must be computed for pandas_ta
        # Recalculate BOS/CHoCH based on the current close and the global max/min of the chunk
        structure.bos = current_close > recent_high
        structure.choch = (not structure.bos) and (current_close < recent_low) # CHoCH is reversal not confirmed by BOS

        # For order blocks, liquidity, volume profile, swing points, it's often more practical
        # to compute on a subset of the data or apply custom Dask logic.
        # For simplicity, we compute on the full ddf if needed or rely on subset in _process_dataframe for smaller chunks.
        # Here, we'll just use the last few candles from the computed df for these
        df_last_n = ddf.tail(self.liquidity_lookback + self.swing_window * 2 + 50).compute() # Fetch relevant tail to calculate specific features
        if not df_last_n.empty and len(df_last_n) >= self.min_candles:
            structure.order_blocks = self._detect_order_blocks(df_last_n)
            structure.fvg = self._detect_strong_fvg(df_last_n) # Add FVG detection for Dask path
            structure.liquidity = self._detect_liquidity_zones(df_last_n)
            structure.volume_profile = self._analyze_volume_profile(df_last_n)
            structure.swing_highs_lows = self._identify_swing_points(df_last_n)
        else:
            logger.warning("Not enough data in Dask tail for full feature extraction after computing.")

        return structure

    def _process_dataframe(self, df: pd.DataFrame) -> MarketStructure:
        """Process standard-sized DataFrame for detailed analysis"""
        # It's crucial that df has enough candles here for .iloc[-2] etc.
        # This check is partly handled by _validate_data, but good to double check.
        if len(df) < 3: # Need at least 3 candles for previous, prev2
            logger.warning("DataFrame too short for detailed processing (need >= 3 candles).")
            # Fallback to a simplified structure or return None if not sufficient
            return MarketStructure(recent_high=np.nan, recent_low=np.nan, current_close=df["close"].iloc[-1] if not df.empty else np.nan)


        # Define recent_high/low for BOS/CHoCH. This could be last known swing point, or prev candle's high/low
        # For simplicity and to match the previous interpretation, using previous candle's high/low for BOS/CHoCH trigger.
        # But for 'swing_high' / 'swing_low' in MarketStructure object, we use the _identify_swing_points results.
        prev_high = df["high"].iloc[-2]
        prev_low = df["low"].iloc[-2]
        current_close = df["close"].iloc[-1]

        structure = MarketStructure(
            recent_high=prev_high, # This refers to the previous candle's high for BOS/CHoCH logic
            recent_low=prev_low,   # This refers to the previous candle's low for BOS/CHoCH logic
            current_close=current_close
        )

        structure.trend_structure = self._detect_trend_structure(df)
        structure.bos = current_close > prev_high
        structure.choch = (not structure.bos) and (current_close < prev_low) # CHoCH implies reversal if BOS not met

        # Detect other features
        structure.fvg = self._detect_strong_fvg(df)
        structure.order_blocks = self._detect_order_blocks(df)
        structure.liquidity = self._detect_liquidity_zones(df)
        structure.volume_profile = self._analyze_volume_profile(df)
        structure.swing_highs_lows = self._identify_swing_points(df)

        # After identifying swings, update recent_high/low in structure object if a more relevant swing exists
        # This makes the 'recent_high' and 'recent_low' in the MarketStructure dataclass more meaningful
        if structure.swing_highs_lows['highs']:
            structure.recent_high = max(s['price'] for s in structure.swing_highs_lows['highs'])
        if structure.swing_highs_lows['lows']:
            structure.recent_low = min(s['price'] for s in structure.swing_highs_lows['lows'])


        return structure

    def _fallback_processing(self, candles: List) -> Optional[MarketStructure]:
        """Simplified processing when main method fails, using raw candle list."""
        logger.warning("Attempting fallback processing due to an error.")
        try:
            # Ensure basic numeric conversion for raw list data
            processed_candles = []
            for candle in candles:
                try:
                    processed_candles.append([
                        candle[0], # Timestamp (no conversion needed if already int/float)
                        float(candle[1]), # Open
                        float(candle[2]), # High
                        float(candle[3]), # Low
                        float(candle[4]), # Close
                        float(candle[5])  # Volume
                    ])
                except (ValueError, TypeError):
                    logger.warning(f"Skipping malformed candle in fallback: {candle}")
                    continue

            if len(processed_candles) < 3: # Need at least 3 for basic calculations
                logger.warning("Not enough valid candles for fallback processing.")
                return None

            # Get the last few candles for simplified recent high/low
            recent_candles = processed_candles[-min(5, len(processed_candles)):] # Use last 5 or fewer if not enough

            current_close = processed_candles[-1][4]
            # Simple max/min from recent subset
            recent_high_val = max(c[2] for c in recent_candles)
            recent_low_val = min(c[3] for c in recent_candles)

            # Simple BOS/CHoCH based on last two candles (basic)
            bos_val = processed_candles[-1][4] > processed_candles[-2][2] # Current close > previous high
            choch_val = processed_candles[-1][4] < processed_candles[-2][3] # Current close < previous low

            return MarketStructure(
                recent_high=recent_high_val,
                recent_low=recent_low_val,
                current_close=current_close,
                bos=bos_val,
                choch=choch_val,
                trend_structure="unknown" # Fallback doesn't determine complex trend
            )
        except Exception as e:
            logger.error(f"Fallback processing itself failed: {str(e)}", exc_info=True)
            return None

    def _retry_processing(self, candles: List, retries: int = 2, is_fallback: bool = False) -> Optional[MarketStructure]:
        """
        Retry mechanism for processing. Avoids infinite recursion by changing processing path.
        If `is_fallback` is True, it means we are already in a fallback scenario and should not retry `detect_structure` directly.
        """
        if is_fallback: # If we are already in a fallback, don't recursively call detect_structure
            logger.error("Already in fallback mode, not retrying detect_structure recursively.")
            return None # Or return the result of the fallback itself if it succeeded

        for i in range(retries):
            try:
                logger.info(f"Retry attempt {i+1} for detect_structure.")
                # Directly call _process_dataframe or _process_with_dask if the issue was not data validation.
                # However, for simplicity and to avoid re-implementing logic, we call _fallback_processing
                # if the original detect_structure failed.
                # A more sophisticated retry would analyze the error type.
                df = self._validate_data(candles)
                if df is None:
                    logger.error("Data validation failed during retry. Cannot proceed.")
                    return None

                if len(df) > 10000:
                    structure = self._process_with_dask(dd.from_pandas(df, npartitions=os.cpu_count() or 4))
                else:
                    structure = self._process_dataframe(df)
                logger.info(f"Retry attempt {i+1} successful.")
                return structure
            except Exception as e:
                logger.error(f"Retry attempt {i+1} failed: {e}", exc_info=True)
                if i == retries - 1: # On last retry attempt, try fallback
                    logger.warning("Max retries reached. Attempting simplified fallback processing.")
                    return self._fallback_processing(candles)
        return None # Should not be reached if fallback is last attempt

    def _detect_trend_structure(self, df: pd.DataFrame) -> str:
        """Determine trend using EMA cross and price structure."""
        if len(df) < max(20, self.swing_window * 2 + 1): # Need enough data for EMAs and swings
            logger.debug(f"Not enough data for detailed trend detection ({len(df)} candles). Returning neutral.")
            return "neutral"

        try:
            # Calculate EMA trend
            ema_trend = self._calculate_ema_trend(df)

            # Check price structure for confirmation
            # A more robust trend detection should analyze higher highs/lows from identified swing points
            swings = self._identify_swing_points(df) # Get recent swing points
            return self._assess_price_structure(swings, ema_trend)
        except Exception as e:
            logger.error(f"Trend detection failed: {str(e)}", exc_info=True)
            return "neutral"

    def _calculate_ema_trend(self, df: pd.DataFrame) -> str:
        """Calculate EMA-based trend direction."""
        # Ensure that EMA columns exist and are not NaN
        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)

        # Check for NaNs at the end of EMA series
        if df['ema_20'].iloc[-1] is np.nan or df['ema_50'].iloc[-1] is np.nan:
            logger.warning("EMA values are NaN, likely due to insufficient data for EMA calculation. Returning neutral.")
            return "neutral"

        if df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1]:
            return "up"
        elif df['ema_20'].iloc[-1] < df['ema_50'].iloc[-1]:
            return "down"
        return "neutral" # If EMAs are equal or other edge cases

    def _assess_price_structure(self, swings: Dict, ema_trend: str) -> str:
        """
        Assess price structure for trend confirmation using identified swing points.
        A stronger definition of trend based on HH/HL or LH/LL.
        """
        recent_highs = sorted(swings['highs'], key=lambda x: x['timestamp'], reverse=True)[:5] # Get top 5 recent highs
        recent_lows = sorted(swings['lows'], key=lambda x: x['timestamp'], reverse=True)[:5] # Get top 5 recent lows

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return ema_trend # Fallback to EMA trend if not enough swing points

        # Check for higher highs (HH) and higher lows (HL) for uptrend
        is_hh = recent_highs[0]['price'] > recent_highs[1]['price']
        is_hl = recent_lows[0]['price'] > recent_lows[1]['price']

        if ema_trend == "up" and is_hh and is_hl:
            return "uptrend"

        # Check for lower highs (LH) and lower lows (LL) for downtrend
        is_lh = recent_highs[0]['price'] < recent_highs[1]['price']
        is_ll = recent_lows[0]['price'] < recent_lows[1]['price']

        if ema_trend == "down" and is_lh and is_ll:
            return "downtrend"

        return "range" # Default if no clear trend structure

    def _detect_strong_fvg(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect significant FVGs with multi-candle check."""
        if len(df) < 3:
            logger.debug("Not enough candles for FVG detection (need >= 3).")
            return None

        try:
            # Look for FVGs in recent 'lookback' candles (e.g., last 5 candles)
            # Use `min(len(df) - 1, 5)` to prevent IndexError if df is short.
            # The loop should go from `idx = -2` backwards to `-min_candles_for_fvg-1`
            # The range should be `range(2, min(len(df), 7))` to look back 5 candles excluding current and previous.
            # A typical FVG involves 3 candles: [Candle1, Candle2 (middle), Candle3]
            # FVG exists if High(Candle1) < Low(Candle3) for bullish, or Low(Candle1) > High(Candle3) for bearish.
            # Here, the current logic appears to be checking for gap between Candle(idx-1) and Candle(idx)

            # Let's adjust the FVG logic to a more standard 3-candle definition.
            # Candle_prev2, Candle_prev1, Candle_current
            # FVG is between High of Candle_prev2 and Low of Candle_current (for bullish)
            # or Low of Candle_prev2 and High of Candle_current (for bearish)
            # This means we need at least 3 candles.
            # We iterate backwards from the current candle (index -1)
            # Let's consider `i` as the offset from the end of the DataFrame.
            # `df.iloc[-1]` is the current candle.
            # `df.iloc[-2]` is the previous candle.
            # `df.iloc[-3]` is the prev2 candle.

            # Iterate from the third last candle backwards, looking for FVG patterns
            # max_lookback_fvg = min(len(df), 20) # Look back up to 20 candles for FVG
            # Start from index -3 (prev2_candle) down to -max_lookback_fvg - 1
            for i in range(1, min(6, len(df)-1)): # i is the offset from end: 2 means current-prev-prev2
                # candle_prev2 = df.iloc[-i-1]
                # candle_prev1 = df.iloc[-i]
                # candle_current = df.iloc[-i+1] # This would be `df.iloc[-1]` if i=2
                # This logic is complex. Let's simplify with fixed indices for recent check,
                # then a loop for historical.

                # Simplified check for most recent FVG (last 3 candles)
                if len(df) >= 3:
                    c0_high = df["high"].iloc[-3]  # Candle -3 (prev2)
                    c0_low = df["low"].iloc[-3]
                    c2_low = df["low"].iloc[-1]    # Candle -1 (current)
                    c2_high = df["high"].iloc[-1]

                    # Bullish FVG: High of c0 < Low of c2
                    if c0_high < c2_low:
                        fvg_low = c0_high # Top of prev2 candle's high
                        fvg_high = c2_low # Bottom of current candle's low
                        strength = (fvg_high - fvg_low) / fvg_low * 100 if fvg_low != 0 else 0
                        if strength >= self.fvg_min_strength:
                            return {
                                "type": "bullish",
                                "range": (fvg_low, fvg_high),
                                "strength": round(strength, 4),
                                "age": 0 # Age refers to how many candles ago it occurred, 0 for most recent
                            }

                    # Bearish FVG: Low of c0 > High of c2
                    elif c0_low > c2_high:
                        fvg_high = c0_low # Bottom of prev2 candle's low
                        fvg_low = c2_high # Top of current candle's high
                        strength = (fvg_high - fvg_low) / fvg_low * 100 if fvg_low != 0 else 0
                        if strength >= self.fvg_min_strength:
                            return {
                                "type": "bearish",
                                "range": (fvg_low, fvg_high),
                                "strength": round(strength, 4),
                                "age": 0 # Age refers to how many candles ago it occurred, 0 for most recent
                            }
            return None # No FVG detected in recent 3 candles

        except IndexError:
            logger.debug("Not enough candles for detailed FVG calculation in _detect_strong_fvg.")
            return None
        except Exception as e:
            logger.error(f"FVG detection failed: {str(e)}", exc_info=True)
            return None

    def _detect_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Detect significant order blocks with volume confirmation."""
        blocks = []
        # Need enough data for rolling mean and a few previous candles
        if len(df) < max(20, self.min_candles): # Ensure at least 20 for vol_ma, or min_candles
            logger.debug("Not enough candles for Order Block detection.")
            return blocks

        try:
            # Calculate volume thresholds
            df['vol_ma'] = df['vol'].rolling(20).mean() # Rolling mean for volume
            # Fill NaN created by rolling window to avoid issues
            df['vol_ma'].fillna(method='bfill', inplace=True) # or fillna(df['vol_ma'].mean())
            df['vol_upper'] = self.volume_multiplier * df['vol_ma']

            # Find significant volume spikes with price confirmation
            # A candle qualifies as a potential OB if its volume is significantly higher than average
            # and it's a clear directional candle (close != open)
            vol_mask = df['vol'] > df['vol_upper']
            price_mask = (df['close'] != df['open']) # Use != 0 to ensure movement

            # Consider only candles that are at least `swing_window` away from the end
            # and within a recent lookback (e.g., last 50 candles for order blocks)
            # This helps avoid 'noisy' OBs at the very end and focuses on relevant ones.
            # Only consider candles up to `len(df) - 1 - self.swing_window`
            # And from `len(df) - 50` for recent focus (arbitrary lookback for blocks)
            recent_ob_lookback = min(len(df), 50) # Look back up to 50 candles for order blocks

            # Iterate from the end of the DataFrame backwards, but exclude the very last few candles for stability
            # Also, only consider candles that form a proper "block" with preceding candles
            # An order block is typically the last down-candle before an impulsive move up (bullish OB)
            # or the last up-candle before an impulsive move down (bearish OB).
            # This simplified version just looks for high volume candles.

            # Iterate from the 'recent_ob_lookback' point to the end of the DataFrame (excluding current candle for OB formation)
            # Iterate through the DataFrame in a vectorized manner for performance, then filter.
            # Using `iloc` for index-based access is often safer with rolling windows.
            for i in range(len(df) - recent_ob_lookback, len(df) - 1): # Check candles in the recent lookback range, excluding the very last one
                if i < 0: # Handle cases where recent_ob_lookback is larger than df
                    continue

                row = df.iloc[i]
                if vol_mask.iloc[i] and price_mask.iloc[i]:
                    # Basic OB detection: high volume and clear direction
                    blocks.append({
                        "type": "bullish" if row['close'] > row['open'] else "bearish",
                        "price": row['close'],
                        "range": (row['low'], row['high']),
                        "volume": row['vol'],
                        "strength": round(row['vol'] / df['vol_ma'].iloc[i], 2),
                        "timestamp": row['ts']
                    })
            return blocks
        except KeyError as ke:
            logger.error(f"Missing column for Order Block detection: {ke}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Order block detection failed: {str(e)}", exc_info=True)
            return []

    def _detect_liquidity_zones(self, df: pd.DataFrame) -> Dict:
        """Liquidity zone detection with tolerance-based matching."""
        zones = {
            "equal_highs": [],
            "equal_lows": [],
            "volume_clusters": []
        }

        if len(df) < self.liquidity_lookback:
            logger.debug("Not enough candles for Liquidity Zone detection.")
            return zones

        try:
            # Lookback period for liquidity zones
            lookback_df = df.iloc[-self.liquidity_lookback:].copy() # Use .copy() to avoid SettingWithCopyWarning

            if lookback_df.empty:
                return zones

            # Find equal highs and lows with tolerance
            # Calculate tolerance based on a percentage of the average high or a fixed small value
            tolerance = lookback_df['high'].mean() * 0.0005 # 0.05% tolerance of mean high

            high_values = lookback_df['high'].values
            low_values = lookback_df['low'].values

            # Find clusters using vectorized operations
            high_clusters = self._find_price_clusters(high_values, tolerance)
            low_clusters = self._find_price_clusters(low_values, tolerance)

            # Process high clusters
            for price, count in high_clusters.items():
                if count > 1: # Only consider clusters with more than one occurrence
                    zones["equal_highs"].append({
                        "price": float(price), # Ensure float conversion for JSON compatibility
                        "occurrences": int(count)
                    })

            # Process low clusters
            for price, count in low_clusters.items():
                if count > 1: # Only consider clusters with more than one occurrence
                    zones["equal_lows"].append({
                        "price": float(price), # Ensure float conversion for JSON compatibility
                        "occurrences": int(count)
                    })

            # Volume-based liquidity zones (POC - Point of Control)
            # Find the price level with the highest volume in the lookback period
            if not lookback_df.empty:
                # Calculate mid-price for volume profile (or use close price)
                lookback_df['mid_price'] = (lookback_df['high'] + lookback_df['low']) / 2
                # Group by rounded mid_price and sum volume
                # Use a smaller bin for volume clusters if needed
                volume_bins = np.linspace(lookback_df['low'].min(), lookback_df['high'].max(), 50) # 50 bins
                lookback_df['price_bin'] = pd.cut(lookback_df['mid_price'], bins=volume_bins, labels=False, include_lowest=True)

                volume_per_bin = lookback_df.groupby('price_bin')['vol'].sum()
                if not volume_per_bin.empty:
                    max_vol_bin_idx = volume_per_bin.idxmax()
                    max_vol_sum = volume_per_bin.max()

                    # Get the price range for this bin
                    price_range_for_bin = [volume_bins[max_vol_bin_idx], volume_bins[max_vol_bin_idx + 1]]
                    poc_price = (price_range_for_bin[0] + price_range_for_bin[1]) / 2

                    zones["volume_clusters"].append({
                        'price': float(poc_price),
                        'volume': float(max_vol_sum),
                        'type': 'POC_Cluster' # Indicate this is a point of control
                    })

            return zones
        except Exception as e:
            logger.error(f"Liquidity zone detection failed: {str(e)}", exc_info=True)
            return zones

    def _find_price_clusters(self, prices: np.ndarray, tolerance: float) -> Dict[float, int]:
        """
        Find price clusters using efficient binning.
        Improved handling for edge cases and precision.
        """
        clusters = {}
        if len(prices) == 0:
            return clusters

        try:
            # Handle cases where min_price and max_price might be the same or close
            min_price, max_price = np.min(prices), np.max(prices)
            if max_price - min_price < tolerance * 2: # If range is too small, treat as one cluster
                if len(prices) > 1:
                    clusters[float(np.mean(prices))] = len(prices)
                return clusters

            # Create bins based on tolerance
            # Add a small epsilon to max_price to ensure max_price falls into a bin
            num_bins = int(np.ceil((max_price - min_price) / tolerance))
            bins = np.linspace(min_price, max_price, num_bins + 1) # num_bins + 1 edges for num_bins
            if len(bins) < 2: # Ensure at least two bin edges
                bins = np.array([min_price, max_price + tolerance]) # Fallback for very small range

            # Digitize prices to bins
            # `right=True` means bins are `(a, b]`. `right=False` means `[a, b)`. Use `right=False` (default) for typical binning.
            indices = np.digitize(prices, bins)

            # Find most common bins
            # Use `pd.Series` value counts for convenience
            bin_counts = pd.Series(indices).value_counts()

            for bin_idx, count in bin_counts.items():
                if count > 1:
                    # Calculate the center of the bin for the cluster price
                    # Ensure bin_idx is within valid range for bins array
                    if 0 < bin_idx <= len(bins):
                        cluster_price = (bins[bin_idx - 1] + bins[bin_idx]) / 2
                        clusters[float(cluster_price)] = int(count)
            return clusters
        except Exception as e:
            logger.error(f"Error in _find_price_clusters: {str(e)}", exc_info=True)
            return {}

    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Volume profile analysis using efficient binning."""
        profile = {
            'high_volume_nodes': [],
            'low_volume_nodes': []
        }

        if len(df) < self.min_candles: # Ensure enough data for meaningful profile
            logger.debug("Not enough candles for Volume Profile analysis.")
            return profile

        try:
            bins = 20 # Number of price bins for the profile
            min_low, max_high = df['low'].min(), df['high'].max()

            if min_low == max_high: # Avoid division by zero if all prices are identical
                return profile

            bin_size = (max_high - min_low) / bins
            if bin_size == 0: # Prevent zero division if all prices are the same
                return profile

            # Calculate quantiles for volume thresholds
            vol_q25 = df['vol'].quantile(0.25)
            vol_q75 = df['vol'].quantile(0.75)

            # Assign each candle to a price bin based on its mid-price
            df['mid_price'] = (df['high'] + df['low']) / 2
            df['price_bin_idx'] = np.floor((df['mid_price'] - min_low) / bin_size).astype(int)
            # Ensure indices are within bounds [0, bins-1]
            df['price_bin_idx'] = df['price_bin_idx'].clip(0, bins - 1)

            # Group by price bin and sum volume
            volume_per_bin = df.groupby('price_bin_idx')['vol'].sum()

            # Process each bin based on volume quantiles
            for bin_idx in range(bins):
                vol_sum = volume_per_bin.get(bin_idx, 0) # Get volume for bin, default to 0 if no candles in bin

                # Calculate the center of the bin for the price level
                price_level = min_low + (bin_idx + 0.5) * bin_size

                if vol_sum > vol_q75:
                    profile['high_volume_nodes'].append({
                        'price_level': float(price_level),
                        'volume': float(vol_sum)
                    })
                elif vol_sum < vol_q25 and vol_sum > 0: # Only consider low volume if it's not zero
                    profile['low_volume_nodes'].append({
                        'price_level': float(price_level),
                        'volume': float(vol_sum)
                    })

            return profile
        except Exception as e:
            logger.error(f"Volume profile analysis failed: {str(e)}", exc_info=True)
            return profile

    def _identify_swing_points(self, df: pd.DataFrame) -> Dict:
        """Identify swing points using efficient windowing."""
        swings = {'highs': [], 'lows': []}
        window = self.swing_window # Use self.swing_window from config

        # A window size of `N` means `N` candles to the left AND `N` candles to the right.
        # So, the rolling window size is `2*N + 1` (N left, 1 center, N right).
        if len(df) < 2 * window + 1:
            logger.debug(f"Not enough candles for swing point detection (need >= {2 * window + 1}).")
            return swings

        try:
            # Use rolling windows for efficiency. min_periods=window*2+1 ensures full window
            # .rolling(window=2*window+1, center=True) means the current point is in the middle of the window
            roll_high = df['high'].rolling(window=2*window+1, center=True, min_periods=2*window+1)
            roll_low = df['low'].rolling(window=2*window+1, center=True, min_periods=2*window+1)

            # Find local maxima/minima
            # Swing High: A candle high that is the highest in its surrounding window.
            # Swing Low: A candle low that is the lowest in its surrounding window.
            # .dropna() is crucial because rolling window will produce NaNs at the edges
            highs_mask = (df['high'] == roll_high.max()).dropna()
            lows_mask = (df['low'] == roll_low.min()).dropna()

            # Process highs
            for idx in highs_mask[highs_mask].index: # Iterate only over True values (where high is max)
                swings['highs'].append({
                    'price': float(df.loc[idx, 'high']),
                    'strength': round(self._calculate_swing_strength(df, idx, 'high'), 4),
                    'timestamp': df.loc[idx, 'ts'].isoformat() # Convert Timestamp to ISO format string
                })

            # Process lows
            for idx in lows_mask[lows_mask].index: # Iterate only over True values (where low is min)
                swings['lows'].append({
                    'price': float(df.loc[idx, 'low']),
                    'strength': round(self._calculate_swing_strength(df, idx, 'low'), 4),
                    'timestamp': df.loc[idx, 'ts'].isoformat() # Convert Timestamp to ISO format string
                })

            # Return only recent swings (e.g., last 20 candles where swing occurred)
            # Filter based on timestamp, not index, for more robust recentness.
            # Get the timestamp threshold for recent swings (e.g., last 20 candles' range or fixed duration)
            if not df.empty:
                recent_threshold_idx = max(0, len(df) - 20) # Look back last 20 candles for recent swings
                recent_timestamp_threshold = df['ts'].iloc[recent_threshold_idx]
            else:
                recent_timestamp_threshold = pd.Timestamp.min # If df is empty, no swings anyway

            return {
                'highs': [s for s in swings['highs'] if pd.to_datetime(s['timestamp']) >= recent_timestamp_threshold],
                'lows': [s for s in swings['lows'] if pd.to_datetime(s['timestamp']) >= recent_timestamp_threshold]
            }
        except KeyError as ke:
            logger.error(f"Missing column for Swing Point detection: {ke}", exc_info=True)
            return swings
        except Exception as e:
            logger.error(f"Swing point detection failed: {str(e)}", exc_info=True)
            return swings

    def _calculate_swing_strength(self, df: pd.DataFrame, idx: int, swing_type: str) -> float:
        """
        Calculate swing strength with price deviation and volume confirmation.
        Ensures division by zero is handled.
        """
        base_price = df['high'].iloc[idx] if swing_type == 'high' else df['low'].iloc[idx]
        price_deviation_strength = 0.0
        volume_strength = 1.0 # Default to 1 if no volume data or mean is zero

        lookback_vol_candles = max(1, min(idx, 5)) # Look back up to 5 candles for volume mean

        # Price deviation strength
        if swing_type == 'high':
            prev_lows = df['low'].iloc[max(0, idx - lookback_vol_candles):idx].mean()
            if prev_lows != 0:
                price_deviation_strength = (base_price - prev_lows) / prev_lows
        else: # swing_type == 'low'
            prev_highs = df['high'].iloc[max(0, idx - lookback_vol_candles):idx].mean()
            if prev_highs != 0:
                price_deviation_strength = (prev_highs - base_price) / prev_highs

        # Volume confirmation
        current_vol = df['vol'].iloc[idx]
        recent_volumes = df['vol'].iloc[max(0, idx - lookback_vol_candles):idx].mean()

        if recent_volumes != 0: # Avoid division by zero
            volume_strength = current_vol / recent_volumes
        elif current_vol > 0: # If recent_volumes is 0 but current has volume, give it some strength
            volume_strength = 2.0 # Arbitrary higher strength for isolated volume spike

        return float(price_deviation_strength * volume_strength) # Ensure float return


class TestSMAnalyzer(unittest.TestCase):
    def setUp(self):
        # Set min_candles low for testing purposes
        self.analyzer = SMAnalyzer(min_candles=3, fvg_min_strength=0.1, liquidity_lookback=5, swing_window=1)
        self.test_candles_valid = [ # Minimum 3 candles required for basic structure
            [1625097600000, 100.0, 105.0, 95.0, 102.0, 1000.0],
            [1625098200000, 102.0, 107.0, 98.0, 104.0, 1200.0],
            [1625098800000, 104.0, 110.0, 100.0, 108.0, 1300.0], # Current close > previous high (BOS scenario)
            [1625099400000, 108.0, 112.0, 106.0, 110.0, 900.0],
            [1625100000000, 110.0, 115.0, 108.0, 112.0, 1500.0],
            # Add more candles to meet min_candles=50 requirement for default config, or reduce min_candles for tests
            # For test_validate_data, we'll use a subset, but for test_detect_structure, ensure enough candles.
            # Adding more candles to meet typical min_candles if not overridden
            *[ # Example of adding more candles
                [1625100000000 + i * 60000, 112.0 + i, 115.0 + i, 108.0 + i, 111.0 + i, 1500.0 + i * 10]
                for i in range(5, 55) # Adds 50 more candles, totaling 55
            ]
        ]

    def test_validate_data(self):
        logger.info("\n--- Running test_validate_data ---")
        # Valid data (adjust min_candles for this test case temporarily if needed)
        original_min_candles = self.analyzer.min_candles
        self.analyzer.min_candles = 3 # Temporarily set low for this test

        result = self.analyzer._validate_data(self.test_candles_valid[:5]) # Use 5 candles for this test
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)
        self.assertTrue(isinstance(result, pd.DataFrame))
        logger.info("Passed: Valid data.")

        # Invalid data (insufficient candles)
        result = self.analyzer._validate_data(self.test_candles_valid[:2])
        self.assertIsNone(result)
        logger.info("Passed: Insufficient data.")

        # Invalid data (corrupted candles - non-numeric price)
        corrupted = self.test_candles_valid[:5].copy()
        corrupted.append([1625100600000, "invalid", 120.0, 110.0, 115.0, 800.0])
        # After correction in _validate_data, this invalid candle will be dropped.
        result = self.analyzer._validate_data(corrupted)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5) # Corrupted row should be removed
        logger.info("Passed: Corrupted non-numeric data removed.")

        # Test high < low
        invalid_candle = [[1625100600000, 100.0, 90.0, 110.0, 105.0, 800.0]] # High < Low
        result = self.analyzer._validate_data(self.test_candles_valid[:5] + invalid_candle)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5) # Invalid candle should be removed
        logger.info("Passed: High < Low candle removed.")

        self.analyzer.min_candles = original_min_candles # Reset to original

    def test_detect_structure(self):
        logger.info("\n--- Running test_detect_structure ---")
        # Ensure enough candles for default min_candles (50) or set min_candles for analyzer
        original_min_candles = self.analyzer.min_candles
        self.analyzer.min_candles = 50 # Or use a test_candles_valid that matches this
        
        test_candles_for_structure = self.test_candles_valid # This now has 55 candles

        result = self.analyzer.detect_structure(test_candles_for_structure)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, MarketStructure)
        # Assuming the generated data is generally trending up
        self.assertEqual(result.trend_structure, "uptrend")
        # BOS/CHoCH will depend on the exact last few candles and swing point logic
        # For this generic data, we just assert it's not both True at the same time
        self.assertFalse(result.bos and result.choch)
        logger.info(f"Passed: Basic structure detection. Trend: {result.trend_structure}, BOS: {result.bos}, CHoCH: {result.choch}")

        # Test with insufficient data for detect_structure
        self.analyzer.min_candles = 50 # Ensure this is still enforced
        result = self.analyzer.detect_structure(self.test_candles_valid[:10]) # Only 10 candles
        self.assertIsNone(result)
        logger.info("Passed: Insufficient data for detect_structure.")
        self.analyzer.min_candles = original_min_candles # Reset

    def test_detect_order_blocks(self):
        logger.info("\n--- Running test_detect_order_blocks ---")
        # Create data with a clear volume spike
        ob_candles = self.test_candles_valid[:25].copy() # Use a subset for faster testing
        # Add a significant bullish candle with high volume
        ob_candles.append([1625100600000, 105.0, 120.0, 104.0, 118.0, 5000.0]) # High volume spike
        ob_df = self.analyzer._validate_data(ob_candles)
        self.assertIsNotNone(ob_df)
        
        blocks = self.analyzer._detect_order_blocks(ob_df)
        self.assertGreater(len(blocks), 0) # Expect at least one block
        self.assertEqual(blocks[-1]["type"], "bullish") # The last added candle should be a bullish OB
        self.assertGreaterEqual(blocks[-1]["strength"], 1.0) # Strength should be >1 if volume is above average
        logger.info(f"Passed: Order block detected. Blocks found: {len(blocks)}")

    def test_detect_fvg(self):
        logger.info("\n--- Running test_detect_fvg ---")
        # Create FVG pattern
        # Bullish FVG: prev2_high < current_low
        fvg_candles = [
            [1, 100.0, 105.0, 99.0, 102.0, 100.0], # Candle 1
            [2, 102.0, 103.0, 98.0, 100.0, 120.0], # Candle 2 (middle)
            [3, 101.0, 108.0, 104.0, 106.0, 130.0], # Candle 3: current_low (104.0) > prev2_high (105.0) -> No, this is not a bullish FVG
        ]
        # Let's fix this for a clear bullish FVG: High of C1 < Low of C3
        fvg_candles_bullish = [
            [1, 100.0, 105.0, 99.0, 104.0, 100.0],  # C1: High 105
            [2, 104.0, 106.0, 103.0, 105.0, 120.0],  # C2
            [3, 105.0, 115.0, 107.0, 110.0, 150.0]   # C3: Low 107. Bullish FVG (105-107) between C1.High and C3.Low
        ]
        df_bullish = self.analyzer._validate_data(fvg_candles_bullish)
        self.assertIsNotNone(df_bullish)
        fvg_b = self.analyzer._detect_strong_fvg(df_bullish)
        self.assertIsNotNone(fvg_b)
        self.assertEqual(fvg_b["type"], "bullish")
        self.assertAlmostEqual(fvg_b["range"][0], 105.0)
        self.assertAlmostEqual(fvg_b["range"][1], 107.0)
        self.assertGreaterEqual(fvg_b["strength"], self.analyzer.fvg_min_strength) # Check against configured strength
        logger.info(f"Passed: Bullish FVG detected. FVG: {fvg_b}")

        # Bearish FVG: prev2_low > current_high
        fvg_candles_bearish = [
            [1, 110.0, 115.0, 108.0, 112.0, 100.0], # C1: Low 108
            [2, 112.0, 110.0, 105.0, 107.0, 120.0], # C2
            [3, 107.0, 98.0, 95.0, 96.0, 150.0]     # C3: High 98. Bearish FVG (108-98) between C1.Low and C3.High
        ]
        df_bearish = self.analyzer._validate_data(fvg_candles_bearish)
        self.assertIsNotNone(df_bearish)
        fvg_br = self.analyzer._detect_strong_fvg(df_bearish)
        self.assertIsNotNone(fvg_br)
        self.assertEqual(fvg_br["type"], "bearish")
        self.assertAlmostEqual(fvg_br["range"][0], 98.0)
        self.assertAlmostEqual(fvg_br["range"][1], 108.0)
        self.assertGreaterEqual(fvg_br["strength"], self.analyzer.fvg_min_strength)
        logger.info(f"Passed: Bearish FVG detected. FVG: {fvg_br}")

        # No FVG
        no_fvg_candles = [
            [1, 100.0, 105.0, 99.0, 102.0, 100.0],
            [2, 102.0, 104.0, 101.0, 103.0, 120.0],
            [3, 103.0, 105.0, 102.0, 104.0, 130.0]
        ]
        df_no_fvg = self.analyzer._validate_data(no_fvg_candles)
        self.assertIsNotNone(df_no_fvg)
        fvg_n = self.analyzer._detect_strong_fvg(df_no_fvg)
        self.assertIsNone(fvg_n)
        logger.info("Passed: No FVG detected.")

    def test_real_time_processing(self):
        logger.info("\n--- Running test_real_time_processing ---")
        # Ensure min_candles for analyzer is lower for RT testing
        rt_analyzer = SMAnalyzer(min_candles=3, real_time_mode=True, fvg_min_strength=0.1)

        # First chunk - not enough data
        result = rt_analyzer.detect_structure(self.test_candles_valid[:2])
        self.assertIsNone(result)  # Should return None or last_structure if not enough data
        logger.info("Passed: RT - First chunk (insufficient data).")

        # Second chunk - now enough data
        result = rt_analyzer.detect_structure(self.test_candles_valid[2:5]) # Candles 3, 4, 5
        self.assertIsNotNone(result)
        self.assertEqual(len(rt_analyzer.state["df"]), 5) # Total candles in state should be 5
        # Depending on the data, BOS or CHoCH should be detectable
        # Current close (112.0) vs prev high (110.0) -> BOS
        self.assertTrue(result.bos)
        logger.info(f"Passed: RT - Second chunk (sufficient data). Result: {result.trend_structure}, BOS: {result.bos}")

        # Third chunk - more data
        result = rt_analyzer.detect_structure(self.test_candles_valid[5:7]) # Add 2 more candles
        self.assertIsNotNone(result)
        # rt_analyzer.state["df"] length might be capped by max_rt_lookback_candles
        # For these small test chunks, it will just accumulate.
        self.assertEqual(len(rt_analyzer.state["df"]), 7)
        logger.info(f"Passed: RT - Third chunk. Result: {result.trend_structure}, BOS: {result.bos}")


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
    # Using unittest.main() directly in __main__ can be problematic with pytest.
    # If running with pytest, __main__ block is usually not executed.
    # For standalone execution, it's fine.
    print("Running Unit Tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False) # Pass a dummy arg to prevent unittest from using sys.argv

    # Example analysis
    print("\n=== Running Example Analysis ===")
    analyzer = SMAnalyzer(min_candles=30, fvg_min_strength=0.1, liquidity_lookback=10, swing_window=3)

    # Generate realistic test data
    np.random.seed(42)
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1H').astype(int) // 10**6
    base_prices = np.cumsum(np.random.normal(0, 0.5, 100)) + 100
    spreads = np.random.uniform(0.1, 0.5, 100)
    highs = base_prices + spreads
    lows = base_prices - spreads
    closes = np.random.uniform(lows, highs)
    opens = np.roll(closes, 1)
    opens[0] = closes[0] - 0.1 # Adjust first open to avoid NaN if roll starts with one
    volumes = np.random.randint(50, 1000, 100) + np.abs(np.random.normal(0, 200, 100))

    test_candles_example = list(zip(
        timestamps,
        opens,
        highs,
        lows,
        closes,
        volumes
    ))

    # Add a clear FVG (Bullish FVG: C1.High < C3.Low)
    # Example: C50: High=X, Low=Y; C51: High=A, Low=B; C52: High=Z, Low=W
    # To create Bullish FVG, we need C50.High < C52.Low
    # Let's say C50.High is 100.0. Make C52.Low 101.0
    original_c50 = list(test_candles_example[50])
    original_c52 = list(test_candles_example[52])

    test_candles_example[50] = (original_c50[0], original_c50[1], 100.0, original_c50[3], original_c50[4], original_c50[5]) # High 100
    test_candles_example[51] = (test_candles_example[51][0], test_candles_example[51][1], 100.5, 99.5, test_candles_example[51][4], test_candles_example[51][5]) # Middle
    test_candles_example[52] = (original_c52[0], original_c52[1], original_c52[2], 101.0, original_c52[4], original_c52[5]) # Low 101
    logger.info("Manually injected a bullish FVG at index 50-52 in example data.")

    # Add an order block (significant volume spike with clear body)
    original_c70 = list(test_candles_example[70])
    test_candles_example[70] = (original_c70[0], 108.0, 110.0, 107.5, 109.5, 2500.0) # High volume bullish candle
    logger.info("Manually injected an order block at index 70 in example data.")


    # Analyze structure
    result = analyzer.detect_structure(test_candles_example)

    if result:
        print("\n=== Enhanced Market Structure Analysis Results ===")
        print(f"Current Close: {result.current_close:.2f}")
        print(f"Recent High: {result.recent_high:.2f}")
        print(f"Recent Low: {result.recent_low:.2f}")
        print(f"Trend Structure: {result.trend_structure}")
        print(f"BOS: {result.bos} | CHoCH: {result.choch}")

        if result.fvg:
            print(f"Significant FVG: {result.fvg['type'].capitalize()} ({result.fvg['range'][0]:.2f}-{result.fvg['range'][1]:.2f}) (Strength: {result.fvg['strength']:.2f}%)")
        else:
            print("No significant FVG detected")

        print(f"Order Blocks Found: {len(result.order_blocks)}")
        for ob in result.order_blocks:
            print(f"  - {ob['type'].capitalize()} OB at {ob['price']:.2f} (Vol: {ob['volume']:.0f}, Strength: {ob['strength']:.2f})")

        print(f"Liquidity Zones: {len(result.liquidity['equal_highs'])} equal highs, "
              f"{len(result.liquidity['equal_lows'])} equal lows, "
              f"{len(result.liquidity['volume_clusters'])} volume clusters")
        for eqh in result.liquidity['equal_highs']:
            print(f"  - Equal High at {eqh['price']:.2f} ({eqh['occurrences']} occurrences)")
        for eql in result.liquidity['equal_lows']:
            print(f"  - Equal Low at {eql['price']:.2f} ({eql['occurrences']} occurrences)")
        for voc in result.liquidity['volume_clusters']:
            print(f"  - Volume Cluster at {voc['price']:.2f} (Vol: {voc['volume']:.0f})")

        print(f"Volume Profile: {len(result.volume_profile['high_volume_nodes'])} high volume nodes, "
              f"{len(result.volume_profile['low_volume_nodes'])} low volume nodes")
        for hvn in result.volume_profile['high_volume_nodes']:
            print(f"  - HVN at {hvn['price_level']:.2f} (Vol: {hvn['volume']:.0f})")
        for lvn in result.volume_profile['low_volume_nodes']:
            print(f"  - LVN at {lvn['price_level']:.2f} (Vol: {lvn['volume']:.0f})")

        print(f"Swing Points: {len(result.swing_highs_lows['highs'])} highs, "
              f"{len(result.swing_highs_lows['lows'])} lows")
        for sh in result.swing_highs_lows['highs']:
            print(f"  - Swing High at {sh['price']:.2f} (Strength: {sh['strength']:.4f}) at {sh['timestamp']}")
        for sl in result.swing_highs_lows['lows']:
            print(f"  - Swing Low at {sl['price']:.2f} (Strength: {sl['strength']:.4f}) at {sl['timestamp']}")
    else:
        print("\n=== Market Structure Analysis Failed ===")


    # Real-time processing example
    print("\n=== Real-time Processing Simulation ===")
    rt_analyzer = SMAnalyzer(min_candles=5, real_time_mode=True, fvg_min_strength=0.1)

    for i in range(0, len(test_candles_example), 10): # Process in chunks of 10 candles
        chunk = test_candles_example[i:i+10]
        print(f"\nProcessing chunk {i//10 + 1}: {len(chunk)} new candles.")
        result_rt = rt_analyzer.detect_structure(chunk)
        if result_rt:
            print(f"  - Current Close: {result_rt.current_close:.2f} | Trend: {result_rt.trend_structure} | BOS: {result_rt.bos} | CHoCH: {result_rt.choch}")
            if result_rt.fvg:
                print(f"    - RT FVG: {result_rt.fvg['type']} ({result_rt.fvg['range'][0]:.2f}-{result_rt.fvg['range'][1]:.2f})")
            print(f"  - Total candles in RT buffer: {len(rt_analyzer.state['df'])}")
        else:
            print(f"  - No structure detected in this chunk (waiting for more data or error).")
