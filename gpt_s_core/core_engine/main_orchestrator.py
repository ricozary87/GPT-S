import time
import logging
from typing import List, Dict, Optional, Tuple, Deque
from dataclasses import dataclass, field
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import platform
import smtplib
from email.mime.text import MIMEText
import unittest
from unittest.mock import patch, MagicMock
from collections import deque # Explicitly import deque
import numpy as np # Import numpy for np.clip and np.isna

# Local imports (assuming these are correctly structured in your project)
from gpt_s_core.data_sources.okx_fetcher import get_candlesticks as get_ohlcv_data
from gpt_s_core.analyzers.structure_analyzer import SMAnalyzer, MarketStructure
from gpt_s_core.config.api_manager import APIVault
from gpt_s_core.utils.monitor import SystemMonitor
from gpt_s_core.utils.alert_manager import AlertManager
# from gpt_s_core.config import settings # Uncomment if you have a settings module

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("orchestrator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("orchestrator")

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # 'BOS', 'CHoCH', 'FVG', 'ORDER_BLOCK'
    level: float
    strength: float
    timestamp: int  # Consistent timestamp: Unix epoch in milliseconds
    confidence: float = 0.0
    timeframe: str = "1H"
    source: str = "SMC"
    current_price: Optional[float] = None
    trend: Optional[str] = None
    # Add a unique ID for easier tracking and duplicate checking if needed
    signal_id: str = field(init=False)

    def __post_init__(self):
        # Generate a simple ID based on key attributes for better tracking
        self.signal_id = f"{self.symbol}-{self.timeframe}-{self.signal_type}-{self.timestamp}-{self.level:.2f}"

# Custom Exceptions
class DataFetchError(Exception):
    """Custom exception for data fetching failures."""
    pass

class AnalysisError(Exception):
    """Custom exception for analysis failures."""
    pass

# --- KRITIK 1: Modifikasi SMAnalyzer.detect_structure untuk menerima DataFrame langsung ---
# INI ADALAH PERUBAHAN KRITIS YANG HARUS DITERAPKAN DI FILE `structure_analyzer.py`
# Asumsi: SMAnalyzer.detect_structure sekarang menerima `pd.DataFrame` sebagai input.
# Definisinya akan berubah dari `def detect_structure(self, candles: List) -> Optional[MarketStructure]:`
# menjadi `def detect_structure(self, df: pd.DataFrame) -> Optional[MarketStructure]:`
# Dan di dalamnya, Anda tidak perlu lagi melakukan `pd.DataFrame(candles, columns=...)` karena sudah DataFrame.

class MainOrchestrator:
    def __init__(self, config_path: str = "trading_config.yml"):
        """Initialize trading orchestrator with all components."""
        self.api_vault = APIVault()
        # Initialize SMAnalyzer with appropriate config.
        # min_candles here should match what the SMAnalyzer needs.
        # This config can also be loaded from the external trading_config.yml.
        self.analyzer = SMAnalyzer(
            min_candles=50, # Example default, will be set by load_config if specified there
            volume_multiplier=2.5,
            fvg_min_strength=0.5,
            liquidity_lookback=20,
            swing_window=5
        )
        self.monitor = EnhancedSystemMonitor()
        self.alert_manager = MultiChannelAlertManager()
        self.executor = ThreadPoolExecutor(max_workers=8) # Default 8, can be from config
        self.load_config(config_path)

        # Using deque for last_signals to limit memory usage and easily manage recent signals
        # Max length of 100 signals per symbol.
        self.last_signals: Dict[str, Deque[TradingSignal]] = {}
        # Initialize deques for existing symbols in config, if any
        for symbol in self.symbols:
            self.last_signals[symbol] = deque(maxlen=100) # Max 100 recent signals per symbol

        self.cycle_count = 0
        self.error_count = 0 # Counter for consecutive errors

        logger.info("MainOrchestrator initialized")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Timeframes: {self.intervals}")
        logger.info(f"Signal confidence threshold: {self.confidence_threshold}")

    def load_config(self, config_path: str):
        """
        Load trading configuration from YAML file.
        For demo purposes, still hardcoded, but structure for file loading is hinted.
        """
        # In a real implementation, you would load from YAML (e.g., using PyYAML)
        # For demo purposes, we'll use hardcoded values
        self.symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT"]
        self.intervals = ["1m", "5m", "15m", "1H", "4H"]
        self.confidence_threshold = 0.65
        self.max_retries = 3
        self.alert_channels = ["telegram", "email", "sms"]
        self.signal_whitelist = ["BOS", "CHoCH", "FVG", "ORDER_BLOCK"]
        self.min_candles_fetch = 200 # Number of candles to fetch for analysis

        # Sync relevant config to SMAnalyzer (important if SMAnalyzer uses these internally)
        self.analyzer.config["min_candles"] = self.min_candles_fetch
        # If SMAnalyzer had more config based on intervals, you would pass them here too
        # For example: self.analyzer.config["fvg_min_strength"] = config.get("fvg_strength", 0.5)

        logger.info(f"Configuration loaded (or hardcoded) from {config_path}")

    def run(self):
        """Main execution loop with enhanced error handling and monitoring."""
        self.cycle_count += 1
        logger.info(f"Starting trading cycle #{self.cycle_count}")

        try:
            # Comprehensive system health check
            health_status = self.monitor.check_system_health()
            if not health_status["ok"]:
                logger.error(f"System health check failed: {health_status['message']}. Details: {health_status['details']}")
                self.send_critical_alert(f"🚨 SYSTEM HEALTH FAILURE: {health_status['message']}")
                # If system is unhealthy, stop processing this cycle.
                return

            # Process symbols with retry mechanism and collect outcomes
            # --- KRITIK 4: No backpressure if all pairs failed ---
            successful_tasks_count = self.process_symbols_with_retry()
            if successful_tasks_count == 0 and len(self.symbols) * len(self.intervals) > 0:
                logger.critical("🔥 All configured symbol-timeframe pairs failed to process. Investigate immediately!")
                self.send_critical_alert("🔥 All trading pairs failed to process. Check logs for details.")
                # Consider raising SystemExit here if this is a critical condition for your system
                # raise SystemExit("Critical: All trading pairs failed.")

            # Log performance metrics
            self.log_performance_metrics()

            # Reset error count on successful cycle
            self.error_count = 0

        except Exception as e:
            self.error_count += 1
            logger.error(f"Critical error in main loop: {e}", exc_info=True)
            self.send_critical_alert(f"🔥 Orchestrator crashed in cycle #{self.cycle_count}: {str(e)}")

            # Emergency shutdown after multiple consecutive errors
            if self.error_count >= 5:
                logger.critical("Too many consecutive errors. Shutting down.")
                self.send_critical_alert("🛑 EMERGENCY SHUTDOWN INITIATED")
                raise SystemExit("Emergency shutdown due to excessive errors.")
        finally:
            # --- KRITIK 3: Logger terlalu verbose per signal, ganti dengan summary ---
            signal_summary = self._get_signal_summary()
            logger.info(f"Cycle #{self.cycle_count} completed. Signal Summary: {signal_summary}")
            # Ensure last_signals is not printed directly here to prevent verbosity.

    def _get_signal_summary(self) -> str:
        """Generates a summary string of signals processed in the current cycle."""
        signal_counts = {}
        total_signals = 0
        for symbol_deque in self.last_signals.values():
            for signal in symbol_deque:
                signal_counts[signal.signal_type] = signal_counts.get(signal.signal_type, 0) + 1
                total_signals += 1

        parts = [f"Processed {total_signals} signals."]
        if signal_counts:
            signal_details = ", ".join([f"{st}={count}" for st, count in signal_counts.items()])
            parts.append(f"[{signal_details}]")
        return " ".join(parts)


    def process_symbols_with_retry(self) -> int:
        """
        Process all symbols with retry mechanism for failed tasks using ThreadPoolExecutor.
        Returns the count of successfully processed symbol-timeframe pairs.
        """
        futures = {}
        for symbol in self.symbols:
            for interval in self.intervals:
                key = f"{symbol}_{interval}"
                futures[key] = self.executor.submit(
                    self.process_symbol_with_retry_wrapper,
                    symbol,
                    interval
                )

        successful_tasks = 0
        # Process results as they complete
        for key, future in futures.items():
            try:
                future.result() # This will re-raise the exception if the wrapper failed
                successful_tasks += 1
            except (DataFetchError, AnalysisError, Exception) as e:
                # Log specific failure for this task
                logger.error(f"Task for {key} failed permanently after all retries: {e}")
                # Critical alert for individual pair failure (optional, can be noisy)
                # self.send_critical_alert(f"Persistent failure for {key}: {str(e)}")
        return successful_tasks

    def process_symbol_with_retry_wrapper(self, symbol: str, interval: str):
        """
        Wrapper to handle retries for a single symbol-timeframe pair.
        Raises the final exception if all retries fail, to be caught by `as_completed`.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                self.process_symbol(symbol, interval)
                logger.debug(f"Successfully processed {symbol} {interval} on attempt {attempt}.")
                return  # Success, exit retry loop
            except DataFetchError as e:
                logger.warning(f"Data fetch error for {symbol} {interval} (attempt {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise # Re-raise on last attempt to be caught by as_completed
            except AnalysisError as e:
                logger.error(f"Analysis failed for {symbol} {interval} (no retry for analysis errors): {e}")
                raise # Re-raise immediately as it might indicate a logic bug
            except Exception as e:
                logger.error(f"Unexpected error processing {symbol} {interval} (attempt {attempt}/{self.max_retries}): {e}", exc_info=True)
                if attempt < self.max_retries:
                    time.sleep(1 + attempt) # Simple linear backoff for unexpected errors
                else:
                    raise # Re-raise on last attempt

    def process_symbol(self, symbol: str, interval: str):
        """Process trading signals for a single symbol and timeframe."""
        logger.debug(f"Initiating process for {symbol} on {interval} timeframe.") # Changed to debug for less verbosity

        # Fetch OHLCV data with timeout
        df = None
        try:
            df = get_ohlcv_data(
                symbol=symbol,
                timeframe=interval,
                limit=self.min_candles_fetch,
                timeout=15
            )
        except Exception as e:
            logger.error(f"API data fetch failed for {symbol} {interval}: {e}", exc_info=True)
            raise DataFetchError(f"API data fetch failed for {symbol} {interval}") from e

        if df is None or df.empty or len(df) < self.analyzer.config["min_candles"]:
            logger.warning(f"Insufficient or no valid data returned for {symbol} {interval} (fetched: {len(df) if df is not None else 0}, needed: {self.analyzer.config['min_candles']}).")
            return

        # Analyze market structure
        analysis_result = None
        try:
            # --- KRITIK 1: SMAnalyzer.detect_structure sekarang menerima DataFrame ---
            # Assume SMAnalyzer.detect_structure is updated to accept pd.DataFrame directly
            analysis_result = self.analyzer.detect_structure(df)
        except Exception as e:
            logger.error(f"Market structure analysis failed for {symbol} {interval}: {e}", exc_info=True)
            raise AnalysisError(f"Market structure analysis failed for {symbol} {interval}") from e

        if not analysis_result: # analysis_result can be None if SMAnalyzer.detect_structure returns None
            logger.warning(f"No meaningful analysis result for {symbol} {interval}.")
            return

        # Generate and process trading signals
        signals = self._generate_signals(symbol, interval, analysis_result)
        for signal in signals:
            self._process_signal(signal)
        logger.debug(f"Completed processing {symbol} {interval}. Generated {len(signals)} signals.") # Changed to debug

    def _generate_signals(self, symbol: str, interval: str, analysis: MarketStructure) -> List[TradingSignal]:
        """Generate trading signals from analysis results with enhanced logic."""
        signals = []

        current_close = analysis.current_close
        # --- KRITIK 2: MarketStructure default values tidak dicek NaN di alert ---
        # Robustness check for critical fields before generating signals
        if pd.isna(current_close) or pd.isna(analysis.recent_high) or pd.isna(analysis.recent_low):
            logger.warning(f"❌ Analysis result for {symbol} {interval} contains NaN in critical price levels. Skipping signal generation.")
            return []

        # 1. BOS/CHoCH signals with confidence calculation
        if analysis.bos and current_close > analysis.recent_high:
            price_break_pct = (current_close - analysis.recent_high) / analysis.recent_high
            # --- KRITIK 3: Confidence terlalu agresif, gunakan np.clip atau logaritmik ---
            # Using log1p (log(1+x)) for more gradual confidence increase and np.clip for bounds
            confidence = np.clip(0.7 + np.log1p(price_break_pct * 10), 0.7, 0.95) # Multiplier of 10 for log scale
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type="BOS",
                level=analysis.recent_high,
                strength=price_break_pct * 100,
                timestamp=int(time.time() * 1000), # Milliseconds timestamp
                confidence=float(confidence), # Ensure float type for dataclass
                timeframe=interval,
                current_price=current_close,
                trend=analysis.trend_structure
            ))

        if analysis.choch and current_close < analysis.recent_low:
            price_break_pct = (analysis.recent_low - current_close) / analysis.recent_low
            confidence = np.clip(0.7 + np.log1p(price_break_pct * 10), 0.7, 0.95)
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type="CHoCH",
                level=analysis.recent_low,
                strength=price_break_pct * 100,
                timestamp=int(time.time() * 1000), # Milliseconds timestamp
                confidence=float(confidence),
                timeframe=interval,
                current_price=current_close,
                trend=analysis.trend_structure
            ))

        # 2. FVG signals
        if analysis.fvg:
            fvg_strength = analysis.fvg.get('strength', 0.0)
            if fvg_strength >= self.analyzer.config.get("fvg_min_strength", 0.0): # Ensure FVG meets its own min_strength
                fvg_confidence = np.clip(0.6 + np.log1p(fvg_strength), 0.6, 0.9) # Base 0.6, scale with log strength
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type="FVG",
                    level=analysis.fvg['range'][0],
                    strength=fvg_strength,
                    timestamp=int(time.time() * 1000), # Milliseconds timestamp
                    confidence=float(fvg_confidence),
                    timeframe=interval,
                    current_price=current_close,
                    trend=analysis.trend_structure
                ))

        # 3. Order Block signals with volume/strength filter
        if analysis.order_blocks:
            for block in analysis.order_blocks:
                # Use block's strength from SMAnalyzer directly, or a custom threshold
                ob_strength = block.get('strength', 0.0)
                # Consider volume threshold defined by SMAnalyzer if block['volume'] is raw volume
                # If SMAnalyzer already filters by volume_multiplier, then ob_strength is sufficient.
                
                # Check for NaN in block level/strength before using
                if pd.isna(block.get('range', [np.nan])[0]) or pd.isna(ob_strength):
                    logger.warning(f"Skipping Order Block with NaN values for {symbol}.")
                    continue

                # Scale confidence based on OB strength
                ob_confidence = np.clip(0.7 + np.log1p(ob_strength), 0.7, 0.95)
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type="ORDER_BLOCK",
                    level=block['range'][0], # Use the start of the block range as level
                    strength=ob_strength,
                    timestamp=block.get('timestamp', int(time.time() * 1000)), # Use block's timestamp if available
                    confidence=float(ob_confidence),
                    timeframe=interval,
                    current_price=current_close,
                    trend=analysis.trend_structure
                ))
        return signals

    def _process_signal(self, signal: TradingSignal):
        """
        Process and validate trading signal with enhanced checks, storage, and checkpointing.
        --- KRITIK 5: Tidak ada checkpointing signal ---
        """
        # --- KRITIK 2: MarketStructure default values tidak dicek NaN di alert (Signal level/strength) ---
        if pd.isna(signal.level) or pd.isna(signal.strength) or pd.isna(signal.confidence):
            logger.warning(f"❌ Signal for {signal.symbol} ({signal.signal_type}) contains NaN values (level/strength/confidence). Ignoring signal.")
            return

        # Check if signal type is in whitelist
        if signal.signal_type not in self.signal_whitelist:
            logger.debug(f"Ignoring non-whitelisted signal type: {signal.signal_type} for {signal.symbol}.")
            return

        # Check confidence threshold
        if signal.confidence < self.confidence_threshold:
            logger.info(f"Low confidence signal ignored: {signal.signal_type} for {signal.symbol} "
                        f"(Confidence: {signal.confidence:.2f} < {self.confidence_threshold}).")
            return

        # Initialize deque for symbol if not exists (already done in __init__)
        # if signal.symbol not in self.last_signals:
        #    self.last_signals[signal.symbol] = deque(maxlen=100) # Max 100 recent signals per symbol

        # Check for duplicate signals
        duplicate = self.is_duplicate_signal(signal, self.last_signals[signal.symbol])

        if duplicate:
            logger.info(f"Ignoring duplicate {signal.signal_type} signal for {signal.symbol} at level {signal.level:.4f}.")
            return

        # Store signal (will automatically pop old ones if deque is full)
        self.last_signals[signal.symbol].append(signal)

        # --- KRITIK 5: Tambahkan checkpointing signal (contoh sederhana ke file JSON/SQLite) ---
        self._checkpoint_signal(signal)

        # Send alert through configured channels
        self.send_signal_alert(signal)
        logger.info(f"New signal processed & stored: {signal.signal_type} for {signal.symbol} @ {signal.level:.4f} (Conf: {signal.confidence:.2f}). ID: {signal.signal_id}")

    def _checkpoint_signal(self, signal: TradingSignal):
        """
        Saves a signal to persistent storage (e.g., SQLite, JSON file).
        For simplicity, this example uses a JSON file append mode.
        In a real system, consider a proper database for querying and reliability.
        """
        checkpoint_file = f"signals_checkpoint_{signal.symbol}.json"
        try:
            # Convert TradingSignal dataclass to a dictionary
            signal_dict = signal.__dict__.copy()
            # Convert timestamp from int (ms) to ISO string for readability if desired, or keep as int
            signal_dict['timestamp_iso'] = pd.to_datetime(signal.timestamp, unit='ms').isoformat()

            with open(checkpoint_file, 'a') as f: # 'a' for append mode
                # Serialize to JSON and write, each signal on a new line
                json.dump(signal_dict, f)
                f.write('\n') # New line for each entry
            logger.debug(f"Signal {signal.signal_id} checkpointed to {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to checkpoint signal {signal.signal_id}: {e}", exc_info=True)


    def is_duplicate_signal(self, signal: TradingSignal, recent_signals: Deque[TradingSignal]) -> bool:
        """
        Check if similar signal was recently processed based on type, timeframe,
        similar level (within tolerance), and within a time window.
        """
        # Time window based on timeframe, or a default
        time_window_seconds = 300 # 5 minutes default for low timeframes
        if signal.timeframe in ["1H", "4H"]:
            time_window_seconds = 3600 # 1 hour for higher timeframes
        elif signal.timeframe in ["1m", "5m"]:
            time_window_seconds = 120 # 2 minutes for very low timeframes


        # Price tolerance based on percentage of the signal level
        price_tolerance = 0.0005 * signal.level if signal.level != 0 else 0.0001 # 0.05% of the price level

        current_time_ms = int(time.time() * 1000)

        for s in recent_signals:
            # Check if it's the exact same signal_id (for re-runs or very quick duplicates)
            if s.signal_id == signal.signal_id:
                return True

            # More flexible duplicate check
            if (s.signal_type == signal.signal_type and
                s.timeframe == signal.timeframe and
                abs(s.level - signal.level) < price_tolerance and # Check level proximity
                (current_time_ms - s.timestamp) < (time_window_seconds * 1000) # Check within time window
            ):
                return True
        return False

    def send_signal_alert(self, signal: TradingSignal):
        """Send signal alert through configured channels."""
        # Enhance message with more details
        message = (
            f"🚀 {signal.symbol} {signal.signal_type} ({signal.timeframe})\n"
            f"  Level: {signal.level:.4f}\n"
            f"  Current Price: {signal.current_price:.4f}\n"
            f"  Strength: {signal.strength:.2f}%\n"
            f"  Confidence: {signal.confidence*100:.0f}%\n"
            f"  Trend: {signal.trend.capitalize() if signal.trend else 'N/A'}\n"
            f"  Timestamp: {pd.to_datetime(signal.timestamp, unit='ms').strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Signal ID: {signal.signal_id}" # Added signal ID for traceability
        )

        for channel in self.alert_channels:
            try:
                self.alert_manager.send_alert(
                    message=message,
                    channel=channel,
                    severity="medium" if signal.confidence < 0.8 else "high"
                )
                logger.info(f"Alert sent via {channel} for {signal.symbol} {signal.signal_type}")
            except Exception as e:
                logger.error(f"Failed to send {channel} alert for {signal.symbol} {signal.signal_type}: {e}", exc_info=True)

    def send_critical_alert(self, message: str):
        """Send critical system alert through all configured channels."""
        full_message = f"🛑 CRITICAL SYSTEM ALERT 🛑\nTime: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\nMessage: {message}"
        for channel in self.alert_channels:
            try:
                self.alert_manager.send_alert(
                    message=full_message,
                    channel=channel,
                    severity="critical"
                )
                logger.error(f"Critical alert sent via {channel}.")
            except Exception as e:
                logger.critical(f"FATAL: Failed to send critical alert via {channel}: {e}", exc_info=True)

    def log_performance_metrics(self):
        """Log system performance metrics."""
        metrics = self.monitor.get_system_metrics()
        logger.info(
            f"Performance Metrics | "
            f"CPU: {metrics['cpu']:.1f}% | "
            f"Memory: {metrics['memory']:.1f}% | "
            f"Disk: {metrics['disk']:.1f}% | "
            f"Network: {metrics['network']['sent']:.2f}MB sent, {metrics['network']['recv']:.2f}MB recv | "
            f"OS: {metrics['os']}"
        )

# ====================================================================================================
# External Classes (provided for completeness in this single file)
# In a real project, these would be in their own respective files.

class SystemMonitor:
    def check_system_health(self) -> dict:
        return {"ok": True, "message": "Base health check"}
    def get_system_metrics(self) -> dict:
        return {"cpu": 0, "memory": 0, "disk": 0, "network": {"sent": 0, "recv": 0}, "boot_time": 0, "os": "N/A"}

class EnhancedSystemMonitor(SystemMonitor):
    """Extended system monitor with additional metrics."""
    def check_system_health(self) -> dict:
        health = {
            "ok": True,
            "message": "All systems normal",
            "details": {}
        }
        try:
            cpu_usage = psutil.cpu_percent(interval=None) # Non-blocking for quick check
            health["details"]["cpu"] = cpu_usage
            if cpu_usage > 90:
                health["ok"] = False
                health["message"] = f"High CPU usage: {cpu_usage:.1f}%"

            mem = psutil.virtual_memory()
            health["details"]["memory"] = mem.percent
            if mem.percent > 90:
                health["ok"] = False
                health["message"] = f"High memory usage: {mem.percent:.1f}%"

            disk = psutil.disk_usage('/')
            health["details"]["disk"] = disk.percent
            if disk.percent > 95:
                health["ok"] = False
                health["message"] = f"Low disk space: {disk.percent:.1f}% used"

            net = psutil.net_io_counters()
            health["details"]["network"] = {
                "sent": net.bytes_sent / (1024*1024),
                "recv": net.bytes_recv / (1024*1024)
            }
        except Exception as e:
            health["ok"] = False
            health["message"] = f"Error during system health check: {e}"
            logger.error(f"Error in EnhancedSystemMonitor.check_system_health: {e}", exc_info=True)
        return health

    def get_system_metrics(self) -> dict:
        """Get detailed system performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=1) # Blocking for 1 second for accurate average
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net = psutil.net_io_counters()

        return {
            "cpu": float(cpu_percent), # Ensure float
            "memory": float(mem.percent), # Ensure float
            "disk": float(disk.percent), # Ensure float
            "network": {
                "sent": float(net.bytes_sent / (1024*1024)),
                "recv": float(net.bytes_recv / (1024*1024))
            },
            "boot_time": pd.to_datetime(psutil.boot_time(), unit='s').strftime('%Y-%m-%d %H:%M:%S'),
            "os": platform.platform()
        }

class APIVault:
    """Mock APIVault for demonstration purposes."""
    def get_secret(self, key: str) -> str:
        # In a real system, this would fetch from a secure vault (e.g., HashiCorp Vault, AWS Secrets Manager)
        # For this demo, just return a dummy string or raise an error for unmocked secrets
        logger.warning(f"MOCK: Attempted to get secret for {key}. Returning dummy value.")
        return "dummy_secret_value" # Or raise NotImplementedError

class AlertManager:
    """Base class for alert managers."""
    def send_alert(self, message: str, channel: str = "telegram", severity: str = "medium"):
        raise NotImplementedError("send_alert method must be implemented by subclasses.")

class MultiChannelAlertManager(AlertManager):
    """Alert manager with support for multiple channels (Telegram, Email, SMS)."""
    def send_alert(self, message: str, channel: str = "telegram", severity: str = "medium"):
        if channel == "telegram":
            self._send_telegram_alert(message, severity)
        elif channel == "email":
            self._send_email_alert(message, severity)
        elif channel == "sms":
            self._send_sms_alert(message, severity)
        else:
            logger.error(f"Unsupported alert channel specified: {channel}. Message: {message}")
            raise ValueError(f"Unsupported channel: {channel}")

    def _send_telegram_alert(self, message: str, severity: str):
        emoji = "⚠️" if severity == "medium" else "🔴" if severity == "high" else "🚨"
        formatted_msg = f"{emoji} {severity.upper()} ALERT: {message}"
        # Actual Telegram sending logic would go here
        logger.info(f"MOCK: Telegram alert sent ({severity}): {formatted_msg}")

    def _send_email_alert(self, message: str, severity: str):
        subject = f"{severity.upper()} Alert: Trading System Notification"
        # Actual email sending logic would go here
        logger.info(f"MOCK: Email alert sent ({severity}): {subject} - {message}")

    def _send_sms_alert(self, message: str, severity: str):
        # Actual SMS sending logic would go here
        logger.info(f"MOCK: SMS alert sent ({severity}): {message}")

# ====================================================================================================

if __name__ == "__main__":
    # For running tests via `python your_script_name.py`
    print("Running built-in unit tests...")
    # Using argv=['first-arg-is-ignored'] to prevent unittest.main from parsing command line args
    # exit=False ensures the script continues after tests, which is useful for combined run
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

    # Main trading loop execution
    try:
        orchestrator = MainOrchestrator()
        logger.info("Starting main trading loop...")

        while True:
            cycle_start = time.time()
            orchestrator.run()

            cycle_time = time.time() - cycle_start
            sleep_duration = max(60 - cycle_time, 10)
            logger.info(f"Cycle completed in {cycle_time:.2f}s. Sleeping for {sleep_duration:.2f}s...")
            time.sleep(sleep_duration)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Shutting down gracefully...")
        if orchestrator.executor:
            orchestrator.executor.shutdown(wait=True)
            logger.info("ThreadPoolExecutor shut down.")
        logger.info("Orchestrator stopped.")
    except SystemExit as se:
        logger.critical(f"System exit triggered: {se}")
    except Exception as e:
        logger.critical(f"UNHANDLED FATAL ERROR IN MAIN ORCHESTRATOR LOOP: {e}", exc_info=True)
        try:
            # Ensure orchestrator exists before attempting to send critical alert
            if 'orchestrator' in locals(): # Check if orchestrator object was successfully created
                 orchestrator.send_critical_alert(f"💀 FATAL ERROR: {str(e)}. Orchestrator Terminated.")
        except Exception as alert_e:
            logger.critical(f"Failed to send final critical alert: {alert_e}", exc_info=True)
        raise

# ===========================================================
# Unit tests for critical methods using unittest framework (can be adapted for pytest)

class TestMainOrchestrator(unittest.TestCase):
    def setUp(self):
        # Patch external dependencies for unit tests
        # Patching where they are imported within the __main__ scope of this script
        self.patcher_fetch = patch('__main__.get_ohlcv_data')
        self.mock_fetch = self.patcher_fetch.start()

        self.patcher_analyzer = patch('__main__.SMAnalyzer')
        self.mock_analyzer_class = self.patcher_analyzer.start()

        self.mock_analyzer_instance = MagicMock()
        # Configure the mock analyzer's detect_structure method to return a controlled result
        self.mock_analyzer_instance.detect_structure.return_value = MagicMock(
            bos=True,
            choch=False,
            fvg={"range": (100.0, 101.0), "strength": 0.8, "type": "bullish"}, # Added type for FVG
            order_blocks=[{"range": (99.0, 99.5), "volume": 1500, "strength": 1.5, "timestamp": int(time.time() * 1000)}],
            recent_high=102.0,
            recent_low=98.5,
            current_close=102.5,
            trend_structure="uptrend",
            # Ensure other attributes used by _generate_signals are present
            swing_highs_lows={'highs': [], 'lows': []}, # Minimal mock for unused attributes
            volume_profile={'high_volume_nodes': [], 'low_volume_nodes': []},
            liquidity={'equal_highs': [], 'equal_lows': [], 'volume_clusters': []}
        )
        # Ensure analyzer's config is accessible if needed for FVG strength check
        self.mock_analyzer_instance.config = {"fvg_min_strength": 0.5, "min_candles": 50}

        self.mock_analyzer_class.return_value = self.mock_analyzer_instance

        self.patcher_monitor = patch('__main__.EnhancedSystemMonitor')
        self.mock_monitor = self.patcher_monitor.start()
        self.mock_monitor.return_value.check_system_health.return_value = {"ok": True, "message": "All systems normal", "details": {}}
        self.mock_monitor.return_value.get_system_metrics.return_value = {
            "cpu": 10.0, "memory": 20.0, "disk": 30.0, "network": {"sent": 100.0, "recv": 200.0},
            "boot_time": "2025-01-01 00:00:00", "os": "Linux"
        }

        self.patcher_alert = patch('__main__.MultiChannelAlertManager')
        self.mock_alert = self.patcher_alert.start()
        self.mock_alert_instance = self.mock_alert.return_value
        self.mock_alert_instance.send_alert.return_value = None

        # Patch APIVault because MainOrchestrator initializes it
        self.patcher_api_vault = patch('__main__.APIVault')
        self.mock_api_vault = self.patcher_api_vault.start()


        # Initialize the orchestrator, which will now use our mocks
        self.orchestrator = MainOrchestrator()
        self.orchestrator.symbols = ["BTC-USDT"]
        self.orchestrator.intervals = ["15m"]
        self.orchestrator.min_candles_fetch = 100
        self.orchestrator.confidence_threshold = 0.65

        # Sample data for fetcher mock (ensure it meets SMAnalyzer's min_candles)
        # Assuming min_candles for analyzer is 50.
        num_mock_candles = 60 # Provide more than min_candles_fetch or analyzer's min_candles
        self.sample_ohlcv_df = pd.DataFrame([
            [int(time.time() * 1000) - (num_mock_candles - 1 - i) * 60000, # Decreasing timestamps
             100.0 + i, 105.0 + i, 95.0 + i, 102.0 + i, 1000.0 + i * 10]
            for i in range(num_mock_candles)
        ], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).set_index('timestamp')

        self.mock_fetch.return_value = self.sample_ohlcv_df

        self.orchestrator.last_signals = {} # Reset for each test
        # Manually initialize deque for the test symbol
        self.orchestrator.last_signals["BTC-USDT"] = deque(maxlen=100)

        # Patch checkpointing to prevent actual file writes during tests
        self.patcher_checkpoint = patch.object(self.orchestrator, '_checkpoint_signal')
        self.mock_checkpoint = self.patcher_checkpoint.start()

    def tearDown(self):
        self.patcher_fetch.stop()
        self.patcher_analyzer.stop()
        self.patcher_monitor.stop()
        self.patcher_alert.stop()
        self.patcher_api_vault.stop()
        self.patcher_checkpoint.stop() # Stop checkpoint patcher

    def test_generate_signals(self):
        logger.info("Running test_generate_signals...")
        # Mock analysis is already configured in setUp to return BOS, FVG, OB
        signals = self.orchestrator._generate_signals("BTC-USDT", "15m", self.mock_analyzer_instance.detect_structure.return_value)
        self.assertEqual(len(signals), 3) # BOS, FVG, ORDER_BLOCK

        bos_signals = [s for s in signals if s.signal_type == "BOS"]
        self.assertEqual(len(bos_signals), 1)
        self.assertAlmostEqual(bos_signals[0].level, 102.0)
        self.assertGreater(bos_signals[0].confidence, 0.7)
        self.assertAlmostEqual(bos_signals[0].current_price, 102.5)

        fvg_signals = [s for s in signals if s.signal_type == "FVG"]
        self.assertEqual(len(fvg_signals), 1)
        self.assertAlmostEqual(fvg_signals[0].level, 100.0)
        self.assertGreaterEqual(fvg_signals[0].confidence, 0.6) # Base confidence for FVG

        ob_signals = [s for s in signals if s.signal_type == "ORDER_BLOCK"]
        self.assertEqual(len(ob_signals), 1)
        self.assertAlmostEqual(ob_signals[0].level, 99.0)
        self.assertGreaterEqual(ob_signals[0].confidence, 0.7)

        logger.info("test_generate_signals PASSED.")

    def test_process_signal_duplicate(self):
        logger.info("Running test_process_signal_duplicate...")
        signal_time = int(time.time() * 1000)
        signal = TradingSignal(
            symbol="BTC-USDT", signal_type="BOS", level=102.0, strength=0.5,
            timestamp=signal_time, confidence=0.8, timeframe="15m", current_price=102.5
        )

        self.orchestrator.last_signals["BTC-USDT"].append(signal) # Add duplicate
        
        with self.assertLogs(logger.name, level="INFO") as cm:
            self.orchestrator._process_signal(signal)
            self.assertIn("Ignoring duplicate BOS signal for BTC-USDT at level 102.0000", cm.output[0])
        self.mock_alert_instance.send_alert.assert_not_called()
        self.mock_checkpoint.assert_not_called() # No checkpoint for duplicates
        logger.info("test_process_signal_duplicate PASSED.")

    def test_process_signal_low_confidence(self):
        logger.info("Running test_process_signal_low_confidence...")
        signal = TradingSignal(
            symbol="BTC-USDT", signal_type="FVG", level=100.0, strength=0.5,
            timestamp=int(time.time() * 1000), confidence=0.5, timeframe="15m", current_price=100.5
        )
        
        with self.assertLogs(logger.name, level="INFO") as cm:
            self.orchestrator._process_signal(signal)
            self.assertIn("Low confidence signal ignored", cm.output[0])
        self.mock_alert_instance.send_alert.assert_not_called()
        self.mock_checkpoint.assert_not_called() # No checkpoint for low confidence
        logger.info("test_process_signal_low_confidence PASSED.")

    @patch.object(MultiChannelAlertManager, 'send_alert')
    def test_valid_signal_processing(self, mock_send_alert):
        logger.info("Running test_valid_signal_processing...")
        signal = TradingSignal(
            symbol="BTC-USDT", signal_type="ORDER_BLOCK", level=99.0, strength=1500.0,
            timestamp=int(time.time() * 1000), confidence=0.85, timeframe="15m", current_price=99.2
        )
        
        self.orchestrator._process_signal(signal)
        
        self.assertIn(signal, self.orchestrator.last_signals["BTC-USDT"])
        mock_send_alert.assert_called_once()
        self.mock_checkpoint.assert_called_once_with(signal) # Checkpoint should be called
        logger.info("test_valid_signal_processing PASSED.")

    @patch('__main__.get_ohlcv_data')
    @patch.object(SMAnalyzer, 'detect_structure')
    @patch.object(MainOrchestrator, '_generate_signals')
    @patch.object(MainOrchestrator, '_process_signal')
    def test_process_symbol_success(self, mock_process_signal, mock_generate_signals, mock_detect_structure, mock_get_ohlcv_data):
        logger.info("Running test_process_symbol_success...")
        mock_get_ohlcv_data.return_value = self.sample_ohlcv_df

        # analysis_result should match the mock from setUp for detect_structure
        mock_analysis_result = self.mock_analyzer_instance.detect_structure.return_value
        mock_detect_structure.return_value = mock_analysis_result

        mock_signals = [MagicMock(spec=TradingSignal, symbol="BTC-USDT", signal_type="BOS")]
        mock_generate_signals.return_value = mock_signals

        self.orchestrator.process_symbol("BTC-USDT", "15m")

        mock_get_ohlcv_data.assert_called_once_with(
            symbol="BTC-USDT", interval="15m", limit=self.orchestrator.min_candles_fetch, timeout=unittest.mock.ANY
        )
        mock_detect_structure.assert_called_once_with(self.sample_ohlcv_df) # Now expects DataFrame
        
        mock_generate_signals.assert_called_once_with("BTC-USDT", "15m", mock_analysis_result)
        mock_process_signal.assert_called_once_with(mock_signals[0])

        logger.info("test_process_symbol_success PASSED.")

    @patch('__main__.get_ohlcv_data')
    def test_process_symbol_data_fetch_failure(self, mock_get_ohlcv_data):
        logger.info("Running test_process_symbol_data_fetch_failure...")
        mock_get_ohlcv_data.side_effect = ConnectionError("Network issue")

        with self.assertRaises(DataFetchError):
            self.orchestrator.process_symbol("BTC-USDT", "15m")
        logger.info("test_process_symbol_data_fetch_failure PASSED.")

    @patch('__main__.get_ohlcv_data')
    @patch.object(SMAnalyzer, 'detect_structure')
    def test_process_symbol_analysis_failure(self, mock_detect_structure, mock_get_ohlcv_data):
        logger.info("Running test_process_symbol_analysis_failure...")
        mock_get_ohlcv_data.return_value = self.sample_ohlcv_df

        mock_detect_structure.side_effect = ValueError("Invalid analysis parameter")

        with self.assertRaises(AnalysisError):
            self.orchestrator.process_symbol("BTC-USDT", "15m")
        logger.info("test_process_symbol_analysis_failure PASSED.")

    def test_process_symbols_with_retry_success(self):
        logger.info("Running test_process_symbols_with_retry_success...")
        with patch.object(self.orchestrator, 'process_symbol_with_retry_wrapper') as mock_wrapper:
            mock_wrapper.return_value = None # Assume success for all calls
            successful_count = self.orchestrator.process_symbols_with_retry()
            expected_calls = len(self.orchestrator.symbols) * len(self.orchestrator.intervals)
            self.assertEqual(mock_wrapper.call_count, expected_calls)
            self.assertEqual(successful_count, expected_calls) # All tasks should be successful
            logger.info("test_process_symbols_with_retry_success PASSED.")

    def test_process_symbols_with_retry_failure(self):
        logger.info("Running test_process_symbols_with_retry_failure...")
        with patch.object(self.orchestrator, 'process_symbol_with_retry_wrapper') as mock_wrapper:
            # Make the wrapper always raise an exception after retries
            mock_wrapper.side_effect = DataFetchError("Simulated persistent fetch error")
            
            # The outer process_symbols_with_retry will catch these and log
            with self.assertLogs(logger.name, level='ERROR') as cm:
                successful_count = self.orchestrator.process_symbols_with_retry()
                # Check for error logs for each failed task
                self.assertGreater(len(cm.output), 0)
                self.assertIn("failed permanently after all retries", cm.output[0])
            self.assertEqual(successful_count, 0) # No tasks should be successful
            logger.info("test_process_symbols_with_retry_failure PASSED.")

    @patch.object(EnhancedSystemMonitor, 'check_system_health')
    @patch.object(MainOrchestrator, 'process_symbols_with_retry')
    @patch.object(MainOrchestrator, 'log_performance_metrics')
    def test_run_system_unhealthy(self, mock_log_perf, mock_process_symbols, mock_check_health):
        logger.info("Running test_run_system_unhealthy...")
        mock_check_health.return_value = {"ok": False, "message": "High CPU", "details": {}}
        
        with self.assertLogs(logger.name, level='ERROR') as cm:
            self.orchestrator.run()
            self.assertIn("System health check failed", cm.output[0])
        
        mock_process_symbols.assert_not_called()
        mock_log_perf.assert_not_called()
        logger.info("test_run_system_unhealthy PASSED.")

    @patch.object(MultiChannelAlertManager, 'send_alert')
    def test_send_critical_alert(self, mock_send_alert):
        logger.info("Running test_send_critical_alert...")
        test_message = "Test critical message"
        self.orchestrator.send_critical_alert(test_message)
        
        self.assertEqual(mock_send_alert.call_count, len(self.orchestrator.alert_channels))
        
        expected_calls = [
            unittest.mock.call(message=unittest.mock.ANY, channel="telegram", severity="critical"),
            unittest.mock.call(message=unittest.mock.ANY, channel="email", severity="critical"),
            unittest.mock.call(message=unittest.mock.ANY, channel="sms", severity="critical")
        ]
        mock_send_alert.assert_has_calls(expected_calls, any_order=True)
        
        call_args, _ = mock_send_alert.call_args_list[0]
        self.assertIn(test_message, call_args[0])
        self.assertIn("CRITICAL SYSTEM ALERT", call_args[0])
        logger.info("test_send_critical_alert PASSED.")
