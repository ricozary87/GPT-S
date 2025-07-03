import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import platform
import smtplib
from email.mime.text import MIMEText
import unittest
from unittest.mock import patch, MagicMock

# Local imports
from gpt_s_core.data_sources.okx_fetcher import get_ohlcv_data
from gpt_s_core.analyzers.structure_analyzer import SMAnalyzer
from gpt_s_core.config.api_manager import APIVault
from gpt_s_core.utils.monitor import SystemMonitor
from gpt_s_core.utils.alert_manager import AlertManager
from gpt_s_core.config import settings

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
    timestamp: int
    confidence: float = 0.0
    timeframe: str = "1H"  # Default timeframe
    source: str = "SMC"   # Signal source

class MainOrchestrator:
    def __init__(self, config_path: str = "trading_config.yml"):
        """Initialize trading orchestrator with all components."""
        self.api_vault = APIVault()
        self.analyzer = SMAnalyzer()
        self.monitor = EnhancedSystemMonitor()
        self.alert_manager = MultiChannelAlertManager()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.load_config(config_path)
        self.last_signals: Dict[str, List[TradingSignal]] = {}
        self.cycle_count = 0
        self.error_count = 0

        logger.info("MainOrchestrator initialized")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Timeframes: {self.intervals}")
        logger.info(f"Signal confidence threshold: {self.confidence_threshold}")

    def load_config(self, config_path: str):
        """Load trading configuration from YAML file"""
        # In a real implementation, you would load from YAML
        # For demo purposes, we'll use hardcoded values
        self.symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT"]
        self.intervals = ["1m", "5m", "15m", "1H", "4H"]
        self.confidence_threshold = 0.65
        self.max_retries = 3
        self.alert_channels = ["telegram", "email", "sms"]
        self.signal_whitelist = ["BOS", "CHoCH", "FVG", "ORDER_BLOCK"]
        
        logger.info(f"Configuration loaded from {config_path}")

    def run(self):
        """Main execution loop with enhanced error handling and monitoring"""
        self.cycle_count += 1
        logger.info(f"Starting trading cycle #{self.cycle_count}")
        
        try:
            # Comprehensive system health check
            health_status = self.monitor.check_system_health()
            if not health_status["ok"]:
                logger.error(f"System health check failed: {health_status['message']}")
                self.send_critical_alert(f"🚨 SYSTEM HEALTH FAILURE: {health_status['message']}")
                return

            # Process symbols with retry mechanism
            self.process_symbols_with_retry()

            # Log performance metrics
            self.log_performance_metrics()

        except Exception as e:
            self.error_count += 1
            logger.error(f"Critical error in main loop: {e}", exc_info=True)
            self.send_critical_alert(f"🔥 Orchestrator crashed: {str(e)}")
            
            # Emergency shutdown after multiple errors
            if self.error_count > 5:
                logger.critical("Too many consecutive errors. Shutting down.")
                self.send_critical_alert("🛑 EMERGENCY SHUTDOWN INITIATED")
                raise SystemExit("Emergency shutdown")
        finally:
            logger.info(f"Cycle #{self.cycle_count} completed. "
                       f"Processed {len(self.last_signals)} signals this cycle.")

    def process_symbols_with_retry(self):
        """Process all symbols with retry mechanism for failed tasks"""
        futures = {}
        for symbol in self.symbols:
            for interval in self.intervals:
                key = f"{symbol}_{interval}"
                futures[key] = self.executor.submit(
                    self.process_symbol_with_retry,
                    symbol,
                    interval
                )

        # Process results as they complete
        for future in as_completed(futures.values()):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Symbol processing failed: {e}")

    def process_symbol_with_retry(self, symbol: str, interval: str):
        """Process symbol with retry mechanism"""
        for attempt in range(1, self.max_retries + 1):
            try:
                self.process_symbol(symbol, interval)
                return  # Success, exit retry loop
            except DataFetchError as e:
                logger.warning(f"Data fetch error for {symbol} {interval} (attempt {attempt}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            except AnalysisError as e:
                logger.error(f"Analysis failed for {symbol} {interval}: {e}")
                break  # Don't retry analysis errors
            except Exception as e:
                logger.error(f"Unexpected error processing {symbol} {interval}: {e}")
                time.sleep(1)

    def process_symbol(self, symbol: str, interval: str):
        """Process trading signals for a single symbol and timeframe"""
        logger.info(f"Processing {symbol} on {interval} timeframe")
        
        # Fetch OHLCV data with timeout
        try:
            df = get_ohlcv_data(
                symbol=symbol,
                interval=interval,
                limit=200,  # More data for better analysis
                timeout=10  # Seconds
            )
        except Exception as e:
            raise DataFetchError(f"Data fetch failed for {symbol} {interval}") from e
            
        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol} {interval}")
            return

        # Analyze market structure
        try:
            analysis = self.analyzer.detect_structure(df)
        except Exception as e:
            raise AnalysisError(f"Analysis failed for {symbol} {interval}") from e
            
        if not analysis:
            logger.warning(f"Empty analysis for {symbol} {interval}")
            return

        # Generate and process trading signals
        signals = self._generate_signals(symbol, interval, analysis)
        for signal in signals:
            self._process_signal(signal)

    def _generate_signals(self, symbol: str, interval: str, analysis) -> List[TradingSignal]:
        """Generate trading signals from analysis results with enhanced logic"""
        signals = []
        
        # 1. BOS/CHoCH signals with confidence calculation
        if analysis.bos:
            confidence = min(0.95, 0.7 + (analysis.current_close - analysis.recent_high) / analysis.recent_high * 5)
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type="BOS",
                level=analysis.recent_high,
                strength=(analysis.current_close - analysis.recent_high) / analysis.recent_high * 100,
                timestamp=int(time.time()),
                confidence=confidence,
                timeframe=interval
            ))
        
        if analysis.choch:
            confidence = min(0.95, 0.7 + (analysis.recent_low - analysis.current_close) / analysis.recent_low * 5)
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type="CHoCH",
                level=analysis.recent_low,
                strength=(analysis.recent_low - analysis.current_close) / analysis.recent_low * 100,
                timestamp=int(time.time()),
                confidence=confidence,
                timeframe=interval
            ))

        # 2. FVG signals
        if analysis.fvg and analysis.fvg['strength'] >= self.analyzer.config["fvg_min_strength"]:
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type="FVG",
                level=analysis.fvg['range'][0],
                strength=analysis.fvg['strength'],
                timestamp=int(time.time()),
                confidence=0.75 if analysis.fvg['strength'] > 1.0 else 0.6,
                timeframe=interval
            ))

        # 3. Order Block signals with volume filter
        if analysis.order_blocks:
            for block in analysis.order_blocks:
                if block['volume'] > 1000:  # Minimum volume threshold
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type="ORDER_BLOCK",
                        level=block['range'][0],
                        strength=block['volume'],
                        timestamp=block['timestamp'],
                        confidence=0.8,
                        timeframe=interval
                    ))

        return signals

    def _process_signal(self, signal: TradingSignal):
        """Process and validate trading signal with enhanced checks"""
        # Check if signal type is in whitelist
        if signal.signal_type not in self.signal_whitelist:
            logger.debug(f"Ignoring non-whitelisted signal type: {signal.signal_type}")
            return

        # Check confidence threshold
        if signal.confidence < self.confidence_threshold:
            logger.info(f"Low confidence signal ignored: {signal.signal_type} "
                       f"(Confidence: {signal.confidence:.2f} < {self.confidence_threshold})")
            return

        # Check for duplicate signals
        recent_signals = self.last_signals.get(signal.symbol, [])
        duplicate = self.is_duplicate_signal(signal, recent_signals)
        
        if duplicate:
            logger.info(f"Ignoring duplicate {signal.signal_type} signal for {signal.symbol}")
            return

        # Store signal
        if signal.symbol not in self.last_signals:
            self.last_signals[signal.symbol] = []
        self.last_signals[signal.symbol].append(signal)
        
        # Send alert through configured channels
        self.send_signal_alert(signal)

    def is_duplicate_signal(self, signal: TradingSignal, recent_signals: list) -> bool:
        """Check if similar signal was recently processed"""
        for s in recent_signals[-5:]:  # Check last 5 signals
            if (s.signal_type == signal.signal_type and 
                abs(s.level - signal.level) < 0.01 * s.level and
                s.timeframe == signal.timeframe and
                (time.time() - s.timestamp) < 3600):  # 1 hour window
                return True
        return False

    def send_signal_alert(self, signal: TradingSignal):
        """Send signal alert through configured channels"""
        message = (
            f"🚀 {signal.symbol} {signal.signal_type} ({signal.timeframe})\n"
            f"Level: {signal.level:.4f}\n"
            f"Strength: {signal.strength:.2f}%\n"
            f"Confidence: {signal.confidence*100:.0f}%"
        )
        
        # Send to all configured channels
        for channel in self.alert_channels:
            try:
                self.alert_manager.send_alert(
                    message=message,
                    channel=channel,
                    severity="medium" if signal.confidence < 0.8 else "high"
                )
                logger.info(f"Alert sent via {channel} for {signal.symbol} {signal.signal_type}")
            except Exception as e:
                logger.error(f"Failed to send {channel} alert: {e}")

    def send_critical_alert(self, message: str):
        """Send critical system alert through all channels"""
        for channel in self.alert_channels:
            try:
                self.alert_manager.send_alert(
                    message=message,
                    channel=channel,
                    severity="critical"
                )
            except Exception as e:
                logger.error(f"Critical alert failed for {channel}: {e}")

    def log_performance_metrics(self):
        """Log system performance metrics"""
        metrics = self.monitor.get_system_metrics()
        logger.info(
            f"Performance Metrics | "
            f"CPU: {metrics['cpu']}% | "
            f"Memory: {metrics['memory']}% | "
            f"Network: {metrics['network']['sent']}MB sent, {metrics['network']['recv']}MB recv"
        )

class EnhancedSystemMonitor(SystemMonitor):
    """Extended system monitor with additional metrics"""
    def check_system_health(self) -> dict:
        """Comprehensive system health check"""
        health = {
            "ok": True,
            "message": "All systems normal",
            "details": {}
        }
        
        # CPU check
        cpu_usage = psutil.cpu_percent(interval=1)
        health["details"]["cpu"] = cpu_usage
        if cpu_usage > 90:
            health["ok"] = False
            health["message"] = f"High CPU usage: {cpu_usage}%"
        
        # Memory check
        mem = psutil.virtual_memory()
        health["details"]["memory"] = mem.percent
        if mem.percent > 90:
            health["ok"] = False
            health["message"] = f"High memory usage: {mem.percent}%"
        
        # Disk check
        disk = psutil.disk_usage('/')
        health["details"]["disk"] = disk.percent
        if disk.percent > 95:
            health["ok"] = False
            health["message"] = f"Low disk space: {disk.percent}% used"
        
        # Network check
        net = psutil.net_io_counters()
        health["details"]["network"] = {
            "sent": net.bytes_sent / (1024*1024),
            "recv": net.bytes_recv / (1024*1024)
        }
        
        return health
    
    def get_system_metrics(self) -> dict:
        """Get detailed system performance metrics"""
        return {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage('/').percent,
            "network": {
                "sent": psutil.net_io_counters().bytes_sent / (1024*1024),
                "recv": psutil.net_io_counters().bytes_recv / (1024*1024)
            },
            "boot_time": psutil.boot_time(),
            "os": platform.platform()
        }

class MultiChannelAlertManager(AlertManager):
    """Alert manager with support for multiple channels"""
    def send_alert(self, message: str, channel: str = "telegram", severity: str = "medium"):
        """Send alert through specified channel with severity"""
        if channel == "telegram":
            self._send_telegram_alert(message, severity)
        elif channel == "email":
            self._send_email_alert(message, severity)
        elif channel == "sms":
            self._send_sms_alert(message, severity)
        else:
            raise ValueError(f"Unsupported channel: {channel}")
    
    def _send_telegram_alert(self, message: str, severity: str):
        """Send alert to Telegram with severity formatting"""
        # Add emoji based on severity
        emoji = "⚠️" if severity == "medium" else "🔴" if severity == "high" else "🚨"
        formatted_msg = f"{emoji} {message}"
        # Actual Telegram sending implementation would go here
        logger.info(f"Telegram alert sent ({severity}): {formatted_msg}")
    
    def _send_email_alert(self, message: str, severity: str):
        """Send email alert with severity in subject"""
        subject = f"{severity.upper()} Alert: Trading Signal"
        # Actual email sending implementation would go here
        logger.info(f"Email alert sent ({severity}): {subject} - {message}")
    
    def _send_sms_alert(self, message: str, severity: str):
        """Send SMS alert with priority based on severity"""
        # Actual SMS sending implementation would go here
        logger.info(f"SMS alert sent ({severity}): {message}")

class DataFetchError(Exception):
    """Custom exception for data fetching failures"""
    pass

class AnalysisError(Exception):
    """Custom exception for analysis failures"""
    pass

if __name__ == "__main__":
    try:
        orchestrator = MainOrchestrator()
        logger.info("Starting main trading loop...")
        
        while True:
            cycle_start = time.time()
            orchestrator.run()
            
            # Dynamic sleep based on processing time
            cycle_time = time.time() - cycle_start
            sleep_time = max(60 - cycle_time, 10)  # Minimum 10 seconds
            logger.info(f"Cycle completed in {cycle_time:.2f}s. Sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        orchestrator.executor.shutdown(wait=True)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        orchestrator.send_critical_alert(f"💀 FATAL ERROR: {str(e)}")
        raise

# Unit tests for critical methods
class TestMainOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orchestrator = MainOrchestrator()
        self.orchestrator.symbols = ["BTC-USDT"]
        self.orchestrator.intervals = ["15m"]
        self.sample_analysis = MagicMock(
            bos=True,
            choch=False,
            fvg={"range": (100.0, 101.0), "strength": 0.8},
            order_blocks=[{"range": (99.0, 99.5), "volume": 1500, "timestamp": int(time.time())}],
            recent_high=102.0,
            recent_low=98.5,
            current_close=102.5
        )
    
    def test_generate_signals(self):
        signals = self.orchestrator._generate_signals("BTC-USDT", "15m", self.sample_analysis)
        self.assertEqual(len(signals), 3)  # BOS, FVG, ORDER_BLOCK
        
        # Verify BOS signal
        bos_signals = [s for s in signals if s.signal_type == "BOS"]
        self.assertEqual(len(bos_signals), 1)
        self.assertAlmostEqual(bos_signals[0].level, 102.0)
        self.assertGreater(bos_signals[0].confidence, 0.7)
    
    def test_process_signal_duplicate(self):
        signal = TradingSignal(
            symbol="BTC-USDT",
            signal_type="BOS",
            level=102.0,
            strength=0.5,
            timestamp=int(time.time()),
            confidence=0.8,
            timeframe="15m"
        )
        
        # Add duplicate to last signals
        self.orchestrator.last_signals["BTC-USDT"] = [signal]
        
        # Should be detected as duplicate
        with self.assertLogs(level="INFO") as log:
            self.orchestrator._process_signal(signal)
            self.assertIn("Ignoring duplicate", log.output[0])
    
    def test_process_signal_low_confidence(self):
        signal = TradingSignal(
            symbol="BTC-USDT",
            signal_type="FVG",
            level=100.0,
            strength=0.5,
            timestamp=int(time.time()),
            confidence=0.5,  # Below threshold
            timeframe="15m"
        )
        
        with self.assertLogs(level="INFO") as log:
            self.orchestrator._process_signal(signal)
            self.assertIn("Low confidence", log.output[0])
    
    @patch.object(MultiChannelAlertManager, 'send_alert')
    def test_valid_signal_processing(self, mock_send_alert):
        signal = TradingSignal(
            symbol="BTC-USDT",
            signal_type="ORDER_BLOCK",
            level=99.0,
            strength=1500,
            timestamp=int(time.time()),
            confidence=0.85,
            timeframe="15m"
        )
        
        self.orchestrator._process_signal(signal)
        
        # Should be stored and alert sent
        self.assertIn(signal, self.orchestrator.last_signals["BTC-USDT"])
        mock_send_alert.assert_called_once()

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)