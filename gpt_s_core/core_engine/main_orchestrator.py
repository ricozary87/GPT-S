import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Local imports
from gpt_s_core.data_sources.okx_fetcher import get_ohlcv_data
from gpt_s_core.analyzers.structure_analyzer import SMAnalyzer
from gpt_s_core.utils.monitor import SystemMonitor
from gpt_s_core.config.api_manager import APIVault
from gpt_s_core.utils.alert_manager import AlertManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("orchestrator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # 'BOS', 'CHoCH', 'FVG', 'ORDER_BLOCK'
    level: float
    strength: float
    timestamp: int
    confidence: float = 0.0

class MainOrchestrator:
    def __init__(self):
        """Initialize trading orchestrator with all components."""
        self.api_vault = APIVault()
        self.analyzer = SMAnalyzer()
        self.monitor = SystemMonitor()
        self.alert_manager = AlertManager()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
        self.intervals = ["1m", "15m", "1H"]
        self.last_signals: Dict[str, List[TradingSignal]] = {}

        logger.info("MainOrchestrator initialized with:")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Timeframes: {self.intervals}")

    def run(self):
        """Main execution loop with error handling and monitoring."""
        try:
            # System health check
            if not self.monitor.check_system_health():
                logger.error("System health check failed")
                self.alert_manager.send_alert("🚨 System health check failed!")
                return

            # Process each symbol and timeframe
            futures = []
            for symbol in self.symbols:
                for interval in self.intervals:
                    futures.append(
                        self.executor.submit(
                            self.process_symbol,
                            symbol,
                            interval
                        )
                    )

            # Wait for all tasks to complete
            for future in futures:
                future.result()

        except Exception as e:
            logger.error(f"Critical error in main loop: {e}")
            self.alert_manager.send_alert(f"🔥 Orchestrator crashed: {str(e)}")
        finally:
            logger.info("Cycle completed. Waiting for next iteration...")

    def process_symbol(self, symbol: str, interval: str):
        """Process trading signals for a single symbol and timeframe."""
        try:
            logger.info(f"Processing {symbol} on {interval} timeframe")
            
            # Fetch OHLCV data
            df = get_ohlcv_data(
                symbol=symbol,
                interval=interval,
                limit=100  # Get last 100 candles
            )
            
            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol} {interval}")
                return

            # Analyze market structure
            analysis = self.analyzer.detect_structure(df)
            
            if not analysis:
                logger.warning(f"Analysis failed for {symbol} {interval}")
                return

            # Generate trading signals
            signals = self._generate_signals(symbol, interval, analysis)
            
            # Process valid signals
            for signal in signals:
                self._process_signal(signal)

        except Exception as e:
            logger.error(f"Error processing {symbol} {interval}: {e}")

    def _generate_signals(self, symbol: str, interval: str, analysis) -> List[TradingSignal]:
        """Generate trading signals from analysis results."""
        signals = []
        
        # 1. BOS/CHoCH signals
        if analysis.bos:
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type="BOS",
                level=analysis.recent_high,
                strength=(analysis.current_close - analysis.recent_high) / analysis.recent_high * 100,
                timestamp=int(time.time()),
                confidence=0.8  # Example confidence score
            ))
        
        if analysis.choch:
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type="CHoCH",
                level=analysis.recent_low,
                strength=(analysis.recent_low - analysis.current_close) / analysis.recent_low * 100,
                timestamp=int(time.time()),
                confidence=0.75
            ))

        # 2. FVG signals
        if analysis.fvg:
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type="FVG",
                level=analysis.fvg['range'][0],
                strength=analysis.fvg['strength'],
                timestamp=int(time.time()),
                confidence=0.7
            ))

        # 3. Order Block signals
        if analysis.order_block:
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type="ORDER_BLOCK",
                level=analysis.order_block['range'][0],
                strength=analysis.order_block['volume'],
                timestamp=analysis.order_block['timestamp'],
                confidence=0.85 if analysis.order_block['volume'] > 1000 else 0.6
            ))

        return signals

    def _process_signal(self, signal: TradingSignal):
        """Process and validate trading signal."""
        # Check if similar signal was recently processed
        recent_signals = self.last_signals.get(signal.symbol, [])
        similar_exists = any(
            s.signal_type == signal.signal_type and 
            abs(s.level - signal.level) < 0.01 * s.level
            for s in recent_signals[-3:]  # Check last 3 signals
        )
        
        if similar_exists:
            logger.info(f"Ignoring duplicate {signal.signal_type} signal for {signal.symbol}")
            return

        # Validate signal strength
        if signal.confidence < 0.6:
            logger.info(f"Low confidence signal ignored: {signal}")
            return

        # Store signal
        if signal.symbol not in self.last_signals:
            self.last_signals[signal.symbol] = []
        self.last_signals[signal.symbol].append(signal)
        
        # Send alert
        message = (
            f"🚀 {signal.symbol} {signal.signal_type}\n"
            f"Level: {signal.level:.2f}\n"
            f"Strength: {signal.strength:.2f}%\n"
            f"Confidence: {signal.confidence*100:.0f}%"
        )
        self.alert_manager.send_alert(message)
        
        logger.info(f"Processed signal: {signal}")

if __name__ == "__main__":
    try:
        orchestrator = MainOrchestrator()
        logger.info("Starting main trading loop...")
        
        while True:
            start_time = time.time()
            orchestrator.run()
            
            # Dynamic sleep based on processing time
            cycle_time = time.time() - start_time
            sleep_time = max(60 - cycle_time, 5)  # At least 5 seconds
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        raise
