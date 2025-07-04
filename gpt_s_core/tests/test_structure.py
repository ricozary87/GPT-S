import logging
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import gpt_s_core.data_sources.okx_fetcher as okx
from gpt_s_core.analyzers.structure_analyzer import SMAnalyzer, MarketStructure

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
logger = logging.getLogger(__name__)

def validate_candle_data_for_test(candles: list) -> bool:
    if not candles:
        return False
    for i, candle in enumerate(candles):
        if len(candle) < 6:
            logger.error(f"Test candle at index {i} has invalid format (less than 6 elements): {candle}")
            return False
        try:
            float(candle[1])
            float(candle[2])
            float(candle[3])
            float(candle[4])
        except (ValueError, TypeError):
            logger.error(f"Test candle at index {i} has non-numeric price values: {candle}")
            return False
    return True

@pytest.fixture
def sm_analyzer():
    return SMAnalyzer(min_candles=3, fvg_min_strength=0.1, liquidity_lookback=5, swing_window=1)

@pytest.mark.parametrize("scenario, candles, expected_bos, expected_choch, check_nan_current_close", [
    ("bos_scenario",
     [[1625097600000, 100.0, 105.0, 95.0, 102.0, 1000.0],
      [1625098200000, 102.0, 107.0, 98.0, 104.0, 1200.0],
      [1625098800000, 104.0, 110.0, 100.0, 108.0, 1300.0]
     ], True, False, False),
    ("choch_scenario",
     [[1625097600000, 100.0, 105.0, 95.0, 102.0, 1000.0],
      [1625098200000, 102.0, 107.0, 98.0, 104.0, 1200.0],
      [1625098800000, 104.0, 106.0, 90.0, 95.0, 1300.0]
     ], False, True, False),
    ("no_change_scenario",
     [[1625097600000, 100.0, 105.0, 95.0, 102.0, 1000.0],
      [1625098200000, 102.0, 107.0, 98.0, 104.0, 1200.0],
      [1625098800000, 104.0, 106.0, 99.0, 101.0, 1300.0]
     ], False, False, False),
    ("insufficient_candles_one",
     [[1625098800000, 104.0, 106.0, 99.0, 101.0, 1300.0]
     ], False, False, True),
    ("insufficient_candles_two",
     [[1625098200000, 102.0, 107.0, 98.0, 104.0, 1200.0],
      [1625098800000, 104.0, 106.0, 99.0, 101.0, 1300.0]
     ], False, False, True),
    ("empty_candles", [], False, False, True),
    ("invalid_data_non_numeric",
     [[1625097600000, 100.0, "ABC", 95.0, 102.0, 1000.0],
      [1625098200000, 102.0, 107.0, 98.0, 104.0, 1200.0],
      [1625098800000, 104.0, 110.0, 100.0, 108.0, 1300.0]
     ], False, False, True),
])
def test_structure_scenarios(sm_analyzer, scenario, candles, expected_bos, expected_choch, check_nan_current_close):
    logger.info(f"\n--- Testing Scenario: {scenario} ---")
    result = sm_analyzer.detect_structure(candles)

    if check_nan_current_close:
        assert result is None or (isinstance(result, MarketStructure) and np.isnan(result.current_close))
        logger.info(f"✅ Scenario '{scenario}' passed (expected None/NaN). Result: {result}")
    else:
        assert isinstance(result, MarketStructure)
        assert result.bos == expected_bos
        assert result.choch == expected_choch
        assert not np.isnan(result.current_close)
        logger.info(f"✅ Scenario '{scenario}' passed. Result: BOS={result.bos}, CHoCH={result.choch}")

@patch('gpt_s_core.data_sources.okx_fetcher.get_candlesticks')
def test_structure_with_mock_fetcher(mock_get_candlesticks, sm_analyzer):
    mock_data = [
        [1719700000000, "98.0", "103.0", "97.0", "102.0", "950.0"],
        [1719800000000, "100.0", "105.0", "99.0", "104.5", "1000.0"],
        [1719900000000, "104.0", "106.0", "103.0", "107.0", "1200.0"]
    ]
    mock_df_return = pd.DataFrame(
        mock_data,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    ).set_index('timestamp')

    mock_get_candlesticks.return_value = mock_df_return

    df_fetched = okx.get_candlesticks(symbol="BTC-USDT", timeframe="5m", limit=10)
    mock_get_candlesticks.assert_called_once_with(symbol="BTC-USDT", timeframe="5m", limit=10)
    assert not df_fetched.empty

    candles = df_fetched.reset_index()[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()

    result = sm_analyzer.detect_structure(candles)

    assert isinstance(result, MarketStructure)
    assert result.bos
    assert not result.choch
    assert float(result.current_close) == 107.0
    assert result.recent_high == 105.0
    assert result.recent_low == 99.0
    logger.info("✅ Struktur dari mock fetcher berhasil diproses")

@pytest.mark.parametrize("symbol, tf", [
    ("BTC-USDT", "5m"),
    ("ETH-USDT", "15m")
])
def test_structure_live(symbol, tf):
    logger.info(f"\n--- [LIVE TEST] Fetching {symbol} ({tf}) ---")

    live_analyzer = SMAnalyzer(min_candles=10)
    df = okx.get_candlesticks(symbol=symbol, timeframe=tf, limit=max(10, live_analyzer.config["min_candles"]))

    if df.empty:
        pytest.fail(f"❌ Gagal mengambil candle live dari OKX untuk {symbol} ({tf}).")

    candles = df.reset_index()[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()

    logger.info(f"Mengambil {len(candles)} candle live untuk {symbol} ({tf}).")
    result = live_analyzer.detect_structure(candles)

    assert isinstance(result, MarketStructure)
    assert "bos" in result.__dict__
    assert "choch" in result.__dict__
    assert not np.isnan(result.current_close)
    logger.info(f"✅ Struktur terdeteksi untuk {symbol} ({tf}): {result}")
