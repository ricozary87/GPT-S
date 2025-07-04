# tests/test_structure.py

from gpt_s_core.data_sources.okx_fetcher import get_candlesticks
from analyzers.structure_analyzer import detect_market_structure
import logging
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Configure logging once
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
logger = logging.getLogger(__name__)

def validate_candlestick_data(candles_data: list) -> bool:
    """
    Validate candlestick data structure
    
    Args:
        candles_data (list): List of candlestick data
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    if not candles_data:
        return False
        
    for candle in candles_data:
        if len(candle) < 6:
            logger.error(f"Invalid candle format: {candle}")
            return False
            
        # Check if numeric fields can be converted to floats
        try:
            float(candle[1])  # open
            float(candle[2])  # high
            float(candle[3])  # low
            float(candle[4])  # close
        except (ValueError, TypeError):
            logger.error(f"Non-numeric value in candle: {candle}")
            return False
            
    return True

def run_structure_analysis(candles_data: list):
    """
    Run market structure detection on candlestick data and log results
    
    Args:
        candles_data (list): List of candlestick data
    """
    if not validate_candlestick_data(candles_data):
        logger.error("❌ Invalid candlestick data format. Analysis aborted.")
        return
        
    structure_result = detect_market_structure(candles_data)

    if structure_result["bos"]:
        logger.info("✅ Struktur pasar: Break of Structure (BOS) terdeteksi!")
        logger.info(f"   - Penutupan Terbaru: {structure_result['current_close']:.2f}")
        logger.info(f"   - High Sebelumnya: {structure_result['recent_high']:.2f}")
    elif structure_result["choch"]:
        logger.info("✅ Struktur pasar: Change of Character (CHoCH) terdeteksi!")
        logger.info(f"   - Penutupan Terbaru: {structure_result['current_close']:.2f}")
        logger.info(f"   - Low Sebelumnya: {structure_result['recent_low']:.2f}")
    elif np.isnan(structure_result["current_close"]):
        logger.warning("⚠️ Tidak ada cukup data candlestick untuk deteksi struktur.")
    else:
        logger.info("ℹ️ Struktur pasar: Tidak ada BOS atau CHoCH yang terdeteksi.")
        logger.info(f"   Detail hasil: {structure_result}")

# ===========================================
# Unit Tests (Pytest compatible)
# ===========================================

@pytest.fixture
def mock_candles_bos():
    """Mock data for BOS scenario"""
    return [
        [1719800000000, "100.0", "105.0", "99.0", "104.5", "1000"],
        [1719700000000, "98.0", "103.0", "97.0", "102.0", "950"],
    ]

@pytest.fixture
def mock_candles_choch():
    """Mock data for CHoCH scenario"""
    return [
        [1719800000000, "100.0", "101.0", "94.0", "95.5", "1000"],
        [1719700000000, "98.0", "103.0", "97.0", "102.0", "950"],
    ]

@pytest.fixture
def mock_candles_no_change():
    """Mock data for no significant change scenario"""
    return [
        [1719800000000, "100.0", "102.0", "96.0", "100.0", "1000"],
        [1719700000000, "98.0", "103.0", "97.0", "102.0", "950"],
    ]

@pytest.fixture
def mock_candles_insufficient():
    """Mock data for insufficient data scenario"""
    return [
        [1719800000000, "100.0", "105.0", "99.0", "104.5", "1000"],
    ]

@pytest.fixture
def mock_candles_empty():
    """Mock data for empty data scenario"""
    return []

def test_detect_market_structure_bos(mock_candles_bos):
    """Test BOS detection with valid mock data"""
    result = detect_market_structure(mock_candles_bos)
    assert result["bos"] is True, "BOS should be detected"
    assert result["current_close"] > result["recent_high"], "Close should be above recent high"
    assert result["choch"] is False, "CHoCH should not be detected"

def test_detect_market_structure_choch(mock_candles_choch):
    """Test CHoCH detection with valid mock data"""
    result = detect_market_structure(mock_candles_choch)
    assert result["choch"] is True, "CHoCH should be detected"
    assert result["current_close"] < result["recent_low"], "Close should be below recent low"
    assert result["bos"] is False, "BOS should not be detected"

def test_detect_market_structure_no_change(mock_candles_no_change):
    """Test scenario with no significant change"""
    result = detect_market_structure(mock_candles_no_change)
    assert result["bos"] is False, "BOS should not be detected"
    assert result["choch"] is False, "CHoCH should not be detected"

def test_detect_market_structure_insufficient(mock_candles_insufficient):
    """Test with insufficient data"""
    result = detect_market_structure(mock_candles_insufficient)
    assert np.isnan(result["current_close"]), "Should return NaN for insufficient data"

def test_detect_market_structure_empty(mock_candles_empty):
    """Test with empty data"""
    result = detect_market_structure(mock_candles_empty)
    assert np.isnan(result["current_close"]), "Should return NaN for empty data"

def test_validate_candlestick_data():
    """Test candlestick data validation"""
    valid_data = [
        [1719800000000, "100.0", "105.0", "99.0", "104.5", "1000"],
        [1719700000000, "98.0", "103.0", "97.0", "102.0", "950"],
    ]
    invalid_data_short = [[1719800000000, "100.0", "105.0"]]
    invalid_data_non_numeric = [[1719800000000, "100.0", "high", "99.0", "104.5", "1000"]]
    
    assert validate_candlestick_data(valid_data) is True
    assert validate_candlestick_data(invalid_data_short) is False
    assert validate_candlestick_data(invalid_data_non_numeric) is False

@pytest.mark.parametrize("symbol, interval, limit", [
    ("BTC-USDT", "5m", 10),
    ("ETH-USDT", "15m", 20),
    ("SOL-USDT", "1H", 15),
])
@patch('data_sources.okx_fetcher.get_candlesticks')
def test_get_candlesticks_mocked(mock_get, symbol, interval, limit):
    """Test get_candlesticks with different parameters using mocking"""
    # Setup mock response
    mock_response = {
        "status": "success",
        "data": [
            [1719800000000, "100.0", "105.0", "99.0", "104.5", "1000"],
            [1719700000000, "98.0", "103.0", "97.0", "102.0", "950"],
        ]
    }
    mock_get.return_value = mock_response
    
    # Call the function
    result = get_candlesticks(instId=symbol, interval=interval, limit=limit)
    
    # Assertions
    assert result["status"] == "success"
    assert len(result["data"]) == 2
    mock_get.assert_called_once_with(instId=symbol, interval=interval, limit=limit)

# ===========================================
# Integration Tests (Live and Mock Scenarios)
# ===========================================

def test_live_structure():
    """Integration test with live data from OKX"""
    logger.info("\n--- Memulai pengujian deteksi struktur pasar (Live Data) ---")
    
    # Get real candlestick data
    fetch_result = get_candlesticks(instId="BTC-USDT", interval="5m", limit=10)
    
    if fetch_result["status"] == "success":
        candles_data = fetch_result["data"]
        logger.info(f"Berhasil mengambil {len(candles_data)} candlestick dari OKX.")
        run_structure_analysis(candles_data)
    else:
        logger.error(f"❌ Gagal mengambil candlestick: {fetch_result.get('message', 'Unknown error')}")

def test_mock_structure():
    """Integration test with mock data scenarios"""
    logger.info("\n--- Memulai pengujian deteksi struktur pasar (Mock Data) ---")
    
    scenarios = [
        ("BOS Bullish", [
            [1719800000000, "100.0", "105.0", "99.0", "104.5", "1000"],
            [1719700000000, "98.0", "103.0", "97.0", "102.0", "950"],
        ]),
        ("CHoCH Bearish", [
            [1719800000000, "100.0", "101.0", "94.0", "95.5", "1000"],
            [1719700000000, "98.0", "103.0", "97.0", "102.0", "950"],
        ]),
        ("Tidak Ada Perubahan", [
            [1719800000000, "100.0", "102.0", "96.0", "100.0", "1000"],
            [1719700000000, "98.0", "103.0", "97.0", "102.0", "950"],
        ]),
        ("Data Tidak Cukup", [
            [1719800000000, "100.0", "105.0", "99.0", "104.5", "1000"],
        ]),
        ("Data Kosong", []),
    ]
    
    for name, data in scenarios:
        logger.info(f"\n--- Skenario: {name} ---")
        run_structure_analysis(data)

# ===========================================
# Main Execution
# ===========================================

if __name__ == "__main__":
    # Run integration tests when executed directly
    logger.info("=" * 60)
    logger.info("MEMULAI PENGUJIAN STRUKTUR PASAR")
    logger.info("=" * 60)
    
    try:
        logger.info("\n[1/2] Menjalankan pengujian dengan data live...")
        test_live_structure()
        
        logger.info("\n[2/2] Menjalankan pengujian dengan data mock...")
        test_mock_structure()
        
        logger.info("\n✅ Semua pengujian selesai!")
    except Exception as e:
        logger.error(f"❌ Pengujian gagal: {str(e)}")
        sys.exit(1)