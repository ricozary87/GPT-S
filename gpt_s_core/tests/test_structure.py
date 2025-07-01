# tests/test_structure.py

from data_sources.okx_fetcher import get_candlesticks
from analyzers.structure_analyzer import detect_market_structure
import logging
import sys
import numpy as np # Import numpy untuk np.nan

# Konfigurasi logging
# Hanya konfigurasikan logging sekali jika belum dikonfigurasi
# Ini mencegah masalah duplikasi handler jika modul diimpor berkali-kali
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(sys.stdout)
                        ])
logger = logging.getLogger(__name__)

def run_structure_analysis(candles_data: list):
    """
    Menjalankan deteksi struktur pasar pada data candlestick yang diberikan
    dan mencetak hasilnya ke log.

    Args:
        candles_data (list): List data candlestick.
    """
    structure_result = detect_market_structure(candles_data)

    if structure_result["bos"]:
        logger.info("✅ Struktur pasar: Break of Structure (BOS) terdeteksi!")
        logger.info(f"   - Penutupan Terbaru: {structure_result['current_close']:.2f}")
        logger.info(f"   - High Sebelumnya: {structure_result['recent_high']:.2f}")
    elif structure_result["choch"]:
        logger.info("✅ Struktur pasar: Change of Character (CHoCH) terdeteksi!")
        logger.info(f"   - Penutupan Terbaru: {structure_result['current_close']:.2f}")
        logger.info(f"   - Low Sebelumnya: {structure_result['recent_low']:.2f}")
    elif np.isnan(structure_result["current_close"]): # Memeriksa np.nan bukan None
        logger.warning("⚠️ Tidak ada cukup data candlestick untuk melakukan deteksi struktur pasar.")
        logger.info(f"   Detail hasil: {structure_result}")
    else:
        logger.info("ℹ️ Struktur pasar: Tidak ada BOS atau CHoCH yang terdeteksi.")
        logger.info(f"   Detail hasil: {structure_result}")

def test_live_structure():
    """
    Mengambil candlestick langsung dari OKX dan menganalisis struktur pasar.
    """
    logger.info("\n--- Memulai pengujian deteksi struktur pasar (Live Data) ---")

    # Mendapatkan data candlestick
    # PERBAIKAN: Menggunakan instId alih-alih symbol
    fetch_result = get_candlesticks(instId="BTC-USDT", interval="5m", limit=10) # Ambil 10 data untuk analisis

    if fetch_result["status"] == "success":
        candles_data = fetch_result["data"]
        logger.info(f"Berhasil mengambil {len(candles_data)} candlestick dari OKX.")
        run_structure_analysis(candles_data)
    else:
        logger.error(f"❌ Gagal mengambil candlestick dari OKX: {fetch_result['message']}")
        logger.error("   Deteksi struktur pasar tidak dapat dilakukan untuk data live.")

def test_mock_structure():
    """
    Menguji deteksi struktur pasar dengan data candlestick simulasi.
    """
    logger.info("\n--- Memulai pengujian deteksi struktur pasar (Mock Data) ---")

    # Contoh data simulasi: [timestamp, open, high, low, close, volume]
    # (Timestamp hanya placeholder, bisa string atau angka)

    # Skenario 1: BOS Bullish
    mock_candles_bos = [
        [1719800000000, "100.0", "105.0", "99.0", "104.5", "1000"], # Terbaru (Close > High sebelumnya)
        [1719700000000, "98.0", "103.0", "97.0", "102.0", "950"],  # Sebelumnya
    ]
    logger.info("--- Skenario: BOS Bullish ---")
    run_structure_analysis(mock_candles_bos)

    # Skenario 2: CHoCH Bearish
    mock_candles_choch = [
        [1719800000000, "100.0", "101.0", "94.0", "95.5", "1000"], # Terbaru (Close < Low sebelumnya)
        [1719700000000, "98.0", "103.0", "97.0", "102.0", "950"],  # Sebelumnya
    ]
    logger.info("\n--- Skenario: CHoCH Bearish ---")
    run_structure_analysis(mock_candles_choch)

    # Skenario 3: Tidak Ada Perubahan Signifikan
    mock_candles_no_change = [
        [1719800000000, "100.0", "102.0", "96.0", "100.0", "1000"], # Terbaru (Close antara Low & High sebelumnya)
        [1719700000000, "98.0", "103.0", "97.0", "102.0", "950"],  # Sebelumnya
    ]
    logger.info("\n--- Skenario: Tidak Ada Perubahan Signifikan ---")
    run_structure_analysis(mock_candles_no_change)

    # Skenario 4: Data Tidak Cukup (1 Candlestick)
    mock_candles_insufficient = [
        [1719800000000, "100.0", "105.0", "99.0", "104.5", "1000"],
    ]
    logger.info("\n--- Skenario: Data Tidak Cukup (1 Candlestick) ---")
    run_structure_analysis(mock_candles_insufficient)

    # Skenario 5: Data Kosong
    mock_candles_empty = []
    logger.info("\n--- Skenario: Data Kosong ---")
    run_structure_analysis(mock_candles_empty)


if __name__ == "__main__":
    # Jalankan pengujian dengan data live dari OKX
    test_live_structure()

    # Jalankan pengujian dengan data simulasi/mock
    test_mock_structure()