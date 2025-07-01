import pandas as pd

def detect_market_structure(candles: list) -> dict:
    """
    Deteksi struktur pasar (BOS, CHoCH) dari data candlestick OKX.

    Fungsi ini mendeteksi BOS (Break of Structure) atau CHoCH (Change of Character)
    berdasarkan perbandingan harga penutupan candlestick terbaru dengan
    harga tertinggi (high) dan terendah (low) dari candlestick sebelumnya.

    Args:
        candles (list): List dari data candlestick. Setiap elemen dalam list
                        diharapkan berformat:
                        [timestamp, open, high, low, close, volume, ...]
                        Minimal harus ada 2 candlestick dalam list untuk deteksi.

    Returns:
        dict: Kamus yang berisi informasi struktur pasar terbaru:
              - "recent_high": Harga tertinggi dari candlestick sebelumnya.
              - "recent_low": Harga terendah dari candlestick sebelumnya.
              - "current_close": Harga penutupan dari candlestick terbaru.
              - "bos": True jika terjadi Break of Structure, False jika tidak.
              - "choch": True jika terjadi Change of Character, False jika tidak.
              Mengembalikan nilai None untuk harga jika data tidak mencukupi.
    """
    # Pastikan ada setidaknya 2 candlestick untuk perbandingan
    if len(candles) < 2:
        print("Peringatan: Tidak ada cukup data candlestick (minimal 2 dibutuhkan) untuk mendeteksi struktur pasar.")
        return {
            "recent_high": None,
            "recent_low": None,
            "current_close": None,
            "bos": False,
            "choch": False
        }

    # Buat DataFrame dari data candlestick
    # Mengasumsikan kolom standar untuk candlestick (ts, open, high, low, close, vol)
    # Tambahan kolom lain akan diabaikan jika tidak disebutkan.
    # Jika ada kolom spesifik setelah 'vol' yang penting, Anda bisa menambahkannya di sini.
    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "vol"] + [f"col{i}" for i in range(len(candles[0]) - 6)])

    # Konversi kolom harga ke float untuk perhitungan yang akurat
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)

    # Urutkan DataFrame dari candlestick paling lama ke paling baru
    # Ini penting agar iloc[-2] dan iloc[-1] merujuk ke data yang benar
    df = df.iloc[::-1].reset_index(drop=True)

    # Ambil nilai yang relevan dari DataFrame
    recent_high = df["high"].iloc[-2]      # High dari candlestick sebelumnya
    recent_low = df["low"].iloc[-2]        # Low dari candlestick sebelumnya
    current_close = df["close"].iloc[-1]   # Close dari candlestick terbaru

    # Inisialisasi dictionary struktur pasar
    structure = {
        "recent_high": recent_high,
        "recent_low": recent_low,
        "current_close": current_close,
        "bos": False,
        "choch": False
    }

    # Deteksi Break of Structure (BOS)
    # Jika harga penutupan terbaru lebih tinggi dari high candlestick sebelumnya
    if current_close > recent_high:
        structure["bos"] = True
    # Deteksi Change of Character (CHoCH)
    # Jika harga penutupan terbaru lebih rendah dari low candlestick sebelumnya
    # Menggunakan 'elif' karena BOS dan CHoCH adalah kejadian eksklusif dalam logika ini
    elif current_close < recent_low:
        structure["choch"] = True

    return structure

# --- Contoh Penggunaan ---

# Contoh data candlestick (format: [timestamp, open, high, low, close, volume, ...])
# Data diurutkan dari yang paling baru ke paling lama seperti yang mungkin diterima dari API
sample_candles_bullish_bos = [
    [1719750400000, "100.0", "105.0", "99.0", "104.5", "1000", "extra1", "extra2"], # Candlestick Terbaru
    [1719750300000, "98.0", "103.0", "97.0", "102.0", "950", "extra1", "extra2"],  # Candlestick Sebelumnya (recent_high 103.0, recent_low 97.0)
    [1719750200000, "95.0", "99.0", "94.0", "98.0", "900", "extra1", "extra2"],
]

sample_candles_bearish_choch = [
    [1719750400000, "100.0", "101.0", "94.0", "95.5", "1000", "extra1", "extra2"], # Candlestick Terbaru
    [1719750300000, "98.0", "103.0", "97.0", "102.0", "950", "extra1", "extra2"],  # Candlestick Sebelumnya (recent_high 103.0, recent_low 97.0)
    [1719750200000, "95.0", "99.0", "94.0", "98.0", "900", "extra1", "extra2"],
]

sample_candles_no_change = [
    [1719750400000, "100.0", "102.0", "96.0", "100.0", "1000", "extra1", "extra2"], # Candlestick Terbaru
    [1719750300000, "98.0", "103.0", "97.0", "102.0", "950", "extra1", "extra2"],  # Candlestick Sebelumnya (recent_high 103.0, recent_low 97.0)
    [1719750200000, "95.0", "99.0", "94.0", "98.0", "900", "extra1", "extra2"],
]

sample_candles_insufficient = [
    [1719750400000, "100.0", "105.0", "99.0", "104.5", "1000", "extra1", "extra2"],
]

sample_candles_empty = []


print("--- Skenario BOS Bullish ---")
result_bos = detect_market_structure(sample_candles_bullish_bos)
print(result_bos)
# Expected: {'recent_high': 103.0, 'recent_low': 97.0, 'current_close': 104.5, 'bos': True, 'choch': False}

print("\n--- Skenario CHoCH Bearish ---")
result_choch = detect_market_structure(sample_candles_bearish_choch)
print(result_choch)
# Expected: {'recent_high': 103.0, 'recent_low': 97.0, 'current_close': 95.5, 'bos': False, 'choch': True}

print("\n--- Skenario Tidak Ada Perubahan Signifikan ---")
result_no_change = detect_market_structure(sample_candles_no_change)
print(result_no_change)
# Expected: {'recent_high': 103.0, 'recent_low': 97.0, 'current_close': 100.0, 'bos': False, 'choch': False}

print("\n--- Skenario Data Tidak Cukup (1 Candlestick) ---")
result_insufficient = detect_market_structure(sample_candles_insufficient)
print(result_insufficient)
# Expected: Peringatan... {'recent_high': None, 'recent_low': None, 'current_close': None, 'bos': False, 'choch': False}

print("\n--- Skenario Data Kosong ---")
result_empty = detect_market_structure(sample_candles_empty)
print(result_empty)
# Expected: Peringatan... {'recent_high': None, 'recent_low': None, 'current_close': None, 'bos': False, 'choch': False}