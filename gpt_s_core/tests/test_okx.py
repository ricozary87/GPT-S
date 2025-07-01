from data_sources.okx_fetcher import get_candlesticks, get_orderbook

def test_okx():
    candles = get_candlesticks()
    print("✅ Candle sample:", candles["data"][0] if candles else "❌ No data")

    book = get_orderbook()
    print("✅ Orderbook sample:", book["data"][0] if book else "❌ No data")

if __name__ == "__main__":
    test_okx()
