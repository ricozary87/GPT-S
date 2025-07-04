import pytest
from gpt_s_core.data_sources.okx_fetcher import get_candlesticks, get_orderbook

# ✅ VALIDATORS
def validate_candle(candle):
    keys = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    return all(k in candle for k in keys)

def validate_orderbook(ob):
    return 'asks' in ob and 'bids' in ob and isinstance(ob['asks'], list) and isinstance(ob['bids'], list)

# ✅ PARAMETERIZED TEST CASES
symbols = [
    ("BTC-USDT", "1H"),
    ("ETH-USDT", "15m"),
    ("SOL-USDT", "4H"),
]

# ✅ TEST CANDLES
@pytest.mark.parametrize("symbol, interval", symbols)
def test_get_candlesticks(symbol, interval):
    df = get_candlesticks(symbol, interval, limit=5)
    assert not df.empty, f"❌ No candle data returned for {symbol} {interval}"
    assert validate_candle(df.iloc[0].to_dict()), f"❌ Invalid candle format for {symbol}"
    print(f"✅ {symbol} {interval} OK - Last Close: {df.iloc[-1]['close']}")

# ✅ TEST ORDERBOOK
@pytest.mark.parametrize("symbol", ["BTC-USDT", "ETH-USDT", "SOL-USDT"])
def test_get_orderbook(symbol):
    ob = get_orderbook(symbol)
    assert 'data' in ob and isinstance(ob['data'], list) and len(ob['data']) > 0, f"❌ No orderbook data for {symbol}"
    assert validate_orderbook(ob['data'][0]), f"❌ Invalid orderbook for {symbol}"
    print(f"✅ {symbol} Orderbook OK - Bids: {len(ob['data'][0]['bids'])} | Asks: {len(ob['data'][0]['asks'])}")
