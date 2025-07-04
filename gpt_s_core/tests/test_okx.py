import pytest
from gpt_s_core.data_sources.okx_fetcher import (
    get_candlesticks,
    get_orderbook,
    get_latest_price
)

# ✅ Validator minimal
def validate_candle_structure(candle: dict) -> bool:
    keys = ['open', 'high', 'low', 'close', 'volume']
    return all(k in candle for k in keys)

# ✅ Test get_candlesticks (parametrized)
@pytest.mark.parametrize("symbol, tf", [
    ("BTC-USDT", "1H"),
    ("ETH-USDT", "1H"),
    ("SOL-USDT", "1H"),
])
def test_get_candlesticks(symbol, tf):
    df = get_candlesticks(symbol, timeframe=tf, limit=100, add_ta=True, use_cache=False)
    assert not df.empty
    assert validate_candle_structure(df.iloc[-1].to_dict())
    print(f"✅ {symbol} candle OK: {df.iloc[-1]['close']}")

# ✅ Test get_orderbook
@pytest.mark.parametrize("symbol", ["BTC-USDT", "ETH-USDT", "SOL-USDT"])
def test_get_orderbook(symbol):
    ob = get_orderbook(symbol)
    assert 'data' in ob
    assert isinstance(ob['data'], list) and len(ob['data']) > 0
    print(f"✅ {symbol} orderbook: {len(ob['data'][0]['bids'])} bids")

# ✅ Test get_latest_price
def test_get_latest_price():
    price = get_latest_price("BTC-USDT")
    assert isinstance(price, float)
    assert price > 0
    print(f"✅ Latest BTC price: {price}")
