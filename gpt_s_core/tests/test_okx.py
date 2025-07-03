from data_sources.okx_fetcher import get_candlesticks, get_orderbook
import sys
import json

def validate_candle(candle):
    """Validate candle structure"""
    required_keys = ['instId', 'open', 'high', 'low', 'close', 'vol', 'ts']
    for key in required_keys:
        if key not in candle:
            return False
    return True

def validate_orderbook(book_entry):
    """Validate orderbook entry structure"""
    required_keys = ['instId', 'asks', 'bids', 'ts']
    if not all(key in book_entry for key in required_keys):
        return False
        
    # Validate price levels
    for side in ['asks', 'bids']:
        if not isinstance(book_entry[side], list):
            return False
        for level in book_entry[side]:
            if len(level) < 2 or not isinstance(level[0], str) or not isinstance(level[1], str):
                return False
    return True

def test_get_candlesticks(symbol='BTC-USDT', interval='1H'):
    """Test candlestick data fetching with parameterized inputs"""
    try:
        candles = get_candlesticks(symbol, interval)
        assert candles, "❌ No data returned for candlesticks"
        assert 'data' in candles, "❌ Missing 'data' key in candlesticks"
        assert candles['data'], "❌ Empty candles data array"
        
        first_candle = candles['data'][0]
        assert validate_candle(first_candle), f"❌ Invalid candle structure: {json.dumps(first_candle, indent=2)}"
        
        print(f"✅ Candlesticks validated ({symbol} {interval}):")
        print(f"  Timestamp: {first_candle.get('ts')}")
        print(f"  O/H/L/C: {first_candle.get('open')}/{first_candle.get('high')}/"
              f"{first_candle.get('low')}/{first_candle.get('close')}")
        return True
    except Exception as e:
        print(f"❌ Candlestick test failed: {str(e)}")
        return False

def test_get_orderbook(symbol='BTC-USDT', depth='5'):
    """Test orderbook data fetching with parameterized inputs"""
    try:
        book = get_orderbook(symbol, depth)
        assert book, "❌ No data returned for orderbook"
        assert 'data' in book, "❌ Missing 'data' key in orderbook"
        assert book['data'], "❌ Empty orderbook data array"
        
        first_book = book['data'][0]
        assert validate_orderbook(first_book), f"❌ Invalid orderbook structure: {json.dumps(first_book, indent=2)}"
        
        print(f"✅ Orderbook validated ({symbol} depth={depth}):")
        print(f"  Best bid: {first_book['bids'][0][0]} @ {first_book['bids'][0][1]}")
        print(f"  Best ask: {first_book['asks'][0][0]} @ {first_book['asks'][0][1]}")
        return True
    except Exception as e:
        print(f"❌ Orderbook test failed: {str(e)}")
        return False

def run_tests():
    """Run parameterized test cases"""
    test_cases = [
        {'symbol': 'BTC-USDT', 'interval': '1H'},
        {'symbol': 'ETH-USDT', 'interval': '15m'},
        {'symbol': 'SOL-USDT', 'interval': '4H'},
    ]
    
    print("="*60)
    print("Starting OKX API Tests")
    print("="*60)
    
    # Run candlestick tests
    print("\n🔍 Testing Candlestick Endpoints:")
    candle_results = []
    for case in test_cases:
        print(f"\nTesting {case['symbol']} ({case['interval']})...")
        candle_results.append(test_get_candlesticks(case['symbol'], case['interval']))
    
    # Run orderbook tests
    print("\n🔍 Testing Orderbook Endpoints:")
    book_results = []
    for case in test_cases:
        print(f"\nTesting {case['symbol']}...")
        book_results.append(test_get_orderbook(case['symbol']))
    
    # Summary report
    print("\n" + "="*60)
    print("Test Summary:")
    print(f"Candlestick Tests: {sum(candle_results)}/{len(candle_results)} passed")
    print(f"Orderbook Tests: {sum(book_results)}/{len(book_results)} passed")
    print("="*60)
    
    return all(candle_results + book_results)

if __name__ == "__main__":
    # Run tests and return proper exit code
    success = run_tests()
    sys.exit(0 if success else 1)