from websocket import WebSocketApp
import json
import threading

class OKXWebSocket:
    def __init__(self, symbols=['BTC-USDT']):
        self.url = "wss://ws.okx.com:8443/ws/v5/public"
        self.symbols = symbols
        self.orderbook = {}
        self.ws = None

    def on_message(self, ws, message):
        data = json.loads(message)
        if 'arg' in data and data['arg']['channel'] == 'books':
            symbol = data['arg']['instId']
            self.orderbook[symbol] = {
                'bids': data['data'][0]['bids'][:5],  # Top 5 bids
                'asks': data['data'][0]['asks'][:5]   # Top 5 asks
            }

    def start(self):
        def run():
            self.ws = WebSocketApp(
                self.url,
                on_message=self.on_message,
                on_error=lambda ws, err: print(f"WS Error: {err}"),
                on_close=lambda ws: print("WS Closed")
            )
            self.ws.on_open = lambda ws: self._subscribe()
            self.ws.run_forever()

        threading.Thread(target=run, daemon=True).start()

    def _subscribe(self):
        for symbol in self.symbols:
            self.ws.send(json.dumps({
                "op": "subscribe",
                "args": [{
                    "channel": "books",
                    "instId": symbol
                }]
            }))

    def get_liquidity_levels(self, symbol):
        """Return key liquidity zones untuk SMC"""
        if symbol not in self.orderbook:
            return None
        return {
            'support': self.orderbook[symbol]['bids'][0][0],  # Best bid
            'resistance': self.orderbook[symbol]['asks'][0][0]  # Best ask
        }
