from websocket import WebSocketApp
import json
import threading
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OKXWebSocket:
    """WebSocket client for OKX public orderbook data with SMC liquidity zones"""
    
    def __init__(self, symbols=['BTC-USDT']):
        """
        Initialize WebSocket client
        :param symbols: List of instrument IDs to subscribe to
        """
        self.url = "wss://ws.okx.com:8443/ws/v5/public"
        self.symbols = symbols
        self.orderbook = {}
        self.ws = None
        self.lock = threading.Lock()
        self._running = False
        self.thread = None

    def on_message(self, ws, message):
        """
        Handle incoming WebSocket messages
        :param ws: WebSocket connection
        :param message: Received message (JSON string)
        """
        try:
            data = json.loads(message)
            if 'event' in data and data['event'] == 'subscribe':
                logging.info(f"Subscribed successfully: {data}")
                return
                
            if 'arg' in data and data['arg']['channel'] == 'books':
                symbol = data['arg']['instId']
                with self.lock:
                    self.orderbook[symbol] = {
                        'bids': data['data'][0]['bids'][:5],  # Top 5 bids
                        'asks': data['data'][0]['asks'][:5]   # Top 5 asks
                    }
        except Exception as e:
            logging.error(f"Message processing error: {e}")

    def on_error(self, ws, error):
        """
        Handle WebSocket errors
        :param ws: WebSocket connection
        :param error: Error object
        """
        logging.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """
        Handle WebSocket closure
        :param ws: WebSocket connection
        :param close_status_code: Connection close code
        :param close_msg: Close message
        """
        logging.warning(f"WebSocket closed (code: {close_status_code}, message: {close_msg})")
        self._reconnect()

    def _subscribe(self, ws):
        """
        Subscribe to orderbook channels for all symbols
        :param ws: WebSocket connection
        """
        for symbol in self.symbols:
            subscription_msg = json.dumps({
                "op": "subscribe",
                "args": [{
                    "channel": "books",
                    "instId": symbol
                }]
            })
            ws.send(subscription_msg)
            logging.info(f"Subscribing to {symbol}")

    def _reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        retry_delay = 2
        while self._running:
            logging.info(f"Reconnecting in {retry_delay} seconds...")
            time.sleep(retry_delay)
            try:
                self._connect()
                return
            except Exception as e:
                logging.error(f"Reconnect failed: {e}")
                retry_delay = min(retry_delay * 2, 60)  # Cap at 60 seconds

    def _connect(self):
        """Internal connection handler"""
        self.ws = WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.on_open = self._subscribe
        self.ws.run_forever()

    def start(self):
        """
        Start WebSocket connection in a background thread
        with automatic reconnection
        """
        if self._running:
            logging.warning("WebSocket is already running")
            return
            
        self._running = True
        self.thread = threading.Thread(target=self._connect, daemon=True)
        self.thread.start()
        logging.info("WebSocket started")

    def stop(self):
        """Gracefully shutdown WebSocket connection"""
        self._running = False
        if self.ws:
            self.ws.close()
        logging.info("WebSocket stopped")

    def update_symbols(self, new_symbols):
        """
        Update subscription symbols and resubscribe
        :param new_symbols: List of new instrument IDs
        """
        with self.lock:
            self.symbols = new_symbols
            
        if self.ws and self.ws.sock and self.ws.sock.connected:
            # Resubscribe with new symbols
            self.ws.close()
            logging.info("Updating symbols subscription")
        else:
            self.start()

    def get_liquidity_levels(self, symbol):
        """
        Get key liquidity zones for SMC analysis
        :param symbol: Instrument ID
        :return: Dictionary with support/resistance levels or None
        """
        if not self._running:
            logging.warning("WebSocket not running")
            return None
            
        with self.lock:
            book = self.orderbook.get(symbol)
            
        if not book or not book['bids'] or not book['asks']:
            logging.warning(f"No orderbook data for {symbol}")
            return None
            
        return {
            'support': book['bids'][0][0],  # Best bid price
            'resistance': book['asks'][0][0]  # Best ask price
        }