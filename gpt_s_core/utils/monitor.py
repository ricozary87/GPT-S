import logging
from prometheus_client import start_http_server, Counter, Gauge

# Metrics
API_CALLS = Counter('okx_api_calls', 'Total OKX API calls', ['endpoint'])
WS_MESSAGES = Counter('okx_ws_messages', 'WebSocket messages received')
SMC_SIGNALS = Counter('smc_signals', 'Signals detected', ['pattern'])

class PerformanceMonitor:
    def __init__(self, port=8000):
        start_http_server(port)
        logging.basicConfig(
            filename='smc_trading.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    @staticmethod
    def log_api_call(endpoint):
        API_CALLS.labels(endpoint=endpoint).inc()
        logging.info(f"API Call: {endpoint}")

    @staticmethod
    def log_signal(pattern):
        SMC_SIGNALS.labels(pattern=pattern).inc()
        logging.warning(f"SMC Signal Detected: {pattern}")
