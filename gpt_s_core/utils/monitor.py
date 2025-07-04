import logging
import os # Added for potential future use (e.g., config for log file path)
import time # Added if time.sleep is used in any health checks within SystemMonitor
from collections import deque # Added if used in SystemMonitor (e.g., for CPU history)
from typing import Optional # Added for Optional type hint

from prometheus_client import start_http_server, Counter, Gauge, generate_latest
import psutil # Assuming SystemMonitor uses psutil
import platform # Assuming SystemMonitor uses platform
import pandas as pd # Assuming SystemMonitor uses pandas for timestamps

# =====================================================================
# Prometheus Metrics Definitions (Global)
# These should be defined once when the module is loaded.
# =====================================================================
API_CALLS = Counter('okx_api_calls_total', 'Total OKX API calls', ['endpoint'])
WS_MESSAGES = Counter('okx_ws_messages_total', 'Total WebSocket messages received')
SMC_SIGNALS = Counter('smc_signals_total', 'Total Smart Money Concept signals detected', ['pattern'])
SYSTEM_HEALTH_GAUGE = Gauge('system_health_status', 'Current system health status (1=ok, 0=not ok)', ['check_type'])
CPU_USAGE_GAUGE = Gauge('system_cpu_usage_percent', 'Current CPU usage in percent')
MEMORY_USAGE_GAUGE = Gauge('system_memory_usage_percent', 'Current memory usage in percent')
DISK_USAGE_GAUGE = Gauge('system_disk_usage_percent', 'Current disk usage in percent') # Added for completeness
NETWORK_SENT_GAUGE = Gauge('system_network_sent_mb', 'Network data sent in MB')
NETWORK_RECV_GAUGE = Gauge('system_network_recv_mb', 'Network data received in MB')


class PerformanceMonitor:
    """
    Manages Prometheus metrics exposure and logging for the application's performance.
    Designed to be initialized once to start the Prometheus HTTP server.
    """
    _server_started = False # Class-level flag to ensure server starts only once
    _logger = logging.getLogger(__name__) # Use specific logger for this module

    def __init__(self, prometheus_port: int = 8000):
        """
        Initializes the PerformanceMonitor and starts the Prometheus HTTP server.
        The Prometheus server should only be started once per application instance.
        """
        if not PerformanceMonitor._server_started:
            try:
                start_http_server(prometheus_port)
                PerformanceMonitor._server_started = True
                self._logger.info(f"Prometheus HTTP server started on port {prometheus_port}.")
            except OSError as e:
                self._logger.error(f"Failed to start Prometheus HTTP server on port {prometheus_port}: {e}. Metrics will not be exposed.", exc_info=True)
                # It might be beneficial to raise an exception here if prometheus server is critical.
                # Or set a flag like self.server_failed_to_start = True
            except Exception as e:
                self._logger.error(f"An unexpected error occurred while starting Prometheus server: {e}", exc_info=True)
        else:
            self._logger.debug("Prometheus HTTP server already started.")
        
        # Logging configuration is handled by the application's main entry point.
        # This __init__ does not reconfigure logging.

    @staticmethod
    def log_api_call(endpoint: str) -> None:
        """Increments API call counter for a specific endpoint."""
        API_CALLS.labels(endpoint=endpoint).inc()
        PerformanceMonitor._logger.debug(f"API Call to endpoint: {endpoint}") # Use debug for high-frequency logs

    @staticmethod
    def log_ws_message() -> None:
        """Increments WebSocket message counter."""
        WS_MESSAGES.inc()
        PerformanceMonitor._logger.debug("WebSocket message received.")

    @staticmethod
    def log_smc_signal(pattern: str, level: str = "INFO") -> None:
        """
        Increments SMC signal counter and logs the detected signal.
        Allows specifying log level for the signal message.
        """
        SMC_SIGNALS.labels(pattern=pattern).inc()
        if level.upper() == "WARNING":
            PerformanceMonitor._logger.warning(f"SMC Signal Detected: {pattern}")
        elif level.upper() == "ERROR":
            PerformanceMonitor._logger.error(f"SMC Signal Detected: {pattern}")
        elif level.upper() == "CRITICAL":
            PerformanceMonitor._logger.critical(f"SMC Signal Detected: {pattern}")
        else: # Default to INFO
            PerformanceMonitor._logger.info(f"SMC Signal Detected: {pattern}")
    
    @staticmethod
    def set_system_health(check_type: str, status: bool) -> None:
        """Sets the system health gauge for a specific check type."""
        SYSTEM_HEALTH_GAUGE.labels(check_type=check_type).set(1 if status else 0)
        PerformanceMonitor._logger.debug(f"System health {check_type}: {'OK' if status else 'NOT OK'}")

    @staticmethod
    def set_cpu_usage(value: float) -> None:
        """Sets the current CPU usage gauge."""
        CPU_USAGE_GAUGE.set(value)

    @staticmethod
    def set_memory_usage(value: float) -> None:
        """Sets the current memory usage gauge."""
        MEMORY_USAGE_GAUGE.set(value)

    @staticmethod
    def set_disk_usage(value: float) -> None:
        """Sets the current disk usage gauge."""
        DISK_USAGE_GAUGE.set(value)

    @staticmethod
    def set_network_metrics(sent_mb: float, recv_mb: float) -> None:
        """Sets current network data sent and received gauges."""
        NETWORK_SENT_GAUGE.set(sent_mb)
        NETWORK_RECV_GAUGE.set(recv_mb)


_monitor_instance: Optional[PerformanceMonitor] = None

# ✅ Solusi 2: Tambahkan fungsi start_monitoring(port=8000) wrapper
def start_monitoring(port: int = 8000):
    """
    Initializes the PerformanceMonitor (and thus the Prometheus HTTP server)
    if it hasn't been initialized yet. Ensures a single instance.
    """
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor(prometheus_port=port)
    return _monitor_instance # Return the instance

# You can add a method to get the raw metrics in Prometheus text format
def get_prometheus_metrics_text() -> bytes:
    """Returns the current Prometheus metrics in exposition format."""
    return generate_latest()

# ✅ Solusi 3: Tidak Ada __all__ untuk modul ini
__all__ = [
    "PerformanceMonitor",
    "start_monitoring",
    "get_prometheus_metrics_text", # Optional: if you want to expose this
    "API_CALLS",
    "WS_MESSAGES",
    "SMC_SIGNALS",
    "SYSTEM_HEALTH_GAUGE",
    "CPU_USAGE_GAUGE",
    "MEMORY_USAGE_GAUGE",
    "DISK_USAGE_GAUGE",
    "NETWORK_SENT_GAUGE",
    "NETWORK_RECV_GAUGE",
    # Add other gauges if they are part of SystemMonitor class
]

# ✅ Solusi 1: Impor logging Harus Menyediakan Formatter Default (jika terpisah)
# This block ensures that if this file is run directly or imported into an app
# that hasn't configured logging yet, it will at least log to console.
if not logging.getLogger().hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    # Also set the level, as default is WARNING
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Default StreamHandler added for monitor module logging.")

# Example of basic SystemMonitor/AlertManager if they are not in separate files
# and EnhancedSystemMonitor/MultiChannelAlertManager inherit from them in orchestrator.
# If they are in separate files, remove these definitions.

class SystemMonitor:
    def check_system_health(self) -> dict:
        return {"ok": True, "message": "Base health check", "details": {}}
    def get_system_metrics(self) -> dict:
        return {"cpu": 0.0, "memory": 0.0, "disk": 0.0,
                "network": {"sent": 0.0, "recv": 0.0}, "boot_time": "N/A", "os": "N/A"}

class AlertManager:
    def send_alert(self, message: str, channel: str = "telegram", severity: str = "medium"):
        logging.getLogger(__name__).info(f"MOCK AlertManager: Sending alert '{message}' via {channel} ({severity})")
