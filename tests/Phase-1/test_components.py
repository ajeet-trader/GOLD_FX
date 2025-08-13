"""
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# test_components.py
from src.utils.logger import LoggerManager
from src.utils.database import DatabaseManager
from src.utils.error_handler import ErrorHandler
from src.core.mt5_manager import MT5Manager

print("Testing Logger...")
logger = LoggerManager({"logging": {"level": "INFO"}})
logger.setup_logging()
logger.info("Test message")

print("Testing Database...")
db = DatabaseManager({"database": {"sqlite": {"path": "data/test_db.db"}}})
db.initialize_database()

print("Testing Error Handler...")
err = ErrorHandler({})
err.start()

print("Testing MT5 Manager...")
mt5 = MT5Manager(symbol="XAUUSDm")
print(f"Symbol: {mt5.symbol}")

"""

# Component tests for Phase 1 modules

import unittest
import sys, os
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.mt5_manager import MT5Manager
from src.utils.database import DatabaseManager
from src.utils.error_handler import ErrorHandler, TradingError, ErrorSeverity
from src.utils.logger import LoggerManager

class TestMT5Manager(unittest.TestCase):
    def setUp(self):
        self.mt5 = MT5Manager('XAUUSDm')

    def test_connection(self):
        result = self.mt5.connect()
        self.assertTrue(result)
        self.mt5.disconnect()

    def test_symbol_validation(self):
        self.mt5.connect()
        symbol = self.mt5.get_valid_symbol("XAUUSD")
        self.assertIn("XAUUSD", symbol)
        self.mt5.disconnect()

    def test_account_info(self):
        self.mt5.connect()
        info = self.mt5._get_account_info()  # or self.mt5.account_info
        self.assertIn('balance', info)
        self.assertIn('equity', info)
        self.mt5.disconnect()


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db = DatabaseManager({"database": {"sqlite": {"path": "data/test_trading.db"}}})
        self.db.initialize_database()

    def test_trade_storage(self):
        trade = {
            "account_id": 1,
            "ticket": 99999,
            "symbol": "XAUUSDm",
            "action": "BUY",
            "volume": 0.01,
            "price_open": 2650.00,
            "open_time": datetime.now(),
            "profit": 10.50
        }
        result = self.db.store_trade(trade)
        self.assertTrue(result)

    def test_signal_storage(self):
        signal = {
            "symbol": "XAUUSDm",
            "strategy": "test",
            "signal_type": "BUY",
            "confidence": 0.75,
            "price": 2650.00,
            "timeframe": "M15"
        }
        result = self.db.store_signal(signal)
        self.assertTrue(result)


class TestErrorHandler(unittest.TestCase):
    def setUp(self):
        self.handler = ErrorHandler({})
        self.handler.start()

    def test_error_handling(self):
        error = TradingError("Test error", ErrorSeverity.LOW)
        result = self.handler.handle_error(error)
        self.assertTrue(result)

    def test_circuit_breaker(self):
        self.handler.setup_circuit_breaker("test", threshold=3, timeout=60)
        for i in range(5):
            error = TradingError(f"Error {i}", ErrorSeverity.MEDIUM)
            self.handler.handle_error(error)
        stats = self.handler.get_error_stats()
        self.assertGreater(stats['total_errors'], 0)


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = LoggerManager({"logging": {"level": "INFO"}})
        self.logger.setup_logging()

    def test_logger_creation(self):
        log = self.logger.get_logger('system')
        self.assertIsNotNone(log)

    def test_trade_logging(self):
        trade = {
            'symbol': 'XAUUSDm',
            'type': 'BUY',
            'volume': 0.01,
            'price': 2650.00
        }
        try:
            self.logger.log_trade(trade['type'], trade['symbol'], trade['volume'], trade['price'])
            success = True
        except:
            success = False
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
