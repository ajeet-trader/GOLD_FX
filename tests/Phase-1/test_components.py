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