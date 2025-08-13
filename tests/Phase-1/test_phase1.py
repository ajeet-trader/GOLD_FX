
# """
# Basic Phase 1 Tests
# ===================
# Simple tests to verify Phase 1 components are working.
# """

# import unittest
# import sys
# from pathlib import Path

# # Add src to path
# #sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
# ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# sys.path.insert(0, str(ROOT_DIR))

# class TestPhase1(unittest.TestCase):
#     """Basic tests for Phase 1 components"""
    
#     def test_imports(self):
#         """Test that all modules can be imported"""
#         try:
#             from src.utils.logger import LoggerManager
#             from src.utils.database import DatabaseManager
#             from src.utils.error_handler import ErrorHandler
#             from src.core.mt5_manager import MT5Manager
#             from src.phase_1_core_integration import CoreSystem
#         except ImportError as e:
#             self.fail(f"Import failed: {e}")
    
#     def test_core_system_creation(self):
#         """Test that CoreSystem can be created"""
#         try:
#             from src.phase_1_core_integration import CoreSystem
#             core = CoreSystem()
#             self.assertIsNotNone(core)
#         except Exception as e:
#             self.fail(f"CoreSystem creation failed: {e}")


# if __name__ == '__main__':
#     unittest.main()

"""
Phase 1 Integration Tests
"""

import unittest
import sys
from pathlib import Path
from datetime import datetime
import time

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.phase_1_core_integration import CoreSystem

class TestPhase1Integration(unittest.TestCase):
    """Test complete Phase 1 integration"""

    def setUp(self):
        """Setup test environment"""
        self.core = CoreSystem('config/master_config.yaml')

    def test_full_initialization(self):
        """Test complete system initialization"""
        result = self.core.initialize()
        self.assertTrue(result)
        self.assertTrue(self.core.initialized)

    def test_mt5_connection_flow(self):
        """Test MT5 connection workflow"""
        self.core.initialize()
        # Updated to use connect_mt5() instead of missing connect()
        result = self.core.connect_mt5()
        self.assertTrue(result)

        # Test account access - account_info is a dict attribute, not callable
        account = self.core.mt5_manager.account_info
        self.assertIsNotNone(account)
        self.assertIn('balance', account)

        self.core.mt5_manager.disconnect()

    def test_health_check(self):
        """Test system health check"""
        self.core.initialize()
        self.core.connect_mt5()

        # _perform_health_check returns bool in CoreSystem, not dict
        health_ok = self.core._perform_health_check()
        self.assertTrue(health_ok)  # just check that system is healthy

        self.core.mt5_manager.disconnect()

    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        self.core.initialize()

        # Simulate error
        try:
            raise ConnectionError("Test connection error")
        except Exception as e:
            handled_context = self.core.error_handler.handle_error(e)
            self.assertIsNotNone(handled_context)

    def test_data_flow(self):
        """Test data flow through system"""
        self.core.initialize()
        self.core.connect_mt5()

        # Fetch data - timeframe should be a string like "M15"
        data = self.core.mt5_manager.get_historical_data("XAUUSDm", "M15", 100)
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

        # Store signal - must match database Signal model fields
        signal = {
            'symbol': 'XAUUSDm',
            'strategy': 'test',
            'signal_type': 'LONG',       # renamed from 'direction'
            'confidence': 0.80,
            'price': 1950.0,
            'timeframe': 'M15',
            'timestamp': datetime.now()
        }
        result = self.core.database_manager.store_signal(signal)
        self.assertIsNotNone(result)

        self.core.mt5_manager.disconnect()


if __name__ == '__main__':
    unittest.main()
