"""
Basic Phase 1 Tests
===================
Simple tests to verify Phase 1 components are working.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestPhase1(unittest.TestCase):
    """Basic tests for Phase 1 components"""
    
    def test_imports(self):
        """Test that all modules can be imported"""
        try:
            from utils.logger import LoggerManager
            from utils.database import DatabaseManager
            from utils.error_handler import ErrorHandler
            from core.mt5_manager import MT5Manager
            from core_system import CoreSystem
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_core_system_creation(self):
        """Test that CoreSystem can be created"""
        try:
            from core_system import CoreSystem
            core = CoreSystem()
            self.assertIsNotNone(core)
        except Exception as e:
            self.fail(f"CoreSystem creation failed: {e}")


if __name__ == '__main__':
    unittest.main()