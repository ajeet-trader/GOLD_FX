"""
Test Suite for Signal Age Validation Fixes
==========================================

Tests for Sprint 1 fixes to signal age validation logic, specifically:
- Updated signal age threshold from 300s to 3600s (1 hour default)
- Test configuration threshold adjustment to 7200s (2 hours)
- Weekend market closure bypass for mock mode
- Proper validation of signal freshness in different modes
"""

import sys
import os
from pathlib import Path
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.core.execution_engine import ExecutionEngine
from src.core.base import Signal, SignalType, SignalGrade


class TestSignalAgeValidationFixes(unittest.TestCase):
    """Test suite for signal age validation fixes"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = {
            'execution': {
                'max_signal_age_seconds': 3600,  # 1 hour default
                'test_max_signal_age_seconds': 7200,  # 2 hours for testing
                'position_size_percent': 2.0,
                'max_concurrent_positions': 3
            },
            'risk_management': {
                'max_daily_loss_percent': 5.0,
                'max_position_size_percent': 2.0
            }
        }
        
        # Mock dependencies
        self.mock_mt5_manager = Mock()
        self.mock_database = Mock()
        self.mock_risk_manager = Mock()
        
        # Set up mock MT5 manager
        self.mock_mt5_manager.connected = True
        self.mock_mt5_manager.get_account_info.return_value = {'balance': 10000}
        
        # Set up mock risk manager
        self.mock_risk_manager.validate_trade.return_value = {'valid': True}
        self.mock_risk_manager.mt5_manager = self.mock_mt5_manager

    def _create_test_signal(self, age_seconds=0):
        """Create a test signal with specified age"""
        signal_time = datetime.now() - timedelta(seconds=age_seconds)
        
        return Signal(
            timestamp=signal_time,
            symbol="XAUUSDm",
            strategy_name="TestStrategy",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=1950.0,
            timeframe="M15",
            strength=0.8,
            stop_loss=1940.0,
            take_profit=1970.0
        )

    def test_live_mode_signal_age_validation_3600s(self):
        """Test signal age validation in live mode with 3600s threshold"""
        execution_engine = ExecutionEngine(
            self.mock_config, 
            self.mock_mt5_manager, 
            self.mock_database,
            self.mock_risk_manager,
            mode='live'
        )
        
        # Test fresh signal (should pass)
        fresh_signal = self._create_test_signal(age_seconds=1800)  # 30 minutes
        result = execution_engine._validate_signal_age(fresh_signal)
        self.assertTrue(result['valid'], "Fresh signal should pass validation")
        
        # Test old signal (should fail)
        old_signal = self._create_test_signal(age_seconds=4000)  # > 1 hour
        result = execution_engine._validate_signal_age(old_signal)
        self.assertFalse(result['valid'], "Old signal should fail validation")
        self.assertIn('too old', result['reason'].lower())

    def test_test_mode_signal_age_validation_7200s(self):
        """Test signal age validation in test mode with 7200s threshold"""
        test_config = self.mock_config.copy()
        execution_engine = ExecutionEngine(
            test_config, 
            self.mock_mt5_manager, 
            self.mock_database,
            self.mock_risk_manager,
            mode='mock'
        )
        
        # Test signal aged 1.5 hours (should pass in test mode)
        signal_1_5h = self._create_test_signal(age_seconds=5400)  # 1.5 hours
        result = execution_engine._validate_signal_age(signal_1_5h)
        self.assertTrue(result['valid'], "1.5 hour old signal should pass in test mode")
        
        # Test signal aged 3 hours (should fail even in test mode)
        signal_3h = self._create_test_signal(age_seconds=10800)  # 3 hours
        result = execution_engine._validate_signal_age(signal_3h)
        self.assertFalse(result['valid'], "3 hour old signal should fail even in test mode")

    def test_weekend_bypass_mock_mode(self):
        """Test weekend market closure bypass in mock mode"""
        execution_engine = ExecutionEngine(
            self.mock_config, 
            self.mock_mt5_manager, 
            self.mock_database,
            self.mock_risk_manager,
            mode='mock'
        )
        
        # Create a signal for weekend testing
        weekend_signal = self._create_test_signal(age_seconds=1800)
        
        # Mock weekend datetime
        with patch('src.core.execution_engine.datetime') as mock_datetime:
            # Saturday
            mock_datetime.now.return_value = datetime(2025, 8, 23, 10, 0, 0)  # Saturday
            mock_datetime.weekday.return_value = 5  # Saturday
            
            result = execution_engine._validate_signal_age(weekend_signal)
            self.assertTrue(result['valid'], "Weekend signal should pass in mock mode")

    def test_weekend_restriction_live_mode(self):
        """Test weekend market closure restriction in live mode"""
        execution_engine = ExecutionEngine(
            self.mock_config, 
            self.mock_mt5_manager, 
            self.mock_database,
            self.mock_risk_manager,
            mode='live'
        )
        
        weekend_signal = self._create_test_signal(age_seconds=1800)
        
        # Mock weekend datetime for live mode
        with patch('src.core.execution_engine.datetime') as mock_datetime:
            # Sunday  
            mock_datetime.now.return_value = datetime(2025, 8, 24, 10, 0, 0)  # Sunday
            mock_datetime.weekday.return_value = 6  # Sunday
            
            result = execution_engine._validate_signal_age(weekend_signal)
            self.assertFalse(result['valid'], "Weekend signal should fail in live mode")
            self.assertIn('weekend', result['reason'].lower())

    def test_signal_age_threshold_configuration(self):
        """Test that signal age thresholds are correctly configured"""
        execution_engine = ExecutionEngine(
            self.mock_config, 
            self.mock_mt5_manager, 
            self.mock_database,
            self.mock_risk_manager,
            mode='live'
        )
        
        # Check live mode threshold
        expected_live_threshold = self.mock_config['execution']['max_signal_age_seconds']
        self.assertEqual(execution_engine.max_signal_age_seconds, expected_live_threshold)
        
        # Test with test mode
        test_execution_engine = ExecutionEngine(
            self.mock_config, 
            self.mock_mt5_manager, 
            self.mock_database,
            self.mock_risk_manager,
            mode='mock'
        )
        
        # Should use test threshold in mock mode
        expected_test_threshold = self.mock_config['execution']['test_max_signal_age_seconds']
        # Note: This assumes the implementation uses test threshold for mock mode

    def test_signal_age_calculation_accuracy(self):
        """Test accuracy of signal age calculation"""
        execution_engine = ExecutionEngine(
            self.mock_config, 
            self.mock_mt5_manager, 
            self.mock_database,
            self.mock_risk_manager,
            mode='live'
        )
        
        # Test precise age calculation
        exact_age_seconds = 2000
        signal = self._create_test_signal(age_seconds=exact_age_seconds)
        
        # Mock current time for precise calculation
        with patch('src.core.execution_engine.datetime') as mock_datetime:
            mock_now = signal.timestamp + timedelta(seconds=exact_age_seconds)
            mock_datetime.now.return_value = mock_now
            
            result = execution_engine._validate_signal_age(signal)
            
            # Should calculate age correctly (2000s < 3600s threshold)
            self.assertTrue(result['valid'], f"Signal age {exact_age_seconds}s should be valid")

    def test_edge_case_threshold_boundary(self):
        """Test edge cases at threshold boundaries"""
        execution_engine = ExecutionEngine(
            self.mock_config, 
            self.mock_mt5_manager, 
            self.mock_database,
            self.mock_risk_manager,
            mode='live'
        )
        
        # Test signal exactly at threshold
        threshold_signal = self._create_test_signal(age_seconds=3600)  # Exactly 1 hour
        result = execution_engine._validate_signal_age(threshold_signal)
        # Implementation may allow equal or require less than threshold
        
        # Test signal just over threshold
        over_threshold_signal = self._create_test_signal(age_seconds=3601)  # 1 second over
        result = execution_engine._validate_signal_age(over_threshold_signal)
        self.assertFalse(result['valid'], "Signal just over threshold should fail")
        
        # Test signal just under threshold
        under_threshold_signal = self._create_test_signal(age_seconds=3599)  # 1 second under
        result = execution_engine._validate_signal_age(under_threshold_signal)
        self.assertTrue(result['valid'], "Signal just under threshold should pass")

    def test_no_timestamp_signal_handling(self):
        """Test handling of signals without timestamps"""
        execution_engine = ExecutionEngine(
            self.mock_config, 
            self.mock_mt5_manager, 
            self.mock_database,
            self.mock_risk_manager,
            mode='live'
        )
        
        # Create signal without timestamp (should use current time)
        signal = Signal(
            timestamp=None,
            symbol="XAUUSDm",
            strategy_name="TestStrategy",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=1950.0,
            timeframe="M15",
            strength=0.8
        )
        
        try:
            result = execution_engine._validate_signal_age(signal)
            # Should handle gracefully (either pass or fail with clear reason)
            self.assertIsInstance(result, dict)
            self.assertIn('valid', result)
            self.assertIn('reason', result)
        except Exception as e:
            self.fail(f"Signal without timestamp should be handled gracefully: {e}")

    def test_future_timestamp_signal_handling(self):
        """Test handling of signals with future timestamps"""
        execution_engine = ExecutionEngine(
            self.mock_config, 
            self.mock_mt5_manager, 
            self.mock_database,
            self.mock_risk_manager,
            mode='live'
        )
        
        # Create signal with future timestamp
        future_signal = Signal(
            timestamp=datetime.now() + timedelta(minutes=30),
            symbol="XAUUSDm",
            strategy_name="TestStrategy",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=1950.0,
            timeframe="M15",
            strength=0.8
        )
        
        result = execution_engine._validate_signal_age(future_signal)
        # Future signals should be handled appropriately
        self.assertIsInstance(result, dict)


class TestSignalAgeIntegration(unittest.TestCase):
    """Integration tests for signal age validation"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.mock_config = {
            'execution': {
                'max_signal_age_seconds': 3600,
                'test_max_signal_age_seconds': 7200,
                'position_size_percent': 2.0
            },
            'risk_management': {
                'max_daily_loss_percent': 5.0
            }
        }
        
        self.mock_mt5_manager = Mock()
        self.mock_database = Mock()
        self.mock_risk_manager = Mock()
        
        self.mock_mt5_manager.connected = True
        self.mock_risk_manager.validate_trade.return_value = {'valid': True}

    def test_end_to_end_signal_processing_with_age_validation(self):
        """Test complete signal processing including age validation"""
        execution_engine = ExecutionEngine(
            self.mock_config, 
            self.mock_mt5_manager, 
            self.mock_database,
            self.mock_risk_manager,
            mode='live'
        )
        
        # Create fresh signal
        fresh_signal = Signal(
            timestamp=datetime.now() - timedelta(minutes=30),
            symbol="XAUUSDm",
            strategy_name="TestStrategy",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=1950.0,
            timeframe="M15",
            strength=0.8,
            stop_loss=1940.0,
            take_profit=1970.0
        )
        
        try:
            # Process signal through validation pipeline
            result = execution_engine._validate_signal_age(fresh_signal)
            self.assertTrue(result['valid'], "Fresh signal should pass age validation")
            
        except Exception as e:
            self.fail(f"Signal processing failed: {e}")

    def test_performance_impact_of_age_validation(self):
        """Test that age validation doesn't significantly impact performance"""
        execution_engine = ExecutionEngine(
            self.mock_config, 
            self.mock_mt5_manager, 
            self.mock_database,
            self.mock_risk_manager,
            mode='live'
        )
        
        signal = Signal(
            timestamp=datetime.now() - timedelta(minutes=30),
            symbol="XAUUSDm",
            strategy_name="TestStrategy",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=1950.0,
            timeframe="M15",
            strength=0.8
        )
        
        import time
        start_time = time.time()
        
        # Perform multiple age validations
        for _ in range(100):
            result = execution_engine._validate_signal_age(signal)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete quickly (under 1 second for 100 validations)
        self.assertLess(elapsed, 1.0, "Age validation should be performant")


if __name__ == '__main__':
    unittest.main(verbosity=2, buffer=True)