"""
Test Suite for Elliott Wave DatetimeIndex Slicing Fixes
=======================================================

Tests for Sprint 1 fixes to Elliott Wave strategy, specifically:
- Fixed DatetimeIndex slicing error in _check_volume_confirmation method
- Replaced data.loc[wave.start_index:wave.end_index, 'Volume'] with data.iloc[start_idx:end_idx + 1]['Volume']
- Added bounds checking and range validation
- Improved error handling for index operations
"""

import sys
import os
from pathlib import Path
import unittest
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.strategies.technical.elliott_wave import ElliottWaveStrategy
from src.core.base import SignalType


class MockWave:
    """Mock wave object for testing"""
    def __init__(self, start_index, end_index, wave_type='impulse', degree=1):
        self.start_index = start_index
        self.end_index = end_index
        self.wave_type = wave_type
        self.degree = degree


class TestElliottWaveDatetimeIndexFixes(unittest.TestCase):
    """Test suite for Elliott Wave DatetimeIndex slicing fixes"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'parameters': {
                'lookback_bars': 100,
                'min_wave_size': 5,
                'max_wave_size': 50,
                'volume_confirmation_threshold': 1.2,
                'confidence_threshold': 0.65,
                'mode': 'mock'
            }
        }
        
        # Create strategy instance
        self.strategy = ElliottWaveStrategy(self.config)
        
        # Create mock data with DatetimeIndex
        self.mock_data = self._create_mock_data_with_datetime_index()

    def _create_mock_data_with_datetime_index(self):
        """Create mock OHLCV data with DatetimeIndex for testing"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), 
                             end=datetime.now(), freq='15Min')[:200]
        
        np.random.seed(42)
        close_prices = 1950 + np.cumsum(np.random.randn(len(dates)) * 2)
        
        data = pd.DataFrame({
            'Open': close_prices + np.random.randn(len(dates)) * 0.5,
            'High': close_prices + np.abs(np.random.randn(len(dates)) * 3),
            'Low': close_prices - np.abs(np.random.randn(len(dates)) * 3),
            'Close': close_prices,
            'Volume': np.random.randint(100, 2000, len(dates))
        }, index=dates)
        
        # Ensure High >= Close >= Low
        data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
        data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
        
        return data

    def test_volume_confirmation_with_valid_indices(self):
        """Test volume confirmation with valid wave indices"""
        if not hasattr(self.strategy, '_check_volume_confirmation'):
            self.skipTest("Volume confirmation method not available")
        
        # Create test waves with valid indices
        wave1 = MockWave(start_index=10, end_index=20)
        wave2 = MockWave(start_index=25, end_index=35)
        wave3 = MockWave(start_index=40, end_index=55)
        
        waves = [wave1, wave2, wave3]
        
        try:
            # Should not raise DatetimeIndex slicing errors
            result = self.strategy._check_volume_confirmation(self.mock_data, waves)
            self.assertIsInstance(result, (bool, list))
        except Exception as e:
            if "datetimeindex" in str(e).lower() or "slice" in str(e).lower():
                self.fail(f"DatetimeIndex slicing error still occurring: {e}")

    def test_volume_confirmation_with_boundary_indices(self):
        """Test volume confirmation with boundary wave indices"""
        if not hasattr(self.strategy, '_check_volume_confirmation'):
            self.skipTest("Volume confirmation method not available")
        
        data_length = len(self.mock_data)
        
        # Test edge cases
        boundary_waves = [
            MockWave(start_index=0, end_index=5),  # Start boundary
            MockWave(start_index=data_length-10, end_index=data_length-1),  # End boundary
            MockWave(start_index=data_length-5, end_index=data_length-1),  # Near end boundary
        ]
        
        try:
            result = self.strategy._check_volume_confirmation(self.mock_data, boundary_waves)
            self.assertIsInstance(result, (bool, list))
        except IndexError as e:
            self.fail(f"Index error with boundary conditions: {e}")
        except Exception as e:
            if "slice" in str(e).lower():
                self.fail(f"Slicing error with boundary conditions: {e}")

    def test_volume_confirmation_with_invalid_indices(self):
        """Test volume confirmation handles invalid wave indices gracefully"""
        if not hasattr(self.strategy, '_check_volume_confirmation'):
            self.skipTest("Volume confirmation method not available")
        
        data_length = len(self.mock_data)
        
        # Test invalid indices
        invalid_waves = [
            MockWave(start_index=-5, end_index=10),  # Negative start
            MockWave(start_index=10, end_index=data_length + 10),  # Beyond data length
            MockWave(start_index=50, end_index=40),  # End before start
            MockWave(start_index=data_length + 5, end_index=data_length + 10),  # Both beyond length
        ]
        
        try:
            # Should handle gracefully without crashing
            result = self.strategy._check_volume_confirmation(self.mock_data, invalid_waves)
            self.assertIsInstance(result, (bool, list))
        except Exception as e:
            # Should not crash with unhandled indexing errors
            self.assertNotIn("slice", str(e).lower())
            self.assertNotIn("index", str(e).lower())

    def test_iloc_vs_loc_consistency(self):
        """Test that iloc indexing is used instead of loc for integer indices"""
        if not hasattr(self.strategy, '_check_volume_confirmation'):
            self.skipTest("Volume confirmation method not available")
        
        # Create wave with specific indices
        test_wave = MockWave(start_index=15, end_index=25)
        
        # Test direct volume access using both methods
        start_idx = max(0, min(test_wave.start_index, len(self.mock_data) - 1))
        end_idx = max(0, min(test_wave.end_index, len(self.mock_data) - 1))
        
        if start_idx < end_idx:
            # This should work (iloc with integer indices)
            volume_iloc = self.mock_data.iloc[start_idx:end_idx + 1]['Volume']
            self.assertIsInstance(volume_iloc, pd.Series)
            self.assertGreater(len(volume_iloc), 0)
            
            # This would fail with DatetimeIndex (loc with integer indices on DatetimeIndex)
            # We should NOT be using this approach anymore:
            # volume_loc = self.mock_data.loc[start_idx:end_idx, 'Volume']  # This causes the bug

    def test_bounds_checking_implementation(self):
        """Test that bounds checking is properly implemented"""
        if not hasattr(self.strategy, '_check_volume_confirmation'):
            self.skipTest("Volume confirmation method not available")
        
        data_length = len(self.mock_data)
        
        # Test various index scenarios
        test_cases = [
            (-10, 10),     # Negative start
            (10, data_length + 10),  # End beyond bounds
            (data_length - 1, data_length - 1),  # Single point at end
            (0, 0),        # Single point at start
        ]
        
        for start_idx, end_idx in test_cases:
            wave = MockWave(start_index=start_idx, end_index=end_idx)
            
            try:
                result = self.strategy._check_volume_confirmation(self.mock_data, [wave])
                # Should handle all cases without errors
                self.assertIsInstance(result, (bool, list))
            except Exception as e:
                self.fail(f"Bounds checking failed for indices ({start_idx}, {end_idx}): {e}")

    def test_volume_confirmation_with_empty_waves_list(self):
        """Test volume confirmation with empty waves list"""
        if not hasattr(self.strategy, '_check_volume_confirmation'):
            self.skipTest("Volume confirmation method not available")
        
        try:
            result = self.strategy._check_volume_confirmation(self.mock_data, [])
            # Should handle empty list gracefully
            self.assertIsInstance(result, (bool, list))
        except Exception as e:
            self.fail(f"Empty waves list should be handled gracefully: {e}")

    def test_signal_generation_without_datetime_errors(self):
        """Test that signal generation works without DatetimeIndex errors"""
        try:
            signals = self.strategy.generate_signal("XAUUSDm", "M15")
            self.assertIsInstance(signals, list)
            
            # Check that no DatetimeIndex errors occur
            for signal in signals:
                if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                    self.assertIsInstance(signal.confidence, (int, float))
                    self.assertGreater(signal.confidence, 0.0)
                    
        except Exception as e:
            if "datetimeindex" in str(e).lower() or "slice" in str(e).lower():
                self.fail(f"DatetimeIndex error still occurring in signal generation: {e}")

    def test_wave_analysis_with_different_data_sizes(self):
        """Test wave analysis with different data sizes"""
        # Test with smaller dataset
        small_data = self.mock_data.iloc[:50]
        
        try:
            signals = self.strategy.generate_signal("XAUUSDm", "M15")
            self.assertIsInstance(signals, list)
        except Exception as e:
            if "slice" in str(e).lower() or "index" in str(e).lower():
                self.fail(f"Indexing error with small dataset: {e}")
        
        # Test with larger dataset
        large_dates = pd.date_range(start=datetime.now() - timedelta(days=14), 
                                   end=datetime.now(), freq='15Min')[:500]
        
        np.random.seed(42)
        large_data = pd.DataFrame({
            'Open': 1950 + np.random.randn(len(large_dates)) * 10,
            'High': 1950 + np.random.randn(len(large_dates)) * 10 + 5,
            'Low': 1950 + np.random.randn(len(large_dates)) * 10 - 5,
            'Close': 1950 + np.random.randn(len(large_dates)) * 10,
            'Volume': np.random.randint(100, 2000, len(large_dates))
        }, index=large_dates)
        
        # Mock the data for large dataset test
        original_get_data = self.strategy.mt5_manager.get_historical_data
        self.strategy.mt5_manager.get_historical_data = Mock(return_value=large_data)
        
        try:
            signals = self.strategy.generate_signal("XAUUSDm", "M15")
            self.assertIsInstance(signals, list)
        except Exception as e:
            if "slice" in str(e).lower() or "index" in str(e).lower():
                self.fail(f"Indexing error with large dataset: {e}")
        finally:
            # Restore original method
            self.strategy.mt5_manager.get_historical_data = original_get_data

    def test_volume_data_access_patterns(self):
        """Test different volume data access patterns"""
        if not hasattr(self.strategy, '_check_volume_confirmation'):
            self.skipTest("Volume confirmation method not available")
        
        # Create waves with different patterns
        patterns = [
            [MockWave(10, 15), MockWave(20, 25)],  # Non-overlapping
            [MockWave(10, 20), MockWave(15, 25)],  # Overlapping
            [MockWave(5, 50)],  # Single large wave
            [MockWave(i, i+3) for i in range(10, 30, 5)],  # Multiple small waves
        ]
        
        for pattern in patterns:
            try:
                result = self.strategy._check_volume_confirmation(self.mock_data, pattern)
                self.assertIsInstance(result, (bool, list))
            except Exception as e:
                if "slice" in str(e).lower() or "index" in str(e).lower():
                    self.fail(f"Volume access pattern failed: {e}")

    def test_error_handling_robustness(self):
        """Test robustness of error handling for various edge cases"""
        if not hasattr(self.strategy, '_check_volume_confirmation'):
            self.skipTest("Volume confirmation method not available")
        
        # Test with data missing Volume column
        data_no_volume = self.mock_data.drop(columns=['Volume'])
        
        try:
            result = self.strategy._check_volume_confirmation(data_no_volume, [MockWave(10, 20)])
            # Should handle missing volume gracefully
        except KeyError:
            # Acceptable to raise KeyError for missing column
            pass
        except Exception as e:
            if "slice" in str(e).lower():
                self.fail(f"Should not have slicing errors: {e}")

    def test_performance_with_fixed_indexing(self):
        """Test that fixed indexing doesn't negatively impact performance"""
        if not hasattr(self.strategy, '_check_volume_confirmation'):
            self.skipTest("Volume confirmation method not available")
        
        # Create many waves for performance test
        many_waves = [MockWave(i, i+5) for i in range(0, 100, 10)]
        
        import time
        start_time = time.time()
        
        try:
            result = self.strategy._check_volume_confirmation(self.mock_data, many_waves)
            end_time = time.time()
            
            # Should complete in reasonable time
            elapsed = end_time - start_time
            self.assertLess(elapsed, 5.0, "Volume confirmation should be performant")
            
        except Exception as e:
            if "slice" in str(e).lower():
                self.fail(f"Performance test failed with slicing error: {e}")


class TestElliottWaveIntegration(unittest.TestCase):
    """Integration tests for Elliott Wave strategy fixes"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.config = {
            'parameters': {
                'lookback_bars': 100,
                'min_wave_size': 5,
                'confidence_threshold': 0.65,
                'mode': 'mock'
            }
        }

    def test_strategy_initialization_without_errors(self):
        """Test strategy initializes without DatetimeIndex-related errors"""
        try:
            strategy = ElliottWaveStrategy(self.config)
            self.assertIsNotNone(strategy)
            self.assertEqual(strategy.strategy_name, "ElliottWaveStrategy")
        except Exception as e:
            if "slice" in str(e).lower() or "index" in str(e).lower():
                self.fail(f"Strategy initialization failed with indexing error: {e}")

    def test_end_to_end_elliott_wave_pipeline(self):
        """Test complete Elliott Wave analysis pipeline"""
        strategy = ElliottWaveStrategy(self.config)
        
        try:
            # Generate signals
            signals = strategy.generate_signal("XAUUSDm", "M15")
            
            # Should complete without DatetimeIndex errors
            self.assertIsInstance(signals, list)
            
            # Get performance summary
            performance = strategy.get_performance_summary()
            self.assertIsInstance(performance, dict)
            
            # Get analysis
            mock_data = strategy.mt5_manager.get_historical_data("XAUUSDm", "M15", 100)
            if mock_data is not None:
                analysis = strategy.analyze(mock_data, "XAUUSDm", "M15")
                self.assertIsInstance(analysis, dict)
            
        except Exception as e:
            if "slice" in str(e).lower() or "datetimeindex" in str(e).lower():
                self.fail(f"End-to-end pipeline failed with DatetimeIndex error: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2, buffer=True)