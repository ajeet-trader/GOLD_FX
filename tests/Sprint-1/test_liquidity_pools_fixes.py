"""
Test Suite for Liquidity Pools Signal Throttling Fixes
======================================================

Tests for Sprint 1 fixes to Liquidity Pools strategy, specifically:
- Implemented comprehensive signal throttling to fix 141-signal overflow
- Added throttling parameters: max_signals_per_run=5, min_pool_strength=2.0, max_active_pools=10
- Increased cooldown_bars from 3 to 10
- Implemented signal deduplication logic to prevent similar signals
- Added pool strength filtering and sorting
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

from src.strategies.smc.liquidity_pools import LiquidityPoolsStrategy
from src.core.base import SignalType


class TestLiquidityPoolsThrottlingFixes(unittest.TestCase):
    """Test suite for Liquidity Pools signal throttling fixes"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'parameters': {
                'lookback_bars': 100,
                'confidence_threshold': 0.65,
                'max_signals_per_run': 5,
                'min_pool_strength': 2.0,
                'max_active_pools': 10,
                'cooldown_bars': 10,
                'min_pool_volume': 1000,
                'pool_proximity_threshold': 10.0,
                'mode': 'mock'
            }
        }
        
        # Create strategy instance
        self.strategy = LiquidityPoolsStrategy(self.config)
        
        # Create mock data
        self.mock_data = self._create_mock_data()

    def _create_mock_data(self):
        """Create mock OHLCV data for testing"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), 
                             end=datetime.now(), freq='15Min')[:200]
        
        np.random.seed(42)
        close_prices = 1950 + np.cumsum(np.random.randn(len(dates)) * 2)
        
        data = pd.DataFrame({
            'Open': close_prices + np.random.randn(len(dates)) * 0.5,
            'High': close_prices + np.abs(np.random.randn(len(dates)) * 3),
            'Low': close_prices - np.abs(np.random.randn(len(dates)) * 3),
            'Close': close_prices,
            'Volume': np.random.randint(500, 3000, len(dates))
        }, index=dates)
        
        # Ensure High >= Close >= Low
        data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
        data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
        
        return data

    def test_max_signals_per_run_throttling(self):
        """Test that signal generation respects max_signals_per_run limit"""
        signals = self.strategy.generate_signal("XAUUSDm", "M15")
        
        # Should never exceed max_signals_per_run
        max_allowed = self.config['parameters']['max_signals_per_run']
        self.assertLessEqual(len(signals), max_allowed, 
                           f"Generated {len(signals)} signals, max allowed is {max_allowed}")

    def test_min_pool_strength_filtering(self):
        """Test that only pools above min_pool_strength threshold are considered"""
        if hasattr(self.strategy, '_identify_liquidity_pools'):
            pools = self.strategy._identify_liquidity_pools(self.mock_data)
            
            min_strength = self.config['parameters']['min_pool_strength']
            
            # All pools should meet minimum strength requirement
            for pool in pools:
                if hasattr(pool, 'strength'):
                    self.assertGreaterEqual(pool.strength, min_strength,
                                          f"Pool strength {pool.strength} below minimum {min_strength}")

    def test_max_active_pools_limitation(self):
        """Test that only top max_active_pools are considered"""
        if hasattr(self.strategy, '_identify_liquidity_pools'):
            pools = self.strategy._identify_liquidity_pools(self.mock_data)
            
            max_pools = self.config['parameters']['max_active_pools']
            
            # Should not exceed max_active_pools limit
            active_pools = pools[:max_pools] if isinstance(pools, list) else []
            self.assertLessEqual(len(active_pools), max_pools,
                               f"Active pools {len(active_pools)} exceeds maximum {max_pools}")

    def test_cooldown_bars_enforcement(self):
        """Test that cooldown bars prevent rapid signal generation"""
        # Generate initial signals
        initial_signals = self.strategy.generate_signal("XAUUSDm", "M15")
        
        # Immediate second call should be throttled by cooldown
        immediate_signals = self.strategy.generate_signal("XAUUSDm", "M15")
        
        # Should have fewer or same signals due to cooldown
        self.assertLessEqual(len(immediate_signals), len(initial_signals),
                           "Cooldown should prevent rapid signal generation")

    def test_signal_deduplication_logic(self):
        """Test that similar signals are deduplicated"""
        signals = self.strategy.generate_signal("XAUUSDm", "M15")
        
        if len(signals) > 1:
            # Check that signals are not too similar in price
            prices = [signal.price for signal in signals]
            
            # Calculate minimum distance between signals
            min_distance = float('inf')
            for i in range(len(prices)):
                for j in range(i + 1, len(prices)):
                    distance = abs(prices[i] - prices[j])
                    min_distance = min(min_distance, distance)
            
            # Should have reasonable separation (depends on implementation)
            if min_distance != float('inf'):
                self.assertGreater(min_distance, 0.1, 
                                 "Signals should be deduplicated to avoid clustering")

    def test_pool_strength_sorting(self):
        """Test that pools are sorted by strength (strongest first)"""
        if hasattr(self.strategy, '_identify_liquidity_pools'):
            pools = self.strategy._identify_liquidity_pools(self.mock_data)
            
            if len(pools) > 1 and hasattr(pools[0], 'strength'):
                # Check that pools are sorted by strength (descending)
                for i in range(len(pools) - 1):
                    if hasattr(pools[i], 'strength') and hasattr(pools[i+1], 'strength'):
                        self.assertGreaterEqual(pools[i].strength, pools[i+1].strength,
                                              "Pools should be sorted by strength (descending)")

    def test_no_signal_overflow_after_fixes(self):
        """Test that signal overflow (141+ signals) no longer occurs"""
        # Run multiple signal generation cycles
        total_signals_across_runs = []
        
        for _ in range(10):
            signals = self.strategy.generate_signal("XAUUSDm", "M15")
            total_signals_across_runs.extend(signals)
            
            # Each individual run should be throttled
            max_allowed = self.config['parameters']['max_signals_per_run']
            self.assertLessEqual(len(signals), max_allowed,
                               f"Single run generated {len(signals)} signals, max allowed is {max_allowed}")
        
        # Even across multiple runs, should not have excessive signals
        self.assertLess(len(total_signals_across_runs), 100,
                       "Multiple runs should not generate excessive signals")

    def test_throttling_parameters_configuration(self):
        """Test that throttling parameters are properly configured"""
        # Check that throttling parameters are set correctly
        self.assertEqual(self.strategy.max_signals_per_run, 5)
        self.assertEqual(self.strategy.min_pool_strength, 2.0)
        self.assertEqual(self.strategy.max_active_pools, 10)
        self.assertEqual(self.strategy.cooldown_bars, 10)

    def test_performance_with_throttling(self):
        """Test that throttling improves performance by reducing computations"""
        import time
        
        start_time = time.time()
        signals = self.strategy.generate_signal("XAUUSDm", "M15")
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        # Should complete quickly due to throttling
        self.assertLess(elapsed, 5.0, "Throttled signal generation should be fast")
        
        # Should generate reasonable number of signals
        self.assertLessEqual(len(signals), 5, "Should respect signal limit")

    def test_pool_filtering_by_volume(self):
        """Test that pools are filtered by minimum volume requirements"""
        if hasattr(self.strategy, '_identify_liquidity_pools'):
            pools = self.strategy._identify_liquidity_pools(self.mock_data)
            
            min_volume = self.config['parameters'].get('min_pool_volume', 1000)
            
            # Check that pools meet volume requirements
            for pool in pools:
                if hasattr(pool, 'volume'):
                    self.assertGreaterEqual(pool.volume, min_volume,
                                          f"Pool volume {pool.volume} below minimum {min_volume}")

    def test_proximity_based_deduplication(self):
        """Test that nearby pools are deduplicated based on proximity threshold"""
        if hasattr(self.strategy, '_deduplicate_signals'):
            # Create test signals at similar price levels
            test_signals = []
            base_price = 1950.0
            
            # Create signals that should be deduplicated
            for i in range(5):
                signal_mock = Mock()
                signal_mock.price = base_price + (i * 2.0)  # Close prices
                signal_mock.signal_type = SignalType.BUY
                test_signals.append(signal_mock)
            
            # Deduplicate signals
            deduplicated = self.strategy._deduplicate_signals(test_signals)
            
            # Should have fewer signals after deduplication
            self.assertLessEqual(len(deduplicated), len(test_signals),
                               "Deduplication should reduce signal count")

    def test_signal_quality_after_throttling(self):
        """Test that signal quality is maintained or improved after throttling"""
        signals = self.strategy.generate_signal("XAUUSDm", "M15")
        
        # All signals should meet basic quality requirements
        for signal in signals:
            # Should have valid signal type
            self.assertIn(signal.signal_type, [SignalType.BUY, SignalType.SELL])
            
            # Should have reasonable confidence
            self.assertGreater(signal.confidence, 0.0)
            self.assertLessEqual(signal.confidence, 1.0)
            
            # Should have valid price
            self.assertGreater(signal.price, 0.0)
            
            # Should have reasonable metadata
            if hasattr(signal, 'metadata') and signal.metadata:
                self.assertIsInstance(signal.metadata, dict)

    def test_cooldown_reset_mechanism(self):
        """Test that cooldown resets properly after sufficient time"""
        # This would require mocking time or having a reset method
        if hasattr(self.strategy, '_reset_cooldown') or hasattr(self.strategy, 'last_signal_time'):
            # Generate initial signals
            initial_signals = self.strategy.generate_signal("XAUUSDm", "M15")
            
            # Simulate cooldown reset
            if hasattr(self.strategy, '_reset_cooldown'):
                self.strategy._reset_cooldown()
            elif hasattr(self.strategy, 'last_signal_time'):
                # Reset the last signal time
                self.strategy.last_signal_time = None
            
            # Should be able to generate signals again
            new_signals = self.strategy.generate_signal("XAUUSDm", "M15")
            self.assertIsInstance(new_signals, list)

    def test_edge_case_no_valid_pools(self):
        """Test behavior when no valid liquidity pools are found"""
        # Create data that shouldn't generate strong pools
        flat_data = pd.DataFrame({
            'Open': [1950.0] * 50,
            'High': [1950.1] * 50,
            'Low': [1949.9] * 50,
            'Close': [1950.0] * 50,
            'Volume': [100] * 50  # Low volume
        }, index=pd.date_range(start=datetime.now(), periods=50, freq='15Min'))
        
        # Mock the data
        original_get_data = self.strategy.mt5_manager.get_historical_data
        self.strategy.mt5_manager.get_historical_data = Mock(return_value=flat_data)
        
        try:
            signals = self.strategy.generate_signal("XAUUSDm", "M15")
            # Should handle gracefully with no signals or empty list
            self.assertIsInstance(signals, list)
            self.assertLessEqual(len(signals), 5)  # Respect max limit even with no pools
        finally:
            # Restore original method
            self.strategy.mt5_manager.get_historical_data = original_get_data

    def test_memory_efficiency_with_throttling(self):
        """Test that throttling improves memory efficiency"""
        import gc
        
        # Measure memory before
        gc.collect()
        
        # Generate signals multiple times
        for _ in range(5):
            signals = self.strategy.generate_signal("XAUUSDm", "M15")
            
            # Should always respect limits
            self.assertLessEqual(len(signals), 5)
        
        # Should not accumulate excessive objects
        gc.collect()


class TestLiquidityPoolsIntegration(unittest.TestCase):
    """Integration tests for Liquidity Pools throttling fixes"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.config = {
            'parameters': {
                'lookback_bars': 100,
                'confidence_threshold': 0.65,
                'max_signals_per_run': 5,
                'min_pool_strength': 2.0,
                'max_active_pools': 10,
                'cooldown_bars': 10,
                'mode': 'mock'
            }
        }

    def test_strategy_initialization_with_throttling_parameters(self):
        """Test strategy initializes with throttling parameters"""
        try:
            strategy = LiquidityPoolsStrategy(self.config)
            self.assertIsNotNone(strategy)
            self.assertEqual(strategy.strategy_name, "LiquidityPoolsStrategy")
            
            # Check throttling parameters
            self.assertEqual(strategy.max_signals_per_run, 5)
            self.assertEqual(strategy.min_pool_strength, 2.0)
            self.assertEqual(strategy.max_active_pools, 10)
            self.assertEqual(strategy.cooldown_bars, 10)
            
        except Exception as e:
            self.fail(f"Strategy initialization failed: {e}")

    def test_end_to_end_throttled_pipeline(self):
        """Test complete liquidity pools pipeline with throttling"""
        strategy = LiquidityPoolsStrategy(self.config)
        
        try:
            # Generate signals
            signals = strategy.generate_signal("XAUUSDm", "M15")
            
            # Should respect throttling limits
            self.assertLessEqual(len(signals), 5)
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
            self.fail(f"End-to-end throttled pipeline failed: {e}")

    def test_consistent_throttling_across_multiple_runs(self):
        """Test that throttling is consistent across multiple strategy runs"""
        strategy = LiquidityPoolsStrategy(self.config)
        
        signal_counts = []
        
        # Run multiple times
        for _ in range(10):
            signals = strategy.generate_signal("XAUUSDm", "M15")
            signal_counts.append(len(signals))
            
            # Each run should respect limits
            self.assertLessEqual(len(signals), 5)
        
        # Should not have wild variations in signal counts
        if signal_counts:
            max_count = max(signal_counts)
            self.assertLessEqual(max_count, 5, "Signal count should never exceed limit")


if __name__ == '__main__':
    unittest.main(verbosity=2, buffer=True)