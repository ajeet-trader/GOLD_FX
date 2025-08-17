#!/usr/bin/env python3
"""
Comprehensive Phase 2 Test Suite
================================

Complete test suite for all Phase 2 components:
- ML Strategies (XGBoost, Ensemble NN, RL Agent)
- Fusion Strategies (Weighted Voting, Confidence Sizing, Regime Detection, Adaptive Ensemble)
- Signal Engine Integration
- Phase 2 Core Integration
- Performance Testing
- Error Handling

Usage:
    python tests/Phase-2/test_phase2_complete.py
"""

import sys
import os
from pathlib import Path
import unittest
import logging
from datetime import datetime
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Test Results Tracking
test_results = {
    'ml_strategies': {},
    'fusion_strategies': {},
    'integration': {},
    'performance': {}
}


class TestMLStrategies(unittest.TestCase):
    """Test all ML strategies"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = {
            'name': 'test_strategy',
            'parameters': {
                'lookback_bars': 100,
                'sequence_length': 60,
                'confidence_threshold': 0.6
            }
        }
    
    def test_xgboost_classifier(self):
        """Test XGBoost Classifier Strategy"""
        print("\n" + "="*60)
        print("TESTING XGBOOST CLASSIFIER STRATEGY")
        print("="*60)
        
        try:
            from src.strategies.ml.xgboost_classifier import XGBoostClassifierStrategy
            
            # Initialize strategy
            strategy = XGBoostClassifierStrategy(self.config)
            self.assertIsNotNone(strategy)
            print("✅ XGBoost strategy initialization: PASSED")
            
            # Test signal generation
            signal = strategy.generate_signal("XAUUSDm", "M15")
            self.assertIsNotNone(signal)
            print(f"✅ Signal generation: PASSED - {signal.signal_type.value} (conf: {signal.confidence:.3f})")
            
            test_results['ml_strategies']['xgboost_classifier'] = 'PASSED'
            
        except Exception as e:
            print(f"❌ XGBoost test failed: {e}")
            test_results['ml_strategies']['xgboost_classifier'] = f'FAILED: {e}'
            raise
    
    def test_ensemble_nn(self):
        """Test Ensemble Neural Network Strategy"""
        print("\n" + "="*60)
        print("TESTING ENSEMBLE NEURAL NETWORK STRATEGY")
        print("="*60)
        
        try:
            from src.strategies.ml.ensemble_nn import EnsembleNNStrategy
            
            # Initialize strategy
            strategy = EnsembleNNStrategy(self.config)
            self.assertIsNotNone(strategy)
            print("✅ Ensemble NN strategy initialization: PASSED")
            
            # Test signal generation
            signal = strategy.generate_signal("XAUUSDm", "M15")
            self.assertIsNotNone(signal)
            print(f"✅ Signal generation: PASSED - {signal.signal_type.value} (conf: {signal.confidence:.3f})")
            
            test_results['ml_strategies']['ensemble_nn'] = 'PASSED'
            
        except Exception as e:
            print(f"❌ Ensemble NN test failed: {e}")
            test_results['ml_strategies']['ensemble_nn'] = f'FAILED: {e}'
            raise
    
    def test_rl_agent(self):
        """Test Reinforcement Learning Agent Strategy"""
        print("\n" + "="*60)
        print("TESTING REINFORCEMENT LEARNING AGENT STRATEGY")
        print("="*60)
        
        try:
            from src.strategies.ml.rl_agent import RLAgentStrategy
            
            # Initialize strategy
            strategy = RLAgentStrategy(self.config)
            self.assertIsNotNone(strategy)
            print("✅ RL Agent strategy initialization: PASSED")
            
            # Test signal generation
            signal = strategy.generate_signal("XAUUSDm", "M15")
            self.assertIsNotNone(signal)
            print(f"✅ Signal generation: PASSED - {signal.signal_type.value} (conf: {signal.confidence:.3f})")
            
            test_results['ml_strategies']['rl_agent'] = 'PASSED'
            
        except Exception as e:
            print(f"❌ RL Agent test failed: {e}")
            test_results['ml_strategies']['rl_agent'] = f'FAILED: {e}'
            raise


class TestFusionStrategies(unittest.TestCase):
    """Test all fusion strategies"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = {
            'name': 'test_fusion',
            'parameters': {
                'lookback_bars': 100,
                'performance_window': 30,
                'confidence_threshold': 0.6
            }
        }
    
    def test_weighted_voting(self):
        """Test Weighted Voting Fusion Strategy"""
        print("\n" + "="*60)
        print("TESTING WEIGHTED VOTING FUSION STRATEGY")
        print("="*60)
        
        try:
            from src.strategies.fusion.weighted_voting import WeightedVotingFusionStrategy
            
            # Initialize strategy
            strategy = WeightedVotingFusionStrategy(self.config)
            self.assertIsNotNone(strategy)
            print("✅ Weighted Voting strategy initialization: PASSED")
            
            # Test signal generation
            signal = strategy.generate_signal("XAUUSDm", "M15")
            self.assertIsNotNone(signal)
            print(f"✅ Signal generation: PASSED - {signal.signal_type.value} (conf: {signal.confidence:.3f})")
            
            test_results['fusion_strategies']['weighted_voting'] = 'PASSED'
            
        except Exception as e:
            print(f"❌ Weighted Voting test failed: {e}")
            test_results['fusion_strategies']['weighted_voting'] = f'FAILED: {e}'
            raise
    
    def test_confidence_sizing(self):
        """Test Confidence Sizing Fusion Strategy"""
        print("\n" + "="*60)
        print("TESTING CONFIDENCE SIZING FUSION STRATEGY")
        print("="*60)
        
        try:
            from src.strategies.fusion.confidence_sizing import ConfidenceSizingFusionStrategy
            
            # Initialize strategy
            strategy = ConfidenceSizingFusionStrategy(self.config)
            self.assertIsNotNone(strategy)
            print("✅ Confidence Sizing strategy initialization: PASSED")
            
            # Test signal generation
            signal = strategy.generate_signal("XAUUSDm", "M15")
            self.assertIsNotNone(signal)
            print(f"✅ Signal generation: PASSED - {signal.signal_type.value} (conf: {signal.confidence:.3f})")
            
            test_results['fusion_strategies']['confidence_sizing'] = 'PASSED'
            
        except Exception as e:
            print(f"❌ Confidence Sizing test failed: {e}")
            test_results['fusion_strategies']['confidence_sizing'] = f'FAILED: {e}'
            raise
    
    def test_regime_detection(self):
        """Test Regime Detection Fusion Strategy"""
        print("\n" + "="*60)
        print("TESTING REGIME DETECTION FUSION STRATEGY")
        print("="*60)
        
        try:
            from src.strategies.fusion.regime_detection import RegimeDetectionFusionStrategy
            
            # Initialize strategy
            strategy = RegimeDetectionFusionStrategy(self.config)
            self.assertIsNotNone(strategy)
            print("✅ Regime Detection strategy initialization: PASSED")
            
            # Test signal generation
            signal = strategy.generate_signal("XAUUSDm", "M15")
            self.assertIsNotNone(signal)
            print(f"✅ Signal generation: PASSED - {signal.signal_type.value} (conf: {signal.confidence:.3f})")
            
            test_results['fusion_strategies']['regime_detection'] = 'PASSED'
            
        except Exception as e:
            print(f"❌ Regime Detection test failed: {e}")
            test_results['fusion_strategies']['regime_detection'] = f'FAILED: {e}'
            raise
    
    def test_adaptive_ensemble(self):
        """Test Adaptive Ensemble Fusion Strategy"""
        print("\n" + "="*60)
        print("TESTING ADAPTIVE ENSEMBLE FUSION STRATEGY")
        print("="*60)
        
        try:
            from src.strategies.fusion.adaptive_ensemble import AdaptiveEnsembleFusionStrategy
            
            # Initialize strategy
            strategy = AdaptiveEnsembleFusionStrategy(self.config)
            self.assertIsNotNone(strategy)
            print("✅ Adaptive Ensemble strategy initialization: PASSED")
            
            # Test signal generation
            signal = strategy.generate_signal("XAUUSDm", "M15")
            self.assertIsNotNone(signal)
            print(f"✅ Signal generation: PASSED - {signal.signal_type.value} (conf: {signal.confidence:.3f})")
            
            # Test weight adaptation
            weights = strategy.get_strategy_weights()
            self.assertIsInstance(weights, dict)
            print(f"✅ Weight adaptation: PASSED - {len(weights)} strategies tracked")
            
            test_results['fusion_strategies']['adaptive_ensemble'] = 'PASSED'
            
        except Exception as e:
            print(f"❌ Adaptive Ensemble test failed: {e}")
            test_results['fusion_strategies']['adaptive_ensemble'] = f'FAILED: {e}'
            raise


class TestSignalEngineIntegration(unittest.TestCase):
    """Test Signal Engine integration with all strategies"""
    
    def test_signal_engine_loading(self):
        """Test Signal Engine loads all strategies correctly"""
        print("\n" + "="*60)
        print("TESTING SIGNAL ENGINE INTEGRATION")
        print("="*60)
        
        try:
            from src.core.signal_engine import SignalEngine
            
            # Test configuration
            config = {
                'strategies': {
                    'technical': {
                        'active_strategies': ['ichimoku', 'harmonic'],
                        'ichimoku': {'tenkan_period': 9},
                        'harmonic': {'min_confidence': 0.7}
                    },
                    'smc': {
                        'active_strategies': ['order_blocks', 'market_structure'],
                        'order_blocks': {'lookback': 50},
                        'market_structure': {'lookback_bars': 200}
                    },
                    'ml': {
                        'active_strategies': ['lstm', 'xgboost_classifier', 'ensemble_nn', 'rl_agent'],
                        'lstm': {'sequence_length': 60},
                        'xgboost_classifier': {'lookback_bars': 200},
                        'ensemble_nn': {'lookback_bars': 200},
                        'rl_agent': {'lookback_bars': 200}
                    },
                    'fusion': {
                        'active_strategies': ['weighted_voting', 'confidence_sizing', 'regime_detection', 'adaptive_ensemble'],
                        'weighted_voting': {'lookback_bars': 200},
                        'confidence_sizing': {'lookback_bars': 200},
                        'regime_detection': {'lookback_bars': 200},
                        'adaptive_ensemble': {'lookback_bars': 200}
                    }
                },
                'risk_management': {'risk_per_trade': 0.02},
                'signal_generation': {'max_signals_per_bar': 5}
            }
            
            # Initialize Signal Engine
            engine = SignalEngine(config, mt5_manager=None, database_manager=None)
            self.assertIsNotNone(engine)
            print("✅ Signal Engine initialization: PASSED")
            
            # Check loaded strategies
            active_strategies = engine.get_active_strategies()
            self.assertIsInstance(active_strategies, dict)
            
            total_strategies = sum(len(strategies) for strategies in active_strategies.values())
            print(f"✅ Strategy loading: PASSED - {total_strategies} strategies loaded")
            
            for category, strategies in active_strategies.items():
                if strategies:
                    print(f"   {category.upper()}: {len(strategies)} strategies - {strategies}")
            
            # Test signal generation
            signals = engine.generate_signals("XAUUSDm", 15)
            self.assertIsInstance(signals, list)
            print(f"✅ Signal generation: PASSED - {len(signals)} signals generated")
            
            test_results['integration']['signal_engine'] = 'PASSED'
            
        except Exception as e:
            print(f"❌ Signal Engine integration test failed: {e}")
            test_results['integration']['signal_engine'] = f'FAILED: {e}'
            raise


class TestPhase2Integration(unittest.TestCase):
    """Test complete Phase 2 system integration"""
    
    def test_phase2_system_initialization(self):
        """Test Phase 2 system initialization"""
        print("\n" + "="*60)
        print("TESTING PHASE 2 SYSTEM INTEGRATION")
        print("="*60)
        
        try:
            from src.phase_2_core_integration import Phase2TradingSystem
            
            # Initialize system
            system = Phase2TradingSystem()
            system.set_mode('paper')
            self.assertIsNotNone(system)
            print("✅ Phase 2 system creation: PASSED")
            
            # Test initialization
            init_result = system.initialize()
            self.assertTrue(init_result)
            print("✅ Phase 2 system initialization: PASSED")
            
            # Test signal engine integration
            if system.signal_engine:
                active_strategies = system.signal_engine.get_active_strategies()
                total_strategies = sum(len(strategies) for strategies in active_strategies.values())
                print(f"✅ Integrated strategies: PASSED - {total_strategies} strategies active")
            
            # Test performance summary
            summary = system.get_performance_summary()
            self.assertIsInstance(summary, dict)
            print("✅ Performance tracking: PASSED")
            
            # Shutdown system
            system.shutdown()
            print("✅ System shutdown: PASSED")
            
            test_results['integration']['phase2_system'] = 'PASSED'
            
        except Exception as e:
            print(f"❌ Phase 2 system integration test failed: {e}")
            test_results['integration']['phase2_system'] = f'FAILED: {e}'
            raise


class TestPerformance(unittest.TestCase):
    """Test system performance and stress testing"""
    
    def test_signal_generation_performance(self):
        """Test signal generation performance"""
        print("\n" + "="*60)
        print("TESTING SIGNAL GENERATION PERFORMANCE")
        print("="*60)
        
        try:
            from src.core.signal_engine import SignalEngine
            import time
            
            # Minimal config for performance testing
            config = {
                'strategies': {
                    'ml': {
                        'active_strategies': ['xgboost_classifier', 'ensemble_nn'],
                        'xgboost_classifier': {'lookback_bars': 100},
                        'ensemble_nn': {'lookback_bars': 100}
                    },
                    'fusion': {
                        'active_strategies': ['weighted_voting'],
                        'weighted_voting': {'lookback_bars': 100}
                    }
                },
                'signal_generation': {'max_signals_per_bar': 5}
            }
            
            engine = SignalEngine(config)
            
            # Performance test - generate signals multiple times
            start_time = time.time()
            signal_count = 0
            
            for i in range(10):
                signals = engine.generate_signals("XAUUSDm", 15)
                signal_count += len(signals)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ Performance test: PASSED")
            print(f"   Generated {signal_count} signals in {duration:.2f} seconds")
            print(f"   Average: {duration/10:.3f} seconds per generation")
            print(f"   Rate: {signal_count/duration:.1f} signals/second")
            
            # Performance should be reasonable
            self.assertLess(duration, 30.0)  # Should complete in under 30 seconds
            
            test_results['performance']['signal_generation'] = 'PASSED'
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            test_results['performance']['signal_generation'] = f'FAILED: {e}'
            raise


def print_test_summary():
    """Print comprehensive test summary"""
    print("\n" + "="*80)
    print("PHASE 2 COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    total_tests = 0
    passed_tests = 0
    
    for category, tests in test_results.items():
        print(f"\n{category.upper().replace('_', ' ')} TESTS:")
        print("-" * 40)
        
        for test_name, result in tests.items():
            total_tests += 1
            if result == 'PASSED':
                passed_tests += 1
                print(f"   ✅ {test_name}: {result}")
            else:
                print(f"   ❌ {test_name}: {result}")
    
    print(f"\n" + "="*80)
    print(f"OVERALL RESULTS: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Phase 2 implementation is complete and functional!")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. Review the failures above.")
    
    print("="*80)


def main():
    """Main test runner"""
    print("PHASE 2 COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Test started at: {datetime.now()}")
    print("=" * 80)
    
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add ML strategy tests
    test_suite.addTest(TestMLStrategies('test_xgboost_classifier'))
    test_suite.addTest(TestMLStrategies('test_ensemble_nn'))
    test_suite.addTest(TestMLStrategies('test_rl_agent'))
    
    # Add fusion strategy tests
    test_suite.addTest(TestFusionStrategies('test_weighted_voting'))
    test_suite.addTest(TestFusionStrategies('test_confidence_sizing'))
    test_suite.addTest(TestFusionStrategies('test_regime_detection'))
    test_suite.addTest(TestFusionStrategies('test_adaptive_ensemble'))
    
    # Add integration tests
    test_suite.addTest(TestSignalEngineIntegration('test_signal_engine_loading'))
    test_suite.addTest(TestPhase2Integration('test_phase2_system_initialization'))
    
    # Add performance tests
    test_suite.addTest(TestPerformance('test_signal_generation_performance'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
    
    try:
        result = runner.run(test_suite)
        
        # Print summary
        print_test_summary()
        
        # Return success/failure
        return len(result.failures) == 0 and len(result.errors) == 0
        
    except Exception as e:
        print(f"❌ Test suite execution failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)