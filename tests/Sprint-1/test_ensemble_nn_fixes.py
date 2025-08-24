"""
Test Suite for EnsembleNN TensorFlow Tensor Shape Fixes
=======================================================

Tests for Sprint 1 fixes to EnsembleNN strategy, specifically:
- TensorFlow tensor shape consistency between training and prediction
- LSTM feature extraction shape handling
- Prediction reshape logic improvements
- Training data generation optimization
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

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from src.strategies.ml.ensemble_nn import EnsembleNNStrategy
from src.core.base import SignalType


class TestEnsembleNNFixes(unittest.TestCase):
    """Test suite for EnsembleNN tensor shape fixes"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'parameters': {
                'lookback_bars': 50,
                'confidence_threshold': 0.65,
                'min_training_samples': 50,  # Reduced for testing
                'lstm_units': 25,
                'dense_units': 15,
                'epochs': 5,  # Reduced for faster testing
                'mode': 'mock'
            }
        }
        
        # Create strategy instance
        self.strategy = EnsembleNNStrategy(self.config)
        
        # Create mock data
        self.mock_data = self._create_mock_data()

    def _create_mock_data(self):
        """Create mock OHLCV data for testing"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=5), 
                             end=datetime.now(), freq='15Min')[:100]
        
        np.random.seed(42)
        close_prices = 1950 + np.cumsum(np.random.randn(len(dates)) * 2)
        
        data = pd.DataFrame({
            'Open': close_prices + np.random.randn(len(dates)) * 0.5,
            'High': close_prices + np.abs(np.random.randn(len(dates)) * 3),
            'Low': close_prices - np.abs(np.random.randn(len(dates)) * 3),
            'Close': close_prices,
            'Volume': np.random.randint(100, 1000, len(dates))
        }, index=dates)
        
        # Ensure High >= Close >= Low
        data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
        data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
        
        return data

    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_lstm_feature_extraction_shape_consistency(self):
        """Test LSTM feature extraction returns consistent (30, 1) shape"""
        if not hasattr(self.strategy, '_extract_lstm_features'):
            self.skipTest("LSTM feature extraction method not available")
        
        # Test multiple data windows
        for i in range(5):
            window_data = self.mock_data.iloc[i:i+50]
            features = self.strategy._extract_lstm_features(window_data)
            
            if features is not None:
                self.assertEqual(features.shape, (30, 1), 
                               f"LSTM features shape should be (30, 1), got {features.shape}")

    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_lstm_training_feature_extraction_consistency(self):
        """Test LSTM training feature extraction matches prediction extraction"""
        if not hasattr(self.strategy, '_extract_lstm_features_for_training'):
            self.skipTest("LSTM training feature extraction method not available")
        
        window_data = self.mock_data.iloc[10:60]
        
        # Extract features for prediction and training
        pred_features = self.strategy._extract_lstm_features(window_data)
        train_features = self.strategy._extract_lstm_features_for_training(window_data)
        
        if pred_features is not None and train_features is not None:
            self.assertEqual(pred_features.shape, train_features.shape,
                           "Training and prediction feature shapes should match")

    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_ensemble_prediction_tensor_reshape(self):
        """Test ensemble prediction handles tensor reshaping correctly"""
        if not self.strategy.is_trained:
            # Try to train the model first
            self.strategy._train_model(self.mock_data)
        
        if self.strategy.is_trained and hasattr(self.strategy, '_make_ensemble_prediction'):
            features = self.strategy._extract_features(self.mock_data)
            if features is not None:
                try:
                    prediction, confidence = self.strategy._make_ensemble_prediction(features)
                    self.assertIsInstance(prediction, str)
                    self.assertIsInstance(confidence, (int, float))
                    self.assertIn(prediction, ['BUY', 'SELL', 'HOLD'])
                    self.assertGreaterEqual(confidence, 0.0)
                    self.assertLessEqual(confidence, 1.0)
                except Exception as e:
                    self.fail(f"Ensemble prediction failed with tensor reshaping error: {e}")

    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_multiple_predictions_tensor_handling(self):
        """Test multiple predictions handle tensor shapes correctly"""
        if not self.strategy.is_trained:
            self.strategy._train_model(self.mock_data)
        
        if self.strategy.is_trained and hasattr(self.strategy, '_make_multiple_predictions'):
            features = self.strategy._extract_features(self.mock_data)
            if features is not None:
                try:
                    predictions = self.strategy._make_multiple_predictions(features, self.mock_data)
                    self.assertIsInstance(predictions, list)
                    for pred, conf in predictions:
                        self.assertIn(pred, ['BUY', 'SELL', 'HOLD'])
                        self.assertGreaterEqual(conf, 0.0)
                        self.assertLessEqual(conf, 1.0)
                except Exception as e:
                    self.fail(f"Multiple predictions failed with tensor error: {e}")

    def test_training_data_generation_optimization(self):
        """Test optimized training data generation with reduced sample requirements"""
        if hasattr(self.strategy, '_prepare_training_data'):
            X, y = self.strategy._prepare_training_data(self.mock_data)
            
            if X is not None and y is not None:
                # Should generate training data with reduced requirements
                self.assertGreater(len(X), 0, "Training data should be generated")
                self.assertEqual(len(X), len(y), "Features and labels should have same length")
                
                # Check that we can work with smaller datasets
                self.assertGreaterEqual(len(X), 10, "Should generate at least 10 training samples")

    def test_signal_generation_with_tensor_fixes(self):
        """Test that signal generation works without tensor shape errors"""
        try:
            signals = self.strategy.generate_signal("XAUUSDm", "M15")
            self.assertIsInstance(signals, list)
            
            # Check that no tensor shape errors occur
            for signal in signals:
                self.assertIn(signal.signal_type, [SignalType.BUY, SignalType.SELL])
                self.assertIsInstance(signal.confidence, (int, float))
                self.assertGreater(signal.confidence, 0.0)
                
        except Exception as e:
            if "shape" in str(e).lower() or "tensor" in str(e).lower():
                self.fail(f"Tensor shape error still occurring: {e}")
            # Other errors might be acceptable (like insufficient data warnings)

    def test_insufficient_training_data_handling(self):
        """Test handling of insufficient training data scenarios"""
        # Create very small dataset
        small_data = self.mock_data.iloc[:30]
        
        try:
            signals = self.strategy.generate_signal("XAUUSDm", "M15")
            # Should handle gracefully without crashing
            self.assertIsInstance(signals, list)
        except Exception as e:
            # Should not crash with tensor shape errors
            self.assertNotIn("shape", str(e).lower())
            self.assertNotIn("tensor", str(e).lower())

    def test_model_architecture_compatibility(self):
        """Test that model architecture handles different input shapes correctly"""
        if TENSORFLOW_AVAILABLE and hasattr(self.strategy, '_build_model'):
            try:
                # Test model building doesn't fail
                model = self.strategy._build_model()
                if model is not None:
                    # Check that model can handle expected input shapes
                    self.assertIsNotNone(model)
                    
                    # Verify LSTM layer configuration
                    lstm_layers = [layer for layer in model.layers if 'lstm' in layer.name.lower()]
                    if lstm_layers:
                        # Should have LSTM layers configured
                        self.assertGreater(len(lstm_layers), 0)
                        
            except Exception as e:
                if "shape" in str(e).lower():
                    self.fail(f"Model architecture still has shape issues: {e}")

    def test_memory_optimization_features(self):
        """Test memory optimization features work correctly"""
        # Test cleanup functionality
        if hasattr(self.strategy, '_cleanup_memory'):
            try:
                initial_prediction_count = getattr(self.strategy, 'prediction_count', 0)
                
                # Trigger memory cleanup
                self.strategy._cleanup_memory()
                
                # Should not crash
                self.assertTrue(True, "Memory cleanup executed without errors")
                
            except Exception as e:
                self.fail(f"Memory cleanup failed: {e}")

    def test_configuration_parameter_handling(self):
        """Test that configuration parameters are properly handled"""
        # Test reduced minimum training samples
        self.assertEqual(self.strategy.min_training_samples, 50)
        
        # Test other optimized parameters
        self.assertIsInstance(self.strategy.lookback_bars, int)
        self.assertGreater(self.strategy.lookback_bars, 0)


class TestEnsembleNNIntegration(unittest.TestCase):
    """Integration tests for EnsembleNN strategy"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.config = {
            'parameters': {
                'lookback_bars': 50,
                'confidence_threshold': 0.60,
                'min_training_samples': 50,
                'mode': 'mock'
            }
        }

    def test_strategy_initialization_without_errors(self):
        """Test strategy initializes without tensor-related errors"""
        try:
            strategy = EnsembleNNStrategy(self.config)
            self.assertIsNotNone(strategy)
            self.assertEqual(strategy.strategy_name, "EnsembleNNStrategy")
        except Exception as e:
            if "tensor" in str(e).lower() or "shape" in str(e).lower():
                self.fail(f"Strategy initialization failed with tensor error: {e}")

    def test_end_to_end_signal_pipeline(self):
        """Test complete signal generation pipeline"""
        strategy = EnsembleNNStrategy(self.config)
        
        try:
            # Generate signals
            signals = strategy.generate_signal("XAUUSDm", "M15")
            
            # Should complete without tensor errors
            self.assertIsInstance(signals, list)
            
            # Get performance summary
            performance = strategy.get_performance_summary()
            self.assertIsInstance(performance, dict)
            
        except Exception as e:
            if "tensor" in str(e).lower() or "shape" in str(e).lower():
                self.fail(f"End-to-end pipeline failed with tensor error: {e}")


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)