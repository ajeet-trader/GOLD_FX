
"""
XGBoost Classifier Strategy - Machine Learning Trading Strategy
==============================================================

Advanced XGBoost-based trading strategy for XAUUSD signal generation.
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for module resolution when run as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

# ML imports with fallback
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost/Scikit-learn not available. Strategy will run in simulation mode.")

# Import base classes directly
from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade, StrategyPerformance

# Import CLI args for mode selection
try:
    from src.utils.cli_args import parse_mode, print_mode_banner
except Exception:
    def parse_mode(*_args, **_kwargs): # type: ignore
        return 'mock'
    def print_mode_banner(_mode): # type: ignore
        pass


class XGBoostClassifierStrategy(AbstractStrategy):
    """XGBoost-based trading strategy for signal classification"""
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """Initialize XGBoost classifier strategy - 8GB RAM optimized"""
        super().__init__(config, mt5_manager, database)
        
        # Determine mode (CLI overrides config)
        self.mode = parse_mode() or self.config.get('mode', 'mock')

        print_mode_banner(self.mode)
        
        # Create appropriate MT5 manager based on mode
        if self.mode == 'live' and mt5_manager is None:
            try:
                from src.core.mt5_manager import MT5Manager
                live_mgr = MT5Manager()
                if hasattr(live_mgr, 'connect') and live_mgr.connect():
                    self.mt5_manager = live_mgr
                    print("✅ Connected to live MT5")
                else:
                    print("⚠️  Failed to connect to live MT5, falling back to mock data")
                    self.mt5_manager = self._create_mock_mt5()
                    self.mode = 'mock'
            except ImportError:
                print("⚠️  MT5Manager not available, using mock data")
                self.mt5_manager = self._create_mock_mt5()
                self.mode = 'mock'
        elif self.mode == 'mock' or mt5_manager is None:
            self.mt5_manager = self._create_mock_mt5()
        else:
            self.mt5_manager = mt5_manager

        # self.strategy_name is already set by AbstractStrategy
        self.lookback_bars = self.config.get('parameters', {}).get('lookback_bars', 120)
        self.min_confidence = self.config.get('parameters', {}).get('min_confidence', 0.15)
        
        # XGBoost parameters - allow override from config
        default_xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 4,
            'learning_rate': 0.15,
            'n_estimators': 50,
            'random_state': 42,
            'max_leaves': 15,
            'tree_method': 'exact'
        }
        self.xgb_params = default_xgb_params
        self.xgb_params.update(self.config.get('parameters', {}).get('xgb_params', {}))
        
        # Memory optimization settings
        self.memory_cleanup_interval = self.config.get('parameters', {}).get('memory_cleanup_interval', 20)
        self.prediction_count = 0 # Tracks total prediction attempts
        self.max_training_samples = self.config.get('parameters', {}).get('max_training_samples', 800)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler() if XGBOOST_AVAILABLE else None
        self.label_encoder = LabelEncoder() if XGBOOST_AVAILABLE else None
        self.is_trained = False
        
        # Performance tracking (ML-specific)
        self.model_accuracy = 0.0
        
        # self.logger is handled by AbstractStrategy
        
        # Initialize model if XGBoost is available
        if XGBOOST_AVAILABLE:
            self._initialize_model()
        
        self.logger.info(f"{self.strategy_name} initialized (XGBoost available: {XGBOOST_AVAILABLE})")
    
    def _create_mock_mt5(self):
        """Create mock MT5 manager with mode-specific data"""
        class MockMT5Manager:
            def __init__(self, mode):
                self.mode = mode
                
            def get_historical_data(self, symbol, timeframe, bars):
                import pandas as pd
                import numpy as np
                from datetime import datetime, timedelta
                
                dates = pd.date_range(start=datetime.now() - timedelta(days=7), 
                                     end=datetime.now(), freq='15Min')[:bars]
                
                np.random.seed(42 if self.mode == 'mock' else 123)
                close_prices = (1950 if self.mode == 'mock' else 1975) + np.cumsum(np.random.randn(len(dates)) * 2)
                
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
        
        return MockMT5Manager(self.mode)

    def _initialize_model(self):
        """Initialize XGBoost model"""
        try:
            if not XGBOOST_AVAILABLE:
                return
            
            self.model = xgb.XGBClassifier(**self.xgb_params)
            
            # Ensure label_encoder is fitted with all possible labels
            # In a real scenario, this would fit on historical data or defined classes
            # For testing/init, ensure it can handle expected labels ('BUY', 'SELL', 'HOLD')
            try:
                self.label_encoder.fit(['BUY', 'SELL', 'HOLD'])
            except Exception as e:
                self.logger.warning(f"LabelEncoder fit failed during init (might be empty): {e}")

            self.logger.info("XGBoost model initialized")
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}", exc_info=True)
            self.model = None
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """Generate trading signals using XGBoost classifier - 8GB RAM optimized"""
        signals = []
        try:
            self.logger.info(f"XGBoost Classifier - Analyzing {symbol} on {timeframe}")
            
            # Get market data with memory optimization
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available for signal generation.")
                return []
            
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_bars)
            if data is None or len(data) < self.lookback_bars:
                self.logger.warning(f"Insufficient data for XGBoost analysis: {len(data) if data is not None else 0}")
                return []
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.warning("Missing required data columns for XGBoost analysis.")
                return []
            
            # Memory cleanup
            self.prediction_count += 1 # Tracks prediction attempts
            if self.prediction_count % self.memory_cleanup_interval == 0:
                self._cleanup_memory()
            
            # Train model if not trained
            if not self.is_trained:
                self.logger.info("Training XGBoost model...")
                self._train_model(data)
            
            # Generate features for prediction
            features = self._extract_features(data)
            if features is None:
                self.logger.warning("Feature extraction failed for prediction.")
                return []
            
            # Make multiple predictions with different feature variations
            predictions = []
            if XGBOOST_AVAILABLE and self.model and self.is_trained:
                self.logger.info("Using XGBoost model for predictions")
                predictions = self._make_multiple_predictions(features, data)
            else:
                self.logger.info("Using fallback predictions (XGBoost not available or not trained)")
                predictions = self._make_fallback_predictions(data)
            
            self.logger.info(f"Generated {len(predictions)} prediction(s): {predictions}")
            
            # Convert predictions to signals
            self.logger.info(f"XGBoost predictions: {predictions}")
            self.logger.info(f"Min confidence threshold: {self.min_confidence}")
            
            for i, (prediction, confidence) in enumerate(predictions):
                self.logger.info(f"  Prediction {i+1}: {prediction} (confidence: {confidence:.3f})")
                if prediction == 'HOLD' or confidence < self.min_confidence:
                    self.logger.info(f"  Skipping prediction {i+1}: prediction={prediction}, confidence={confidence:.3f} < {self.min_confidence}")
                    continue
                
                signal_type = SignalType.BUY if prediction == 'BUY' else SignalType.SELL
                
                # Calculate risk parameters
                atr = self._calculate_atr(data)
                
                # Default ATR if calculation fails
                if atr is None:
                    atr = data['Close'].iloc[-1] * 0.01

                current_price = float(data['Close'].iloc[-1])
                risk_factor = 1.0 + (i * 0.2)  # Vary risk for different signals
                
                if signal_type == SignalType.BUY:
                    stop_loss = current_price - (0.5 * atr * risk_factor)  # Further reduced from 1.0x
                    take_profit = current_price + (2.0 * atr * confidence)   # Increased from 1.5x
                else:
                    stop_loss = current_price + (0.5 * atr * risk_factor)  # Further reduced from 1.0x
                    take_profit = current_price - (2.0 * atr * confidence)   # Increased from 1.5x
                
                # Validate risk-reward ratio
                if signal_type == SignalType.BUY:
                    risk = current_price - stop_loss
                    reward = take_profit - current_price
                else:
                    risk = stop_loss - current_price
                    reward = current_price - take_profit
                
                ratio = reward/risk if risk > 0 else float('inf')
                self.logger.info(f"  Risk-Reward validation: risk={risk:.5f}, reward={reward:.5f}, ratio={ratio:.2f}")
                
                if risk > 0 and reward > 0 and (reward / risk) >= 0.5:  # Further reduced from 0.8 to 0.5
                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name=self.strategy_name,
                        signal_type=signal_type,
                        confidence=confidence,
                        price=current_price,
                        timeframe=timeframe,
                        strength=confidence,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'signal_reason': f'xgboost_prediction_{i+1}',
                            'model_prediction': prediction,
                            'model_accuracy': self.model_accuracy,
                            'prediction_index': i
                        }
                    )
                    # Use base class validation: This appends to self.signal_history if valid
                    self.logger.info(f"  Created signal: {signal_type.value} at {current_price:.2f}, confidence: {confidence:.3f}")
                    if self.validate_signal(signal):
                        self.logger.info(f"  Signal validated successfully")
                        signals.append(signal) # Append to local list to be returned
                    else:
                        self.logger.info(f"  Signal validation failed")
                else:
                    ratio_text = f"{reward/risk:.2f}" if risk > 0 else "invalid"
                    self.logger.info(f"  Risk-reward ratio insufficient: {ratio_text} < 0.5")
            
            # Print results like technical strategies
            if signals:
                avg_confidence = np.mean([s.confidence for s in signals])
                self.logger.info(f"Generated {len(signals)} signals (avg confidence: {avg_confidence:.2f})")
                for signal in signals:
                    self.logger.info(f"      - {signal.signal_type.value} at {signal.price:.2f} (conf: {signal.confidence:.2f})")
            else:
                self.logger.info("No valid signals generated")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}", exc_info=True)
            return []
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze market conditions and XGBoost model performance"""
        try:
            return {
                'strategy': self.strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'xgboost_available': XGBOOST_AVAILABLE,
                'model_trained': self.is_trained,
                'model_accuracy': self.model_accuracy,
                'predictions_attempted': self.prediction_count, # Renamed for consistency
                'memory_optimized': True,
                'lookback_bars': self.lookback_bars,
                'max_training_samples': self.max_training_samples,
                'latest_training_time': self.performance.last_signal_time.isoformat() if self.performance.last_signal_time else 'N/A'
            }
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            return {'strategy': self.strategy_name, 'error': str(e)}

    def _extract_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features for ML model"""
        try:
            if len(data) < 50:
                self.logger.warning("Insufficient data for feature extraction.")
                return None
            
            close = data['Close']
            volume = data['Volume']
            
            features = []
            
            # Technical indicators
            sma_10 = close.rolling(min(10, len(close))).mean().iloc[-1]
            sma_20 = close.rolling(min(20, len(close))).mean().iloc[-1]
            sma_50 = close.rolling(min(50, len(close))).mean().iloc[-1]
            
            if pd.isna(sma_10) or pd.isna(sma_20) or pd.isna(sma_50):
                self.logger.warning("NaN in SMA calculations for feature extraction.")
                return None
            
            features.extend([
                close.iloc[-1] / sma_10 - 1,
                close.iloc[-1] / sma_20 - 1,
                close.iloc[-1] / sma_50 - 1,
                sma_10 / sma_20 - 1,
                sma_20 / sma_50 - 1
            ])
            
            # RSI
            rsi = self._calculate_rsi(close)
            features.append(rsi / 100.0)
            
            # Returns
            returns_1 = close.pct_change(1).iloc[-1]
            returns_5 = close.pct_change(5).iloc[-1]
            returns_10 = close.pct_change(10).iloc[-1]
            
            features.extend([
                returns_1 if not pd.isna(returns_1) else 0,
                returns_5 if not pd.isna(returns_5) else 0,
                returns_10 if not pd.isna(returns_10) else 0
            ])
            
            # Volatility
            volatility = close.rolling(min(20, len(close))).std().iloc[-1] if len(close) >= 20 else 0
            features.append(volatility / close.iloc[-1] if not pd.isna(volatility) else 0)
            
            # Volume features
            vol_sma = volume.rolling(min(20, len(volume))).mean().iloc[-1] if len(volume) >= 20 else 0
            vol_ratio = volume.iloc[-1] / vol_sma - 1 if not pd.isna(vol_sma) and vol_sma > 0 else 0
            features.append(vol_ratio)
            
            # ATR
            atr = self._calculate_atr(data)
            features.append(atr / close.iloc[-1] if atr is not None else 0) # Handle None from ATR
            
            # Convert to numpy array
            feature_array = np.array(features).reshape(1, -1)
            
            # Scale features if scaler is available and fitted
            if self.scaler and XGBOOST_AVAILABLE and hasattr(self.scaler, 'scale_') and self.is_trained:
                try:
                    feature_array = self.scaler.transform(feature_array)
                except Exception as e:
                    self.logger.warning(f"Scaler transform failed during feature extraction: {e}. Possible unfitted scaler or incompatible features.")
                    pass
            elif self.scaler and XGBOOST_AVAILABLE and not hasattr(self.scaler, 'scale_'):
                self.logger.warning("Scaler not fitted yet for feature extraction, returning unscaled features.")

            return feature_array
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}", exc_info=True)
            return None
    
    def _train_model(self, data: pd.DataFrame):
        """Train the XGBoost model"""
        try:
            if not XGBOOST_AVAILABLE:
                self.logger.warning("XGBoost not available, skipping training.")
                return
            
            self.logger.info("Training XGBoost model...")
            
            # Prepare training data
            X, y = self._prepare_training_data(data)
            
            if X is None or len(X) < 50:
                self.logger.warning("Insufficient or invalid training data for XGBoost.")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.xgb_params.get('random_state', 42)
            )
            
            # Scale features (fit only on training data)
            if self.scaler is None: # Initialize scaler if not already
                self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            self.model_accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            self.logger.info(f"Model trained. Accuracy: {self.model_accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}", exc_info=True)
            self.is_trained = False
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data with features and labels"""
        try:
            if len(data) < 100:
                self.logger.warning("Insufficient data for training data preparation.")
                return None, None
            
            X_list = []
            y_list = []
            
            buy_count = 0
            sell_count = 0
            hold_count = 0
            
            # Generate features and labels for training
            # Ensure window_data is large enough for future price, and past 50 bars for features
            for i in range(50, len(data) - 5): 
                window_data = data.iloc[i-50:i]
                features = self._extract_simple_features(window_data)
                
                if features is None:
                    continue
                
                # Generate label based on future price movement
                current_price = data['Close'].iloc[i]
                future_price = data['Close'].iloc[i+5] # Predict 5 bars into the future
                price_change = (future_price - current_price) / current_price
                
                # Reduced thresholds for more signal generation - 0.05% instead of 0.2%
                if price_change > 0.0005:
                    label = 'BUY'
                    buy_count += 1
                elif price_change < -0.0005:
                    label = 'SELL'
                    sell_count += 1
                else:
                    label = 'HOLD'
                    hold_count += 1
                
                X_list.append(features)
                y_list.append(label)
            
            self.logger.info(f"Training data labels: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}")
            
            if len(X_list) == 0:
                self.logger.warning("No training samples generated during data preparation.")
                return None, None
            
            X = np.array(X_list)
            
            # Ensure label_encoder is fitted before transforming
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
            if not hasattr(self.label_encoder, 'classes_'):
                self.label_encoder.fit(['BUY', 'SELL', 'HOLD']) # Fit if not already fitted
            y = self.label_encoder.transform(y_list)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed: {e}", exc_info=True)
            return None, None
    
    def _extract_simple_features(self, data: pd.DataFrame) -> Optional[List[float]]:
        """Extract simplified features for training - match main extraction"""
        try:
            if len(data) < 50:
                return None
            
            close = data['Close']
            volume = data['Volume']
            
            features = []
            
            # Moving averages
            sma_10 = close.rolling(min(10, len(close))).mean().iloc[-1]
            sma_20 = close.rolling(min(20, len(close))).mean().iloc[-1]
            sma_50 = close.rolling(min(50, len(close))).mean().iloc[-1]
            
            if any(pd.isna(val) for val in [sma_10, sma_20, sma_50]):
                return None
            
            features.extend([
                close.iloc[-1] / sma_10 - 1,
                close.iloc[-1] / sma_20 - 1,
                close.iloc[-1] / sma_50 - 1,
                sma_10 / sma_20 - 1,
                sma_20 / sma_50 - 1
            ])
            
            # RSI
            rsi = self._calculate_rsi(close)
            features.append(rsi / 100.0)
            
            # Returns
            returns_1 = close.pct_change(1).iloc[-1]
            returns_5 = close.pct_change(5).iloc[-1]
            returns_10 = close.pct_change(10).iloc[-1]
            
            features.extend([
                returns_1 if not pd.isna(returns_1) else 0,
                returns_5 if not pd.isna(returns_5) else 0,
                returns_10 if not pd.isna(returns_10) else 0
            ])
            
            # Volatility
            volatility = close.rolling(min(20, len(close))).std().iloc[-1] if len(close) >= 20 else 0
            features.append(volatility / close.iloc[-1] if not pd.isna(volatility) else 0)
            
            # Volume features
            vol_sma = volume.rolling(min(20, len(volume))).mean().iloc[-1] if len(volume) >= 20 else 0
            features.append(volume.iloc[-1] / vol_sma - 1 if not pd.isna(vol_sma) and vol_sma > 0 else 0)
            
            # ATR
            atr = self._calculate_atr(data)
            features.append(atr / close.iloc[-1] if atr is not None else 0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Simple feature extraction for training failed: {e}", exc_info=True)
            return None
    
    def _make_prediction(self, features: np.ndarray) -> Tuple[str, float]:
        """Make prediction using trained model"""
        try:
            if not self.is_trained or self.model is None:
                self.logger.warning("Model not trained or available for prediction, using HOLD fallback.")
                return 'HOLD', 0.5
            
            probabilities = self.model.predict_proba(features)[0]
            classes = self.label_encoder.classes_ if self.label_encoder and hasattr(self.label_encoder, 'classes_') else ['HOLD', 'BUY', 'SELL']
            
            self.logger.info(f"    Prediction probabilities: {dict(zip(classes, probabilities))}")
            
            # Create probability mapping
            prob_dict = dict(zip(classes, probabilities))
            
            # Generate signals based on individual class probabilities
            # Check if BUY or SELL probabilities exceed threshold
            buy_prob = prob_dict.get('BUY', 0.0)
            sell_prob = prob_dict.get('SELL', 0.0)
            hold_prob = prob_dict.get('HOLD', 1.0)
            
            # Signal threshold - generate signal if BUY/SELL probability > 15%
            signal_threshold = 0.15
            
            if buy_prob > signal_threshold and buy_prob > sell_prob:
                prediction = 'BUY'
                confidence = float(buy_prob)
            elif sell_prob > signal_threshold and sell_prob > buy_prob:
                prediction = 'SELL'
                confidence = float(sell_prob)
            else:
                prediction = 'HOLD'
                confidence = float(hold_prob)
            
            self.logger.info(f"    Final prediction: {prediction} with confidence: {confidence:.3f}")
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            return 'HOLD', 0.5
    
    def _make_fallback_prediction(self, data: pd.DataFrame) -> Tuple[str, float]:
        """Make simple prediction when ML model is not available or trained"""
        try:
            close = data['Close']
            
            if len(close) < 20:
                self.logger.warning("Insufficient data for fallback prediction.")
                return 'HOLD', 0.5
            
            sma_10 = close.rolling(min(10, len(close))).mean().iloc[-1]
            sma_20 = close.rolling(min(20, len(close))).mean().iloc[-1]
            rsi = self._calculate_rsi(close)
            
            if not pd.isna(sma_10) and not pd.isna(sma_20):
                if sma_10 > sma_20 * 1.001:
                    return 'BUY', 0.6
                elif sma_10 < sma_20 * 0.999:
                    return 'SELL', 0.6
            
            return 'HOLD', 0.5
            
        except Exception as e:
            self.logger.error(f"Fallback prediction failed: {e}", exc_info=True)
            return 'HOLD', 0.5
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50.0
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Avoid division by zero for rs
            rs = gain / loss
            # Replace inf with large number, or NaN with 0, then calculate RSI
            rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0) # Handle inf values in rs calculation
            
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except Exception as e:
            self.logger.error(f"RSI calculation failed: {e}", exc_info=True)
            return 50.0
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate Average True Range"""
        try:
            if len(data) < period + 1:
                return None # Not enough data
            
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return float(atr) if not pd.isna(atr) else None
        except Exception as e:
            self.logger.error(f"ATR calculation failed: {e}", exc_info=True)
            return None
    
    def _make_multiple_predictions(self, features: np.ndarray, data: pd.DataFrame) -> List[Tuple[str, float]]:
        """Make multiple predictions with slight variations"""
        try:
            predictions = []
            
            # Primary prediction
            primary_pred, primary_conf = self._make_prediction(features)
            predictions.append((primary_pred, primary_conf))
            
            # Generate feature variations for additional predictions
            # Ensure features.shape[1] is valid for arithmetic
            if features.shape[1] > 0:
                for i in range(2):
                    varied_features = features.copy()
                    # Add noise, ensure noise dimension matches feature dimension
                    noise_factor = 0.02 * (i + 1)
                    varied_features[0] += np.random.normal(0, noise_factor, features.shape[1])
                    
                    pred, conf = self._make_prediction(varied_features)
                    conf *= (0.95 - i * 0.05)  # Reduce confidence for variations
                    predictions.append((pred, conf))
            
            return predictions[:3]  # Maximum 3 predictions
            
        except Exception as e:
            self.logger.error(f"Multiple predictions failed: {e}", exc_info=True)
            return [('HOLD', 0.5)]
    
    def _make_fallback_predictions(self, data: pd.DataFrame) -> List[Tuple[str, float]]:
        """Make multiple fallback predictions when XGBoost is not available"""
        try:
            predictions = []
            close = data['Close']
            
            if len(close) < 20:
                self.logger.warning("Insufficient data for fallback predictions.")
                return [('HOLD', 0.5)]
            
            sma_10 = close.rolling(min(10, len(close))).mean().iloc[-1]
            sma_20 = close.rolling(min(20, len(close))).mean().iloc[-1]
            rsi = self._calculate_rsi(close)
            
            if not pd.isna(sma_10) and not pd.isna(sma_20):
                # Primary signal
                if sma_10 > sma_20 * 1.0015 and rsi < 70:
                    predictions.append(('BUY', 0.65))
                elif sma_10 < sma_20 * 0.9985 and rsi > 30:
                    predictions.append(('SELL', 0.65))
                
                # Secondary signal with different thresholds
                if sma_10 > sma_20 * 1.001 and rsi < 75:
                    predictions.append(('BUY', 0.62))
                elif sma_10 < sma_20 * 0.999 and rsi > 25:
                    predictions.append(('SELL', 0.62))
            
            return predictions if predictions else [('HOLD', 0.5)]
            
        except Exception as e:
            self.logger.error(f"Fallback predictions failed: {e}", exc_info=True)
            return [('HOLD', 0.5)]
    
    def _cleanup_memory(self):
        """Clean up memory for 8GB RAM optimization"""
        try:
            import gc
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info(f"Memory cleanup completed (prediction #{self.prediction_count})")
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {str(e)}", exc_info=True)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the XGBoost Classifier Strategy,
        including its parameters, ML-specific performance, and overall trading performance.
        """
        # Get overall trading performance from AbstractStrategy base class
        base_trading_performance = self.get_performance_summary()

        info = {
            'name': 'XGBoost Classifier Strategy',
            'version': '1.0.0', # Assuming this is its version based on typical ML strategy lifecycle
            'type': 'Machine Learning',
            'description': 'Advanced XGBoost-based trading strategy for XAUUSD signal generation.',
            'parameters': {
                'lookback_bars': self.lookback_bars,
                'min_confidence': self.min_confidence,
                'xgb_params': self.xgb_params, # Full XGBoost params
                'memory_cleanup_interval': self.memory_cleanup_interval,
                'max_training_samples': self.max_training_samples,
            },
            'ml_specific_metrics': {
                'xgboost_available': XGBOOST_AVAILABLE,
                'model_trained': self.is_trained,
                'model_accuracy': self.model_accuracy,
                'predictions_attempted': self.prediction_count,
            },
            'overall_trading_performance': {
                'total_signals_generated': base_trading_performance['total_signals'],
                'win_rate': base_trading_performance['win_rate'],
                'profit_factor': base_trading_performance['profit_factor']
            },
            'memory_optimized_flags': { # Explicitly state flags for memory
                'optimized_for_8gb_ram': True
            }
        }
        return info


# Testing function
if __name__ == "__main__":
    # Setup logging for the test environment
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test configuration
    test_config = {
        'parameters': { # Group parameters under 'parameters' key
            'confidence_threshold': 0.15,
            'lookback_bars': 120,
            'max_training_samples': 500, # Reduced for faster test
            'memory_cleanup_interval': 10, # Reduced for faster test
            'xgb_params': { # Specific XGBoost params override
                'n_estimators': 10, # Very small for fast test
                'max_depth': 3,     # Small for fast test
                'learning_rate': 0.2
            },
            'mode': 'mock' # Added mode parameter to test config
        }
    }
    
    # Create strategy instance
    strategy = XGBoostClassifierStrategy(test_config, mt5_manager=None, database=None) # Pass mt5_manager=None to trigger internal mock creation
    
    print("============================================================")
    print("TESTING MODIFIED XGBOOST CLASSIFIER STRATEGY")
    print("============================================================")
    print(f"Running in {strategy.mode.upper()} mode") # Print mode

    # 1. Testing signal generation
    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - Signal: {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
        if signal.metadata:
            print(f"     Reason: {signal.metadata.get('signal_reason', 'N/A')}")
    
    # 2. Testing analysis method
    print("\n2. Testing analysis method:")
    # Get mock data using the strategy's internal MT5 manager
    mock_data = strategy.mt5_manager.get_historical_data("XAUUSDm", "M15", 120)
    analysis_results = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis results keys: {analysis_results.keys()}")
    print(f"   XGBoost Available: {analysis_results.get('xgboost_available')}")
    print(f"   Model Trained: {analysis_results.get('model_trained')}")
    print(f"   Model Accuracy: {analysis_results.get('model_accuracy', 'N/A'):.3f}")
    
    # 3. Testing performance tracking (from AbstractStrategy)
    print("\n3. Testing performance tracking:")
    summary = strategy.get_performance_summary()
    print(f"   {summary}")
    
    # 4. Strategy Information (comprehensive)
    print("\n4. Strategy Information:")
    info = strategy.get_strategy_info()
    print(f"   Name: {info['name']}")
    print(f"   Version: {info['version']}")
    print(f"   Type: {info['type']}")
    print(f"   Description: {info['description']}")
    print(f"   Parameters:")
    for param, value in info['parameters'].items():
        if param == 'xgb_params':
            print(f"     - {param}: {value}")
        else:
            print(f"     - {param}: {value}")
    print(f"   ML Specific Metrics:")
    for key, value in info['ml_specific_metrics'].items():
        if 'accuracy' in key or 'confidence' in key:
            print(f"     - {key}: {value:.3f}" if isinstance(value, (float, int)) else f"     - {key}: {value}")
        else:
            print(f"     - {key}: {value}")
    print(f"   Overall Trading Performance:")
    print(f"     Total Signals Generated: {info['overall_trading_performance']['total_signals_generated']}")
    print(f"     Win Rate: {info['overall_trading_performance']['win_rate']:.2%}")
    print(f"     Profit Factor: {info['overall_trading_performance']['profit_factor']:.2f}")

    print("\n============================================================")
    print("XGBOOST CLASSIFIER STRATEGY TEST COMPLETED!")
    print("============================================================")
