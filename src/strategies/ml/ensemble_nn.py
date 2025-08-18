from __future__ import annotations

"""
Ensemble Neural Network Strategy - Advanced ML Trading Strategy
===============================================================

Multi-model ensemble neural network for XAUUSD trading signal generation.
Combines multiple neural network architectures for robust predictions.
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for module resolution when run as script
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

# ML imports with fallback
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Strategy will run in simulation mode.")

# Import base classes directly
from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade, StrategyPerformance


class EnsembleNNStrategy(AbstractStrategy):
    """Ensemble Neural Network strategy combining multiple NN architectures"""
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """Initialize ensemble neural network strategy - 8GB RAM optimized"""
        super().__init__(config, mt5_manager, database)
        
        # self.strategy_name is already set by AbstractStrategy
        self.lookback_bars = self.config.get('parameters', {}).get('lookback_bars', 150)
        self.sequence_length = self.config.get('parameters', {}).get('sequence_length', 20)
        self.min_confidence = self.config.get('parameters', {}).get('min_confidence', 0.65)
        self.ensemble_size = self.config.get('parameters', {}).get('ensemble_size', 2)
        
        # Model parameters - 8GB RAM optimized
        self.epochs = self.config.get('parameters', {}).get('epochs', 20)
        self.batch_size = self.config.get('parameters', {}).get('batch_size', 8)
        self.learning_rate = self.config.get('parameters', {}).get('learning_rate', 0.001)
        
        # Memory optimization settings
        self.max_training_samples = self.config.get('parameters', {}).get('max_training_samples', 800)
        self.memory_cleanup_interval = self.config.get('parameters', {}).get('memory_cleanup_interval', 30)
        self.prediction_count = 0 # Tracks total prediction attempts
        
        # Model components
        self.models = []
        self.scaler = StandardScaler() if TENSORFLOW_AVAILABLE else None
        self.label_encoder = LabelEncoder() if TENSORFLOW_AVAILABLE else None
        self.is_trained = False
        
        # Performance tracking (ML-specific)
        self.ensemble_accuracy = 0.0
        
        # self.logger is handled by AbstractStrategy
        
        # Initialize models if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            self._initialize_models()
        
        self.logger.info(f"{self.strategy_name} initialized (TensorFlow available: {TENSORFLOW_AVAILABLE})")
    
    def _initialize_models(self):
        """Initialize ensemble of neural network models"""
        try:
            if not TENSORFLOW_AVAILABLE:
                return
            
            # Ensure label_encoder is fitted with all possible labels
            # In a real scenario, this would fit on historical data or defined classes
            # For testing/init, ensure it can handle expected labels ('BUY', 'SELL', 'HOLD')
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
            try:
                self.label_encoder.fit(['BUY', 'SELL', 'HOLD'])
            except Exception as e:
                self.logger.warning(f"LabelEncoder fit failed during init (might be empty or already fitted): {e}")
            
            # Create different model architectures - 8GB RAM optimized
            self.models = [
                self._create_dense_model(),
                self._create_lstm_model()
            ]
            
            # Filter out any None models if creation failed
            self.models = [model for model in self.models if model is not None]

            self.logger.info(f"Initialized {len(self.models)} neural network models")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}", exc_info=True)
            self.models = []
    
    def _create_dense_model(self) -> Optional[Sequential]:
        """Create dense feedforward neural network"""
        try:
            model = Sequential([
                Dense(128, activation='relu', input_shape=(15,)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(3, activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Dense model creation failed: {e}", exc_info=True)
            return None
    
    def _create_lstm_model(self) -> Optional[Sequential]:
        """Create LSTM neural network for sequence prediction"""
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(30, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(3, activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"LSTM model creation failed: {e}", exc_info=True)
            return None
    
    def _create_hybrid_model(self) -> Optional[Model]:
        """Create hybrid model combining dense and LSTM layers (not used in current init but kept)"""
        try:
            # Dense input branch
            dense_input = Input(shape=(15,))
            dense_branch = Dense(64, activation='relu')(dense_input)
            dense_branch = Dropout(0.3)(dense_branch)
            dense_branch = Dense(32, activation='relu')(dense_branch)
            
            # LSTM input branch
            lstm_input = Input(shape=(30, 1))
            lstm_branch = LSTM(32, return_sequences=False)(lstm_input)
            lstm_branch = Dropout(0.2)(lstm_branch)
            
            # Combine branches
            combined = concatenate([dense_branch, lstm_branch])
            combined = Dense(32, activation='relu')(combined)
            combined = Dropout(0.2)(combined)
            output = Dense(3, activation='softmax')(combined)
            
            model = Model(inputs=[dense_input, lstm_input], outputs=output)
            
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Hybrid model creation failed: {e}", exc_info=True)
            return None
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """Generate trading signals using ensemble neural networks - 8GB RAM optimized"""
        signals = []
        try:
            self.logger.info(f"Ensemble NN - Analyzing {symbol} on {timeframe}")
            
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available for signal generation.")
                return []
            
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_bars)
            if data is None or len(data) < self.lookback_bars:
                self.logger.warning(f"Insufficient data for Ensemble NN analysis: {len(data) if data is not None else 0}")
                return []
            
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.warning("Missing required data columns for Ensemble NN analysis.")
                return []
            
            self.prediction_count += 1 # Tracks prediction attempts
            if self.prediction_count % self.memory_cleanup_interval == 0:
                self._cleanup_memory()
            
            if not self.is_trained:
                self.logger.info("Training ensemble models...")
                self._train_ensemble(data)
            
            current_price = float(data['Close'].iloc[-1])
            
            dense_features, lstm_features = self._extract_features(data)
            if dense_features is None:
                self.logger.warning("Feature extraction failed, cannot make predictions.")
                return []
            
            predictions = []
            if TENSORFLOW_AVAILABLE and self.models and self.is_trained:
                predictions = self._make_multiple_predictions(dense_features, lstm_features, data)
            else:
                predictions = self._make_fallback_predictions(data)
            
            for i, (prediction, confidence) in enumerate(predictions):
                if prediction == 'HOLD' or confidence < self.min_confidence:
                    continue
                
                signal_type = SignalType.BUY if prediction == 'BUY' else SignalType.SELL
                
                atr = self._calculate_atr(data)
                
                # Default ATR if calculation fails
                if atr is None or atr == 0: # Ensure ATR is not None or zero
                    atr = data['Close'].iloc[-1] * 0.01

                current_price = float(data['Close'].iloc[-1])
                risk_factor = 1.0 + (i * 0.2)  # Vary risk for different signals
                
                if signal_type == SignalType.BUY:
                    stop_loss = current_price - (1.5 * atr * risk_factor)
                    take_profit = current_price + (2.5 * atr * confidence)
                else:
                    stop_loss = current_price + (1.5 * atr * risk_factor)
                    take_profit = current_price - (2.5 * atr * confidence)
                
                # Validate risk-reward ratio
                if signal_type == SignalType.BUY:
                    risk = current_price - stop_loss
                    reward = take_profit - current_price
                else:
                    risk = stop_loss - current_price
                    reward = current_price - take_profit
                
                if risk > 0 and reward > 0 and (reward / risk) >= 1.2:
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
                            'signal_reason': f'ensemble_nn_prediction_{i+1}',
                            'ensemble_prediction': prediction,
                            'ensemble_accuracy': self.ensemble_accuracy,
                            'models_count': len(self.models),
                            'prediction_index': i
                        }
                    )
                    # Use base class validation: This appends to self.signal_history if valid
                    if self.validate_signal(signal):
                        signals.append(signal) # Append to local list to be returned
            
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
        """Analyze market conditions and ensemble performance"""
        try:
            return {
                'strategy': self.strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'tensorflow_available': TENSORFLOW_AVAILABLE,
                'ensemble_trained': self.is_trained,
                'ensemble_accuracy': self.ensemble_accuracy,
                'predictions_attempted': self.prediction_count, # Use prediction_count for attempts
                'models_count': len(self.models),
                'memory_optimized': True,
                'lookback_bars': self.lookback_bars,
                'sequence_length': self.sequence_length,
                'latest_training_time': self.performance.last_signal_time.isoformat() if self.performance.last_signal_time else 'N/A'
            }
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            return {'strategy': self.strategy_name, 'error': str(e)}

    def _extract_features(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract features for both dense and LSTM models"""
        try:
            if len(data) < 60:
                return None, None
            
            dense_features = self._extract_dense_features(data)
            lstm_features = self._extract_lstm_features(data)
            
            return dense_features, lstm_features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}", exc_info=True)
            return None, None
    
    def _extract_dense_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract technical indicator features for dense model"""
        try:
            close = data['Close']
            
            features = []
            
            # Moving averages
            # Use min() for rolling window period to handle shorter data at start of training
            sma_5 = close.rolling(min(5, len(close))).mean().iloc[-1]
            sma_10 = close.rolling(min(10, len(close))).mean().iloc[-1]
            sma_20 = close.rolling(min(20, len(close))).mean().iloc[-1]
            sma_50 = close.rolling(min(50, len(close))).mean().iloc[-1]
            
            # Ensure SMA values are not NaN before using
            if any(pd.isna(val) for val in [sma_5, sma_10, sma_20, sma_50]):
                return None
            
            features.extend([
                close.iloc[-1] / sma_5 - 1,
                close.iloc[-1] / sma_10 - 1,
                close.iloc[-1] / sma_20 - 1,
                close.iloc[-1] / sma_50 - 1,
                sma_5 / sma_10 - 1,
                sma_10 / sma_20 - 1,
                sma_20 / sma_50 - 1
            ])
            
            # Technical indicators
            rsi = self._calculate_rsi(close)
            macd, macd_signal = self._calculate_macd(close)
            bb_upper, bb_lower = self._calculate_bollinger_bands(close)
            
            features.extend([
                rsi / 100.0,
                macd if not pd.isna(macd) else 0,
                macd_signal if not pd.isna(macd_signal) else 0,
                (macd - macd_signal) if not pd.isna(macd) and not pd.isna(macd_signal) else 0
            ])
            
            # Bollinger band position
            if not pd.isna(bb_upper) and not pd.isna(bb_lower) and (bb_upper - bb_lower) != 0:
                bb_position = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower)
                features.append(bb_position)
            else:
                features.append(0.5) # Default if bands are invalid
            
            # Price action features
            returns_1 = close.pct_change(1).iloc[-1]
            returns_5 = close.pct_change(5).iloc[-1]
            # Ensure enough data for rolling std
            volatility = close.rolling(min(20, len(close))).std().iloc[-1] / close.iloc[-1] if len(close) >= 20 else 0
            
            features.extend([
                returns_1 if not pd.isna(returns_1) else 0,
                returns_5 if not pd.isna(returns_5) else 0,
                volatility if not pd.isna(volatility) else 0
            ])
            
            feature_array = np.array(features).reshape(1, -1)
            
            # Ensure scaler is fitted before transforming
            if self.scaler and TENSORFLOW_AVAILABLE and hasattr(self.scaler, 'scale_') and self.is_trained: # Only transform if model is trained and scaler is fitted
                try:
                    feature_array = self.scaler.transform(feature_array)
                except Exception as e:
                    self.logger.warning(f"Scaler transform failed: {e}. Possible unfitted scaler.")
                    pass
            elif self.scaler and TENSORFLOW_AVAILABLE and not hasattr(self.scaler, 'scale_'):
                self.logger.warning("Scaler not fitted yet for feature extraction, returning unscaled features.")


            return feature_array
            
        except Exception as e:
            self.logger.error(f"Dense feature extraction failed: {e}", exc_info=True)
            return None
    
    def _extract_lstm_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract price sequence features for LSTM model"""
        try:
            close = data['Close'].values
            
            if len(close) < 30:
                return None
            
            price_sequence = close[-30:]
            
            # Avoid division by zero if std is zero
            std_dev = np.std(price_sequence)
            normalized_sequence = (price_sequence - np.mean(price_sequence)) / (std_dev if std_dev != 0 else 1)
            
            return normalized_sequence.reshape(1, 30, 1)
            
        except Exception as e:
            self.logger.error(f"LSTM feature extraction failed: {e}", exc_info=True)
            return None
    
    def _train_ensemble(self, data: pd.DataFrame):
        """Train the ensemble of neural networks"""
        try:
            if not TENSORFLOW_AVAILABLE or len(self.models) == 0:
                self.logger.warning("TensorFlow not available or no models initialized for training.")
                return
            
            self.logger.info("Starting ensemble neural networks training...")
            
            X_dense, X_lstm, y = self._prepare_training_data(data)
            
            if X_dense is None or len(X_dense) < 100:
                self.logger.warning("Insufficient or invalid training data for ensemble training.")
                return
            
            indices = np.arange(len(X_dense))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
            
            X_dense_train, X_dense_test = X_dense[train_idx], X_dense[test_idx]
            X_lstm_train, X_lstm_test = X_lstm[train_idx], X_lstm[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit scaler once on training data for dense features
            if self.scaler is None: # Initialize scaler if not already
                self.scaler = StandardScaler()
            
            X_dense_train = self.scaler.fit_transform(X_dense_train)
            X_dense_test = self.scaler.transform(X_dense_test)
            
            accuracies = []
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            for i, model in enumerate(self.models):
                if model is None:
                    continue
                
                try:
                    if i == 0:  # Dense model
                        history = model.fit(
                            X_dense_train, y_train,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=(X_dense_test, y_test),
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        y_pred = model.predict(X_dense_test, verbose=0)
                        
                    elif i == 1:  # LSTM model
                        history = model.fit(
                            X_lstm_train, y_train,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=(X_lstm_test, y_test),
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        y_pred = model.predict(X_lstm_test, verbose=0)
                        
                    else: # Fallback for hybrid if re-enabled/misconfigured (should not happen with current models list)
                        self.logger.warning(f"Unexpected model index {i} during training, skipping.")
                        continue
                    
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    accuracy = accuracy_score(y_test, y_pred_classes)
                    accuracies.append(accuracy)
                    
                    self.logger.info(f"Model {i+1} trained. Accuracy: {accuracy:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"Training model {i+1} failed: {e}", exc_info=True)
                    accuracies.append(0.0)
            
            self.ensemble_accuracy = np.mean(accuracies) if accuracies else 0.0
            self.is_trained = True
            
            self.logger.info(f"Ensemble training completed. Average accuracy: {self.ensemble_accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Ensemble training failed: {e}", exc_info=True)
            self.is_trained = False
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data for ensemble models"""
        try:
            if len(data) < 150:
                self.logger.warning("Insufficient data for training data preparation.")
                return None, None, None
            
            X_dense_list = []
            X_lstm_list = []
            y_list = []
            
            for i in range(60, len(data) - 5): # Iterate through data to create samples
                window_data = data.iloc[i-60:i] # Use historical window for features
                
                dense_features = self._extract_dense_features_for_training(window_data)
                lstm_features = self._extract_lstm_features_for_training(window_data)
                
                if dense_features is None or lstm_features is None:
                    continue # Skip if feature extraction fails for this window
                
                current_price = data['Close'].iloc[i]
                future_price = data['Close'].iloc[i+5] # Predict 5 bars into the future
                price_change = (future_price - current_price) / current_price
                
                if price_change > 0.003:
                    label = 'BUY'
                elif price_change < -0.003:
                    label = 'SELL'
                else:
                    label = 'HOLD'
                
                X_dense_list.append(dense_features)
                X_lstm_list.append(lstm_features)
                y_list.append(label)
            
            if len(X_dense_list) == 0: # No samples generated
                self.logger.warning("No training samples generated for ensemble.")
                return None, None, None
            
            X_dense = np.array(X_dense_list).squeeze() # Squeeze if necessary for dense features
            X_lstm = np.array(X_lstm_list).squeeze() # Squeeze if necessary for lstm features
            
            if not TENSORFLOW_AVAILABLE: # Fallback for label encoding if TF not available
                unique_labels = sorted(list(set(y_list)))
                label_map = {label: i for i, label in enumerate(unique_labels)}
                y = np.array([label_map[label] for label in y_list])
            else:
                if self.label_encoder is None:
                    self.label_encoder = LabelEncoder()
                if not hasattr(self.label_encoder, 'classes_'): # Check if encoder is fitted
                    self.label_encoder.fit(['BUY', 'SELL', 'HOLD']) 
                y = self.label_encoder.transform(y_list)
            
            return X_dense, X_lstm, y
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed: {e}", exc_info=True)
            return None, None, None
    
    def _extract_dense_features_for_training(self, data: pd.DataFrame) -> Optional[List[float]]:
        """Extract simplified dense features for training"""
        try:
            if len(data) < 50:
                return None
            
            close = data['Close']
            
            # Basic technical features
            sma_10 = close.rolling(min(10, len(close))).mean().iloc[-1]
            sma_20 = close.rolling(min(20, len(close))).mean().iloc[-1]
            sma_50 = close.rolling(min(50, len(close))).mean().iloc[-1]
            
            if any(pd.isna(val) for val in [sma_10, sma_20, sma_50]):
                return None
            
            features = [
                close.iloc[-1] / sma_10 - 1,
                close.iloc[-1] / sma_20 - 1,
                close.iloc[-1] / sma_50 - 1,
                sma_10 / sma_20 - 1,
                sma_20 / sma_50 - 1,
                self._calculate_rsi(close) / 100.0,
                close.pct_change(1).iloc[-1] if not pd.isna(close.pct_change(1).iloc[-1]) else 0,
                close.pct_change(5).iloc[-1] if not pd.isna(close.pct_change(5).iloc[-1]) else 0,
                close.rolling(min(20, len(close))).std().iloc[-1] / close.iloc[-1] if len(close) >= 20 else 0
            ]
            
            # Pad to 15 features if less (should not happen with 9 features extracted)
            # This ensures consistent input_shape for the Dense model
            while len(features) < 15:
                features.append(0.0)
            
            return features[:15] # Return exactly 15 features
            
        except Exception as e:
            self.logger.warning(f"Dense feature extraction for training failed: {e}", exc_info=True)
            return None
    
    def _extract_lstm_features_for_training(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract LSTM features for training"""
        try:
            close = data['Close'].values
            
            if len(close) < 30:
                return None
            
            price_sequence = close[-30:]
            std_dev = np.std(price_sequence)
            normalized_sequence = (price_sequence - np.mean(price_sequence)) / (std_dev if std_dev != 0 else 1) # Avoid div by zero
            
            return normalized_sequence.reshape(30, 1)
            
        except Exception as e:
            self.logger.warning(f"LSTM feature extraction for training failed: {e}", exc_info=True)
            return None
    
    def _make_ensemble_prediction(self, dense_features: np.ndarray, lstm_features: np.ndarray) -> Tuple[str, float]:
        """Make prediction using ensemble of models"""
        try:
            if not self.is_trained or len(self.models) == 0:
                self.logger.warning("Models not trained or available for ensemble prediction, using HOLD fallback.")
                return 'HOLD', 0.5
            
            predictions = []
            
            for i, model in enumerate(self.models):
                if model is None:
                    continue
                
                try:
                    if i == 0:  # Dense model
                        # Ensure dense_features is correctly scaled before prediction
                        if self.scaler and hasattr(self.scaler, 'scale_'): # Check if scaler is fitted
                            scaled_dense_features = self.scaler.transform(dense_features)
                        else:
                            scaled_dense_features = dense_features # Use unscaled if scaler not ready
                            self.logger.warning("Dense features not scaled for prediction: Scaler not fitted.")

                        # Use explicit numpy array with concrete shape (1, features)
                        dense_input = np.asarray(scaled_dense_features, dtype=np.float32).reshape(1, -1)
                        pred = model.predict(dense_input, verbose=0)[0]
                    elif i == 1:  # LSTM model
                        # Ensure concrete numpy shape (1, timesteps, features)
                        lstm_input = np.asarray(lstm_features, dtype=np.float32)
                        if lstm_input.ndim == 2:
                            lstm_input = lstm_input.reshape(1, lstm_input.shape[0], lstm_input.shape[1])
                        elif lstm_input.ndim == 1:
                            lstm_input = lstm_input.reshape(1, 30, 1)
                        pred = model.predict(lstm_input, verbose=0)[0]
                    else: # Fallback for hybrid if re-enabled/misconfigured (should not happen with current models list)
                        self.logger.warning(f"Unexpected model index {i} during prediction, skipping.")
                        continue
                    
                    predictions.append(pred)
                    
                except Exception as e:
                    self.logger.error(f"Model {i+1} prediction failed: {e}", exc_info=True)
                    continue
            
            if not predictions:
                self.logger.warning("No successful predictions from ensemble models, returning HOLD.")
                return 'HOLD', 0.5
            
            avg_prediction = np.mean(predictions, axis=0)
            
            classes = self.label_encoder.classes_ if self.label_encoder and hasattr(self.label_encoder, 'classes_') else ['HOLD', 'BUY', 'SELL']
            max_prob_idx = np.argmax(avg_prediction)
            
            # Ensure max_prob_idx is within bounds of classes
            if max_prob_idx < len(classes):
                prediction = classes[max_prob_idx]
            else:
                prediction = 'HOLD' # Fallback if index out of bounds
                self.logger.warning(f"max_prob_idx {max_prob_idx} out of bounds for classes {classes}, defaulting to HOLD.")

            confidence = avg_prediction[max_prob_idx]
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}", exc_info=True)
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
            if len(prices) < period + 1: # Not enough data for initial RSI
                return 50.0
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            # Replace inf with large number, or NaN with 0, then calculate RSI
            rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0) # Handle inf values in rs calculation
            
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except Exception as e:
            self.logger.error(f"RSI calculation failed: {e}", exc_info=True)
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD indicator"""
        try:
            if len(prices) < max(fast, slow, signal) + 1: # Not enough data for MACD
                return 0.0, 0.0
            ema_fast = prices.ewm(span=fast, adjust=False).mean() # adjust=False for classic EMA
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            
            return macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0, macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0.0
        except Exception as e:
            self.logger.error(f"MACD calculation failed: {e}", exc_info=True)
            return 0.0, 0.0
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[float, float]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period: # Not enough data for Bollinger Bands
                return prices.iloc[-1], prices.iloc[-1] # Return current price as both bands
            
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return upper_band.iloc[-1] if not pd.isna(upper_band.iloc[-1]) else prices.iloc[-1], \
                   lower_band.iloc[-1] if not pd.isna(lower_band.iloc[-1]) else prices.iloc[-1]
        except Exception as e:
            self.logger.error(f"Bollinger Bands calculation failed: {e}", exc_info=True)
            return prices.iloc[-1] * 1.02, prices.iloc[-1] * 0.98 # Fallback
    
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
    
    def _make_multiple_predictions(self, dense_features: np.ndarray, lstm_features: np.ndarray, 
                                  data: pd.DataFrame) -> List[Tuple[str, float]]:
        """Make multiple predictions for enhanced signal generation"""
        try:
            predictions = []
            
            primary_pred, primary_conf = self._make_ensemble_prediction(dense_features, lstm_features)
            predictions.append((primary_pred, primary_conf))
            
            if len(self.models) > 1:
                for i, model in enumerate(self.models[:self.ensemble_size]): # Use configured ensemble_size
                    if model is None:
                        continue
                    
                    try:
                        if i == 0:  # Dense model
                            if self.scaler and hasattr(self.scaler, 'scale_'): # Check if scaler is fitted
                                scaled_dense_features = self.scaler.transform(dense_features)
                            else:
                                scaled_dense_features = dense_features # Use unscaled if scaler not ready
                            # Use numpy with explicit shape
                            dense_input = np.asarray(scaled_dense_features, dtype=np.float32).reshape(1, -1)
                            pred = model.predict(dense_input, verbose=0)[0]
                        elif i == 1:  # LSTM model
                            # Ensure numpy array with shape (1, timesteps, features)
                            lstm_input = np.asarray(lstm_features, dtype=np.float32)
                            if lstm_input.ndim == 2:
                                lstm_input = lstm_input.reshape(1, lstm_input.shape[0], lstm_input.shape[1])
                            elif lstm_input.ndim == 1:
                                lstm_input = lstm_input.reshape(1, 30, 1)
                            pred = model.predict(lstm_input, verbose=0)[0]
                        else:
                            continue # Skip unexpected models
                        
                        classes = self.label_encoder.classes_ if self.label_encoder and hasattr(self.label_encoder, 'classes_') else ['HOLD', 'BUY', 'SELL']
                        max_prob_idx = np.argmax(pred)
                        
                        if max_prob_idx < len(classes):
                            prediction = classes[max_prob_idx]
                        else:
                            prediction = 'HOLD' # Fallback
                        confidence = pred[max_prob_idx] * 0.9 # Slightly reduce individual confidence
                        
                        predictions.append((prediction, confidence))
                        
                    except Exception as e:
                        self.logger.error(f"Individual model prediction failed: {e}", exc_info=True)
                        continue
            
            return predictions[:self.ensemble_size] # Return up to ensemble_size predictions
            
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
            
            if TENSORFLOW_AVAILABLE and hasattr(tf, 'keras'):
                tf.keras.backend.clear_session()
            
            gc.collect()
            
            self.logger.info(f"Memory cleanup completed (prediction #{self.prediction_count})")
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {str(e)}", exc_info=True)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the Ensemble NN Strategy,
        including its parameters, ML-specific performance, and overall trading performance.
        """
        # Get overall trading performance from AbstractStrategy base class
        base_trading_performance = self.get_performance_summary()

        return {
            'name': 'Ensemble Neural Network Strategy',
            'version': '2.0.0',
            'type': 'Machine Learning',
            'description': 'Combines multiple neural network architectures for robust predictions.',
            'parameters': {
                'lookback_bars': self.lookback_bars,
                'sequence_length': self.sequence_length,
                'min_confidence': self.min_confidence,
                'ensemble_size': self.ensemble_size,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'max_training_samples': self.max_training_samples,
                'memory_cleanup_interval': self.memory_cleanup_interval
            },
            'ml_specific_metrics': {
                'tensorflow_available': TENSORFLOW_AVAILABLE,
                'ensemble_trained': self.is_trained,
                'ensemble_accuracy': self.ensemble_accuracy,
                'predictions_attempted': self.prediction_count,
                'models_count': len(self.models),
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
            'lookback_bars': 150,
            'sequence_length': 20,
            'min_confidence': 0.65,
            'ensemble_size': 2,
            'epochs': 2, # Reduced epochs for faster testing
            'batch_size': 8,
            'learning_rate': 0.001,
            'max_training_samples': 800,
            'memory_cleanup_interval': 30
        }
    }
    
    # Mock MT5 manager for testing
    class MockMT5Manager:
        def get_historical_data(self, symbol, timeframe, bars):
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                 end=datetime.now(), freq='15Min')
            
            np.random.seed(42) # For consistent mock data
            # Generate more complex price data for better NN training simulation
            price_series = np.cumsum(np.random.randn(len(dates)) * 0.5) + np.sin(np.linspace(0, 100, len(dates))) * 5
            close_prices = 1950 + price_series
            
            data = pd.DataFrame({
                'Open': close_prices + np.random.randn(len(dates)) * 0.2,
                'High': close_prices + np.abs(np.random.randn(len(dates)) * 0.5),
                'Low': close_prices - np.abs(np.random.randn(len(dates)) * 0.5),
                'Close': close_prices,
                'Volume': np.random.randint(100, 1000, len(dates))
            }, index=dates)
            
            # Ensure High >= Close >= Low
            data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
            data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
            
            return data
    
    # Create strategy instance
    mock_mt5 = MockMT5Manager()
    strategy = EnsembleNNStrategy(test_config, mock_mt5, database=None)
    
    # Output header matching other strategy files
    print("============================================================")
    print("TESTING MODIFIED ENSEMBLE NN STRATEGY")
    print("============================================================")

    # 1. Testing signal generation
    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - Signal: {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
        if signal.metadata:
            print(f"     Ensemble Prediction: {signal.metadata.get('ensemble_prediction', 'N/A')}")
            print(f"     Ensemble Accuracy: {signal.metadata.get('ensemble_accuracy', 'N/A'):.2f}")
    
    # 2. Testing analysis method
    print("\n2. Testing analysis method:")
    mock_data = mock_mt5.get_historical_data("XAUUSDm", "M15", 200)
    analysis_results = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis results keys: {analysis_results.keys()}")
    print(f"   TensorFlow Available: {analysis_results.get('tensorflow_available')}")
    print(f"   Ensemble Trained: {analysis_results.get('ensemble_trained')}")
    print(f"   Ensemble Accuracy: {analysis_results.get('ensemble_accuracy'):.2f}")
    
    # 3. Testing performance tracking
    print("\n3. Testing performance tracking:")
    summary = strategy.get_performance_summary()
    print(f"   {summary}")
    
    # 4. Strategy Information
    print("\n4. Strategy Information:")
    info = strategy.get_strategy_info()
    print(f"   Name: {info['name']}")
    print(f"   Version: {info['version']}")
    print(f"   Type: {info['type']}")
    print(f"   Description: {info['description']}")
    print(f"   Parameters: {info['parameters']}")
    print(f"   ML Specific Metrics:")
    for key, value in info['ml_specific_metrics'].items():
        if 'accuracy' in key or 'confidence' in key:
            print(f"     - {key}: {value:.2f}" if isinstance(value, (float, int)) else f"     - {key}: {value}")
        else:
            print(f"     - {key}: {value}")
    print(f"   Overall Trading Performance:")
    print(f"     Total Signals Generated: {info['overall_trading_performance']['total_signals_generated']}")
    print(f"     Win Rate: {info['overall_trading_performance']['win_rate']:.2%}")
    print(f"     Profit Factor: {info['overall_trading_performance']['profit_factor']:.2f}")

    # Footer matching other strategy files
    print("\n============================================================")
    print("ENSEMBLE NN STRATEGY TEST COMPLETED!")
    print("============================================================")