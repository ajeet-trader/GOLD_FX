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

# Add project root to path
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

# Import base classes
try:
    from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade
except ImportError:
    from core.base import AbstractStrategy, Signal, SignalType, SignalGrade


class EnsembleNN(AbstractStrategy):
    """Ensemble Neural Network strategy combining multiple NN architectures"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize ensemble neural network strategy"""
        super().__init__(config)
        
        self.strategy_name = "EnsembleNN"
        self.lookback_bars = config.get('lookback_bars', 300)
        self.sequence_length = config.get('sequence_length', 60)
        self.min_confidence = config.get('min_confidence', 0.70)
        self.ensemble_size = config.get('ensemble_size', 3)
        
        # Model parameters
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Model components
        self.models = []
        self.scaler = StandardScaler() if TENSORFLOW_AVAILABLE else None
        self.label_encoder = LabelEncoder() if TENSORFLOW_AVAILABLE else None
        self.is_trained = False
        
        # Performance tracking
        self.predictions_made = 0
        self.ensemble_accuracy = 0.0
        
        # Logger
        self.logger = logging.getLogger(self.strategy_name)
        
        # Initialize models if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            self._initialize_models()
        
        self.logger.info(f"{self.strategy_name} initialized (TensorFlow available: {TENSORFLOW_AVAILABLE})")
    
    def _initialize_models(self):
        """Initialize ensemble of neural network models"""
        try:
            if not TENSORFLOW_AVAILABLE:
                return
            
            self.label_encoder.fit(['BUY', 'SELL', 'HOLD'])
            
            # Create different model architectures
            self.models = [
                self._create_dense_model(),
                self._create_lstm_model(),
                self._create_hybrid_model()
            ]
            
            self.logger.info(f"Initialized {len(self.models)} neural network models")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
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
            self.logger.error(f"Dense model creation failed: {e}")
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
            self.logger.error(f"LSTM model creation failed: {e}")
            return None
    
    def _create_hybrid_model(self) -> Optional[Model]:
        """Create hybrid model combining dense and LSTM layers"""
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
            self.logger.error(f"Hybrid model creation failed: {e}")
            return None
    
    def generate_signal(self, data: pd.DataFrame, symbol: str = "XAUUSDm", 
                       timeframe: str = "M15") -> Optional[Signal]:
        """Generate trading signal using ensemble neural networks"""
        try:
            if data is None or len(data) < self.lookback_bars:
                return None
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                return self._generate_fallback_signal(data, symbol, timeframe)
            
            # Train models if not trained
            if not self.is_trained:
                self._train_ensemble(data)
            
            # Generate features for prediction
            dense_features, lstm_features = self._extract_features(data)
            if dense_features is None:
                return self._generate_fallback_signal(data, symbol, timeframe)
            
            # Make ensemble prediction
            if TENSORFLOW_AVAILABLE and self.models and self.is_trained:
                prediction, confidence = self._make_ensemble_prediction(dense_features, lstm_features)
            else:
                prediction, confidence = self._make_fallback_prediction(data)
            
            # Convert prediction to signal
            if prediction == 'HOLD' or confidence < self.min_confidence:
                return None
            
            signal_type = SignalType.BUY if prediction == 'BUY' else SignalType.SELL
            current_price = float(data['Close'].iloc[-1])
            
            # Calculate stop loss and take profit
            atr = self._calculate_atr(data)
            if signal_type == SignalType.BUY:
                stop_loss = current_price - (2.5 * atr)
                take_profit = current_price + (3.5 * atr)
            else:
                stop_loss = current_price + (2.5 * atr)
                take_profit = current_price - (3.5 * atr)
            
            # Create signal
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
                    'ensemble_prediction': prediction,
                    'ensemble_accuracy': self.ensemble_accuracy,
                    'predictions_made': self.predictions_made,
                    'models_count': len(self.models),
                    'tensorflow_available': TENSORFLOW_AVAILABLE
                }
            )
            
            self.predictions_made += 1
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return self._generate_fallback_signal(data, symbol, timeframe)
    
    def _extract_features(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract features for both dense and LSTM models"""
        try:
            if len(data) < 60:
                return None, None
            
            # Dense features (technical indicators)
            dense_features = self._extract_dense_features(data)
            
            # LSTM features (price sequences)
            lstm_features = self._extract_lstm_features(data)
            
            return dense_features, lstm_features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None, None
    
    def _extract_dense_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract technical indicator features for dense model"""
        try:
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            features = []
            
            # Moving averages
            sma_5 = close.rolling(5).mean().iloc[-1]
            sma_10 = close.rolling(10).mean().iloc[-1]
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            
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
            if not pd.isna(bb_upper) and not pd.isna(bb_lower) and bb_upper != bb_lower:
                bb_position = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower)
                features.append(bb_position)
            else:
                features.append(0.5)
            
            # Price action features
            returns_1 = close.pct_change(1).iloc[-1]
            returns_5 = close.pct_change(5).iloc[-1]
            volatility = close.rolling(20).std().iloc[-1] / close.iloc[-1]
            
            features.extend([
                returns_1 if not pd.isna(returns_1) else 0,
                returns_5 if not pd.isna(returns_5) else 0,
                volatility if not pd.isna(volatility) else 0
            ])
            
            # Convert to numpy array and scale
            feature_array = np.array(features).reshape(1, -1)
            
            if self.scaler and TENSORFLOW_AVAILABLE:
                try:
                    feature_array = self.scaler.transform(feature_array)
                except:
                    pass
            
            return feature_array
            
        except Exception as e:
            self.logger.error(f"Dense feature extraction failed: {e}")
            return None
    
    def _extract_lstm_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract price sequence features for LSTM model"""
        try:
            close = data['Close'].values
            
            if len(close) < 30:
                return None
            
            # Use last 30 normalized price changes
            price_sequence = close[-30:]
            normalized_sequence = (price_sequence - np.mean(price_sequence)) / np.std(price_sequence)
            
            return normalized_sequence.reshape(1, 30, 1)
            
        except Exception as e:
            self.logger.error(f"LSTM feature extraction failed: {e}")
            return None
    
    def _train_ensemble(self, data: pd.DataFrame):
        """Train the ensemble of neural networks"""
        try:
            if not TENSORFLOW_AVAILABLE or len(self.models) == 0:
                return
            
            self.logger.info("Training ensemble neural networks...")
            
            # Prepare training data
            X_dense, X_lstm, y = self._prepare_training_data(data)
            
            if X_dense is None or len(X_dense) < 100:
                return
            
            # Split data
            indices = np.arange(len(X_dense))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
            
            X_dense_train, X_dense_test = X_dense[train_idx], X_dense[test_idx]
            X_lstm_train, X_lstm_test = X_lstm[train_idx], X_lstm[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale dense features
            X_dense_train = self.scaler.fit_transform(X_dense_train)
            X_dense_test = self.scaler.transform(X_dense_test)
            
            # Train each model in the ensemble
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
                        
                    else:  # Hybrid model
                        history = model.fit(
                            [X_dense_train, X_lstm_train], y_train,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=([X_dense_test, X_lstm_test], y_test),
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        y_pred = model.predict([X_dense_test, X_lstm_test], verbose=0)
                    
                    # Calculate accuracy
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    accuracy = accuracy_score(y_test, y_pred_classes)
                    accuracies.append(accuracy)
                    
                    self.logger.info(f"Model {i+1} trained. Accuracy: {accuracy:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"Training model {i+1} failed: {e}")
                    accuracies.append(0.0)
            
            # Calculate ensemble accuracy
            self.ensemble_accuracy = np.mean(accuracies) if accuracies else 0.0
            self.is_trained = True
            
            self.logger.info(f"Ensemble training completed. Average accuracy: {self.ensemble_accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Ensemble training failed: {e}")
            self.is_trained = False
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data for ensemble models"""
        try:
            if len(data) < 150:
                return None, None, None
            
            X_dense_list = []
            X_lstm_list = []
            y_list = []
            
            # Generate training samples
            for i in range(60, len(data) - 5):
                window_data = data.iloc[i-60:i]
                
                # Extract features
                dense_features = self._extract_dense_features_for_training(window_data)
                lstm_features = self._extract_lstm_features_for_training(window_data)
                
                if dense_features is None or lstm_features is None:
                    continue
                
                # Generate label
                current_price = data['Close'].iloc[i]
                future_price = data['Close'].iloc[i+5]
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
            
            if len(X_dense_list) == 0:
                return None, None, None
            
            X_dense = np.array(X_dense_list)
            X_lstm = np.array(X_lstm_list)
            y = self.label_encoder.transform(y_list)
            
            return X_dense, X_lstm, y
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed: {e}")
            return None, None, None
    
    def _extract_dense_features_for_training(self, data: pd.DataFrame) -> Optional[List[float]]:
        """Extract simplified dense features for training"""
        try:
            if len(data) < 50:
                return None
            
            close = data['Close']
            
            # Basic technical features
            sma_10 = close.rolling(10).mean().iloc[-1]
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            
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
                close.rolling(20).std().iloc[-1] / close.iloc[-1] if not pd.isna(close.rolling(20).std().iloc[-1]) else 0
            ]
            
            # Pad to 15 features
            while len(features) < 15:
                features.append(0.0)
            
            return features[:15]
            
        except Exception:
            return None
    
    def _extract_lstm_features_for_training(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract LSTM features for training"""
        try:
            close = data['Close'].values
            
            if len(close) < 30:
                return None
            
            price_sequence = close[-30:]
            normalized_sequence = (price_sequence - np.mean(price_sequence)) / np.std(price_sequence)
            
            return normalized_sequence.reshape(30, 1)
            
        except Exception:
            return None
    
    def _make_ensemble_prediction(self, dense_features: np.ndarray, lstm_features: np.ndarray) -> Tuple[str, float]:
        """Make prediction using ensemble of models"""
        try:
            if not self.is_trained or len(self.models) == 0:
                return 'HOLD', 0.5
            
            predictions = []
            
            for i, model in enumerate(self.models):
                if model is None:
                    continue
                
                try:
                    if i == 0:  # Dense model
                        pred = model.predict(dense_features, verbose=0)[0]
                    elif i == 1:  # LSTM model
                        pred = model.predict(lstm_features, verbose=0)[0]
                    else:  # Hybrid model
                        pred = model.predict([dense_features, lstm_features], verbose=0)[0]
                    
                    predictions.append(pred)
                    
                except Exception as e:
                    self.logger.error(f"Model {i+1} prediction failed: {e}")
                    continue
            
            if not predictions:
                return 'HOLD', 0.5
            
            # Average ensemble predictions
            avg_prediction = np.mean(predictions, axis=0)
            
            # Get class with highest probability
            classes = self.label_encoder.classes_
            max_prob_idx = np.argmax(avg_prediction)
            prediction = classes[max_prob_idx]
            confidence = avg_prediction[max_prob_idx]
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            return 'HOLD', 0.5
    
    def _make_fallback_prediction(self, data: pd.DataFrame) -> Tuple[str, float]:
        """Make simple prediction when neural networks are not available"""
        try:
            close = data['Close']
            
            if len(close) >= 20:
                sma_10 = close.rolling(10).mean().iloc[-1]
                sma_20 = close.rolling(20).mean().iloc[-1]
                rsi = self._calculate_rsi(close)
                
                if not pd.isna(sma_10) and not pd.isna(sma_20):
                    # Combined SMA and RSI signal
                    if sma_10 > sma_20 * 1.002 and rsi < 70:
                        return 'BUY', 0.65
                    elif sma_10 < sma_20 * 0.998 and rsi > 30:
                        return 'SELL', 0.65
            
            return 'HOLD', 0.5
            
        except Exception:
            return 'HOLD', 0.5
    
    def _generate_fallback_signal(self, data: pd.DataFrame, symbol: str, 
                                 timeframe: str) -> Optional[Signal]:
        """Generate fallback signal when main logic fails"""
        try:
            if len(data) < 20:
                return None
            
            prediction, confidence = self._make_fallback_prediction(data)
            
            if prediction == 'HOLD' or confidence < 0.6:
                return None
            
            signal_type = SignalType.BUY if prediction == 'BUY' else SignalType.SELL
            current_price = float(data['Close'].iloc[-1])
            
            price_range = current_price * 0.015
            
            if signal_type == SignalType.BUY:
                stop_loss = current_price - price_range
                take_profit = current_price + (price_range * 2)
            else:
                stop_loss = current_price + price_range
                take_profit = current_price - (price_range * 2)
            
            return Signal(
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
                metadata={'fallback_mode': True}
            )
            
        except Exception:
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1]
        except:
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            
            return macd.iloc[-1], macd_signal.iloc[-1]
        except:
            return 0.0, 0.0
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[float, float]:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return upper_band.iloc[-1], lower_band.iloc[-1]
        except:
            return prices.iloc[-1] * 1.02, prices.iloc[-1] * 0.98
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else data['Close'].iloc[-1] * 0.01
        except:
            return data['Close'].iloc[-1] * 0.01
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze market conditions and ensemble performance"""
        return {
            'strategy': self.strategy_name,
            'symbol': symbol,
            'timeframe': timeframe,
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'ensemble_trained': self.is_trained,
            'ensemble_accuracy': self.ensemble_accuracy,
            'predictions_made': self.predictions_made,
            'models_count': len(self.models)
        }
