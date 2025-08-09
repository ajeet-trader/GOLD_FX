"""
LSTM Predictor - Advanced Machine Learning Strategy
==================================================
Author: XAUUSD Trading System
Version: 2.0.0
Date: 2025-08-08

Advanced LSTM neural network for price prediction and signal generation:
- Multi-layer LSTM architecture
- Feature engineering with technical indicators
- Price direction and magnitude prediction
- Dynamic model retraining
- Confidence-based signal filtering

Features:
- Bidirectional LSTM for better context
- Multiple timeframe feature extraction
- Ensemble predictions
- Adaptive learning rate
- Early stopping and regularization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("TensorFlow/Scikit-learn not available. LSTM strategy will run in simulation mode.")

from enum import Enum
from dataclasses import dataclass

# Import base classes
try:
    from ..signal_engine import Signal, SignalType, SignalGrade
except ImportError:
    # Fallback for testing
    class SignalType(Enum):
        BUY = "BUY"
        SELL = "SELL"
        HOLD = "HOLD"
    
    class SignalGrade(Enum):
        A = "A"
        B = "B" 
        C = "C"
        D = "D"
    
    @dataclass
    class Signal:
        timestamp: datetime
        symbol: str
        strategy_name: str
        signal_type: SignalType
        confidence: float
        price: float
        timeframe: str
        strength: float = 0.0
        grade: Optional[SignalGrade] = None
        stop_loss: Optional[float] = None
        take_profit: Optional[float] = None
        metadata: Dict[str, Any] = None

# Fallback classes when ML libraries are not available
if not ML_AVAILABLE:
    class Sequential:
        pass
    
    class Model:
        pass
    
class LSTMPredictor:
    """
    Advanced LSTM Neural Network for Gold Price Prediction
    
    This strategy uses deep learning to predict:
    - Price direction (up/down/sideways)
    - Price magnitude (how much movement)
    - Market volatility
    - Optimal entry/exit points
    
    Features:
    - Multi-timeframe analysis
    - Technical indicator features
    - Ensemble model predictions
    - Dynamic retraining
    - Risk-adjusted signals
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager):
        """Initialize LSTM Predictor"""
        self.config = config
        self.mt5_manager = mt5_manager
        
        # Strategy parameters
        self.min_confidence = config.get('parameters', {}).get('confidence_threshold', 0.75)
        self.prediction_horizon = config.get('parameters', {}).get('prediction_horizon', 12)
        self.feature_lookback = config.get('parameters', {}).get('feature_lookback', 50)
        self.retrain_frequency = config.get('parameters', {}).get('retrain_frequency', 'weekly')
        self.min_training_samples = config.get('parameters', {}).get('min_training_samples', 1000)
        
        # Model architecture parameters
        self.sequence_length = 60  # 60 bars for LSTM input
        self.lstm_units = [128, 64, 32]  # Multi-layer LSTM
        self.dropout_rate = 0.3
        self.learning_rate = 0.001
        
        # Data preprocessing
        self.price_scaler = MinMaxScaler(feature_range=(0, 1)) if ML_AVAILABLE else None
        self.feature_scaler = StandardScaler() if ML_AVAILABLE else None
        
        # Models
        self.direction_model = None  # Predicts price direction
        self.magnitude_model = None  # Predicts price magnitude
        self.volatility_model = None  # Predicts volatility
        
        # Training data storage
        self.training_data = {
            'features': [],
            'direction_targets': [],
            'magnitude_targets': [],
            'volatility_targets': []
        }
        
        # Performance tracking
        self.model_performance = {
            'direction_accuracy': 0.0,
            'magnitude_mae': 0.0,
            'last_training': None,
            'training_samples': 0
        }
        
        # Feature importance
        self.feature_names = []
        
        # Logger
        self.logger = logging.getLogger('lstm_predictor')
        
        if not ML_AVAILABLE:
            self.logger.warning("ML libraries not available. Running in simulation mode.")
    
    def generate_signals(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """Generate ML-based trading signals"""
        try:
            # Get market data
            data = self.mt5_manager.get_historical_data(symbol, timeframe, 1000)
            if data is None or len(data) < self.sequence_length + 50:
                self.logger.warning(f"Insufficient data for LSTM analysis: {len(data) if data is not None else 0}")
                return []
            
            # Prepare features
            features = self._prepare_features(data, symbol, timeframe)
            if features is None or len(features) == 0:
                self.logger.warning("Feature preparation failed")
                return []
            
            # Check if models need training/retraining
            if self._should_retrain():
                self._train_models(data, symbol, timeframe)
            
            # Generate predictions
            predictions = self._generate_predictions(features)
            if predictions is None:
                return []
            
            # Convert predictions to signals
            signals = self._predictions_to_signals(predictions, data, symbol, timeframe)
            
            # Update training data for future retraining
            self._update_training_data(features, data)
            
            self.logger.info(f"LSTM generated {len(signals)} signals with avg confidence {np.mean([s.confidence for s in signals]):.2f}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"LSTM signal generation failed: {str(e)}")
            return []
    
    def _prepare_features(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[np.ndarray]:
        """Prepare comprehensive feature set for LSTM"""
        try:
            if not ML_AVAILABLE:
                # Return dummy features for simulation
                return np.random.rand(len(data) - self.sequence_length, 20)
            
            # Technical indicators
            features_df = pd.DataFrame()
            
            # Price features
            features_df['returns'] = data['Close'].pct_change()
            features_df['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
            features_df['price_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
            features_df['typical_price'] = (data['High'] + data['Low'] + data['Close']) / 3
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                features_df[f'sma_{period}'] = data['Close'].rolling(period).mean()
                features_df[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
                features_df[f'price_vs_sma_{period}'] = data['Close'] / features_df[f'sma_{period}'] - 1
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            sma_20 = data['Close'].rolling(bb_period).mean()
            bb_std_dev = data['Close'].rolling(bb_period).std()
            features_df['bb_upper'] = sma_20 + (bb_std_dev * bb_std)
            features_df['bb_lower'] = sma_20 - (bb_std_dev * bb_std)
            features_df['bb_position'] = (data['Close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
            features_df['bb_width'] = (features_df['bb_upper'] - features_df['bb_lower']) / sma_20
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            features_df['macd'] = exp1 - exp2
            features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # ATR (Average True Range)
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            features_df['atr'] = true_range.rolling(14).mean()
            features_df['atr_ratio'] = true_range / features_df['atr']
            
            # Volume features (if available)
            if 'Volume' in data.columns:
                features_df['volume_sma'] = data['Volume'].rolling(20).mean()
                features_df['volume_ratio'] = data['Volume'] / features_df['volume_sma']
                features_df['price_volume'] = data['Close'] * data['Volume']
            else:
                features_df['volume_sma'] = 0
                features_df['volume_ratio'] = 1
                features_df['price_volume'] = data['Close']
            
            # Time-based features
            features_df['hour'] = data.index.hour / 23.0
            features_df['day_of_week'] = data.index.dayofweek / 6.0
            features_df['day_of_month'] = data.index.day / 31.0
            
            # Volatility features
            features_df['volatility'] = data['Close'].rolling(20).std()
            features_df['volatility_ratio'] = features_df['volatility'] / features_df['volatility'].rolling(50).mean()
            
            # Support/Resistance levels
            features_df['local_high'] = data['High'].rolling(10, center=True).max()
            features_df['local_low'] = data['Low'].rolling(10, center=True).min()
            features_df['resistance_distance'] = (features_df['local_high'] - data['Close']) / data['Close']
            features_df['support_distance'] = (data['Close'] - features_df['local_low']) / data['Close']
            
            # Higher timeframe context (if possible)
            # This would require additional data fetching in a real implementation
            features_df['trend_strength'] = features_df['sma_5'] - features_df['sma_20']
            features_df['momentum'] = data['Close'] - data['Close'].shift(10)
            
            # Clean and prepare data
            #features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            features_df = features_df.ffill().bfill()

            features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Store feature names for later reference
            self.feature_names = features_df.columns.tolist()
            
            # Convert to numpy array
            features_array = features_df.values
            
            # Normalize features
            if len(features_array) > 0:
                features_scaled = self.feature_scaler.fit_transform(features_array)
                
                # Create sequences for LSTM
                sequences = []
                for i in range(self.sequence_length, len(features_scaled)):
                    sequences.append(features_scaled[i-self.sequence_length:i])
                
                return np.array(sequences)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {str(e)}")
            return None
    
    def _should_retrain(self) -> bool:
        """Check if models need retraining"""
        try:
            # Check if models exist
            if (self.direction_model is None or 
                self.magnitude_model is None or 
                self.volatility_model is None):
                return True
            
            # Check if enough time has passed since last training
            if self.model_performance['last_training'] is None:
                return True
            
            time_since_training = datetime.now() - self.model_performance['last_training']
            
            if self.retrain_frequency == 'daily' and time_since_training.days >= 1:
                return True
            elif self.retrain_frequency == 'weekly' and time_since_training.days >= 7:
                return True
            elif self.retrain_frequency == 'monthly' and time_since_training.days >= 30:
                return True
            
            # Check if enough new data is available
            if len(self.training_data['features']) >= self.min_training_samples * 1.5:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Retrain check failed: {str(e)}")
            return False
    
    def _train_models(self, data: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """Train the LSTM models"""
        try:
            if not ML_AVAILABLE:
                self.logger.info("ML libraries not available. Simulating model training.")
                self.model_performance['last_training'] = datetime.now()
                return True
            
            self.logger.info("Starting LSTM model training...")
            
            # Prepare training data
            features = self._prepare_features(data, symbol, timeframe)
            if features is None or len(features) < 100:
                self.logger.warning("Insufficient data for training")
                return False
            
            # Prepare targets
            targets = self._prepare_targets(data)
            if targets is None:
                return False
            
            # Align features and targets
            min_length = min(len(features), len(targets['direction']))
            X = features[:min_length]
            y_direction = targets['direction'][:min_length]
            y_magnitude = targets['magnitude'][:min_length]
            y_volatility = targets['volatility'][:min_length]
            
            # Split data for training/validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_dir_train, y_dir_val = y_direction[:split_idx], y_direction[split_idx:]
            y_mag_train, y_mag_val = y_magnitude[:split_idx], y_magnitude[split_idx:]
            y_vol_train, y_vol_val = y_volatility[:split_idx], y_volatility[split_idx:]
            
            # Train direction model (classification)
            self.direction_model = self._build_direction_model(X_train.shape)
            self.direction_model.fit(
                X_train, y_dir_train,
                validation_data=(X_val, y_dir_val),
                epochs=50,
                batch_size=32,
                callbacks=self._get_callbacks(),
                verbose=0
            )
            
            # Train magnitude model (regression)
            self.magnitude_model = self._build_magnitude_model(X_train.shape)
            self.magnitude_model.fit(
                X_train, y_mag_train,
                validation_data=(X_val, y_mag_val),
                epochs=50,
                batch_size=32,
                callbacks=self._get_callbacks(),
                verbose=0
            )
            
            # Train volatility model (regression)
            self.volatility_model = self._build_volatility_model(X_train.shape)
            self.volatility_model.fit(
                X_train, y_vol_train,
                validation_data=(X_val, y_vol_val),
                epochs=50,
                batch_size=32,
                callbacks=self._get_callbacks(),
                verbose=0
            )
            
            # Evaluate models
            dir_pred = self.direction_model.predict(X_val, verbose=0)
            dir_accuracy = accuracy_score(y_dir_val, np.argmax(dir_pred, axis=1))
            
            mag_pred = self.magnitude_model.predict(X_val, verbose=0)
            mag_mae = np.mean(np.abs(mag_pred.flatten() - y_mag_val))
            
            # Update performance metrics
            self.model_performance.update({
                'direction_accuracy': dir_accuracy,
                'magnitude_mae': mag_mae,
                'last_training': datetime.now(),
                'training_samples': len(X_train)
            })
            
            self.logger.info(f"Model training completed. Direction accuracy: {dir_accuracy:.3f}, Magnitude MAE: {mag_mae:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            return False
    
    def _prepare_targets(self, data: pd.DataFrame) -> Optional[Dict[str, np.ndarray]]:
        """Prepare target variables for training"""
        try:
            if not ML_AVAILABLE:
                # Return dummy targets for simulation
                n_samples = len(data) - self.sequence_length - self.prediction_horizon
                return {
                    'direction': np.random.randint(0, 3, n_samples),
                    'magnitude': np.random.rand(n_samples) * 0.02,
                    'volatility': np.random.rand(n_samples) * 0.01
                }
            
            # Calculate future returns for direction prediction
            future_returns = []
            magnitude_targets = []
            volatility_targets = []
            
            for i in range(self.sequence_length, len(data) - self.prediction_horizon):
                current_price = data['Close'].iloc[i]
                future_price = data['Close'].iloc[i + self.prediction_horizon]
                
                # Calculate return
                return_pct = (future_price - current_price) / current_price
                future_returns.append(return_pct)
                
                # Magnitude (absolute return)
                magnitude_targets.append(abs(return_pct))
                
                # Volatility (standard deviation of recent returns)
                recent_returns = data['Close'].iloc[max(0, i-20):i].pct_change().dropna()
                volatility = recent_returns.std() if len(recent_returns) > 5 else 0.01
                volatility_targets.append(volatility)
            
            # Convert returns to direction classes
            # 0: Down (< -0.5%), 1: Sideways (-0.5% to 0.5%), 2: Up (> 0.5%)
            direction_targets = []
            for ret in future_returns:
                if ret < -0.005:
                    direction_targets.append(0)  # Down
                elif ret > 0.005:
                    direction_targets.append(2)  # Up
                else:
                    direction_targets.append(1)  # Sideways
            
            return {
                'direction': np.array(direction_targets),
                'magnitude': np.array(magnitude_targets),
                'volatility': np.array(volatility_targets)
            }
            
        except Exception as e:
            self.logger.error(f"Target preparation failed: {str(e)}")
            return None
    
    def _build_direction_model(self, input_shape: Tuple) -> Sequential:
        """Build LSTM model for direction prediction"""
        if not ML_AVAILABLE:
            return None
        
        model = Sequential([
            Bidirectional(LSTM(self.lstm_units[0], return_sequences=True, 
                             kernel_regularizer=l2(0.01)), 
                         input_shape=(input_shape[1], input_shape[2])),
            Dropout(self.dropout_rate),
            
            Bidirectional(LSTM(self.lstm_units[1], return_sequences=True,
                             kernel_regularizer=l2(0.01))),
            Dropout(self.dropout_rate),
            
            LSTM(self.lstm_units[2], kernel_regularizer=l2(0.01)),
            Dropout(self.dropout_rate),
            
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(self.dropout_rate),
            
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes: Down, Sideways, Up
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_magnitude_model(self, input_shape: Tuple) -> Sequential:
        """Build LSTM model for magnitude prediction"""
        if not ML_AVAILABLE:
            return None
        
        model = Sequential([
            Bidirectional(LSTM(self.lstm_units[0], return_sequences=True,
                             kernel_regularizer=l2(0.01)), 
                         input_shape=(input_shape[1], input_shape[2])),
            Dropout(self.dropout_rate),
            
            Bidirectional(LSTM(self.lstm_units[1], return_sequences=True,
                             kernel_regularizer=l2(0.01))),
            Dropout(self.dropout_rate),
            
            LSTM(self.lstm_units[2], kernel_regularizer=l2(0.01)),
            Dropout(self.dropout_rate),
            
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(self.dropout_rate),
            
            Dense(32, activation='relu'),
            Dense(1, activation='linear')  # Regression output
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_volatility_model(self, input_shape: Tuple) -> Sequential:
        """Build LSTM model for volatility prediction"""
        if not ML_AVAILABLE:
            return None
        
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True,
                             kernel_regularizer=l2(0.01)), 
                         input_shape=(input_shape[1], input_shape[2])),
            Dropout(self.dropout_rate),
            
            LSTM(32, kernel_regularizer=l2(0.01)),
            Dropout(self.dropout_rate),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # Regression output
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _get_callbacks(self) -> List:
        """Get training callbacks"""
        if not ML_AVAILABLE:
            return []
        
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
    
    def _generate_predictions(self, features: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Generate predictions using trained models"""
        try:
            if not ML_AVAILABLE or features is None or len(features) == 0:
                # Return dummy predictions for simulation
                n_samples = len(features) if features is not None else 10
                return {
                    'direction_probs': np.random.rand(n_samples, 3),
                    'direction_class': np.random.randint(0, 3, n_samples),
                    'magnitude': np.random.rand(n_samples) * 0.02,
                    'volatility': np.random.rand(n_samples) * 0.01,
                    'confidence': np.random.rand(n_samples)
                }
            
            # Get recent features for prediction
            recent_features = features[-10:]  # Last 10 sequences
            
            # Direction predictions
            direction_probs = self.direction_model.predict(recent_features, verbose=0)
            direction_class = np.argmax(direction_probs, axis=1)
            
            # Magnitude predictions
            magnitude_pred = self.magnitude_model.predict(recent_features, verbose=0)
            
            # Volatility predictions
            volatility_pred = self.volatility_model.predict(recent_features, verbose=0)
            
            # Calculate prediction confidence
            confidence = self._calculate_prediction_confidence(
                direction_probs, magnitude_pred, volatility_pred
            )
            
            return {
                'direction_probs': direction_probs,
                'direction_class': direction_class,
                'magnitude': magnitude_pred.flatten(),
                'volatility': volatility_pred.flatten(),
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {str(e)}")
            return None
    
    def _calculate_prediction_confidence(self, direction_probs: np.ndarray, 
                                       magnitude_pred: np.ndarray, 
                                       volatility_pred: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for predictions"""
        try:
            confidence_scores = []
            
            for i in range(len(direction_probs)):
                # Direction confidence (max probability)
                dir_confidence = np.max(direction_probs[i])
                
                # Magnitude confidence (based on predicted volatility)
                mag = magnitude_pred[i] if len(magnitude_pred.shape) > 1 else magnitude_pred[i]
                vol = volatility_pred[i] if len(volatility_pred.shape) > 1 else volatility_pred[i]
                
                # Higher magnitude relative to volatility = higher confidence
                mag_confidence = min(abs(mag) / (vol + 0.001), 1.0) if vol > 0 else 0.5
                
                # Model performance factor
                perf_factor = min(self.model_performance.get('direction_accuracy', 0.5) * 2, 1.0)
                
                # Combined confidence
                combined_confidence = (dir_confidence * 0.5 + 
                                     mag_confidence * 0.3 + 
                                     perf_factor * 0.2)
                
                confidence_scores.append(combined_confidence)
            
            return np.array(confidence_scores)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {str(e)}")
            return np.array([0.5] * len(direction_probs))
    
    def _predictions_to_signals(self, predictions: Dict[str, np.ndarray], 
                              data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Convert predictions to trading signals"""
        signals = []
        
        try:
            current_price = data['Close'].iloc[-1]
            current_time = data.index[-1]
            
            # Use the most recent prediction
            latest_direction = predictions['direction_class'][-1]
            latest_magnitude = predictions['magnitude'][-1]
            latest_volatility = predictions['volatility'][-1]
            latest_confidence = predictions['confidence'][-1]
            
            # Only generate signal if confidence is above threshold
            if latest_confidence < self.min_confidence:
                return signals
            
            # Determine signal type
            if latest_direction == 2:  # Up
                signal_type = SignalType.BUY
            elif latest_direction == 0:  # Down
                signal_type = SignalType.SELL
            else:  # Sideways
                return signals  # No signal for sideways prediction
            
            # Calculate stop loss and take profit based on predictions
            atr = self._calculate_atr(data, 14)
            volatility_factor = max(latest_volatility * 100, 0.01)  # Convert to percentage
            
            if signal_type == SignalType.BUY:
                # Use predicted magnitude and volatility for targets
                predicted_move = latest_magnitude * current_price
                stop_loss = current_price - (atr * 2 * volatility_factor)
                take_profit = current_price + max(predicted_move, atr * 3)
                
            else:  # SELL
                predicted_move = latest_magnitude * current_price
                stop_loss = current_price + (atr * 2 * volatility_factor)
                take_profit = current_price - max(predicted_move, atr * 3)
            
            # Ensure minimum risk-reward ratio
            if signal_type == SignalType.BUY:
                risk = current_price - stop_loss
                reward = take_profit - current_price
            else:
                risk = stop_loss - current_price
                reward = current_price - take_profit
            
            if risk > 0 and (reward / risk) >= 1.5:  # Minimum 1.5:1 RR
                
                # Determine signal grade
                grade = self._determine_signal_grade(latest_confidence, latest_magnitude)
                
                signal = Signal(
                    timestamp=current_time,
                    symbol=symbol,
                    strategy_name="lstm_predictor",
                    signal_type=signal_type,
                    confidence=latest_confidence,
                    price=current_price,
                    timeframe=timeframe,
                    strength=latest_magnitude,
                    grade=grade,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'predicted_direction': latest_direction,
                        'predicted_magnitude': latest_magnitude,
                        'predicted_volatility': latest_volatility,
                        'model_accuracy': self.model_performance.get('direction_accuracy', 0.0),
                        'prediction_horizon': self.prediction_horizon,
                        'direction_probabilities': predictions['direction_probs'][-1].tolist(),
                        'ml_model': 'LSTM'
                    }
                )
                
                signals.append(signal)
                
        except Exception as e:
            self.logger.error(f"Signal conversion failed: {str(e)}")
        
        return signals
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(data) < period + 1:
                return data['Close'].iloc[-1] * 0.01  # 1% as default
            
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else data['Close'].iloc[-1] * 0.01
            
        except Exception as e:
            self.logger.error(f"ATR calculation failed: {str(e)}")
            return data['Close'].iloc[-1] * 0.01
    
    def _determine_signal_grade(self, confidence: float, magnitude: float) -> SignalGrade:
        """Determine signal grade based on confidence and predicted magnitude"""
        try:
            # Combine confidence and magnitude for grading
            combined_score = (confidence * 0.7) + (min(magnitude * 50, 1.0) * 0.3)
            
            if combined_score >= 0.85:
                return SignalGrade.A
            elif combined_score >= 0.75:
                return SignalGrade.B
            elif combined_score >= 0.65:
                return SignalGrade.C
            else:
                return SignalGrade.D
                
        except Exception as e:
            self.logger.error(f"Signal grade determination failed: {str(e)}")
            return SignalGrade.C
    
    def _update_training_data(self, features: np.ndarray, data: pd.DataFrame) -> None:
        """Update training data for future retraining"""
        try:
            if features is None or len(features) == 0:
                return
            
            # Add recent features to training data storage
            self.training_data['features'].extend(features[-10:].tolist())
            
            # Prepare corresponding targets for the recent data
            targets = self._prepare_targets(data)
            if targets is not None:
                self.training_data['direction_targets'].extend(targets['direction'][-10:].tolist())
                self.training_data['magnitude_targets'].extend(targets['magnitude'][-10:].tolist())
                self.training_data['volatility_targets'].extend(targets['volatility'][-10:].tolist())
            
            # Limit storage size to prevent memory issues
            max_storage = self.min_training_samples * 2
            if len(self.training_data['features']) > max_storage:
                # Keep only the most recent data
                self.training_data['features'] = self.training_data['features'][-max_storage:]
                self.training_data['direction_targets'] = self.training_data['direction_targets'][-max_storage:]
                self.training_data['magnitude_targets'] = self.training_data['magnitude_targets'][-max_storage:]
                self.training_data['volatility_targets'] = self.training_data['volatility_targets'][-max_storage:]
                
        except Exception as e:
            self.logger.error(f"Training data update failed: {str(e)}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and performance metrics"""
        return {
            'name': 'LSTM Predictor',
            'version': '2.0.0',
            'type': 'Machine Learning',
            'ml_available': ML_AVAILABLE,
            'models_trained': all([
                self.direction_model is not None,
                self.magnitude_model is not None,
                self.volatility_model is not None
            ]),
            'model_performance': self.model_performance,
            'min_confidence': self.min_confidence,
            'prediction_horizon': self.prediction_horizon,
            'sequence_length': self.sequence_length,
            'feature_count': len(self.feature_names),
            'training_data_size': len(self.training_data['features']),
            'parameters': {
                'lstm_units': self.lstm_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'retrain_frequency': self.retrain_frequency
            }
        }
    
    def save_models(self, filepath: str) -> bool:
        """Save trained models to disk"""
        try:
            if not ML_AVAILABLE:
                return False
            
            model_data = {
                'direction_model': self.direction_model,
                'magnitude_model': self.magnitude_model,
                'volatility_model': self.volatility_model,
                'feature_scaler': self.feature_scaler,
                'price_scaler': self.price_scaler,
                'performance': self.model_performance,
                'feature_names': self.feature_names
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Models saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        """Load trained models from disk"""
        try:
            if not ML_AVAILABLE:
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.direction_model = model_data['direction_model']
            self.magnitude_model = model_data['magnitude_model']
            self.volatility_model = model_data['volatility_model']
            self.feature_scaler = model_data['feature_scaler']
            self.price_scaler = model_data['price_scaler']
            self.model_performance = model_data['performance']
            self.feature_names = model_data['feature_names']
            
            self.logger.info(f"Models loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            return False


# Testing function
if __name__ == "__main__":
    """Test the LSTM Predictor strategy"""
    
    # Test configuration
    test_config = {
        'parameters': {
            'confidence_threshold': 0.75,
            'prediction_horizon': 12,
            'feature_lookback': 50,
            'retrain_frequency': 'weekly',
            'min_training_samples': 1000
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
            
            # Generate sample OHLCV data with trend
            np.random.seed(42)
            trend = np.linspace(1950, 1980, len(dates))
            noise = np.cumsum(np.random.randn(len(dates)) * 2)
            close_prices = trend + noise
            
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
    
    # Create strategy instance
    mock_mt5 = MockMT5Manager()
    strategy = LSTMPredictor(test_config, mock_mt5)
    
    # Generate signals
    signals = strategy.generate_signals("XAUUSDm", "M15")
    
    print(f"Generated {len(signals)} LSTM signals")
    for signal in signals:
        print(f"Signal: {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
    
    # Get strategy info
    info = strategy.get_strategy_info()
    print(f"\nStrategy Info:")
    print(f"ML Available: {info['ml_available']}")
    print(f"Models Trained: {info['models_trained']}")
    print(f"Performance: {info['model_performance']}")
    
    print("LSTM Predictor strategy test completed!")
