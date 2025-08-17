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
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost/Scikit-learn not available. Strategy will run in simulation mode.")

# Import base classes
try:
    from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade
except ImportError:
    from core.base import AbstractStrategy, Signal, SignalType, SignalGrade


class XGBoostClassifier(AbstractStrategy):
    """XGBoost-based trading strategy for signal classification"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize XGBoost classifier strategy"""
        super().__init__(config)
        
        self.strategy_name = "XGBoostClassifier"
        self.lookback_bars = config.get('lookback_bars', 200)
        self.min_confidence = config.get('min_confidence', 0.65)
        
        # XGBoost parameters
        self.xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
        
        # Model components
        self.model = None
        self.scaler = StandardScaler() if XGBOOST_AVAILABLE else None
        self.label_encoder = LabelEncoder() if XGBOOST_AVAILABLE else None
        self.is_trained = False
        
        # Performance tracking
        self.predictions_made = 0
        self.model_accuracy = 0.0
        
        # Logger
        self.logger = logging.getLogger(self.strategy_name)
        
        # Initialize model if XGBoost is available
        if XGBOOST_AVAILABLE:
            self._initialize_model()
        
        self.logger.info(f"{self.strategy_name} initialized (XGBoost available: {XGBOOST_AVAILABLE})")
    
    def _initialize_model(self):
        """Initialize XGBoost model"""
        try:
            if XGBOOST_AVAILABLE:
                self.model = xgb.XGBClassifier(**self.xgb_params)
                self.label_encoder.fit(['BUY', 'SELL', 'HOLD'])
                self.logger.info("XGBoost model initialized")
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            self.model = None
    
    def generate_signal(self, data: pd.DataFrame, symbol: str = "XAUUSDm", 
                       timeframe: str = "M15") -> Optional[Signal]:
        """Generate trading signal using XGBoost classifier"""
        try:
            if data is None or len(data) < self.lookback_bars:
                return None
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                return self._generate_fallback_signal(data, symbol, timeframe)
            
            # Train model if not trained
            if not self.is_trained:
                self._train_model(data)
            
            # Generate features for prediction
            features = self._extract_features(data)
            if features is None:
                return self._generate_fallback_signal(data, symbol, timeframe)
            
            # Make prediction
            if XGBOOST_AVAILABLE and self.model and self.is_trained:
                prediction, confidence = self._make_prediction(features)
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
                stop_loss = current_price - (2.0 * atr)
                take_profit = current_price + (3.0 * atr)
            else:
                stop_loss = current_price + (2.0 * atr)
                take_profit = current_price - (3.0 * atr)
            
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
                    'model_prediction': prediction,
                    'model_accuracy': self.model_accuracy,
                    'predictions_made': self.predictions_made,
                    'xgboost_available': XGBOOST_AVAILABLE
                }
            )
            
            self.predictions_made += 1
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return self._generate_fallback_signal(data, symbol, timeframe)
    
    def _extract_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features for ML model"""
        try:
            if len(data) < 50:
                return None
            
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            features = []
            
            # Technical indicators
            sma_10 = close.rolling(10).mean().iloc[-1]
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            
            if pd.isna(sma_10) or pd.isna(sma_20) or pd.isna(sma_50):
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
            volatility = close.rolling(20).std().iloc[-1]
            features.append(volatility / close.iloc[-1] if not pd.isna(volatility) else 0)
            
            # Volume features
            vol_sma = volume.rolling(20).mean().iloc[-1]
            features.append(volume.iloc[-1] / vol_sma - 1 if not pd.isna(vol_sma) and vol_sma > 0 else 0)
            
            # ATR
            atr = self._calculate_atr(data)
            features.append(atr / close.iloc[-1])
            
            # Convert to numpy array
            feature_array = np.array(features).reshape(1, -1)
            
            # Scale features if scaler is available
            if self.scaler and XGBOOST_AVAILABLE:
                try:
                    feature_array = self.scaler.transform(feature_array)
                except:
                    pass
            
            return feature_array
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
    
    def _train_model(self, data: pd.DataFrame):
        """Train the XGBoost model"""
        try:
            if not XGBOOST_AVAILABLE:
                return
            
            self.logger.info("Training XGBoost model...")
            
            # Prepare training data
            X, y = self._prepare_training_data(data)
            
            if X is None or len(X) < 50:
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
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
            self.logger.error(f"Model training failed: {e}")
            self.is_trained = False
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data with features and labels"""
        try:
            if len(data) < 100:
                return None, None
            
            X_list = []
            y_list = []
            
            # Generate features and labels for training
            for i in range(50, len(data) - 5):
                window_data = data.iloc[i-50:i]
                features = self._extract_simple_features(window_data)
                
                if features is None:
                    continue
                
                # Generate label based on future price movement
                current_price = data['Close'].iloc[i]
                future_price = data['Close'].iloc[i+5]
                
                price_change = (future_price - current_price) / current_price
                
                if price_change > 0.002:
                    label = 'BUY'
                elif price_change < -0.002:
                    label = 'SELL'
                else:
                    label = 'HOLD'
                
                X_list.append(features)
                y_list.append(label)
            
            if len(X_list) == 0:
                return None, None
            
            X = np.array(X_list)
            y = self.label_encoder.transform(y_list)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed: {e}")
            return None, None
    
    def _extract_simple_features(self, data: pd.DataFrame) -> Optional[List[float]]:
        """Extract simplified features for training"""
        try:
            if len(data) < 20:
                return None
            
            close = data['Close']
            
            # Basic features
            sma_10 = close.rolling(10).mean().iloc[-1]
            sma_20 = close.rolling(20).mean().iloc[-1]
            
            if pd.isna(sma_10) or pd.isna(sma_20):
                return None
            
            features = [
                close.iloc[-1] / sma_10 - 1,
                close.iloc[-1] / sma_20 - 1,
                sma_10 / sma_20 - 1,
                close.pct_change(1).iloc[-1] if not pd.isna(close.pct_change(1).iloc[-1]) else 0,
                close.pct_change(5).iloc[-1] if not pd.isna(close.pct_change(5).iloc[-1]) else 0,
                self._calculate_rsi(close) / 100.0
            ]
            
            return features
            
        except Exception:
            return None
    
    def _make_prediction(self, features: np.ndarray) -> Tuple[str, float]:
        """Make prediction using trained model"""
        try:
            if not self.is_trained or self.model is None:
                return 'HOLD', 0.5
            
            probabilities = self.model.predict_proba(features)[0]
            classes = self.label_encoder.classes_
            
            max_prob_idx = np.argmax(probabilities)
            prediction = classes[max_prob_idx]
            confidence = probabilities[max_prob_idx]
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return 'HOLD', 0.5
    
    def _make_fallback_prediction(self, data: pd.DataFrame) -> Tuple[str, float]:
        """Make simple prediction when ML model is not available"""
        try:
            close = data['Close']
            
            if len(close) >= 20:
                sma_10 = close.rolling(10).mean().iloc[-1]
                sma_20 = close.rolling(20).mean().iloc[-1]
                
                if not pd.isna(sma_10) and not pd.isna(sma_20):
                    if sma_10 > sma_20 * 1.001:
                        return 'BUY', 0.6
                    elif sma_10 < sma_20 * 0.999:
                        return 'SELL', 0.6
            
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
            
            price_range = current_price * 0.01
            
            if signal_type == SignalType.BUY:
                stop_loss = current_price - price_range
                take_profit = current_price + (price_range * 1.5)
            else:
                stop_loss = current_price + price_range
                take_profit = current_price - (price_range * 1.5)
            
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
        """Analyze market conditions"""
        return {
            'strategy': self.strategy_name,
            'symbol': symbol,
            'timeframe': timeframe,
            'xgboost_available': XGBOOST_AVAILABLE,
            'model_trained': self.is_trained,
            'model_accuracy': self.model_accuracy,
            'predictions_made': self.predictions_made
        }