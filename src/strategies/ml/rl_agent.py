"""
Reinforcement Learning Agent Strategy - Advanced ML Trading Strategy
====================================================================

Deep Q-Network (DQN) based reinforcement learning agent for XAUUSD trading.
Uses experience replay and target networks for stable learning.
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
from typing import Dict, List, Any, Optional, Tuple, Deque
from collections import deque
import logging
import random

# ML imports with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, clone_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
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


class RLAgent(AbstractStrategy):
    """Reinforcement Learning agent using Deep Q-Network for trading"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize RL agent strategy"""
        super().__init__(config)
        
        self.strategy_name = "RLAgent"
        self.lookback_bars = config.get('lookback_bars', 100)
        self.min_confidence = config.get('min_confidence', 0.60)
        
        # RL parameters
        self.state_size = config.get('state_size', 10)
        self.action_size = 3  # BUY, SELL, HOLD
        self.memory_size = config.get('memory_size', 2000)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epsilon = config.get('epsilon', 1.0)  # Exploration rate
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.gamma = config.get('gamma', 0.95)  # Discount factor
        self.update_target_freq = config.get('update_target_freq', 100)
        
        # Model components
        self.q_network = None
        self.target_network = None
        self.memory = deque(maxlen=self.memory_size)
        self.is_trained = False
        
        # Performance tracking
        self.total_reward = 0.0
        self.episode_count = 0
        self.predictions_made = 0
        self.training_steps = 0
        
        # Trading state
        self.last_action = 2  # Start with HOLD
        self.last_price = 0.0
        self.position_value = 0.0
        
        # Logger
        self.logger = logging.getLogger(self.strategy_name)
        
        # Initialize networks if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            self._initialize_networks()
        
        self.logger.info(f"{self.strategy_name} initialized (TensorFlow available: {TENSORFLOW_AVAILABLE})")
    
    def _initialize_networks(self):
        """Initialize Q-network and target network"""
        try:
            if not TENSORFLOW_AVAILABLE:
                return
            
            # Create main Q-network
            self.q_network = Sequential([
                Dense(64, activation='relu', input_shape=(self.state_size,)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(self.action_size, activation='linear')
            ])
            
            self.q_network.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse'
            )
            
            # Create target network (copy of main network)
            self.target_network = clone_model(self.q_network)
            self.target_network.set_weights(self.q_network.get_weights())
            
            self.logger.info("RL networks initialized")
            
        except Exception as e:
            self.logger.error(f"Network initialization failed: {e}")
            self.q_network = None
            self.target_network = None
    
    def generate_signal(self, data: pd.DataFrame, symbol: str = "XAUUSDm", 
                       timeframe: str = "M15") -> Optional[Signal]:
        """Generate trading signal using RL agent"""
        try:
            if data is None or len(data) < self.lookback_bars:
                return None
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                return self._generate_fallback_signal(data, symbol, timeframe)
            
            # Extract state features
            state = self._extract_state(data)
            if state is None:
                return self._generate_fallback_signal(data, symbol, timeframe)
            
            # Get action from RL agent
            if TENSORFLOW_AVAILABLE and self.q_network:
                action, confidence = self._get_action(state)
            else:
                action, confidence = self._get_fallback_action(data)
            
            # Convert action to signal
            current_price = float(data['Close'].iloc[-1])
            
            # Calculate reward for previous action and store experience
            if self.last_price > 0:
                reward = self._calculate_reward(current_price)
                self._store_experience(reward, state)
                
                # Train the agent periodically
                if len(self.memory) > self.batch_size and self.training_steps % 10 == 0:
                    self._replay_training()
            
            # Update state for next iteration
            self.last_price = current_price
            self.last_action = action
            
            # Convert action to signal type
            if action == 0:  # BUY
                signal_type = SignalType.BUY
                prediction = 'BUY'
            elif action == 1:  # SELL
                signal_type = SignalType.SELL
                prediction = 'SELL'
            else:  # HOLD
                return None
            
            # Check confidence threshold
            if confidence < self.min_confidence:
                return None
            
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
                    'rl_action': action,
                    'rl_prediction': prediction,
                    'epsilon': self.epsilon,
                    'total_reward': self.total_reward,
                    'episode_count': self.episode_count,
                    'memory_size': len(self.memory),
                    'tensorflow_available': TENSORFLOW_AVAILABLE
                }
            )
            
            self.predictions_made += 1
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return self._generate_fallback_signal(data, symbol, timeframe)
    
    def _extract_state(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract state features for RL agent"""
        try:
            if len(data) < 50:
                return None
            
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            features = []
            
            # Price-based features
            current_price = close.iloc[-1]
            
            # Moving averages
            sma_5 = close.rolling(5).mean().iloc[-1]
            sma_10 = close.rolling(10).mean().iloc[-1]
            sma_20 = close.rolling(20).mean().iloc[-1]
            
            if pd.isna(sma_5) or pd.isna(sma_10) or pd.isna(sma_20):
                return None
            
            features.extend([
                (current_price - sma_5) / current_price,
                (current_price - sma_10) / current_price,
                (current_price - sma_20) / current_price,
                (sma_5 - sma_10) / current_price,
                (sma_10 - sma_20) / current_price
            ])
            
            # Technical indicators
            rsi = self._calculate_rsi(close)
            features.append((rsi - 50) / 50)  # Normalize RSI
            
            # Price momentum
            returns_1 = close.pct_change(1).iloc[-1]
            returns_5 = close.pct_change(5).iloc[-1]
            
            features.extend([
                returns_1 if not pd.isna(returns_1) else 0,
                returns_5 if not pd.isna(returns_5) else 0
            ])
            
            # Volatility
            volatility = close.rolling(10).std().iloc[-1] / current_price
            features.append(volatility if not pd.isna(volatility) else 0)
            
            # Volume indicator
            vol_sma = volume.rolling(10).mean().iloc[-1]
            vol_ratio = volume.iloc[-1] / vol_sma if not pd.isna(vol_sma) and vol_sma > 0 else 1
            features.append(min(max(vol_ratio - 1, -2), 2))  # Clamp between -2 and 2
            
            # Ensure we have exactly state_size features
            while len(features) < self.state_size:
                features.append(0.0)
            
            return np.array(features[:self.state_size]).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"State extraction failed: {e}")
            return None
    
    def _get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Get action from RL agent using epsilon-greedy policy"""
        try:
            if not TENSORFLOW_AVAILABLE or self.q_network is None:
                return self._get_fallback_action_simple()
            
            # Epsilon-greedy action selection
            if np.random.random() <= self.epsilon:
                # Random action (exploration)
                action = random.randrange(self.action_size)
                confidence = 0.6  # Moderate confidence for random actions
            else:
                # Best action from Q-network (exploitation)
                q_values = self.q_network.predict(state, verbose=0)[0]
                action = np.argmax(q_values)
                
                # Calculate confidence based on Q-value difference
                sorted_q = np.sort(q_values)[::-1]
                if len(sorted_q) > 1:
                    confidence = min(0.9, 0.5 + abs(sorted_q[0] - sorted_q[1]) / 2)
                else:
                    confidence = 0.6
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            return action, confidence
            
        except Exception as e:
            self.logger.error(f"Action selection failed: {e}")
            return self._get_fallback_action_simple()
    
    def _get_fallback_action(self, data: pd.DataFrame) -> Tuple[int, float]:
        """Get fallback action when RL agent is not available"""
        try:
            close = data['Close']
            
            if len(close) >= 20:
                sma_10 = close.rolling(10).mean().iloc[-1]
                sma_20 = close.rolling(20).mean().iloc[-1]
                rsi = self._calculate_rsi(close)
                
                if not pd.isna(sma_10) and not pd.isna(sma_20):
                    # Simple moving average crossover with RSI filter
                    if sma_10 > sma_20 * 1.001 and rsi < 70:
                        return 0, 0.65  # BUY
                    elif sma_10 < sma_20 * 0.999 and rsi > 30:
                        return 1, 0.65  # SELL
            
            return 2, 0.5  # HOLD
            
        except Exception:
            return 2, 0.5  # HOLD
    
    def _get_fallback_action_simple(self) -> Tuple[int, float]:
        """Simple fallback action"""
        return 2, 0.5  # HOLD
    
    def _calculate_reward(self, current_price: float) -> float:
        """Calculate reward based on price movement and action taken"""
        try:
            if self.last_price <= 0:
                return 0.0
            
            price_change = (current_price - self.last_price) / self.last_price
            
            # Reward based on action and price movement
            if self.last_action == 0:  # BUY
                reward = price_change * 100  # Positive if price went up
            elif self.last_action == 1:  # SELL
                reward = -price_change * 100  # Positive if price went down
            else:  # HOLD
                reward = -abs(price_change) * 10  # Small penalty for missing moves
            
            # Add small penalty for transaction costs
            if self.last_action != 2:  # Not HOLD
                reward -= 0.1
            
            self.total_reward += reward
            return reward
            
        except Exception as e:
            self.logger.error(f"Reward calculation failed: {e}")
            return 0.0
    
    def _store_experience(self, reward: float, next_state: np.ndarray):
        """Store experience in replay memory"""
        try:
            if hasattr(self, 'last_state') and self.last_state is not None:
                experience = (
                    self.last_state,
                    self.last_action,
                    reward,
                    next_state,
                    False  # done flag (always False for continuous trading)
                )
                self.memory.append(experience)
            
            self.last_state = next_state.copy()
            
        except Exception as e:
            self.logger.error(f"Experience storage failed: {e}")
    
    def _replay_training(self):
        """Train the RL agent using experience replay"""
        try:
            if not TENSORFLOW_AVAILABLE or self.q_network is None or len(self.memory) < self.batch_size:
                return
            
            # Sample random batch from memory
            batch = random.sample(self.memory, self.batch_size)
            
            states = np.array([e[0].flatten() for e in batch])
            actions = np.array([e[1] for e in batch])
            rewards = np.array([e[2] for e in batch])
            next_states = np.array([e[3].flatten() for e in batch])
            dones = np.array([e[4] for e in batch])
            
            # Current Q-values
            current_q_values = self.q_network.predict(states, verbose=0)
            
            # Next Q-values from target network
            next_q_values = self.target_network.predict(next_states, verbose=0)
            
            # Calculate target Q-values
            target_q_values = current_q_values.copy()
            
            for i in range(self.batch_size):
                if dones[i]:
                    target_q_values[i][actions[i]] = rewards[i]
                else:
                    target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
            
            # Train the network
            self.q_network.fit(states, target_q_values, epochs=1, verbose=0)
            
            self.training_steps += 1
            
            # Update target network periodically
            if self.training_steps % self.update_target_freq == 0:
                self.target_network.set_weights(self.q_network.get_weights())
                self.logger.info(f"Target network updated at step {self.training_steps}")
            
        except Exception as e:
            self.logger.error(f"Replay training failed: {e}")
    
    def _generate_fallback_signal(self, data: pd.DataFrame, symbol: str, 
                                 timeframe: str) -> Optional[Signal]:
        """Generate fallback signal when main logic fails"""
        try:
            if len(data) < 20:
                return None
            
            action, confidence = self._get_fallback_action(data)
            
            if action == 2 or confidence < 0.6:  # HOLD or low confidence
                return None
            
            signal_type = SignalType.BUY if action == 0 else SignalType.SELL
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
    
    def reset_episode(self):
        """Reset episode for new trading session"""
        self.episode_count += 1
        self.last_action = 2  # HOLD
        self.last_price = 0.0
        self.position_value = 0.0
        if hasattr(self, 'last_state'):
            delattr(self, 'last_state')
        
        self.logger.info(f"Episode {self.episode_count} started")
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze RL agent performance and state"""
        return {
            'strategy': self.strategy_name,
            'symbol': symbol,
            'timeframe': timeframe,
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'is_trained': self.is_trained,
            'total_reward': self.total_reward,
            'episode_count': self.episode_count,
            'predictions_made': self.predictions_made,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_steps': self.training_steps
        }
