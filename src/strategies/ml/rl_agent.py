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

# Add src to path for module resolution when run as script
#sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

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

# Import base classes directly
from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade, StrategyPerformance


class RLAgentStrategy(AbstractStrategy):
    """Reinforcement Learning agent using Deep Q-Network for trading"""
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """Initialize RL agent strategy - 8GB RAM optimized"""
        super().__init__(config, mt5_manager, database)
        
        # self.strategy_name is already set by AbstractStrategy
        self.lookback_bars = self.config.get('parameters', {}).get('lookback_bars', 80)
        self.min_confidence = self.config.get('parameters', {}).get('min_confidence', 0.55)
        
        # RL parameters - 8GB RAM optimized
        self.state_size = self.config.get('parameters', {}).get('state_size', 8)
        self.action_size = 3  # BUY, SELL, HOLD
        self.memory_size = self.config.get('parameters', {}).get('memory_size', 1000)
        self.batch_size = self.config.get('parameters', {}).get('batch_size', 16)
        self.learning_rate = self.config.get('parameters', {}).get('learning_rate', 0.001)
        self.epsilon = self.config.get('parameters', {}).get('epsilon', 0.8)
        self.epsilon_min = self.config.get('parameters', {}).get('epsilon_min', 0.05)
        self.epsilon_decay = self.config.get('parameters', {}).get('epsilon_decay', 0.99)
        self.gamma = self.config.get('parameters', {}).get('gamma', 0.9)
        self.update_target_freq = self.config.get('parameters', {}).get('update_target_freq', 50)
        
        # Memory optimization settings
        self.memory_cleanup_interval = self.config.get('parameters', {}).get('memory_cleanup_interval', 25)
        self.prediction_count = 0
        self.max_training_frequency = self.config.get('parameters', {}).get('max_training_frequency', 20)
        
        # Model components
        self.q_network = None
        self.target_network = None
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.memory_size)
        self.is_trained = False
        
        # Performance tracking (RL-specific)
        self.total_reward = 0.0
        self.episode_count = 0
        self.predictions_made = 0
        self.training_steps = 0
        
        # Trading state
        self.last_action: int = 2  # Start with HOLD (0=BUY, 1=SELL, 2=HOLD)
        self.last_price: float = 0.0
        self.position_value: float = 0.0
        self.last_state: Optional[np.ndarray] = None # Store last state for experience replay
        
        # self.logger is provided by AbstractStrategy
        
        # Initialize networks if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            self._initialize_networks()
        
        self.logger.info(f"{self.strategy_name} initialized (TensorFlow available: {TENSORFLOW_AVAILABLE})")
    
    def _initialize_networks(self):
        """Initialize Q-network and target network"""
        try:
            if not TENSORFLOW_AVAILABLE:
                return
            
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
            
            self.target_network = clone_model(self.q_network)
            self.target_network.set_weights(self.q_network.get_weights())
            
            self.logger.info("RL networks initialized")
            
        except Exception as e:
            self.logger.error(f"Network initialization failed: {e}", exc_info=True)
            self.q_network = None
            self.target_network = None
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """Generate trading signals using RL agent - 8GB RAM optimized"""
        signals = []
        try:
            self.logger.info(f"RL Agent - Analyzing {symbol} on {timeframe}")
            
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available for signal generation.")
                return []
            
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_bars)
            if data is None or len(data) < self.lookback_bars:
                self.logger.warning(f"Insufficient data for RL analysis: {len(data) if data is not None else 0}")
                return []
            
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.warning("Missing required data columns for RL analysis.")
                return []
            
            self.prediction_count += 1
            if self.prediction_count % self.memory_cleanup_interval == 0:
                self._cleanup_memory()
            
            state = self._extract_state(data)
            if state is None:
                self.logger.warning("State extraction failed, cannot generate signals.")
                return []
            
            current_price = float(data['Close'].iloc[-1])
            
            # Calculate reward for previous action and store experience
            if self.last_price > 0 and self.last_state is not None:
                reward = self._calculate_reward(current_price)
                self._store_experience(self.last_state, self.last_action, reward, state)
                
                # Train the agent (less frequently for speed)
                if (len(self.memory) > self.batch_size and 
                    self.training_steps % self.max_training_frequency == 0):
                    self._replay_training()
                    self.is_trained = True # Mark as trained after first training batch
            
            # Update last state and price for next iteration's reward calculation
            self.last_state = state.copy()
            self.last_price = current_price

            # Generate multiple signals with different exploration levels
            for i in range(3):  # Generate up to 3 signals
                action, confidence = self._get_action_enhanced(state, i)
                
                if action == 0:  # BUY
                    signal_type = SignalType.BUY
                    prediction_label = 'BUY'
                elif action == 1:  # SELL
                    signal_type = SignalType.SELL
                    prediction_label = 'SELL'
                else:  # HOLD
                    continue # No signal generated for HOLD
                
                if confidence < self.min_confidence:
                    continue
                
                atr = self._calculate_atr(data)
                if atr is None or atr == 0: # Ensure ATR is valid
                    atr = current_price * 0.01 # Fallback if ATR calculation fails

                risk_factor = 1 + (i * 0.3)
                
                if signal_type == SignalType.BUY:
                    stop_loss = current_price - (1.5 * atr * risk_factor)
                    take_profit = current_price + (2.5 * atr * confidence)
                else:
                    stop_loss = current_price + (1.5 * atr * risk_factor)
                    take_profit = current_price - (2.5 * atr * confidence)
                
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
                            'signal_reason': f'rl_agent_action_{i+1}',
                            'rl_action': action,
                            'rl_prediction_label': prediction_label,
                            'epsilon': self.epsilon,
                            'total_reward': self.total_reward,
                            'memory_size': len(self.memory),
                            'prediction_index': i
                        }
                    )
                    signals.append(signal)
            
            # For logging and performance tracking, use the first generated signal's action
            if signals:
                self.last_action = signals[0].metadata['rl_action']
            else:
                self.last_action = 2 # Default to HOLD if no signal generated

            self.predictions_made += 1 # Count prediction attempts
            
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
        """
        Analyze current market conditions and RL agent's operational state.
        This method serves as the required analyze method from AbstractStrategy.
        """
        try:
            current_price = float(data['Close'].iloc[-1]) if 'Close' in data.columns and not data.empty else 0.0

            analysis_output = {
                'strategy': self.strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_time': datetime.now().isoformat(),
                'current_price': current_price,
                'tensorflow_available': TENSORFLOW_AVAILABLE,
                'rl_agent_trained': self.is_trained,
                'total_episodes': self.episode_count,
                'total_predictions_made': self.predictions_made,
                'current_epsilon': round(self.epsilon, 3),
                'current_total_reward': round(self.total_reward, 2),
                'memory_usage_percent': round((len(self.memory) / self.memory_size) * 100, 2) if self.memory_size > 0 else 0,
                'training_steps_performed': self.training_steps,
                'last_action_taken': self.last_action,
                'lookback_bars_used': self.lookback_bars
            }
            return analysis_output
        except Exception as e:
            self.logger.error(f"Analysis method failed: {e}", exc_info=True)
            return {'strategy': self.strategy_name, 'error': str(e)}

    def _extract_state(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract state features for RL agent"""
        try:
            if len(data) < 50:
                self.logger.warning("Insufficient data for state extraction.")
                return None
            
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            features = []
            
            # Price-based features
            current_price = close.iloc[-1]
            
            # Moving averages
            sma_5 = close.rolling(min(5, len(close))).mean().iloc[-1]
            sma_10 = close.rolling(min(10, len(close))).mean().iloc[-1]
            sma_20 = close.rolling(min(20, len(close))).mean().iloc[-1]
            
            if pd.isna(sma_5) or pd.isna(sma_10) or pd.isna(sma_20):
                self.logger.warning("NaN in SMA calculations for state extraction.")
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
            volatility = close.rolling(min(10, len(close))).std().iloc[-1] / current_price if len(close) >= 10 else 0
            features.append(volatility if not pd.isna(volatility) else 0)
            
            # Volume indicator
            vol_sma = volume.rolling(min(10, len(volume))).mean().iloc[-1] if len(volume) >= 10 else 0
            vol_ratio = volume.iloc[-1] / vol_sma if not pd.isna(vol_sma) and vol_sma > 0 else 1
            features.append(min(max(vol_ratio - 1, -2), 2))  # Clamp between -2 and 2
            
            # Ensure we have exactly state_size features
            while len(features) < self.state_size:
                features.append(0.0)
            
            return np.array(features[:self.state_size]).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"State extraction failed: {e}", exc_info=True)
            return None
    
    def _get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Get action from RL agent using epsilon-greedy policy"""
        try:
            if not TENSORFLOW_AVAILABLE or self.q_network is None:
                return self._get_fallback_action_simple()
            
            # Epsilon-greedy action selection
            if np.random.random() <= self.epsilon:
                action = random.randrange(self.action_size)
                confidence = 0.6
            else:
                q_values = self.q_network.predict(state, verbose=0)[0]
                action = np.argmax(q_values)
                
                sorted_q = np.sort(q_values)[::-1]
                if len(sorted_q) > 1:
                    confidence = min(0.9, 0.5 + abs(sorted_q[0] - sorted_q[1]) / 2)
                else:
                    confidence = 0.6
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            return action, confidence
            
        except Exception as e:
            self.logger.error(f"Action selection failed: {e}", exc_info=True)
            return self._get_fallback_action_simple()
    
    def _get_fallback_action(self, data: pd.DataFrame) -> Tuple[int, float]:
        """Get fallback action when RL agent is not available"""
        try:
            close = data['Close']
            
            if len(close) >= 20:
                sma_10 = close.rolling(min(10, len(close))).mean().iloc[-1]
                sma_20 = close.rolling(min(20, len(close))).mean().iloc[-1]
                rsi = self._calculate_rsi(close)
                
                if not pd.isna(sma_10) and not pd.isna(sma_20):
                    if sma_10 > sma_20 * 1.001 and rsi < 70:
                        return 0, 0.65
                    elif sma_10 < sma_20 * 0.999 and rsi > 30:
                        return 1, 0.65
            
            return 2, 0.5
            
        except Exception as e:
            self.logger.error(f"Fallback action failed: {e}", exc_info=True)
            return 2, 0.5
    
    def _get_fallback_action_simple(self) -> Tuple[int, float]:
        """Simple fallback action"""
        return 2, 0.5
    
    def _calculate_reward(self, current_price: float) -> float:
        """Calculate reward based on price movement and action taken"""
        try:
            if self.last_price <= 0:
                return 0.0
            
            price_change = (current_price - self.last_price) / self.last_price
            
            if self.last_action == 0:
                reward = price_change * 100
            elif self.last_action == 1:
                reward = -price_change * 100
            else:
                reward = -abs(price_change) * 10
            
            if self.last_action != 2:
                reward -= 0.1
            
            self.total_reward += reward
            return reward
            
        except Exception as e:
            self.logger.error(f"Reward calculation failed: {e}", exc_info=True)
            return 0.0
    
    def _store_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Store experience in replay memory"""
        try:
            experience = (state, action, reward, next_state, False)
            self.memory.append(experience)
            
        except Exception as e:
            self.logger.error(f"Experience storage failed: {e}", exc_info=True)
    
    def _replay_training(self):
        """Train the RL agent using experience replay"""
        try:
            if not TENSORFLOW_AVAILABLE or self.q_network is None or len(self.memory) < self.batch_size:
                return
            
            batch = random.sample(self.memory, self.batch_size)
            
            states = np.array([e[0].flatten() for e in batch])
            actions = np.array([e[1] for e in batch])
            rewards = np.array([e[2] for e in batch])
            next_states = np.array([e[3].flatten() for e in batch])
            dones = np.array([e[4] for e in batch])
            
            current_q_values = self.q_network.predict(states, verbose=0)
            next_q_values = self.target_network.predict(next_states, verbose=0)
            
            target_q_values = current_q_values.copy()
            
            for i in range(self.batch_size):
                if dones[i]:
                    target_q_values[i][actions[i]] = rewards[i]
                else:
                    target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
            
            self.q_network.fit(states, target_q_values, epochs=1, verbose=0)
            
            self.training_steps += 1
            
            if self.training_steps % self.update_target_freq == 0:
                self.target_network.set_weights(self.q_network.get_weights())
                self.logger.info(f"Target network updated at step {self.training_steps}")
            
        except Exception as e:
            self.logger.error(f"Replay training failed: {e}", exc_info=True)
    
    def _generate_fallback_signal(self, data: pd.DataFrame, symbol: str, 
                                 timeframe: str) -> Optional[Signal]:
        """Generate fallback signal when main logic fails"""
        # This function is not called by the main generate_signal, but kept for completeness
        try:
            if len(data) < 20:
                return None
            
            action, confidence = self._get_fallback_action(data)
            
            if action == 2 or confidence < 0.6:
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
            
        except Exception as e:
            self.logger.error(f"Fallback signal generation failed: {e}", exc_info=True)
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
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
    
    def reset_episode(self):
        """Reset episode for new trading session"""
        self.episode_count += 1
        self.last_action = 2
        self.last_price = 0.0
        self.position_value = 0.0
        self.last_state = None # Ensure last_state is reset
        
        self.logger.info(f"Episode {self.episode_count} started")
    
    def _get_action_enhanced(self, state: np.ndarray, variation_index: int) -> Tuple[int, float]:
        """Get enhanced action with variations for multiple signals"""
        try:
            if not TENSORFLOW_AVAILABLE or self.q_network is None:
                return self._get_fallback_action_simple()
            
            exploration_rate = self.epsilon * (1 + variation_index * 0.1)
            
            if np.random.random() <= exploration_rate:
                action = random.randrange(self.action_size)
                confidence = 0.55 + (variation_index * 0.05)
            else:
                q_values = self.q_network.predict(state, verbose=0)[0]
                
                if variation_index > 0 and len(q_values) > 1:
                    sorted_indices = np.argsort(q_values)[::-1]
                    action = sorted_indices[min(variation_index, len(sorted_indices)-1)]
                else:
                    action = np.argmax(q_values)
                
                sorted_q = np.sort(q_values)[::-1]
                if len(sorted_q) > 1:
                    confidence = min(0.85, 0.5 + abs(sorted_q[0] - sorted_q[1]) / 2)
                else:
                    confidence = 0.6
                
                confidence *= (1 - variation_index * 0.1)
            
            return action, confidence
            
        except Exception as e:
            self.logger.error(f"Enhanced action selection failed: {e}", exc_info=True)
            return self._get_fallback_action_simple()
    
    def _get_fallback_action_enhanced(self, data: pd.DataFrame, variation_index: int) -> Tuple[int, float]:
        """Get enhanced fallback action with variations"""
        try:
            close = data['Close']
            
            if len(close) >= 20:
                sma_10 = close.rolling(min(10, len(close))).mean().iloc[-1]
                sma_20 = close.rolling(min(20, len(close))).mean().iloc[-1]
                rsi = self._calculate_rsi(close)
                
                if not pd.isna(sma_10) and not pd.isna(sma_20):
                    buy_threshold = 1.001 + (variation_index * 0.0005)
                    sell_threshold = 0.999 - (variation_index * 0.0005)
                    rsi_upper = 70 + (variation_index * 5)
                    rsi_lower = 30 - (variation_index * 5)
                    
                    if sma_10 > sma_20 * buy_threshold and rsi < rsi_upper:
                        confidence = 0.65 - (variation_index * 0.05)
                        return 0, confidence
                    elif sma_10 < sma_20 * sell_threshold and rsi > rsi_lower:
                        confidence = 0.65 - (variation_index * 0.05)
                        return 1, confidence
            
            return 2, 0.5
            
        except Exception as e:
            self.logger.error(f"Enhanced fallback action failed: {e}", exc_info=True)
            return 2, 0.5
    
    def _cleanup_memory(self):
        """Clean up memory for 8GB RAM optimization"""
        try:
            import gc
            
            if TENSORFLOW_AVAILABLE and hasattr(tf, 'keras'):
                tf.keras.backend.clear_session()
            
            if len(self.memory) > self.memory_size // 2:
                memory_list = list(self.memory)
                self.memory = deque(memory_list[-self.memory_size//2:], maxlen=self.memory_size)
            
            gc.collect()
            
            self.logger.info(f"Memory cleanup completed (prediction #{self.prediction_count})")
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {str(e)}", exc_info=True)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the RLAgentStrategy,
        including its parameters, RL-specific metrics, and overall trading performance.
        """
        # Get overall trading performance from AbstractStrategy base class
        base_trading_performance = self.get_performance_summary()

        info = {
            'name': 'Reinforcement Learning Agent Strategy',
            'version': '2.0.0',
            'type': 'Machine Learning',
            'description': 'Deep Q-Network (DQN) based RL agent for XAUUSD trading.',
            'parameters': {
                'lookback_bars': self.lookback_bars,
                'min_confidence': self.min_confidence,
                'state_size': self.state_size,
                'action_size': self.action_size,
                'memory_size': self.memory_size,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'gamma': self.gamma,
                'update_target_freq': self.update_target_freq,
                'memory_cleanup_interval': self.memory_cleanup_interval,
                'max_training_frequency': self.max_training_frequency
            },
            'rl_specific_metrics': {
                'tensorflow_available': TENSORFLOW_AVAILABLE,
                'rl_agent_trained': self.is_trained,
                'total_reward_accumulated': round(self.total_reward, 2),
                'total_episodes_completed': self.episode_count,
                'total_predictions_made': self.predictions_made,
                'current_epsilon_value': round(self.epsilon, 3),
                'current_memory_usage': len(self.memory),
                'total_training_steps': self.training_steps,
            },
            'overall_trading_performance': {
                'total_signals_generated': base_trading_performance['total_signals'],
                'win_rate': base_trading_performance['win_rate'],
                'profit_factor': base_trading_performance['profit_factor']
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
            'lookback_bars': 80,
            'min_confidence': 0.55,
            'state_size': 8,
            'memory_size': 100, # Reduced for faster test
            'batch_size': 8,
            'learning_rate': 0.001,
            'epsilon': 0.8,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.99,
            'gamma': 0.9,
            'update_target_freq': 10, # Reduced for faster test
            'memory_cleanup_interval': 5, # Reduced for faster test
            'max_training_frequency': 5 # Reduced for faster test
        }
    }
    
    # Mock MT5 manager for testing
    class MockMT5Manager:
        def get_historical_data(self, symbol, timeframe, bars):
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            dates = pd.date_range(start=datetime.now() - timedelta(days=5), 
                                 end=datetime.now(), freq='15Min')
            
            np.random.seed(42) # For consistent mock data
            
            # Generate sample OHLCV data with some trend
            price_series = 1950 + np.cumsum(np.random.randn(len(dates)) * 0.5)
            # Ensure price series is long enough for lookback
            if len(price_series) < bars:
                price_series = np.concatenate((price_series, price_series[-1] + np.cumsum(np.random.randn(bars - len(price_series)) * 0.5)))
            price_series = price_series[:bars] # Trim to exact bars needed
            
            data = pd.DataFrame({
                'Open': price_series + np.random.randn(len(price_series)) * 0.2,
                'High': price_series + np.abs(np.random.randn(len(price_series)) * 0.5),
                'Low': price_series - np.abs(np.random.randn(len(price_series)) * 0.5),
                'Close': price_series,
                'Volume': np.random.randint(100, 1000, len(price_series))
            }, index=dates[:len(price_series)]) # Ensure index matches length
            
            data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
            data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
            
            return data
    
    # Create strategy instance
    mock_mt5 = MockMT5Manager()
    strategy = RLAgentStrategy(test_config, mock_mt5, database=None)
    
    # Output header matching other strategy files
    print("============================================================")
    print("TESTING MODIFIED REINFORCEMENT LEARNING AGENT STRATEGY")
    print("============================================================")

    # 1. Testing signal generation
    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - Signal: {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
        if signal.metadata:
            print(f"     RL Action: {signal.metadata.get('rl_prediction_label', 'N/A')}")
            print(f"     Total Reward: {signal.metadata.get('total_reward', 'N/A'):.2f}")
    
    # 2. Testing analysis method
    print("\n2. Testing analysis method:")
    mock_data = mock_mt5.get_historical_data("XAUUSDm", "M15", 90) # Use slightly more data than lookback
    analysis_results = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis results keys: {analysis_results.keys()}")
    print(f"   TensorFlow Available: {analysis_results.get('tensorflow_available')}")
    print(f"   RL Agent Trained: {analysis_results.get('rl_agent_trained')}")
    print(f"   Current Epsilon: {analysis_results.get('current_epsilon')}")
    print(f"   Memory Usage: {analysis_results.get('memory_usage_percent')}%")
    
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
    print(f"   Parameters: {info['parameters']}")
    print(f"   RL Specific Metrics:")
    for key, value in info['rl_specific_metrics'].items():
        if isinstance(value, float):
            print(f"     - {key}: {value:.2f}")
        else:
            print(f"     - {key}: {value}")
    print(f"   Overall Trading Performance:")
    print(f"     Total Signals Generated: {info['overall_trading_performance']['total_signals_generated']}")
    print(f"     Win Rate: {info['overall_trading_performance']['win_rate']:.2%}")
    print(f"     Profit Factor: {info['overall_trading_performance']['profit_factor']:.2f}")

    # Footer matching other strategy files
    print("\n============================================================")
    print("REINFORCEMENT LEARNING AGENT STRATEGY TEST COMPLETED!")
    print("============================================================")