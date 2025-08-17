"""
Weighted Voting Fusion Strategy - Signal Combination Strategy
============================================================

Combines signals from multiple strategies using weighted voting mechanism.
Dynamically adjusts weights based on strategy performance.
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
from collections import defaultdict

# Import base classes
try:
    from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade
except ImportError:
    from core.base import AbstractStrategy, Signal, SignalType, SignalGrade


class WeightedVoting(AbstractStrategy):
    """Weighted voting fusion strategy for combining multiple strategy signals"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize weighted voting fusion strategy"""
        super().__init__(config)
        
        self.strategy_name = "WeightedVoting"
        self.min_signals = config.get('min_signals', 2)
        self.min_confidence = config.get('min_confidence', 0.65)
        self.performance_window = config.get('performance_window', 100)
        
        # Weight adjustment parameters
        self.initial_weight = config.get('initial_weight', 1.0)
        self.weight_decay = config.get('weight_decay', 0.95)
        self.weight_boost = config.get('weight_boost', 1.1)
        self.min_weight = config.get('min_weight', 0.1)
        self.max_weight = config.get('max_weight', 3.0)
        
        # Strategy weights and performance tracking
        self.strategy_weights = defaultdict(lambda: self.initial_weight)
        self.strategy_performance = defaultdict(list)
        self.strategy_predictions = defaultdict(int)
        self.strategy_correct = defaultdict(int)
        
        # Signal history for performance evaluation
        self.signal_history = []
        self.max_history = config.get('max_history', 1000)
        
        # Logger
        self.logger = logging.getLogger(self.strategy_name)
        
        self.logger.info(f"{self.strategy_name} initialized")
    
    def fuse_signals(self, signals: List[Signal], data: pd.DataFrame = None, 
                    symbol: str = "XAUUSDm", timeframe: str = "M15") -> Optional[Signal]:
        """Fuse multiple signals using weighted voting"""
        try:
            if not signals or len(signals) < self.min_signals:
                return None
            
            # Filter valid signals
            valid_signals = [s for s in signals if s is not None and s.confidence > 0.3]
            
            if len(valid_signals) < self.min_signals:
                return None
            
            # Calculate weighted votes
            buy_weight = 0.0
            sell_weight = 0.0
            hold_weight = 0.0
            
            total_weight = 0.0
            signal_details = []
            
            for signal in valid_signals:
                strategy_name = signal.strategy_name
                weight = self.strategy_weights[strategy_name]
                confidence = signal.confidence
                
                # Weight by both strategy performance and signal confidence
                effective_weight = weight * confidence
                
                if signal.signal_type == SignalType.BUY:
                    buy_weight += effective_weight
                elif signal.signal_type == SignalType.SELL:
                    sell_weight += effective_weight
                else:
                    hold_weight += effective_weight
                
                total_weight += effective_weight
                
                signal_details.append({
                    'strategy': strategy_name,
                    'signal_type': signal.signal_type.value,
                    'confidence': confidence,
                    'weight': weight,
                    'effective_weight': effective_weight
                })
            
            if total_weight == 0:
                return None
            
            # Normalize weights
            buy_score = buy_weight / total_weight
            sell_score = sell_weight / total_weight
            hold_score = hold_weight / total_weight
            
            # Determine final signal
            max_score = max(buy_score, sell_score, hold_score)
            
            if max_score < self.min_confidence:
                return None
            
            if max_score == buy_score:
                final_signal_type = SignalType.BUY
                final_confidence = buy_score
            elif max_score == sell_score:
                final_signal_type = SignalType.SELL
                final_confidence = sell_score
            else:
                return None  # Don't generate HOLD signals
            
            # Calculate price and risk parameters
            current_price = self._get_current_price(valid_signals, data)
            if current_price is None:
                return None
            
            # Calculate stop loss and take profit from constituent signals
            stop_loss, take_profit = self._calculate_risk_parameters(
                valid_signals, final_signal_type, current_price
            )
            
            # Create fused signal
            fused_signal = Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                strategy_name=self.strategy_name,
                signal_type=final_signal_type,
                confidence=final_confidence,
                price=current_price,
                timeframe=timeframe,
                strength=final_confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'fusion_method': 'weighted_voting',
                    'constituent_signals': len(valid_signals),
                    'buy_score': buy_score,
                    'sell_score': sell_score,
                    'hold_score': hold_score,
                    'signal_details': signal_details,
                    'strategy_weights': dict(self.strategy_weights)
                }
            )
            
            # Store signal for performance tracking
            self._store_signal_for_tracking(fused_signal, valid_signals)
            
            self.logger.info(f"Fused signal: {final_signal_type.value} with confidence {final_confidence:.3f}")
            
            return fused_signal
            
        except Exception as e:
            self.logger.error(f"Signal fusion failed: {e}")
            return None
    
    def generate_signal(self, data: pd.DataFrame, symbol: str = "XAUUSDm", 
                       timeframe: str = "M15") -> Optional[Signal]:
        """Generate signal - not used directly, fusion happens via fuse_signals"""
        # This fusion strategy doesn't generate signals independently
        # It combines signals from other strategies
        return None
    
    def update_performance(self, signal: Signal, actual_outcome: float):
        """Update strategy performance based on signal outcomes"""
        try:
            if 'signal_details' not in signal.metadata:
                return
            
            # Update performance for each contributing strategy
            for detail in signal.metadata['signal_details']:
                strategy_name = detail['strategy']
                
                # Determine if prediction was correct
                predicted_direction = 1 if detail['signal_type'] == 'BUY' else -1
                actual_direction = 1 if actual_outcome > 0 else -1
                
                was_correct = predicted_direction == actual_direction
                
                # Update tracking
                self.strategy_predictions[strategy_name] += 1
                if was_correct:
                    self.strategy_correct[strategy_name] += 1
                
                # Update performance history
                self.strategy_performance[strategy_name].append(1.0 if was_correct else 0.0)
                
                # Keep only recent performance
                if len(self.strategy_performance[strategy_name]) > self.performance_window:
                    self.strategy_performance[strategy_name].pop(0)
                
                # Update weights based on performance
                self._update_strategy_weight(strategy_name, was_correct)
            
            self.logger.info(f"Performance updated for signal from {signal.timestamp}")
            
        except Exception as e:
            self.logger.error(f"Performance update failed: {e}")
    
    def _update_strategy_weight(self, strategy_name: str, was_correct: bool):
        """Update individual strategy weight based on performance"""
        try:
            current_weight = self.strategy_weights[strategy_name]
            
            if was_correct:
                # Boost weight for correct predictions
                new_weight = min(current_weight * self.weight_boost, self.max_weight)
            else:
                # Reduce weight for incorrect predictions
                new_weight = max(current_weight * self.weight_decay, self.min_weight)
            
            self.strategy_weights[strategy_name] = new_weight
            
            # Calculate accuracy for logging
            if strategy_name in self.strategy_performance and self.strategy_performance[strategy_name]:
                accuracy = np.mean(self.strategy_performance[strategy_name])
                self.logger.debug(f"Strategy {strategy_name}: weight={new_weight:.3f}, accuracy={accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Weight update failed for {strategy_name}: {e}")
    
    def _get_current_price(self, signals: List[Signal], data: pd.DataFrame = None) -> Optional[float]:
        """Get current price from signals or data"""
        try:
            # Try to get price from data first
            if data is not None and len(data) > 0 and 'Close' in data.columns:
                return float(data['Close'].iloc[-1])
            
            # Fall back to signal prices
            if signals:
                prices = [s.price for s in signals if s.price > 0]
                if prices:
                    return np.mean(prices)  # Average of signal prices
            
            return None
            
        except Exception as e:
            self.logger.error(f"Price extraction failed: {e}")
            return None
    
    def _calculate_risk_parameters(self, signals: List[Signal], signal_type: SignalType, 
                                 current_price: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit from constituent signals"""
        try:
            stop_losses = []
            take_profits = []
            
            for signal in signals:
                if signal.signal_type == signal_type:
                    if signal.stop_loss is not None:
                        stop_losses.append(signal.stop_loss)
                    if signal.take_profit is not None:
                        take_profits.append(signal.take_profit)
            
            # Calculate weighted averages or use conservative estimates
            if stop_losses:
                if signal_type == SignalType.BUY:
                    # For BUY signals, use the highest (most conservative) stop loss
                    stop_loss = max(stop_losses)
                else:
                    # For SELL signals, use the lowest (most conservative) stop loss
                    stop_loss = min(stop_losses)
            else:
                # Default stop loss (1% of current price)
                if signal_type == SignalType.BUY:
                    stop_loss = current_price * 0.99
                else:
                    stop_loss = current_price * 1.01
            
            if take_profits:
                if signal_type == SignalType.BUY:
                    # For BUY signals, use the lowest (most conservative) take profit
                    take_profit = min(take_profits)
                else:
                    # For SELL signals, use the highest (most conservative) take profit
                    take_profit = max(take_profits)
            else:
                # Default take profit (2% of current price)
                if signal_type == SignalType.BUY:
                    take_profit = current_price * 1.02
                else:
                    take_profit = current_price * 0.98
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Risk parameter calculation failed: {e}")
            # Return default values
            if signal_type == SignalType.BUY:
                return current_price * 0.99, current_price * 1.02
            else:
                return current_price * 1.01, current_price * 0.98
    
    def _store_signal_for_tracking(self, fused_signal: Signal, constituent_signals: List[Signal]):
        """Store signal information for performance tracking"""
        try:
            signal_record = {
                'timestamp': fused_signal.timestamp,
                'signal_type': fused_signal.signal_type,
                'confidence': fused_signal.confidence,
                'price': fused_signal.price,
                'constituent_count': len(constituent_signals),
                'strategy_names': [s.strategy_name for s in constituent_signals]
            }
            
            self.signal_history.append(signal_record)
            
            # Keep only recent history
            if len(self.signal_history) > self.max_history:
                self.signal_history.pop(0)
            
        except Exception as e:
            self.logger.error(f"Signal storage failed: {e}")
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        return dict(self.strategy_weights)
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get strategy performance statistics"""
        performance_stats = {}
        
        for strategy_name in self.strategy_weights.keys():
            stats = {
                'weight': self.strategy_weights[strategy_name],
                'predictions': self.strategy_predictions[strategy_name],
                'correct': self.strategy_correct[strategy_name],
                'accuracy': 0.0,
                'recent_accuracy': 0.0
            }
            
            # Calculate overall accuracy
            if stats['predictions'] > 0:
                stats['accuracy'] = stats['correct'] / stats['predictions']
            
            # Calculate recent accuracy
            if strategy_name in self.strategy_performance and self.strategy_performance[strategy_name]:
                stats['recent_accuracy'] = np.mean(self.strategy_performance[strategy_name])
            
            performance_stats[strategy_name] = stats
        
        return performance_stats
    
    def reset_weights(self):
        """Reset all strategy weights to initial values"""
        for strategy_name in self.strategy_weights.keys():
            self.strategy_weights[strategy_name] = self.initial_weight
        
        self.logger.info("Strategy weights reset to initial values")
    
    def set_strategy_weight(self, strategy_name: str, weight: float):
        """Manually set weight for a specific strategy"""
        weight = max(self.min_weight, min(weight, self.max_weight))
        self.strategy_weights[strategy_name] = weight
        
        self.logger.info(f"Strategy {strategy_name} weight set to {weight:.3f}")
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze fusion strategy performance"""
        try:
            performance_stats = self.get_strategy_performance()
            
            analysis = {
                'strategy': self.strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'total_signals_processed': len(self.signal_history),
                'active_strategies': len(self.strategy_weights),
                'strategy_performance': performance_stats,
                'current_weights': self.get_strategy_weights(),
                'fusion_parameters': {
                    'min_signals': self.min_signals,
                    'min_confidence': self.min_confidence,
                    'performance_window': self.performance_window,
                    'weight_decay': self.weight_decay,
                    'weight_boost': self.weight_boost
                }
            }
            
            # Calculate overall fusion performance
            if self.signal_history:
                recent_signals = self.signal_history[-50:]  # Last 50 signals
                analysis['recent_signal_count'] = len(recent_signals)
                
                # Average confidence of recent signals
                if recent_signals:
                    avg_confidence = np.mean([s['confidence'] for s in recent_signals])
                    analysis['average_confidence'] = avg_confidence
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                'strategy': self.strategy_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
