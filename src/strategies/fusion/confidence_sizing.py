"""
Confidence-Based Position Sizing Fusion Strategy
===============================================

Adjusts position sizes based on signal confidence and market conditions.
Combines multiple signals with dynamic position sizing.
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

# Import base classes
try:
    from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade
except ImportError:
    from core.base import AbstractStrategy, Signal, SignalType, SignalGrade


class ConfidenceSizing(AbstractStrategy):
    """Confidence-based position sizing fusion strategy"""
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """Initialize confidence sizing fusion strategy - 8GB RAM optimized"""
        super().__init__(config, mt5_manager, database)
        
        self.strategy_name = "ConfidenceSizing"
        self.min_signals = config.get('min_signals', 2)
        self.min_confidence = config.get('min_confidence', 0.55)  # Reduced for more signals
        self.base_position_size = config.get('base_position_size', 0.01)  # 1% base size
        self.max_position_size = config.get('max_position_size', 0.05)    # 5% max size
        self.confidence_multiplier = config.get('confidence_multiplier', 2.0)
        
        # Volatility adjustment parameters
        self.volatility_window = config.get('volatility_window', 15)  # Reduced from 20
        self.volatility_threshold = config.get('volatility_threshold', 0.02)
        self.volatility_adjustment = config.get('volatility_adjustment', 0.5)
        
        # Signal correlation parameters
        self.correlation_penalty = config.get('correlation_penalty', 0.25)  # Reduced from 0.3
        self.max_correlation = config.get('max_correlation', 0.75)  # Reduced from 0.8
        
        # Performance tracking - 8GB RAM optimized
        self.position_history = []
        self.max_history = config.get('max_history', 300)  # Reduced from 500
        
        # Memory optimization settings
        self.memory_cleanup_interval = 20
        self.prediction_count = 0
        
        # Logger
        self.logger = logging.getLogger(self.strategy_name)
        
        self.logger.info(f"{self.strategy_name} initialized")
    
    def fuse_signals(self, signals: List[Signal], data: pd.DataFrame = None, 
                    symbol: str = "XAUUSDm", timeframe: str = "M15") -> Optional[Signal]:
        """Fuse signals with confidence-based position sizing"""
        try:
            if not signals or len(signals) < self.min_signals:
                return None
            
            # Filter valid signals
            valid_signals = [s for s in signals if s is not None and s.confidence > 0.3]
            
            if len(valid_signals) < self.min_signals:
                return None
            
            # Calculate signal consensus
            buy_signals = [s for s in valid_signals if s.signal_type == SignalType.BUY]
            sell_signals = [s for s in valid_signals if s.signal_type == SignalType.SELL]
            
            # Determine dominant signal direction
            if len(buy_signals) > len(sell_signals):
                dominant_signals = buy_signals
                signal_type = SignalType.BUY
            elif len(sell_signals) > len(buy_signals):
                dominant_signals = sell_signals
                signal_type = SignalType.SELL
            else:
                # Equal signals - use confidence to break tie
                buy_confidence = np.mean([s.confidence for s in buy_signals]) if buy_signals else 0
                sell_confidence = np.mean([s.confidence for s in sell_signals]) if sell_signals else 0
                
                if buy_confidence > sell_confidence:
                    dominant_signals = buy_signals
                    signal_type = SignalType.BUY
                elif sell_confidence > buy_confidence:
                    dominant_signals = sell_signals
                    signal_type = SignalType.SELL
                else:
                    return None  # No clear direction
            
            # Calculate combined confidence
            combined_confidence = self._calculate_combined_confidence(dominant_signals, valid_signals)
            
            if combined_confidence < self.min_confidence:
                return None
            
            # Calculate position size based on confidence and market conditions
            position_size = self._calculate_position_size(
                combined_confidence, dominant_signals, data
            )
            
            # Get current price
            current_price = self._get_current_price(valid_signals, data)
            if current_price is None:
                return None
            
            # Calculate risk parameters
            stop_loss, take_profit = self._calculate_risk_parameters(
                dominant_signals, signal_type, current_price, position_size
            )
            
            # Create fused signal with position sizing
            fused_signal = Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                strategy_name=self.strategy_name,
                signal_type=signal_type,
                confidence=combined_confidence,
                price=current_price,
                timeframe=timeframe,
                strength=combined_confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'fusion_method': 'confidence_sizing',
                    'position_size': position_size,
                    'base_position_size': self.base_position_size,
                    'confidence_multiplier': combined_confidence / self.min_confidence,
                    'dominant_signals': len(dominant_signals),
                    'total_signals': len(valid_signals),
                    'buy_signals': len(buy_signals),
                    'sell_signals': len(sell_signals),
                    'signal_details': [
                        {
                            'strategy': s.strategy_name,
                            'signal_type': s.signal_type.value,
                            'confidence': s.confidence
                        } for s in dominant_signals
                    ]
                }
            )
            
            # Store position for tracking
            self._store_position_record(fused_signal, dominant_signals)
            
            self.logger.info(f"Fused signal: {signal_type.value} with confidence {combined_confidence:.3f}, position size {position_size:.4f}")
            
            return fused_signal
            
        except Exception as e:
            self.logger.error(f"Signal fusion failed: {e}")
            return None
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """Generate confidence-sized fusion signals - Enhanced like technical strategies"""
        signals = []
        try:
            # Print consistent status message like technical strategies
            print(f"Confidence Sizing - Analyzing {symbol} on {timeframe}")
            
            # Get market data
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available for signal generation.")
                return []
            
            data = self.mt5_manager.get_historical_data(symbol, timeframe, 80)
            if data is None or len(data) < 40:
                self.logger.warning(f"Insufficient data for Confidence Sizing analysis: {len(data) if data is not None else 0}")
                return []
            
            # Memory cleanup
            self.prediction_count += 1
            if self.prediction_count % self.memory_cleanup_interval == 0:
                self._cleanup_memory()
            
            # Generate simulated strategy signals for fusion
            strategy_signals = self._generate_simulated_signals(data, symbol, timeframe)
            
            if not strategy_signals:
                print("   No strategy signals to fuse")
                return []
            
            # Apply confidence-based sizing to generate multiple signals
            for i in range(3):  # Generate up to 3 signals with different sizing
                sized_signal = self._fuse_signals_enhanced(strategy_signals, data, symbol, timeframe, i)
                if sized_signal and self.validate_signal(sized_signal):
                    signals.append(sized_signal)
                    self.signal_history.append(sized_signal)
            
            # Print results like technical strategies
            if signals:
                avg_confidence = np.mean([s.confidence for s in signals])
                print(f"   Generated {len(signals)} signals (avg confidence: {avg_confidence:.2f})")
                for signal in signals:
                    print(f"      - {signal.signal_type.value} at {signal.price:.2f} (conf: {signal.confidence:.2f})")
                    if 'position_size' in signal.metadata:
                        print(f"        Position Size: {signal.metadata['position_size']:.4f}")
            else:
                print("   No valid signals generated")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Confidence sizing signal generation failed: {e}")
            print(f"   Confidence Sizing Error: {str(e)}")
            return []
    
    def _calculate_combined_confidence(self, dominant_signals: List[Signal], 
                                     all_signals: List[Signal]) -> float:
        """Calculate combined confidence from multiple signals"""
        try:
            if not dominant_signals:
                return 0.0
            
            # Base confidence from dominant signals
            base_confidence = np.mean([s.confidence for s in dominant_signals])
            
            # Consensus bonus (more signals in same direction = higher confidence)
            consensus_ratio = len(dominant_signals) / len(all_signals)
            consensus_bonus = (consensus_ratio - 0.5) * 0.2  # Up to 10% bonus
            
            # Signal correlation penalty
            correlation_penalty = self._calculate_correlation_penalty(dominant_signals)
            
            # Combined confidence
            combined_confidence = base_confidence + consensus_bonus - correlation_penalty
            
            # Ensure within valid range
            return max(0.0, min(1.0, combined_confidence))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def _calculate_correlation_penalty(self, signals: List[Signal]) -> float:
        """Calculate penalty for highly correlated signals"""
        try:
            if len(signals) < 2:
                return 0.0
            
            # Simple correlation estimation based on strategy types
            strategy_types = [s.strategy_name for s in signals]
            
            # Count similar strategy types
            similar_strategies = 0
            for i, strategy1 in enumerate(strategy_types):
                for j, strategy2 in enumerate(strategy_types[i+1:], i+1):
                    # Simple similarity check based on strategy name patterns
                    if self._are_strategies_similar(strategy1, strategy2):
                        similar_strategies += 1
            
            # Calculate penalty
            max_pairs = len(signals) * (len(signals) - 1) // 2
            correlation_ratio = similar_strategies / max_pairs if max_pairs > 0 else 0
            
            return correlation_ratio * self.correlation_penalty
            
        except Exception as e:
            self.logger.error(f"Correlation penalty calculation failed: {e}")
            return 0.0
    
    def _are_strategies_similar(self, strategy1: str, strategy2: str) -> bool:
        """Check if two strategies are similar (simple heuristic)"""
        # Group similar strategy types
        technical_strategies = ['RSI', 'MACD', 'BollingerBands', 'MovingAverage']
        ml_strategies = ['LSTM', 'XGBoost', 'EnsembleNN', 'RLAgent']
        smc_strategies = ['SMC', 'OrderBlock', 'FairValueGap', 'LiquidityGrab']
        
        def get_strategy_group(strategy_name):
            for group_name, strategies in [
                ('technical', technical_strategies),
                ('ml', ml_strategies),
                ('smc', smc_strategies)
            ]:
                if any(s.lower() in strategy_name.lower() for s in strategies):
                    return group_name
            return 'other'
        
        return get_strategy_group(strategy1) == get_strategy_group(strategy2)
    
    def _calculate_position_size(self, confidence: float, signals: List[Signal], 
                               data: pd.DataFrame = None) -> float:
        """Calculate position size based on confidence and market conditions"""
        try:
            # Base position size adjusted by confidence
            confidence_multiplier = (confidence / self.min_confidence) ** self.confidence_multiplier
            position_size = self.base_position_size * confidence_multiplier
            
            # Volatility adjustment
            if data is not None and len(data) >= self.volatility_window:
                volatility_adjustment = self._calculate_volatility_adjustment(data)
                position_size *= volatility_adjustment
            
            # Signal strength adjustment
            avg_strength = np.mean([s.strength for s in signals if hasattr(s, 'strength') and s.strength > 0])
            if avg_strength > 0:
                strength_multiplier = min(2.0, avg_strength / 0.7)  # Cap at 2x
                position_size *= strength_multiplier
            
            # Ensure within limits
            position_size = max(0.001, min(position_size, self.max_position_size))
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return self.base_position_size
    
    def _calculate_volatility_adjustment(self, data: pd.DataFrame) -> float:
        """Calculate position size adjustment based on market volatility"""
        try:
            if 'Close' not in data.columns or len(data) < self.volatility_window:
                return 1.0
            
            # Calculate recent volatility
            returns = data['Close'].pct_change().dropna()
            recent_volatility = returns.tail(self.volatility_window).std()
            
            # Adjust position size inversely to volatility
            if recent_volatility > self.volatility_threshold:
                # Reduce position size in high volatility
                volatility_ratio = recent_volatility / self.volatility_threshold
                adjustment = 1.0 / (1.0 + (volatility_ratio - 1.0) * self.volatility_adjustment)
            else:
                # Slightly increase position size in low volatility
                volatility_ratio = self.volatility_threshold / recent_volatility
                adjustment = min(1.2, 1.0 + (volatility_ratio - 1.0) * 0.1)
            
            return max(0.5, min(1.5, adjustment))  # Cap adjustment between 0.5x and 1.5x
            
        except Exception as e:
            self.logger.error(f"Volatility adjustment calculation failed: {e}")
            return 1.0
    
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
                    return np.mean(prices)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Price extraction failed: {e}")
            return None
    
    def _calculate_risk_parameters(self, signals: List[Signal], signal_type: SignalType, 
                                 current_price: float, position_size: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit adjusted for position size"""
        try:
            # Collect risk parameters from signals
            stop_losses = []
            take_profits = []
            
            for signal in signals:
                if signal.stop_loss is not None:
                    stop_losses.append(signal.stop_loss)
                if signal.take_profit is not None:
                    take_profits.append(signal.take_profit)
            
            # Calculate average risk parameters
            if stop_losses:
                avg_stop_loss = np.mean(stop_losses)
            else:
                # Default stop loss based on position size (larger positions = tighter stops)
                stop_loss_pct = 0.01 * (1.0 + (position_size / self.base_position_size - 1.0) * 0.5)
                if signal_type == SignalType.BUY:
                    avg_stop_loss = current_price * (1.0 - stop_loss_pct)
                else:
                    avg_stop_loss = current_price * (1.0 + stop_loss_pct)
            
            if take_profits:
                avg_take_profit = np.mean(take_profits)
            else:
                # Default take profit based on position size (larger positions = closer targets)
                take_profit_pct = 0.02 * (2.0 - (position_size / self.base_position_size - 1.0) * 0.3)
                if signal_type == SignalType.BUY:
                    avg_take_profit = current_price * (1.0 + take_profit_pct)
                else:
                    avg_take_profit = current_price * (1.0 - take_profit_pct)
            
            return avg_stop_loss, avg_take_profit
            
        except Exception as e:
            self.logger.error(f"Risk parameter calculation failed: {e}")
            # Return default values
            if signal_type == SignalType.BUY:
                return current_price * 0.99, current_price * 1.02
            else:
                return current_price * 1.01, current_price * 0.98
    
    def _store_position_record(self, signal: Signal, constituent_signals: List[Signal]):
        """Store position record for performance tracking"""
        try:
            position_record = {
                'timestamp': signal.timestamp,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'position_size': signal.metadata.get('position_size', 0),
                'price': signal.price,
                'constituent_count': len(constituent_signals),
                'strategy_names': [s.strategy_name for s in constituent_signals]
            }
            
            self.position_history.append(position_record)
            
            # Keep only recent history
            if len(self.position_history) > self.max_history:
                self.position_history.pop(0)
            
        except Exception as e:
            self.logger.error(f"Position record storage failed: {e}")
    
    def get_position_statistics(self) -> Dict[str, Any]:
        """Get position sizing statistics"""
        try:
            if not self.position_history:
                return {}
            
            position_sizes = [p['position_size'] for p in self.position_history]
            confidences = [p['confidence'] for p in self.position_history]
            
            return {
                'total_positions': len(self.position_history),
                'avg_position_size': np.mean(position_sizes),
                'max_position_size': np.max(position_sizes),
                'min_position_size': np.min(position_sizes),
                'avg_confidence': np.mean(confidences),
                'position_size_std': np.std(position_sizes),
                'base_position_size': self.base_position_size,
                'max_allowed_size': self.max_position_size
            }
            
        except Exception as e:
            self.logger.error(f"Position statistics calculation failed: {e}")
            return {}
    
    def _generate_simulated_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Generate simulated strategy signals for confidence sizing"""
        try:
            signals = []
            current_price = data['close'].iloc[-1]
            timestamp = datetime.now()
            
            # Simulate various strategy signals with different confidence levels
            strategy_configs = [
                ("high_confidence_technical", 0.85, SignalType.BUY, 'strong_technical'),
                ("medium_confidence_smc", 0.70, SignalType.BUY, 'smc_signal'),
                ("low_confidence_ml", 0.60, SignalType.SELL, 'ml_prediction'),
                ("momentum_signal", 0.75, SignalType.BUY, 'momentum_based'),
                ("volume_signal", 0.65, SignalType.SELL, 'volume_based')
            ]
            
            for strategy_name, base_confidence, signal_type, reason in strategy_configs:
                # Add some randomness
                confidence = min(max(base_confidence * np.random.uniform(0.9, 1.1), 0.1), 0.95)
                
                signal = Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    strategy_name=strategy_name,
                    signal_type=signal_type,
                    confidence=confidence,
                    price=current_price,
                    timeframe=timeframe,
                    strength=confidence,
                    metadata={'signal_reason': reason}
                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating simulated signals: {e}")
            return []
    
    def _fuse_signals_enhanced(self, signals: List[Signal], data: pd.DataFrame, 
                              symbol: str, timeframe: str, variation_index: int) -> Optional[Signal]:
        """Enhanced confidence-based signal fusion with variations"""
        try:
            # Apply the original fuse_signals logic with enhancements
            fused_signal = self.fuse_signals(signals, data, symbol, timeframe)
            
            if fused_signal is None:
                return None
            
            # Apply variation-based adjustments
            if variation_index > 0:
                # Adjust confidence and position size based on variation
                confidence_factor = 1.0 - (variation_index * 0.1)  # Reduce confidence for variations
                fused_signal.confidence *= confidence_factor
                
                # Adjust position size in metadata
                if 'position_size' in fused_signal.metadata:
                    original_size = fused_signal.metadata['position_size']
                    varied_size = original_size * (1.0 + variation_index * 0.2)  # Increase size slightly
                    fused_signal.metadata['position_size'] = min(varied_size, self.max_position_size)
                
                # Update strategy name for variation
                fused_signal.strategy_name = f"confidence_sizing_{variation_index+1}"
                
                # Update metadata
                fused_signal.metadata.update({
                    'signal_reason': f'confidence_sizing_fusion_{variation_index+1}',
                    'variation_index': variation_index,
                    'confidence_factor': confidence_factor
                })
            
            return fused_signal
            
        except Exception as e:
            self.logger.error(f"Enhanced confidence fusion failed: {e}")
            return None
    
    def _cleanup_memory(self):
        """Clean up memory for 8GB RAM optimization"""
        try:
            import gc
            
            # Limit position history
            if len(self.position_history) > self.max_history:
                self.position_history = self.position_history[-self.max_history//2:]
            
            # Force garbage collection
            gc.collect()
            
            print(f"   Memory cleanup completed (prediction #{self.prediction_count})")
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {str(e)}")
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze confidence sizing strategy performance"""
        try:
            position_stats = self.get_position_statistics()
            
            analysis = {
                'strategy': self.strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'position_statistics': position_stats,
                'sizing_parameters': {
                    'base_position_size': self.base_position_size,
                    'max_position_size': self.max_position_size,
                    'confidence_multiplier': self.confidence_multiplier,
                    'min_confidence': self.min_confidence,
                    'volatility_threshold': self.volatility_threshold
                },
                'memory_optimized': True,
                'max_history': self.max_history
            }
            
            # Recent performance analysis
            if self.position_history:
                recent_positions = self.position_history[-20:]  # Last 20 positions
                analysis['recent_performance'] = {
                    'recent_positions': len(recent_positions),
                    'avg_recent_size': np.mean([p['position_size'] for p in recent_positions]),
                    'avg_recent_confidence': np.mean([p['confidence'] for p in recent_positions])
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                'strategy': self.strategy_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Testing function
if __name__ == "__main__":
    """Test the Confidence Sizing strategy"""
    
    # Test configuration
    test_config = {
        'min_signals': 2,
        'min_confidence': 0.55,
        'base_position_size': 0.01,
        'max_position_size': 0.05
    }
    
    # Mock MT5 manager for testing
    class MockMT5Manager:
        def get_historical_data(self, symbol, timeframe, bars):
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            dates = pd.date_range(start=datetime.now() - timedelta(days=3), 
                                 end=datetime.now(), freq='15Min')[:bars]
            
            # Generate sample OHLCV data
            np.random.seed(42)
            close_prices = 1950 + np.cumsum(np.random.randn(len(dates)) * 2)
            
            data = pd.DataFrame({
                'open': close_prices + np.random.randn(len(dates)) * 0.5,
                'high': close_prices + np.abs(np.random.randn(len(dates)) * 3),
                'low': close_prices - np.abs(np.random.randn(len(dates)) * 3),
                'close': close_prices,
                'volume': np.random.randint(100, 1000, len(dates))
            }, index=dates)
            
            # Ensure compatibility
            data['Close'] = data['close']  # Add uppercase for compatibility
            
            return data
    
    # Create strategy instance
    mock_mt5 = MockMT5Manager()
    strategy = ConfidenceSizing(test_config, mock_mt5, database=None)
    
    print("============================================================")
    print("TESTING OPTIMIZED CONFIDENCE SIZING STRATEGY")
    print("============================================================")

    # 1. Testing signal generation
    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - Signal: {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
        if signal.metadata:
            print(f"     Reason: {signal.metadata.get('signal_reason', 'N/A')}")
            print(f"     Position Size: {signal.metadata.get('position_size', 'N/A')}")
    
    # 2. Testing analysis method
    print("\n2. Testing analysis method:")
    mock_data = mock_mt5.get_historical_data("XAUUSDm", "M15", 80)
    analysis_results = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis results keys: {analysis_results.keys()}")
    print(f"   Memory Optimized: {analysis_results.get('memory_optimized')}")
    
    # 3. Testing performance tracking
    print("\n3. Testing performance tracking:")
    summary = strategy.get_performance_summary()
    print(f"   {summary}")
    
    print("\n============================================================")
    print("CONFIDENCE SIZING STRATEGY TEST COMPLETED!")
    print("============================================================")
