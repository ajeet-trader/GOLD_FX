
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

# Add src to path for module resolution when run as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict

# Import base classes directly
from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade, StrategyPerformance


class WeightedVoting(AbstractStrategy):
    """Weighted voting fusion strategy for combining multiple strategy signals"""
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """Initialize weighted voting fusion strategy - 8GB RAM optimized"""
        super().__init__(config, mt5_manager, database)
        
        # self.strategy_name is set by AbstractStrategy
        self.min_signals = self.config.get('parameters', {}).get('min_signals', 2)
        self.min_confidence = self.config.get('parameters', {}).get('min_confidence', 0.60)
        self.performance_window = self.config.get('parameters', {}).get('performance_window', 50)
        
        # Weight adjustment parameters
        self.initial_weight = self.config.get('parameters', {}).get('initial_weight', 1.0)
        self.weight_decay = self.config.get('parameters', {}).get('weight_decay', 0.95)
        self.weight_boost = self.config.get('parameters', {}).get('weight_boost', 1.1)
        self.min_weight = self.config.get('parameters', {}).get('min_weight', 0.1)
        self.max_weight = self.config.get('parameters', {}).get('max_weight', 3.0)
        
        # Strategy weights and performance tracking (for component strategies)
        self.strategy_weights = defaultdict(lambda: self.initial_weight)
        self.strategy_performance = defaultdict(list) # Stores recent accuracy (0 or 1) for components
        self.strategy_predictions = defaultdict(int) # Total predictions for components
        self.strategy_correct = defaultdict(int) # Correct predictions for components
        
        # Internal signal history for performance evaluation of fused signals (not AbstractStrategy.signal_history)
        self.fusion_signal_records = []
        self.max_history = self.config.get('parameters', {}).get('max_history', 500)
        
        # Memory optimization settings
        self.memory_cleanup_interval = self.config.get('parameters', {}).get('memory_cleanup_interval', 25)
        self.prediction_count = 0
        
        # self.logger is provided by AbstractStrategy
        self.logger.info(f"{self.strategy_name} initialized")
    
    # Keeping fuse_signals as a helper method for enhanced internal fusion
    def fuse_signals(self, signals: List[Signal], data: pd.DataFrame = None, 
                    symbol: str = "XAUUSDm", timeframe: str = "M15") -> Optional[Signal]:
        """Fuse multiple signals using weighted voting"""
        try:
            if not signals or len(signals) < self.min_signals:
                return None
            
            valid_signals = [s for s in signals if s is not None and s.confidence > 0.3]
            
            if len(valid_signals) < self.min_signals:
                return None
            
            buy_weight = 0.0
            sell_weight = 0.0
            hold_weight = 0.0
            
            total_weight = 0.0
            signal_details = []
            
            for signal in valid_signals:
                strategy_name = signal.strategy_name
                weight = self.strategy_weights[strategy_name]
                confidence = signal.confidence
                
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
            
            buy_score = buy_weight / total_weight
            sell_score = sell_weight / total_weight
            hold_score = hold_weight / total_weight
            
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
                return None
            
            current_price = self._get_current_price(valid_signals, data)
            if current_price is None:
                return None
            
            stop_loss, take_profit = self._calculate_risk_parameters(
                valid_signals, final_signal_type, current_price
            )
            
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
            
            # Store signal for internal tracking of fusion results
            self._store_fusion_signal_record(fused_signal, valid_signals)
            
            self.logger.info(f"Fused signal: {final_signal_type.value} with confidence {final_confidence:.3f}")
            
            return fused_signal
            
        except Exception as e:
            self.logger.error(f"Signal fusion failed: {e}", exc_info=True)
            return None
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """Generate weighted voting fusion signals - Primary entry point for SignalEngine"""
        signals = []
        try:
            self.logger.info(f"Weighted Voting - Analyzing {symbol} on {timeframe}")
            
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available for signal generation.")
                return []
            
            data = self.mt5_manager.get_historical_data(symbol, timeframe, 100)
            if data is None or len(data) < 50:
                self.logger.warning(f"Insufficient data for Weighted Voting analysis: {len(data) if data is not None else 0}")
                return []
            
            self.prediction_count += 1
            if self.prediction_count % self.memory_cleanup_interval == 0:
                self._cleanup_memory()
            
            # Generate simulated strategy signals for fusion
            strategy_signals = self._generate_simulated_signals(data, symbol, timeframe)
            
            if not strategy_signals:
                self.logger.info("No strategy signals to fuse")
                return []
            
            # Apply weighted voting fusion to generate multiple signals
            for i in range(3):
                fused_signal = self._fuse_signals_enhanced(strategy_signals, data, symbol, timeframe, i)
                if fused_signal and self.validate_signal(fused_signal): # Validate using base class method
                    signals.append(fused_signal)
            
            if signals:
                avg_confidence = np.mean([s.confidence for s in signals])
                self.logger.info(f"Generated {len(signals)} signals (avg confidence: {avg_confidence:.2f})")
            else:
                self.logger.info("No valid signals generated")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Weighted voting signal generation failed: {e}", exc_info=True)
            return []
    
    # Renamed to _update_component_performance_metrics to clarify its role
    def _update_component_performance_metrics(self, signal: Signal, actual_outcome: float):
        """Update component strategy performance based on signal outcomes"""
        try:
            if 'signal_details' not in signal.metadata:
                return
            
            for detail in signal.metadata['signal_details']:
                strategy_name = detail['strategy']
                
                predicted_direction = 1 if detail['signal_type'] == 'BUY' else -1
                actual_direction = 1 if actual_outcome > 0 else -1
                
                was_correct = predicted_direction == actual_direction
                
                self.strategy_predictions[strategy_name] += 1
                if was_correct:
                    self.strategy_correct[strategy_name] += 1
                
                self.strategy_performance[strategy_name].append(1.0 if was_correct else 0.0)
                
                if len(self.strategy_performance[strategy_name]) > self.performance_window:
                    self.strategy_performance[strategy_name].pop(0)
                
                self._update_strategy_weight(strategy_name, was_correct)
            
            self.logger.info(f"Component performance updated for signal from {signal.timestamp}")
            
        except Exception as e:
            self.logger.error(f"Component performance update failed: {e}", exc_info=True)
    
    def _update_strategy_weight(self, strategy_name: str, was_correct: bool):
        """Update individual strategy weight based on performance"""
        try:
            current_weight = self.strategy_weights[strategy_name]
            
            if was_correct:
                new_weight = min(current_weight * self.weight_boost, self.max_weight)
            else:
                new_weight = max(current_weight * self.weight_decay, self.min_weight)
            
            self.strategy_weights[strategy_name] = new_weight
            
            if strategy_name in self.strategy_performance and self.strategy_performance[strategy_name]:
                accuracy = np.mean(self.strategy_performance[strategy_name])
                self.logger.debug(f"Strategy {strategy_name}: weight={new_weight:.3f}, accuracy={accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Weight update failed for {strategy_name}: {e}", exc_info=True)
    
    def _get_current_price(self, signals: List[Signal], data: pd.DataFrame = None) -> Optional[float]:
        """Get current price from signals or data"""
        try:
            if data is not None and len(data) > 0 and 'Close' in data.columns:
                return float(data['Close'].iloc[-1])
            
            if signals:
                prices = [s.price for s in signals if s.price > 0]
                if prices:
                    return np.mean(prices)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Price extraction failed: {e}", exc_info=True)
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
            
            if stop_losses:
                if signal_type == SignalType.BUY:
                    stop_loss = max(stop_losses)
                else:
                    stop_loss = min(stop_losses)
            else:
                if signal_type == SignalType.BUY:
                    stop_loss = current_price * 0.99
                else:
                    stop_loss = current_price * 1.01
            
            if take_profits:
                if signal_type == SignalType.BUY:
                    take_profit = min(take_profits)
                else:
                    take_profit = max(take_profits)
            else:
                if signal_type == SignalType.BUY:
                    take_profit = current_price * 1.02
                else:
                    take_profit = current_price * 0.98
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Risk parameter calculation failed: {e}", exc_info=True)
            if signal_type == SignalType.BUY:
                return current_price * 0.99, current_price * 1.02
            else:
                return current_price * 1.01, current_price * 0.98
    
    def _store_fusion_signal_record(self, fused_signal: Signal, constituent_signals: List[Signal]):
        """Store fused signal information for internal performance tracking"""
        try:
            signal_record = {
                'timestamp': fused_signal.timestamp,
                'signal_type': fused_signal.signal_type,
                'confidence': fused_signal.confidence,
                'price': fused_signal.price,
                'constituent_count': len(constituent_signals),
                'strategy_names': [s.strategy_name for s in constituent_signals]
            }
            
            self.fusion_signal_records.append(signal_record)
            
            if len(self.fusion_signal_records) > self.max_history:
                self.fusion_signal_records.pop(0)
            
        except Exception as e:
            self.logger.error(f"Fused signal record storage failed: {e}", exc_info=True)
    
    def get_component_strategy_weights(self) -> Dict[str, float]:
        """Get current weights of component strategies"""
        return dict(self.strategy_weights)
    
    def get_component_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics of component strategies"""
        performance_stats = {}
        
        for strategy_name in self.strategy_weights.keys():
            stats = {
                'weight': self.strategy_weights[strategy_name],
                'predictions': self.strategy_predictions[strategy_name],
                'correct': self.strategy_correct[strategy_name],
                'accuracy': 0.0,
                'recent_accuracy': 0.0
            }
            
            if stats['predictions'] > 0:
                stats['accuracy'] = stats['correct'] / stats['predictions']
            
            if strategy_name in self.strategy_performance and self.strategy_performance[strategy_name]:
                stats['recent_accuracy'] = np.mean(self.strategy_performance[strategy_name])
            
            performance_stats[strategy_name] = stats
        
        return performance_stats
    
    def reset_weights(self):
        """Reset all component strategy weights to initial values"""
        for strategy_name in self.strategy_weights.keys():
            self.strategy_weights[strategy_name] = self.initial_weight
        
        self.logger.info("Component strategy weights reset to initial values")
    
    def set_strategy_weight(self, strategy_name: str, weight: float):
        """Manually set weight for a specific component strategy"""
        weight = max(self.min_weight, min(weight, self.max_weight))
        self.strategy_weights[strategy_name] = weight
        
        self.logger.info(f"Strategy {strategy_name} weight set to {weight:.3f}")
    
    def _generate_simulated_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Generate simulated strategy signals for fusion testing"""
        try:
            signals = []
            current_price = data['close'].iloc[-1]
            timestamp = datetime.now()
            
            strategy_configs = [
                ("technical_momentum", 0.72, SignalType.BUY, 'momentum_signal'),
                ("technical_trend", 0.68, SignalType.SELL, 'trend_signal'),
                ("smc_structure", 0.75, SignalType.BUY, 'structure_signal'),
                ("ml_prediction", 0.70, SignalType.BUY, 'ml_signal'),
                ("volume_analysis", 0.65, SignalType.SELL, 'volume_signal')
            ]
            
            for strategy_name, base_confidence, signal_type, reason in strategy_configs:
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
            self.logger.error(f"Error generating simulated signals: {e}", exc_info=True)
            return []
    
    def _fuse_signals_enhanced(self, signals: List[Signal], data: pd.DataFrame, 
                              symbol: str, timeframe: str, variation_index: int) -> Optional[Signal]:
        """Enhanced signal fusion with variations"""
        try:
            # Re-use the main fuse_signals logic
            fused_signal = self.fuse_signals(signals, data, symbol, timeframe)
            
            if fused_signal is None:
                return None
            
            # Apply variation-based adjustments to confidence/parameters if needed
            if variation_index > 0:
                fused_signal.confidence = min(fused_signal.confidence * (1.0 + variation_index * 0.05), 0.99)
                fused_signal.strength = fused_signal.confidence # Adjust strength too

                # Adjust risk parameters for variations
                current_price = self._get_current_price(signals, data)
                if current_price is not None:
                    atr = current_price * 0.01 # Simple ATR
                    risk_multiplier = 1.0 + (variation_index * 0.2)
                    
                    if fused_signal.signal_type == SignalType.BUY:
                        fused_signal.stop_loss = current_price - (atr * 2.0 * risk_multiplier)
                        fused_signal.take_profit = current_price + (atr * 3.0 * fused_signal.confidence)
                    else:
                        fused_signal.stop_loss = current_price + (atr * 2.0 * risk_multiplier)
                        fused_signal.take_profit = current_price - (atr * 3.0 * fused_signal.confidence)
                
                fused_signal.strategy_name = f"{self.strategy_name}_{variation_index+1}" # Unique name for variations
                fused_signal.metadata['variation_index'] = variation_index
                fused_signal.metadata['signal_reason'] = f'weighted_voting_fusion_variation_{variation_index+1}'

            return fused_signal
            
        except Exception as e:
            self.logger.error(f"Enhanced signal fusion failed: {e}", exc_info=True)
            return None
    
    def _cleanup_memory(self):
        """Clean up memory for 8GB RAM optimization"""
        try:
            import gc
            
            if len(self.fusion_signal_records) > self.max_history:
                self.fusion_signal_records = self.fusion_signal_records[-self.max_history//2:]
            
            for strategy_name in list(self.strategy_performance.keys()):
                if len(self.strategy_performance[strategy_name]) > self.performance_window:
                    self.strategy_performance[strategy_name] = self.strategy_performance[strategy_name][-self.performance_window//2:]
            
            gc.collect()
            
            self.logger.info(f"Memory cleanup completed (prediction #{self.prediction_count})")
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {str(e)}", exc_info=True)
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze fusion strategy performance"""
        try:
            component_perf_stats = self.get_component_strategy_performance()
            
            analysis = {
                'strategy': self.strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'total_fused_signals_recorded': len(self.fusion_signal_records),
                'active_component_strategies': len(self.strategy_weights),
                'component_strategy_performance_metrics': component_perf_stats,
                'current_component_weights': self.get_component_strategy_weights(),
                'fusion_parameters': {
                    'min_signals': self.min_signals,
                    'min_confidence': self.min_confidence,
                    'performance_window': self.performance_window,
                    'weight_decay': self.weight_decay,
                    'weight_boost': self.weight_boost
                },
                'memory_optimized': True,
                'max_fusion_records_history': self.max_history
            }
            
            if self.fusion_signal_records:
                recent_signals = self.fusion_signal_records[-50:]
                analysis['recent_fused_signal_count'] = len(recent_signals)
                
                if recent_signals:
                    avg_confidence = np.mean([s['confidence'] for s in recent_signals])
                    analysis['average_fused_confidence'] = avg_confidence
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            return {
                'strategy': self.strategy_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the Weighted Voting Fusion Strategy,
        including its parameters, component strategy insights, and overall performance.
        """
        # Get overall trading performance of the signals generated by this strategy
        base_trading_performance = self.get_performance_summary()
        
        # Get detailed component strategy performance metrics
        component_perf_stats = self.get_component_strategy_performance()

        info = {
            'name': 'Weighted Voting Fusion Strategy',
            'version': '2.0.0',
            'type': 'Fusion',
            'description': 'Combines signals from multiple strategies using weighted voting, dynamically adjusting weights.',
            'parameters': {
                'min_signals': self.min_signals,
                'min_confidence': self.min_confidence,
                'performance_window': self.performance_window,
                'initial_weight': self.initial_weight,
                'weight_decay': self.weight_decay,
                'weight_boost': self.weight_boost,
                'min_weight': self.min_weight,
                'max_weight': self.max_weight,
                'max_history_records': self.max_history
            },
            'overall_trading_performance': {
                'total_signals_generated': base_trading_performance['total_signals'],
                'win_rate': base_trading_performance['win_rate'],
                'profit_factor': base_trading_performance['profit_factor']
            },
            'component_strategy_insights': {
                'active_strategies_count': len(self.strategy_weights),
                'current_weights': self.get_component_strategy_weights(),
                'performance_by_strategy': component_perf_stats
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
            'min_signals': 2,
            'min_confidence': 0.60,
            'performance_window': 50,
            'initial_weight': 1.0,
            'weight_decay': 0.95,
            'weight_boost': 1.1,
            'min_weight': 0.1,
            'max_weight': 3.0,
            'max_history': 500,
            'memory_cleanup_interval': 25
        }
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
                'Open': close_prices + np.random.randn(len(dates)) * 0.5,
                'High': close_prices + np.abs(np.random.randn(len(dates)) * 3),
                'Low': close_prices - np.abs(np.random.randn(len(dates)) * 3),
                'Close': close_prices,
                'Volume': np.random.randint(100, 1000, len(dates))
            }, index=dates)
            
            return data
    
    # Create strategy instance
    mock_mt5 = MockMT5Manager()
    strategy = WeightedVoting(test_config, mock_mt5, database=None)
    
    # Output header matching other strategy files
    print("============================================================")
    print("TESTING MODIFIED WEIGHTED VOTING STRATEGY")
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
            # print(f"     Component Weights: {signal.metadata.get('strategy_weights', 'N/A')}") # Example of accessing metadata
    
    # 2. Testing analysis method
    print("\n2. Testing analysis method:")
    mock_data = mock_mt5.get_historical_data("XAUUSDm", "M15", 100)
    analysis_results = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis results keys: {analysis_results.keys()}")
    print(f"   Total Fused Signals Recorded: {analysis_results.get('total_fused_signals_recorded', 'N/A')}")
    print(f"   Active Component Strategies: {analysis_results.get('active_component_strategies', 'N/A')}")
    
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
    print(f"   Overall Trading Performance:")
    print(f"     Total Signals Generated: {info['overall_trading_performance']['total_signals_generated']}")
    print(f"     Win Rate: {info['overall_trading_performance']['win_rate']:.2%}")
    print(f"     Profit Factor: {info['overall_trading_performance']['profit_factor']:.2f}")
    print(f"   Component Strategy Insights:")
    print(f"     Active Strategies Count: {info['component_strategy_insights']['active_strategies_count']}")
    # print(f"     Current Weights: {info['component_strategy_insights']['current_weights']}") # Can be verbose
    # print(f"     Performance By Strategy: {info['component_strategy_insights']['performance_by_strategy']}") # Can be verbose

    # Footer matching other strategy files
    print("\n============================================================")
    print("WEIGHTED VOTING STRATEGY TEST COMPLETED!")
    print("============================================================")
