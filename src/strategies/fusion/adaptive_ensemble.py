#!/usr/bin/env python3
"""
Adaptive Ensemble Fusion Strategy
=================================

This fusion strategy dynamically adapts strategy weights based on:
- Recent performance metrics
- Market regime changes
- Signal correlation analysis
- Volatility adjustments
- Time-based decay factors

The strategy continuously learns from performance and adjusts the ensemble
composition to maximize returns while minimizing risk.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade
except ImportError:
    # Fallback definitions
    from abc import ABC, abstractmethod
    from enum import Enum
    from dataclasses import dataclass
    
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
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    class AbstractStrategy(ABC):
        def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
            self.config = config
            self.mt5_manager = mt5_manager
            self.database = database
            self.logger = logging.getLogger(self.__class__.__name__)


@dataclass
class PerformanceMetrics:
    """Performance tracking for strategies"""
    total_signals: int = 0
    winning_signals: int = 0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    volatility: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class MarketRegime:
    """Market regime information"""
    regime_type: str = "NEUTRAL"
    volatility_level: str = "MEDIUM"
    trend_strength: float = 0.0
    momentum: float = 0.0
    confidence: float = 0.0
    duration: int = 0


class AdaptiveEnsembleFusionStrategy(AbstractStrategy):
    """
    Adaptive Ensemble Fusion Strategy
    
    Dynamically adjusts strategy weights based on:
    - Performance metrics with time decay
    - Market regime detection and adaptation
    - Signal correlation and diversification
    - Risk-adjusted returns
    - Momentum and trend following
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        super().__init__(config, mt5_manager, database)
        
        # Configuration
        self.lookback_bars = config.get('parameters', {}).get('lookback_bars', 200)
        self.performance_window = config.get('parameters', {}).get('performance_window', 50)
        self.adaptation_rate = config.get('parameters', {}).get('adaptation_rate', 0.1)
        self.min_signals_for_weight = config.get('parameters', {}).get('min_signals_for_weight', 10)
        self.correlation_threshold = config.get('parameters', {}).get('correlation_threshold', 0.7)
        self.decay_factor = config.get('parameters', {}).get('decay_factor', 0.95)
        
        # Strategy tracking
        self.strategy_performance: Dict[str, PerformanceMetrics] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.signal_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.performance_window))
        self.return_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.performance_window))
        
        # Market regime tracking
        self.current_regime = MarketRegime()
        self.regime_history: deque = deque(maxlen=20)
        self.regime_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Signal correlation matrix
        self.correlation_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.last_correlation_update = datetime.now()
        
        # Ensemble state
        self.last_signals: Dict[str, Signal] = {}
        self.ensemble_performance = PerformanceMetrics()
        
        self.logger.info("Adaptive Ensemble Fusion Strategy initialized")
    
    def generate_signal(self, symbol: str, timeframe: str) -> Optional[Signal]:
        """
        Generate ensemble signal by adaptively combining strategy signals
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            
        Returns:
            Fused signal or None
        """
        try:
            # Get market data for regime detection
            market_data = self._get_market_data(symbol, timeframe)
            if market_data is None or market_data.empty:
                return self._create_fallback_signal(symbol, timeframe)
            
            # Update market regime
            self._update_market_regime(market_data)
            
            # Get signals from other strategies (simulated for now)
            strategy_signals = self._get_strategy_signals(symbol, timeframe, market_data)
            
            if not strategy_signals:
                return None
            
            # Update strategy performance
            self._update_strategy_performance(strategy_signals)
            
            # Update correlation matrix
            self._update_correlation_matrix(strategy_signals)
            
            # Calculate adaptive weights
            adaptive_weights = self._calculate_adaptive_weights(strategy_signals)
            
            # Generate ensemble signal
            ensemble_signal = self._generate_ensemble_signal(
                strategy_signals, adaptive_weights, symbol, timeframe, market_data
            )
            
            # Update tracking
            if ensemble_signal:
                self._track_ensemble_signal(ensemble_signal, strategy_signals)
            
            return ensemble_signal
            
        except Exception as e:
            self.logger.error(f"Error in adaptive ensemble signal generation: {e}")
            return self._create_fallback_signal(symbol, timeframe)
    
    def _get_market_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get market data for analysis"""
        try:
            if self.mt5_manager:
                return self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_bars)
            else:
                # Simulate market data for testing
                dates = pd.date_range(end=datetime.now(), periods=self.lookback_bars, freq='15min')
                np.random.seed(42)
                
                price = 2000.0
                data = []
                
                for i in range(len(dates)):
                    change = np.random.normal(0, 2)
                    price += change
                    
                    high = price + abs(np.random.normal(0, 1))
                    low = price - abs(np.random.normal(0, 1))
                    open_price = price - np.random.normal(0, 0.5)
                    close = price
                    volume = np.random.randint(1000, 10000)
                    
                    data.append({
                        'timestamp': dates[i],
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'close': close,
                        'volume': volume
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None
    
    def _update_market_regime(self, data: pd.DataFrame) -> None:
        """Update current market regime"""
        try:
            # Calculate regime indicators
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # Trend strength using ADX-like calculation
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Moving averages for trend
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            trend_strength = abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / atr if atr > 0 else 0
            
            # Momentum
            momentum = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
            
            # Determine regime
            if volatility > 0.25:
                regime_type = "HIGH_VOLATILITY"
            elif volatility < 0.10:
                regime_type = "LOW_VOLATILITY"
            elif trend_strength > 2.0:
                if momentum > 0:
                    regime_type = "STRONG_UPTREND"
                else:
                    regime_type = "STRONG_DOWNTREND"
            elif abs(momentum) < 0.02:
                regime_type = "SIDEWAYS"
            else:
                regime_type = "TRENDING"
            
            # Update regime
            self.current_regime = MarketRegime(
                regime_type=regime_type,
                volatility_level="HIGH" if volatility > 0.20 else "LOW" if volatility < 0.15 else "MEDIUM",
                trend_strength=trend_strength,
                momentum=momentum,
                confidence=min(trend_strength / 3.0, 1.0),
                duration=self.current_regime.duration + 1 if self.current_regime.regime_type == regime_type else 1
            )
            
            self.regime_history.append(self.current_regime)
            
        except Exception as e:
            self.logger.error(f"Error updating market regime: {e}")
    
    def _get_strategy_signals(self, symbol: str, timeframe: str, data: pd.DataFrame) -> List[Signal]:
        """Get signals from various strategies (simulated)"""
        try:
            signals = []
            current_price = data['close'].iloc[-1]
            timestamp = datetime.now()
            
            # Simulate signals from different strategy types
            strategy_configs = [
                ("technical_trend", 0.75, SignalType.BUY),
                ("technical_momentum", 0.65, SignalType.SELL),
                ("smc_structure", 0.80, SignalType.BUY),
                ("ml_lstm", 0.70, SignalType.HOLD),
                ("ml_xgboost", 0.85, SignalType.BUY)
            ]
            
            for strategy_name, base_confidence, signal_type in strategy_configs:
                # Add some randomness and regime influence
                regime_modifier = self._get_regime_modifier(strategy_name)
                confidence = min(max(base_confidence * regime_modifier, 0.1), 0.95)
                
                signal = Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    strategy_name=strategy_name,
                    signal_type=signal_type,
                    confidence=confidence,
                    price=current_price,
                    timeframe=timeframe,
                    strength=confidence,
                    metadata={
                        'regime': self.current_regime.regime_type,
                        'volatility': self.current_regime.volatility_level
                    }
                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error getting strategy signals: {e}")
            return []
    
    def _get_regime_modifier(self, strategy_name: str) -> float:
        """Get regime-based modifier for strategy performance"""
        regime = self.current_regime.regime_type
        
        # Strategy performance modifiers based on regime
        regime_modifiers = {
            "STRONG_UPTREND": {
                "technical_trend": 1.2,
                "technical_momentum": 1.1,
                "smc_structure": 0.9,
                "ml_lstm": 1.0,
                "ml_xgboost": 1.1
            },
            "STRONG_DOWNTREND": {
                "technical_trend": 1.2,
                "technical_momentum": 1.1,
                "smc_structure": 0.9,
                "ml_lstm": 1.0,
                "ml_xgboost": 1.1
            },
            "SIDEWAYS": {
                "technical_trend": 0.8,
                "technical_momentum": 0.9,
                "smc_structure": 1.2,
                "ml_lstm": 1.1,
                "ml_xgboost": 0.9
            },
            "HIGH_VOLATILITY": {
                "technical_trend": 0.9,
                "technical_momentum": 1.2,
                "smc_structure": 0.8,
                "ml_lstm": 0.9,
                "ml_xgboost": 1.0
            }
        }
        
        return regime_modifiers.get(regime, {}).get(strategy_name, 1.0)
    
    def _update_strategy_performance(self, signals: List[Signal]) -> None:
        """Update performance metrics for strategies"""
        try:
            for signal in signals:
                strategy_name = signal.strategy_name
                
                # Initialize if new strategy
                if strategy_name not in self.strategy_performance:
                    self.strategy_performance[strategy_name] = PerformanceMetrics()
                
                # Add signal to history
                self.signal_history[strategy_name].append(signal)
                
                # Simulate return (in real implementation, this would come from trade results)
                simulated_return = np.random.normal(0.001, 0.01) * signal.confidence
                self.return_history[strategy_name].append(simulated_return)
                
                # Update metrics
                perf = self.strategy_performance[strategy_name]
                perf.total_signals += 1
                perf.total_return += simulated_return
                
                if simulated_return > 0:
                    perf.winning_signals += 1
                
                # Calculate derived metrics
                if perf.total_signals > 0:
                    perf.win_rate = perf.winning_signals / perf.total_signals
                    perf.avg_return = perf.total_return / perf.total_signals
                
                # Calculate volatility and Sharpe ratio
                if len(self.return_history[strategy_name]) > 10:
                    returns = list(self.return_history[strategy_name])
                    perf.volatility = np.std(returns)
                    perf.sharpe_ratio = perf.avg_return / perf.volatility if perf.volatility > 0 else 0
                
                perf.last_update = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {e}")
    
    def _update_correlation_matrix(self, signals: List[Signal]) -> None:
        """Update signal correlation matrix"""
        try:
            # Update correlation every 10 signals
            if (datetime.now() - self.last_correlation_update).seconds < 600:
                return
            
            # Calculate correlations between strategy returns
            strategy_names = list(self.return_history.keys())
            
            for i, strategy1 in enumerate(strategy_names):
                for j, strategy2 in enumerate(strategy_names):
                    if i <= j:
                        continue
                    
                    returns1 = list(self.return_history[strategy1])
                    returns2 = list(self.return_history[strategy2])
                    
                    if len(returns1) > 10 and len(returns2) > 10:
                        # Align lengths
                        min_len = min(len(returns1), len(returns2))
                        returns1 = returns1[-min_len:]
                        returns2 = returns2[-min_len:]
                        
                        correlation = np.corrcoef(returns1, returns2)[0, 1]
                        
                        self.correlation_matrix[strategy1][strategy2] = correlation
                        self.correlation_matrix[strategy2][strategy1] = correlation
            
            self.last_correlation_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating correlation matrix: {e}")
    
    def _calculate_adaptive_weights(self, signals: List[Signal]) -> Dict[str, float]:
        """Calculate adaptive weights for strategies"""
        try:
            weights = {}
            total_score = 0.0
            
            for signal in signals:
                strategy_name = signal.strategy_name
                
                # Base score from performance
                perf = self.strategy_performance.get(strategy_name, PerformanceMetrics())
                
                if perf.total_signals < self.min_signals_for_weight:
                    # Use confidence for new strategies
                    score = signal.confidence
                else:
                    # Combine multiple factors
                    performance_score = perf.win_rate * 0.3 + (perf.sharpe_ratio + 1) * 0.2
                    confidence_score = signal.confidence * 0.3
                    regime_score = self._get_regime_modifier(strategy_name) * 0.2
                    
                    score = performance_score + confidence_score + regime_score
                
                # Apply time decay
                time_decay = self.decay_factor ** ((datetime.now() - perf.last_update).days)
                score *= time_decay
                
                # Diversification bonus (reduce weight for highly correlated strategies)
                diversification_bonus = self._calculate_diversification_bonus(strategy_name)
                score *= diversification_bonus
                
                weights[strategy_name] = max(score, 0.01)  # Minimum weight
                total_score += weights[strategy_name]
            
            # Normalize weights
            if total_score > 0:
                for strategy_name in weights:
                    weights[strategy_name] /= total_score
            
            # Store weights
            self.strategy_weights.update(weights)
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive weights: {e}")
            # Return equal weights as fallback
            return {signal.strategy_name: 1.0/len(signals) for signal in signals}
    
    def _calculate_diversification_bonus(self, strategy_name: str) -> float:
        """Calculate diversification bonus for strategy"""
        try:
            if strategy_name not in self.correlation_matrix:
                return 1.0
            
            # Calculate average correlation with other strategies
            correlations = []
            for other_strategy, correlation in self.correlation_matrix[strategy_name].items():
                if abs(correlation) > self.correlation_threshold:
                    correlations.append(abs(correlation))
            
            if not correlations:
                return 1.0
            
            avg_correlation = np.mean(correlations)
            # Higher correlation = lower bonus
            bonus = 1.0 - (avg_correlation - self.correlation_threshold) * 0.5
            
            return max(bonus, 0.5)
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification bonus: {e}")
            return 1.0
    
    def _generate_ensemble_signal(self, signals: List[Signal], weights: Dict[str, float], 
                                symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[Signal]:
        """Generate final ensemble signal"""
        try:
            if not signals:
                return None
            
            # Calculate weighted signal components
            weighted_confidence = 0.0
            signal_votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
            total_weight = 0.0
            
            for signal in signals:
                weight = weights.get(signal.strategy_name, 0.0)
                weighted_confidence += signal.confidence * weight
                signal_votes[signal.signal_type.value] += weight
                total_weight += weight
            
            if total_weight == 0:
                return None
            
            # Normalize
            weighted_confidence /= total_weight
            
            # Determine final signal type
            final_signal_type = max(signal_votes, key=signal_votes.get)
            signal_strength = signal_votes[final_signal_type] / total_weight
            
            # Apply minimum confidence threshold
            if weighted_confidence < 0.6 or signal_strength < 0.4:
                final_signal_type = "HOLD"
                weighted_confidence = max(weighted_confidence, 0.5)
            
            # Calculate risk parameters
            current_price = data['close'].iloc[-1]
            atr = self._calculate_atr(data)
            
            stop_loss = None
            take_profit = None
            
            if final_signal_type == "BUY":
                stop_loss = current_price - (atr * 2.0 * (1 + self.current_regime.volatility_level == "HIGH"))
                take_profit = current_price + (atr * 3.0 * weighted_confidence)
            elif final_signal_type == "SELL":
                stop_loss = current_price + (atr * 2.0 * (1 + self.current_regime.volatility_level == "HIGH"))
                take_profit = current_price - (atr * 3.0 * weighted_confidence)
            
            # Create ensemble signal
            ensemble_signal = Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                strategy_name="adaptive_ensemble",
                signal_type=SignalType(final_signal_type),
                confidence=weighted_confidence,
                price=current_price,
                timeframe=timeframe,
                strength=signal_strength,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'regime': self.current_regime.regime_type,
                    'volatility': self.current_regime.volatility_level,
                    'strategy_weights': weights.copy(),
                    'signal_votes': signal_votes.copy(),
                    'component_signals': len(signals)
                }
            )
            
            return ensemble_signal
            
        except Exception as e:
            self.logger.error(f"Error generating ensemble signal: {e}")
            return None
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else data['close'].iloc[-1] * 0.01
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return data['close'].iloc[-1] * 0.01
    
    def _track_ensemble_signal(self, ensemble_signal: Signal, component_signals: List[Signal]) -> None:
        """Track ensemble signal for performance analysis"""
        try:
            self.last_signals['ensemble'] = ensemble_signal
            
            for signal in component_signals:
                self.last_signals[signal.strategy_name] = signal
            
            # Update ensemble performance tracking
            self.ensemble_performance.total_signals += 1
            
        except Exception as e:
            self.logger.error(f"Error tracking ensemble signal: {e}")
    
    def _create_fallback_signal(self, symbol: str, timeframe: str) -> Signal:
        """Create fallback signal when normal processing fails"""
        return Signal(
            timestamp=datetime.now(),
            symbol=symbol,
            strategy_name="adaptive_ensemble",
            signal_type=SignalType.HOLD,
            confidence=0.5,
            price=2000.0,  # Default price
            timeframe=timeframe,
            strength=0.5,
            metadata={'fallback': True, 'reason': 'data_unavailable'}
        )
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        return self.strategy_weights.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies"""
        summary = {
            'ensemble_performance': {
                'total_signals': self.ensemble_performance.total_signals,
                'win_rate': self.ensemble_performance.win_rate,
                'total_return': self.ensemble_performance.total_return,
                'sharpe_ratio': self.ensemble_performance.sharpe_ratio
            },
            'strategy_performance': {},
            'current_regime': {
                'type': self.current_regime.regime_type,
                'volatility': self.current_regime.volatility_level,
                'trend_strength': self.current_regime.trend_strength,
                'momentum': self.current_regime.momentum
            },
            'strategy_weights': self.strategy_weights.copy()
        }
        
        for strategy_name, perf in self.strategy_performance.items():
            summary['strategy_performance'][strategy_name] = {
                'total_signals': perf.total_signals,
                'win_rate': perf.win_rate,
                'total_return': perf.total_return,
                'sharpe_ratio': perf.sharpe_ratio,
                'volatility': perf.volatility
            }
        
        return summary


def test_adaptive_ensemble_strategy():
    """Test the Adaptive Ensemble Fusion Strategy"""
    print("\n" + "="*60)
    print("TESTING ADAPTIVE ENSEMBLE FUSION STRATEGY")
    print("="*60)
    
    # Test configuration
    config = {
        'name': 'adaptive_ensemble',
        'parameters': {
            'lookback_bars': 100,
            'performance_window': 30,
            'adaptation_rate': 0.1,
            'min_signals_for_weight': 5,
            'correlation_threshold': 0.7,
            'decay_factor': 0.95
        }
    }
    
    try:
        # Initialize strategy
        print("\n1. Initializing Adaptive Ensemble Strategy...")
        strategy = AdaptiveEnsembleFusionStrategy(config)
        print("   ✅ Strategy initialized successfully")
        
        # Test signal generation
        print("\n2. Testing signal generation...")
        signal = strategy.generate_signal("XAUUSDm", "M15")
        
        if signal:
            print(f"   ✅ Signal generated: {signal.signal_type.value}")
            print(f"      Confidence: {signal.confidence:.3f}")
            print(f"      Strength: {signal.strength:.3f}")
            print(f"      Price: {signal.price:.2f}")
            print(f"      Regime: {signal.metadata.get('regime', 'N/A')}")
            print(f"      Component signals: {signal.metadata.get('component_signals', 0)}")
        else:
            print("   ⚠️  No signal generated")
        
        # Test multiple signals for weight adaptation
        print("\n3. Testing weight adaptation...")
        for i in range(10):
            test_signal = strategy.generate_signal("XAUUSDm", "M15")
            if test_signal:
                print(f"   Signal {i+1}: {test_signal.signal_type.value} (conf: {test_signal.confidence:.3f})")
        
        # Show strategy weights
        print("\n4. Current strategy weights:")
        weights = strategy.get_strategy_weights()
        for strategy_name, weight in weights.items():
            print(f"   {strategy_name}: {weight:.3f}")
        
        # Show performance summary
        print("\n5. Performance summary:")
        summary = strategy.get_performance_summary()
        
        print(f"   Ensemble signals: {summary['ensemble_performance']['total_signals']}")
        print(f"   Current regime: {summary['current_regime']['type']}")
        print(f"   Volatility level: {summary['current_regime']['volatility']}")
        
        for strategy_name, perf in summary['strategy_performance'].items():
            print(f"   {strategy_name}: {perf['total_signals']} signals, "
                  f"win rate: {perf['win_rate']:.2%}")
        
        print("\n✅ Adaptive Ensemble Fusion Strategy test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    success = test_adaptive_ensemble_strategy()
    exit(0 if success else 1)
