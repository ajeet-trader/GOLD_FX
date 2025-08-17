"""
Regime Detection Fusion Strategy - Market Condition Adaptive Strategy
=====================================================================

Detects market regimes and adapts signal fusion based on current market conditions.
Filters and weights signals based on regime-specific performance.
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
from typing import Dict, List, Any, Optional, Tuple, Enum
import logging
from collections import defaultdict

# Import base classes
try:
    from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade
except ImportError:
    from core.base import AbstractStrategy, Signal, SignalType, SignalGrade


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


class RegimeDetection(AbstractStrategy):
    """Regime detection fusion strategy for market-adaptive signal combination"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize regime detection fusion strategy"""
        super().__init__(config)
        
        self.strategy_name = "RegimeDetection"
        self.min_signals = config.get('min_signals', 2)
        self.min_confidence = config.get('min_confidence', 0.60)
        
        # Regime detection parameters
        self.lookback_period = config.get('lookback_period', 50)
        self.trend_threshold = config.get('trend_threshold', 0.02)
        self.volatility_window = config.get('volatility_window', 20)
        self.volatility_threshold = config.get('volatility_threshold', 0.015)
        self.breakout_threshold = config.get('breakout_threshold', 2.0)
        
        # Regime-specific strategy weights
        self.regime_weights = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        
        # Current market state
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        self.regime_history = []
        self.max_regime_history = config.get('max_regime_history', 200)
        
        # Performance tracking
        self.signal_history = []
        self.max_history = config.get('max_history', 1000)
        
        # Logger
        self.logger = logging.getLogger(self.strategy_name)
        
        self.logger.info(f"{self.strategy_name} initialized")
    
    def fuse_signals(self, signals: List[Signal], data: pd.DataFrame = None, 
                    symbol: str = "XAUUSDm", timeframe: str = "M15") -> Optional[Signal]:
        """Fuse signals based on detected market regime"""
        try:
            if not signals or len(signals) < self.min_signals:
                return None
            
            # Detect current market regime
            if data is not None and len(data) >= self.lookback_period:
                self.current_regime, self.regime_confidence = self._detect_regime(data)
            else:
                self.current_regime = MarketRegime.UNKNOWN
                self.regime_confidence = 0.5
            
            # Store regime for history
            self._store_regime_record()
            
            # Filter and weight signals based on regime
            filtered_signals = self._filter_signals_by_regime(signals)
            
            if len(filtered_signals) < self.min_signals:
                return None
            
            # Calculate regime-weighted signal combination
            combined_signal = self._combine_signals_by_regime(filtered_signals, data, symbol, timeframe)
            
            if combined_signal is None:
                return None
            
            # Add regime information to metadata
            combined_signal.metadata.update({
                'fusion_method': 'regime_detection',
                'current_regime': self.current_regime.value,
                'regime_confidence': self.regime_confidence,
                'filtered_signals': len(filtered_signals),
                'original_signals': len(signals),
                'regime_weights': dict(self.regime_weights[self.current_regime])
            })
            
            # Store signal for performance tracking
            self._store_signal_record(combined_signal, filtered_signals)
            
            self.logger.info(f"Fused signal in {self.current_regime.value} regime: {combined_signal.signal_type.value} with confidence {combined_signal.confidence:.3f}")
            
            return combined_signal
            
        except Exception as e:
            self.logger.error(f"Regime-based signal fusion failed: {e}")
            return None
    
    def generate_signal(self, data: pd.DataFrame, symbol: str = "XAUUSDm", 
                       timeframe: str = "M15") -> Optional[Signal]:
        """Generate signal - not used directly, fusion happens via fuse_signals"""
        return None
    
    def _detect_regime(self, data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Detect current market regime from price data"""
        try:
            if len(data) < self.lookback_period:
                return MarketRegime.UNKNOWN, 0.5
            
            close = data['Close'].tail(self.lookback_period)
            high = data['High'].tail(self.lookback_period)
            low = data['Low'].tail(self.lookback_period)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(close)
            
            # Calculate volatility
            volatility = self._calculate_volatility(close)
            
            # Calculate breakout potential
            breakout_score = self._calculate_breakout_score(high, low, close)
            
            # Determine regime based on multiple factors
            regime, confidence = self._classify_regime(trend_strength, volatility, breakout_score)
            
            return regime, confidence
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return MarketRegime.UNKNOWN, 0.5
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength (-1 to 1, negative = downtrend, positive = uptrend)"""
        try:
            # Linear regression slope
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices.values, 1)
            
            # Normalize by price level
            normalized_slope = slope / prices.mean()
            
            # R-squared for trend strength
            y_pred = slope * x + np.mean(prices.values)
            ss_res = np.sum((prices.values - y_pred) ** 2)
            ss_tot = np.sum((prices.values - np.mean(prices.values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Combine slope and R-squared
            trend_strength = normalized_slope * r_squared
            
            # Clamp to [-1, 1]
            return max(-1.0, min(1.0, trend_strength * 100))
            
        except Exception as e:
            self.logger.error(f"Trend strength calculation failed: {e}")
            return 0.0
    
    def _calculate_volatility(self, prices: pd.Series) -> float:
        """Calculate normalized volatility"""
        try:
            returns = prices.pct_change().dropna()
            volatility = returns.std()
            
            # Annualize volatility (assuming daily data, adjust for timeframe)
            annualized_vol = volatility * np.sqrt(252)
            
            return annualized_vol
            
        except Exception as e:
            self.logger.error(f"Volatility calculation failed: {e}")
            return 0.02  # Default volatility
    
    def _calculate_breakout_score(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate breakout potential score"""
        try:
            # Bollinger Bands
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            # Current position relative to bands
            current_close = close.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            if pd.isna(current_upper) or pd.isna(current_lower):
                return 0.0
            
            # Calculate breakout score
            if current_close > current_upper:
                breakout_score = (current_close - current_upper) / (current_upper - current_lower)
            elif current_close < current_lower:
                breakout_score = (current_lower - current_close) / (current_upper - current_lower)
            else:
                # Inside bands - calculate compression
                band_width = (current_upper - current_lower) / sma.iloc[-1]
                avg_band_width = ((upper_band - lower_band) / sma).tail(20).mean()
                
                if not pd.isna(avg_band_width) and avg_band_width > 0:
                    compression = 1.0 - (band_width / avg_band_width)
                    breakout_score = max(0, compression)
                else:
                    breakout_score = 0.0
            
            return min(2.0, max(0.0, breakout_score))
            
        except Exception as e:
            self.logger.error(f"Breakout score calculation failed: {e}")
            return 0.0
    
    def _classify_regime(self, trend_strength: float, volatility: float, breakout_score: float) -> Tuple[MarketRegime, float]:
        """Classify market regime based on calculated metrics"""
        try:
            # Initialize scores for each regime
            regime_scores = {
                MarketRegime.TRENDING_UP: 0.0,
                MarketRegime.TRENDING_DOWN: 0.0,
                MarketRegime.SIDEWAYS: 0.0,
                MarketRegime.HIGH_VOLATILITY: 0.0,
                MarketRegime.LOW_VOLATILITY: 0.0,
                MarketRegime.BREAKOUT: 0.0
            }
            
            # Trending regimes
            if trend_strength > self.trend_threshold:
                regime_scores[MarketRegime.TRENDING_UP] = abs(trend_strength)
            elif trend_strength < -self.trend_threshold:
                regime_scores[MarketRegime.TRENDING_DOWN] = abs(trend_strength)
            else:
                regime_scores[MarketRegime.SIDEWAYS] = 1.0 - abs(trend_strength) / self.trend_threshold
            
            # Volatility regimes
            if volatility > self.volatility_threshold:
                regime_scores[MarketRegime.HIGH_VOLATILITY] = min(1.0, volatility / self.volatility_threshold)
            else:
                regime_scores[MarketRegime.LOW_VOLATILITY] = 1.0 - (volatility / self.volatility_threshold)
            
            # Breakout regime
            if breakout_score > self.breakout_threshold:
                regime_scores[MarketRegime.BREAKOUT] = min(1.0, breakout_score / self.breakout_threshold)
            
            # Find dominant regime
            dominant_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[dominant_regime]
            
            # Ensure minimum confidence
            confidence = max(0.1, min(1.0, confidence))
            
            return dominant_regime, confidence
            
        except Exception as e:
            self.logger.error(f"Regime classification failed: {e}")
            return MarketRegime.UNKNOWN, 0.5
    
    def _filter_signals_by_regime(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals based on regime-specific performance"""
        try:
            if self.current_regime == MarketRegime.UNKNOWN:
                return signals
            
            filtered_signals = []
            
            for signal in signals:
                strategy_name = signal.strategy_name
                
                # Get regime-specific weight for this strategy
                regime_weight = self.regime_weights[self.current_regime][strategy_name]
                
                # Calculate regime-specific performance
                regime_performance = self.regime_performance[self.current_regime][strategy_name]
                if regime_performance:
                    avg_performance = np.mean(regime_performance)
                    # Only include signals from strategies with decent performance in this regime
                    if avg_performance > 0.4:  # 40% success rate threshold
                        filtered_signals.append(signal)
                else:
                    # Include signals from strategies without regime history (benefit of doubt)
                    filtered_signals.append(signal)
            
            return filtered_signals if filtered_signals else signals  # Fallback to all signals
            
        except Exception as e:
            self.logger.error(f"Signal filtering failed: {e}")
            return signals
    
    def _combine_signals_by_regime(self, signals: List[Signal], data: pd.DataFrame = None, 
                                 symbol: str = "XAUUSDm", timeframe: str = "M15") -> Optional[Signal]:
        """Combine signals using regime-specific weights"""
        try:
            if not signals:
                return None
            
            # Calculate weighted votes
            buy_weight = 0.0
            sell_weight = 0.0
            total_weight = 0.0
            
            signal_details = []
            
            for signal in signals:
                strategy_name = signal.strategy_name
                regime_weight = self.regime_weights[self.current_regime][strategy_name]
                confidence = signal.confidence
                
                # Effective weight combines regime weight and signal confidence
                effective_weight = regime_weight * confidence * self.regime_confidence
                
                if signal.signal_type == SignalType.BUY:
                    buy_weight += effective_weight
                elif signal.signal_type == SignalType.SELL:
                    sell_weight += effective_weight
                
                total_weight += effective_weight
                
                signal_details.append({
                    'strategy': strategy_name,
                    'signal_type': signal.signal_type.value,
                    'confidence': confidence,
                    'regime_weight': regime_weight,
                    'effective_weight': effective_weight
                })
            
            if total_weight == 0:
                return None
            
            # Determine final signal
            buy_score = buy_weight / total_weight
            sell_score = sell_weight / total_weight
            
            if buy_score > sell_score and buy_score > self.min_confidence:
                signal_type = SignalType.BUY
                final_confidence = buy_score
            elif sell_score > buy_score and sell_score > self.min_confidence:
                signal_type = SignalType.SELL
                final_confidence = sell_score
            else:
                return None
            
            # Get current price
            current_price = self._get_current_price(signals, data)
            if current_price is None:
                return None
            
            # Calculate risk parameters
            stop_loss, take_profit = self._calculate_regime_risk_parameters(
                signals, signal_type, current_price
            )
            
            # Create combined signal
            combined_signal = Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                strategy_name=self.strategy_name,
                signal_type=signal_type,
                confidence=final_confidence,
                price=current_price,
                timeframe=timeframe,
                strength=final_confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'buy_score': buy_score,
                    'sell_score': sell_score,
                    'signal_details': signal_details
                }
            )
            
            return combined_signal
            
        except Exception as e:
            self.logger.error(f"Signal combination failed: {e}")
            return None
    
    def _calculate_regime_risk_parameters(self, signals: List[Signal], signal_type: SignalType, 
                                        current_price: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate risk parameters adjusted for current regime"""
        try:
            # Base risk parameters from signals
            stop_losses = [s.stop_loss for s in signals if s.stop_loss is not None]
            take_profits = [s.take_profit for s in signals if s.take_profit is not None]
            
            # Regime-specific adjustments
            if self.current_regime == MarketRegime.HIGH_VOLATILITY:
                # Wider stops in high volatility
                stop_multiplier = 1.5
                tp_multiplier = 1.3
            elif self.current_regime == MarketRegime.LOW_VOLATILITY:
                # Tighter stops in low volatility
                stop_multiplier = 0.8
                tp_multiplier = 0.9
            elif self.current_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                # Wider targets in trending markets
                stop_multiplier = 1.0
                tp_multiplier = 1.4
            elif self.current_regime == MarketRegime.BREAKOUT:
                # Wider stops for breakouts
                stop_multiplier = 1.8
                tp_multiplier = 1.6
            else:
                # Default for sideways/unknown
                stop_multiplier = 1.0
                tp_multiplier = 1.0
            
            # Calculate adjusted risk parameters
            if stop_losses:
                base_stop = np.mean(stop_losses)
                if signal_type == SignalType.BUY:
                    adjusted_stop = current_price - (current_price - base_stop) * stop_multiplier
                else:
                    adjusted_stop = current_price + (base_stop - current_price) * stop_multiplier
            else:
                # Default stop loss
                stop_pct = 0.01 * stop_multiplier
                if signal_type == SignalType.BUY:
                    adjusted_stop = current_price * (1.0 - stop_pct)
                else:
                    adjusted_stop = current_price * (1.0 + stop_pct)
            
            if take_profits:
                base_tp = np.mean(take_profits)
                if signal_type == SignalType.BUY:
                    adjusted_tp = current_price + (base_tp - current_price) * tp_multiplier
                else:
                    adjusted_tp = current_price - (current_price - base_tp) * tp_multiplier
            else:
                # Default take profit
                tp_pct = 0.02 * tp_multiplier
                if signal_type == SignalType.BUY:
                    adjusted_tp = current_price * (1.0 + tp_pct)
                else:
                    adjusted_tp = current_price * (1.0 - tp_pct)
            
            return adjusted_stop, adjusted_tp
            
        except Exception as e:
            self.logger.error(f"Risk parameter calculation failed: {e}")
            # Return default values
            if signal_type == SignalType.BUY:
                return current_price * 0.99, current_price * 1.02
            else:
                return current_price * 1.01, current_price * 0.98
    
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
            self.logger.error(f"Price extraction failed: {e}")
            return None
    
    def _store_regime_record(self):
        """Store current regime for history tracking"""
        try:
            regime_record = {
                'timestamp': datetime.now(),
                'regime': self.current_regime,
                'confidence': self.regime_confidence
            }
            
            self.regime_history.append(regime_record)
            
            # Keep only recent history
            if len(self.regime_history) > self.max_regime_history:
                self.regime_history.pop(0)
            
        except Exception as e:
            self.logger.error(f"Regime record storage failed: {e}")
    
    def _store_signal_record(self, signal: Signal, constituent_signals: List[Signal]):
        """Store signal record for performance tracking"""
        try:
            signal_record = {
                'timestamp': signal.timestamp,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'regime': self.current_regime,
                'regime_confidence': self.regime_confidence,
                'constituent_count': len(constituent_signals),
                'strategy_names': [s.strategy_name for s in constituent_signals]
            }
            
            self.signal_history.append(signal_record)
            
            # Keep only recent history
            if len(self.signal_history) > self.max_history:
                self.signal_history.pop(0)
            
        except Exception as e:
            self.logger.error(f"Signal record storage failed: {e}")
    
    def update_regime_performance(self, signal: Signal, actual_outcome: float):
        """Update regime-specific strategy performance"""
        try:
            if 'signal_details' not in signal.metadata:
                return
            
            signal_regime = signal.metadata.get('current_regime', 'unknown')
            
            # Convert string back to enum
            try:
                regime = MarketRegime(signal_regime)
            except ValueError:
                regime = MarketRegime.UNKNOWN
            
            # Update performance for each contributing strategy
            for detail in signal.metadata['signal_details']:
                strategy_name = detail['strategy']
                
                # Determine if prediction was correct
                predicted_direction = 1 if detail['signal_type'] == 'BUY' else -1
                actual_direction = 1 if actual_outcome > 0 else -1
                
                was_correct = predicted_direction == actual_direction
                
                # Update regime-specific performance
                self.regime_performance[regime][strategy_name].append(1.0 if was_correct else 0.0)
                
                # Keep only recent performance
                if len(self.regime_performance[regime][strategy_name]) > 50:
                    self.regime_performance[regime][strategy_name].pop(0)
                
                # Update regime-specific weights
                self._update_regime_weight(regime, strategy_name, was_correct)
            
            self.logger.info(f"Regime performance updated for {regime.value}")
            
        except Exception as e:
            self.logger.error(f"Regime performance update failed: {e}")
    
    def _update_regime_weight(self, regime: MarketRegime, strategy_name: str, was_correct: bool):
        """Update regime-specific strategy weight"""
        try:
            current_weight = self.regime_weights[regime][strategy_name]
            
            if was_correct:
                new_weight = min(2.0, current_weight * 1.05)  # Gradual increase
            else:
                new_weight = max(0.2, current_weight * 0.95)  # Gradual decrease
            
            self.regime_weights[regime][strategy_name] = new_weight
            
        except Exception as e:
            self.logger.error(f"Regime weight update failed: {e}")
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime detection and performance statistics"""
        try:
            stats = {
                'current_regime': self.current_regime.value,
                'regime_confidence': self.regime_confidence,
                'regime_history_length': len(self.regime_history)
            }
            
            # Regime distribution
            if self.regime_history:
                regime_counts = defaultdict(int)
                for record in self.regime_history:
                    regime_counts[record['regime'].value] += 1
                
                stats['regime_distribution'] = dict(regime_counts)
            
            # Strategy performance by regime
            regime_performance_summary = {}
            for regime, strategies in self.regime_performance.items():
                regime_performance_summary[regime.value] = {}
                for strategy, performance in strategies.items():
                    if performance:
                        regime_performance_summary[regime.value][strategy] = {
                            'accuracy': np.mean(performance),
                            'count': len(performance),
                            'weight': self.regime_weights[regime][strategy]
                        }
            
            stats['regime_performance'] = regime_performance_summary
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Regime statistics calculation failed: {e}")
            return {}
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze regime detection strategy performance"""
        try:
            regime_stats = self.get_regime_statistics()
            
            analysis = {
                'strategy': self.strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'regime_statistics': regime_stats,
                'detection_parameters': {
                    'lookback_period': self.lookback_period,
                    'trend_threshold': self.trend_threshold,
                    'volatility_threshold': self.volatility_threshold,
                    'breakout_threshold': self.breakout_threshold
                },
                'total_signals_processed': len(self.signal_history)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                'strategy': self.strategy_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
