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

# Add src to path for module resolution when run as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging
from collections import defaultdict

# Import base classes directly
from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade, StrategyPerformance


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
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """Initialize regime detection fusion strategy - 8GB RAM optimized"""
        super().__init__(config, mt5_manager, database)
        
        # self.strategy_name is already set by AbstractStrategy to self.__class__.__name__
        self.min_signals = self.config.get('parameters', {}).get('min_signals', 2)
        self.min_confidence = self.config.get('parameters', {}).get('min_confidence', 0.55)
        
        # Regime detection parameters - 8GB RAM optimized
        self.lookback_period = self.config.get('parameters', {}).get('lookback_period', 30)
        self.trend_threshold = self.config.get('parameters', {}).get('trend_threshold', 0.015)
        self.volatility_window = self.config.get('parameters', {}).get('volatility_window', 15)
        self.volatility_threshold = self.config.get('parameters', {}).get('volatility_threshold', 0.012)
        self.breakout_threshold = self.config.get('parameters', {}).get('breakout_threshold', 1.5)
        
        # Regime-specific strategy weights
        self.regime_weights = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        
        # Current market state
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        self.regime_history = []
        self.max_regime_history = self.config.get('parameters', {}).get('max_regime_history', 100)
        
        # Performance tracking (for component signals if fusion is used, not main signal_history from AbstractStrategy)
        self.component_signal_history = [] # Renamed to avoid conflict with AbstractStrategy.signal_history
        self.max_history = self.config.get('parameters', {}).get('max_history', 400)
        
        # Memory optimization settings
        self.memory_cleanup_interval = self.config.get('parameters', {}).get('memory_cleanup_interval', 15)
        self.prediction_count = 0
        
        # self.logger is provided by AbstractStrategy
        self.logger.info(f"{self.strategy_name} initialized")
    
    # Keeping fuse_signals as a helper, but the primary entry for SignalEngine is generate_signal
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
            
            # Store signal for performance tracking (for constituent signals if applicable)
            self._store_signal_record(combined_signal, filtered_signals)
            
            self.logger.info(f"Fused signal in {self.current_regime.value} regime: {combined_signal.signal_type.value} with confidence {combined_signal.confidence:.3f}")
            
            return combined_signal
            
        except Exception as e:
            self.logger.error(f"Regime-based signal fusion failed: {e}", exc_info=True)
            return None
    
    # This is the main generate_signal method required by AbstractStrategy
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """Generate regime-adaptive signals - Primary entry point for SignalEngine"""
        signals = []
        try:
            self.logger.info(f"Regime Detection - Analyzing {symbol} on {timeframe}")
            
            # Get market data
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available for signal generation.")
                return []
            
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_period * 3)
            if data is None or len(data) < self.lookback_period:
                self.logger.warning(f"Insufficient data for regime analysis: {len(data) if data is not None else 0}")
                return []
            
            # Memory cleanup
            self.prediction_count += 1
            if self.prediction_count % self.memory_cleanup_interval == 0:
                self._cleanup_memory()
            
            # Detect current regime
            self.current_regime, self.regime_confidence = self._detect_regime(data)
            self._store_regime_record()
            
            # Generate regime-specific signals
            raw_signals = self._generate_regime_signals(data, symbol, timeframe)
            
            # Validate and return signals using base class validation
            validated_signals = []
            for signal in raw_signals:
                if self.validate_signal(signal):
                    validated_signals.append(signal)
            
            if validated_signals:
                avg_confidence = np.mean([s.confidence for s in validated_signals])
                self.logger.info(f"Generated {len(validated_signals)} signals (avg confidence: {avg_confidence:.2f}) in {self.current_regime.value} regime")
            else:
                self.logger.info(f"No valid signals generated in {self.current_regime.value} regime")
            
            return validated_signals
            
        except Exception as e:
            self.logger.error(f"Regime-based signal generation failed: {e}", exc_info=True)
            return []
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze regime detection strategy performance and state."""
        try:
            if data is None or data.empty:
                return {'status': 'No data provided for analysis'}

            # Detect current regime without generating signals
            current_regime_type, current_regime_confidence = self._detect_regime(data)
            
            # Get other regime statistics
            regime_stats = self.get_regime_statistics()
            
            analysis = {
                'strategy': self.strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'current_regime_detected': current_regime_type.value,
                'current_regime_confidence': current_regime_confidence,
                'regime_statistics_history': regime_stats, # Includes weights and performance
                'detection_parameters': {
                    'lookback_period': self.lookback_period,
                    'trend_threshold': self.trend_threshold,
                    'volatility_threshold': self.volatility_threshold,
                    'breakout_threshold': self.breakout_threshold
                },
                'total_component_signals_processed': len(self.component_signal_history),
                'memory_optimized': True,
                'max_regime_history': self.max_regime_history,
                'max_history_component_signals': self.max_history
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            return {
                'strategy': self.strategy_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _detect_regime(self, data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Detect current market regime from price data"""
        try:
            if len(data) < self.lookback_period:
                return MarketRegime.UNKNOWN, 0.5
            
            close = data['Close'].tail(self.lookback_period)
            high = data['High'].tail(self.lookback_period)
            low = data['Low'].tail(self.lookback_period)
            
            trend_strength = self._calculate_trend_strength(close)
            volatility = self._calculate_volatility(close)
            breakout_score = self._calculate_breakout_score(high, low, close)
            
            regime, confidence = self._classify_regime(trend_strength, volatility, breakout_score)
            
            return regime, confidence
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}", exc_info=True)
            return MarketRegime.UNKNOWN, 0.5
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength (-1 to 1, negative = downtrend, positive = uptrend)"""
        try:
            if len(prices) < 2: return 0.0 # Handle too few data points
            x = np.arange(len(prices))
            slope, intercept = np.polyfit(x, prices.values, 1) # Get slope and intercept
            
            normalized_slope = slope / prices.mean()
            
            # R-squared for trend strength
            y_pred = slope * x + intercept # Use intercept from polyfit
            ss_res = np.sum((prices.values - y_pred) ** 2)
            ss_tot = np.sum((prices.values - np.mean(prices.values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            trend_strength = normalized_slope * r_squared
            
            return max(-1.0, min(1.0, trend_strength * 100)) # Clamped to [-1, 1]
            
        except Exception as e:
            self.logger.error(f"Trend strength calculation failed: {e}", exc_info=True)
            return 0.0
    
    def _calculate_volatility(self, prices: pd.Series) -> float:
        """Calculate normalized volatility"""
        try:
            if len(prices) < 2: return 0.0 # Handle too few data points
            returns = prices.pct_change().dropna()
            if returns.empty: return 0.0 # Handle no returns after dropna
            volatility = returns.std()
            
            annualized_vol = volatility * np.sqrt(252)
            
            return annualized_vol
            
        except Exception as e:
            self.logger.error(f"Volatility calculation failed: {e}", exc_info=True)
            return 0.02
    
    def _calculate_breakout_score(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate breakout potential score"""
        try:
            if len(close) < 20: return 0.0 # Need at least 20 for Bollinger Bands
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            current_close = close.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            if pd.isna(current_upper) or pd.isna(current_lower) or (current_upper - current_lower) == 0:
                return 0.0
            
            if current_close > current_upper:
                breakout_score = (current_close - current_upper) / (current_upper - current_lower)
            elif current_close < current_lower:
                breakout_score = (current_lower - current_close) / (current_upper - current_lower)
            else:
                band_width = (current_upper - current_lower) / sma.iloc[-1]
                avg_band_width = ((upper_band - lower_band) / sma).tail(20).mean()
                
                if not pd.isna(avg_band_width) and avg_band_width > 0:
                    compression = 1.0 - (band_width / avg_band_width)
                    breakout_score = max(0, compression)
                else:
                    breakout_score = 0.0
            
            return min(2.0, max(0.0, breakout_score))
            
        except Exception as e:
            self.logger.error(f"Breakout score calculation failed: {e}", exc_info=True)
            return 0.0
    
    def _classify_regime(self, trend_strength: float, volatility: float, breakout_score: float) -> Tuple[MarketRegime, float]:
        """Classify market regime based on calculated metrics"""
        try:
            regime_scores = {
                MarketRegime.TRENDING_UP: 0.0,
                MarketRegime.TRENDING_DOWN: 0.0,
                MarketRegime.SIDEWAYS: 0.0,
                MarketRegime.HIGH_VOLATILITY: 0.0,
                MarketRegime.LOW_VOLATILITY: 0.0,
                MarketRegime.BREAKOUT: 0.0
            }
            
            if trend_strength > self.trend_threshold:
                regime_scores[MarketRegime.TRENDING_UP] = abs(trend_strength)
            elif trend_strength < -self.trend_threshold:
                regime_scores[MarketRegime.TRENDING_DOWN] = abs(trend_strength)
            else:
                regime_scores[MarketRegime.SIDEWAYS] = 1.0 - abs(trend_strength) / self.trend_threshold
            
            if volatility > self.volatility_threshold:
                regime_scores[MarketRegime.HIGH_VOLATILITY] = min(1.0, volatility / self.volatility_threshold)
            else:
                regime_scores[MarketRegime.LOW_VOLATILITY] = 1.0 - (volatility / self.volatility_threshold)
            
            if breakout_score > self.breakout_threshold:
                regime_scores[MarketRegime.BREAKOUT] = min(1.0, breakout_score / self.breakout_threshold)
            
            dominant_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[dominant_regime]
            
            confidence = max(0.1, min(1.0, confidence))
            
            return dominant_regime, confidence
            
        except Exception as e:
            self.logger.error(f"Regime classification failed: {e}", exc_info=True)
            return MarketRegime.UNKNOWN, 0.5
    
    def _filter_signals_by_regime(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals based on regime-specific performance"""
        try:
            if self.current_regime == MarketRegime.UNKNOWN:
                return signals
            
            filtered_signals = []
            
            for signal in signals:
                strategy_name = signal.strategy_name
                
                regime_weight = self.regime_weights[self.current_regime][strategy_name]
                
                regime_performance = self.regime_performance[self.current_regime][strategy_name]
                if regime_performance:
                    avg_performance = np.mean(regime_performance)
                    if avg_performance > 0.4:
                        filtered_signals.append(signal)
                else:
                    filtered_signals.append(signal)
            
            return filtered_signals if filtered_signals else signals
            
        except Exception as e:
            self.logger.error(f"Signal filtering failed: {e}", exc_info=True)
            return signals
    
    def _combine_signals_by_regime(self, signals: List[Signal], data: pd.DataFrame = None, 
                                 symbol: str = "XAUUSDm", timeframe: str = "M15") -> Optional[Signal]:
        """Combine signals using regime-specific weights"""
        try:
            if not signals:
                return None
            
            buy_weight = 0.0
            sell_weight = 0.0
            total_weight = 0.0
            
            signal_details = []
            
            for signal in signals:
                strategy_name = signal.strategy_name
                regime_weight = self.regime_weights[self.current_regime][strategy_name]
                confidence = signal.confidence
                
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
            
            current_price = self._get_current_price(signals, data)
            if current_price is None:
                return None
            
            stop_loss, take_profit = self._calculate_regime_risk_parameters(
                signals, signal_type, current_price
            )
            
            combined_signal = Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                strategy_name=self.strategy_name, # Use self.strategy_name
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
            self.logger.error(f"Signal combination failed: {e}", exc_info=True)
            return None
    
    def _calculate_regime_risk_parameters(self, signals: List[Signal], signal_type: SignalType, 
                                        current_price: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate risk parameters adjusted for current regime"""
        try:
            stop_losses = [s.stop_loss for s in signals if s.stop_loss is not None]
            take_profits = [s.take_profit for s in signals if s.take_profit is not None]
            
            if self.current_regime == MarketRegime.HIGH_VOLATILITY:
                stop_multiplier = 1.5
                tp_multiplier = 1.3
            elif self.current_regime == MarketRegime.LOW_VOLATILITY:
                stop_multiplier = 0.8
                tp_multiplier = 0.9
            elif self.current_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                stop_multiplier = 1.0
                tp_multiplier = 1.4
            elif self.current_regime == MarketRegime.BREAKOUT:
                stop_multiplier = 1.8
                tp_multiplier = 1.6
            else:
                stop_multiplier = 1.0
                tp_multiplier = 1.0
            
            if stop_losses:
                base_stop = np.mean(stop_losses)
                if signal_type == SignalType.BUY:
                    adjusted_stop = current_price - (current_price - base_stop) * stop_multiplier
                else:
                    adjusted_stop = current_price + (base_stop - current_price) * stop_multiplier
            else:
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
                tp_pct = 0.02 * tp_multiplier
                if signal_type == SignalType.BUY:
                    adjusted_tp = current_price * (1.0 + tp_pct)
                else:
                    adjusted_tp = current_price * (1.0 - tp_pct)
            
            return adjusted_stop, adjusted_tp
            
        except Exception as e:
            self.logger.error(f"Risk parameter calculation failed: {e}", exc_info=True)
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
            self.logger.error(f"Price extraction failed: {e}", exc_info=True)
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
            
            if len(self.regime_history) > self.max_regime_history:
                self.regime_history.pop(0)
            
        except Exception as e:
            self.logger.error(f"Regime record storage failed: {e}", exc_info=True)
    
    def _store_signal_record(self, signal: Signal, constituent_signals: List[Signal]):
        """Store signal record for performance tracking (for constituent signals)"""
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
            
            self.component_signal_history.append(signal_record)
            
            if len(self.component_signal_history) > self.max_history:
                self.component_signal_history.pop(0)
            
        except Exception as e:
            self.logger.error(f"Signal record storage failed: {e}", exc_info=True)
    
    def update_regime_performance(self, signal: Signal, actual_outcome: float):
        """Update regime-specific strategy performance"""
        try:
            if 'signal_details' not in signal.metadata:
                return
            
            signal_regime = signal.metadata.get('current_regime', 'unknown')
            
            try:
                regime = MarketRegime(signal_regime)
            except ValueError:
                regime = MarketRegime.UNKNOWN
            
            for detail in signal.metadata['signal_details']:
                strategy_name = detail['strategy']
                
                predicted_direction = 1 if detail['signal_type'] == 'BUY' else -1
                actual_direction = 1 if actual_outcome > 0 else -1
                
                was_correct = predicted_direction == actual_direction
                
                self.regime_performance[regime][strategy_name].append(1.0 if was_correct else 0.0)
                
                if len(self.regime_performance[regime][strategy_name]) > 50:
                    self.regime_performance[regime][strategy_name].pop(0)
                
                self._update_regime_weight(regime, strategy_name, was_correct)
            
            self.logger.info(f"Regime performance updated for {regime.value}")
            
        except Exception as e:
            self.logger.error(f"Regime performance update failed: {e}", exc_info=True)
    
    def _update_regime_weight(self, regime: MarketRegime, strategy_name: str, was_correct: bool):
        """Update regime-specific strategy weight"""
        try:
            current_weight = self.regime_weights[regime][strategy_name]
            
            if was_correct:
                new_weight = min(2.0, current_weight * 1.05)
            else:
                new_weight = max(0.2, current_weight * 0.95)
            
            self.regime_weights[regime][strategy_name] = new_weight
            
        except Exception as e:
            self.logger.error(f"Regime weight update failed: {e}", exc_info=True)
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime detection and performance statistics"""
        try:
            stats = {
                'current_regime': self.current_regime.value,
                'regime_confidence': self.regime_confidence,
                'regime_history_length': len(self.regime_history)
            }
            
            if self.regime_history:
                regime_counts = defaultdict(int)
                for record in self.regime_history:
                    regime_counts[record['regime'].value] += 1
                
                stats['regime_distribution'] = dict(regime_counts)
            
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
            
            stats['regime_performance_by_component_strategy'] = regime_performance_summary
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Regime statistics calculation failed: {e}", exc_info=True)
            return {}
    
    def _generate_regime_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Generate signals based on current market regime"""
        signals = []
        try:
            current_price = data['Close'].iloc[-1]
            atr = self._calculate_atr(data)
            
            if self.current_regime == MarketRegime.TRENDING_UP:
                signals = self._generate_trend_following_signals(data, symbol, timeframe, 'bullish')
            elif self.current_regime == MarketRegime.TRENDING_DOWN:
                signals = self._generate_trend_following_signals(data, symbol, timeframe, 'bearish')
            elif self.current_regime == MarketRegime.SIDEWAYS:
                signals = self._generate_mean_reversion_signals(data, symbol, timeframe)
            elif self.current_regime == MarketRegime.BREAKOUT:
                signals = self._generate_breakout_signals(data, symbol, timeframe)
            elif self.current_regime == MarketRegime.HIGH_VOLATILITY:
                signals = self._generate_volatility_signals(data, symbol, timeframe, 'high')
            elif self.current_regime == MarketRegime.LOW_VOLATILITY:
                signals = self._generate_volatility_signals(data, symbol, timeframe, 'low')
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Regime signal generation failed: {e}", exc_info=True)
            return []
    
    def _generate_trend_following_signals(self, data: pd.DataFrame, symbol: str, timeframe: str, direction: str) -> List[Signal]:
        """Generate trend following signals"""
        signals = []
        try:
            current_price = data['Close'].iloc[-1]
            atr = self._calculate_atr(data)
            
            for i in range(2): # Generate two signals for example
                if direction == 'bullish':
                    signal_type = SignalType.BUY
                    confidence = min(0.75 + (self.regime_confidence * 0.15), 0.90) - (i * 0.05)
                    stop_loss = current_price - (atr * (2.0 + i * 0.5))
                    take_profit = current_price + (atr * (3.5 + i * 0.5))
                else:
                    signal_type = SignalType.SELL
                    confidence = min(0.75 + (self.regime_confidence * 0.15), 0.90) - (i * 0.05)
                    stop_loss = current_price + (atr * (2.0 + i * 0.5))
                    take_profit = current_price - (atr * (3.5 + i * 0.5))
                
                if confidence >= self.min_confidence:
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
                            'signal_reason': f'regime_trend_following_{direction}_{i+1}',
                            'regime': self.current_regime.value,
                            'regime_confidence': self.regime_confidence
                        }
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Trend following signal generation failed: {e}", exc_info=True)
            return []
    
    def _generate_mean_reversion_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Generate mean reversion signals for sideways markets"""
        signals = []
        try:
            current_price = data['Close'].iloc[-1]
            close = data['Close']
            atr = self._calculate_atr(data)
            
            sma_20 = close.rolling(min(20, len(close))).mean().iloc[-1]
            std_20 = close.rolling(min(20, len(close))).std().iloc[-1]
            
            if pd.isna(sma_20) or pd.isna(std_20) or std_20 == 0:
                return signals
            
            upper_band = sma_20 + (2 * std_20)
            lower_band = sma_20 - (2 * std_20)
            
            if current_price > upper_band:
                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=SignalType.SELL,
                    confidence=min(0.70 + self.regime_confidence * 0.1, 0.85),
                    price=current_price,
                    timeframe=timeframe,
                    strength=0.7,
                    stop_loss=current_price + (atr * 1.5),
                    take_profit=sma_20,
                    metadata={
                        'signal_reason': 'regime_mean_reversion_sell',
                        'regime': self.current_regime.value,
                        'bb_position': 'above_upper'
                    }
                )
                signals.append(signal)
                
            elif current_price < lower_band:
                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=SignalType.BUY,
                    confidence=min(0.70 + self.regime_confidence * 0.1, 0.85),
                    price=current_price,
                    timeframe=timeframe,
                    strength=0.7,
                    stop_loss=current_price - (atr * 1.5),
                    take_profit=sma_20,
                    metadata={
                        'signal_reason': 'regime_mean_reversion_buy',
                        'regime': self.current_regime.value,
                        'bb_position': 'below_lower'
                    }
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Mean reversion signal generation failed: {e}", exc_info=True)
            return []
    
    def _generate_breakout_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Generate breakout signals"""
        signals = []
        try:
            current_price = data['Close'].iloc[-1]
            close = data['Close']
            atr = self._calculate_atr(data)
            
            recent_high = close.tail(10).max()
            recent_low = close.tail(10).min()
            
            if current_price > recent_high * 1.001:
                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=SignalType.BUY,
                    confidence=min(0.80 + self.regime_confidence * 0.1, 0.90),
                    price=current_price,
                    timeframe=timeframe,
                    strength=0.8,
                    stop_loss=recent_high - (atr * 0.5),
                    take_profit=current_price + (atr * 4),
                    metadata={
                        'signal_reason': 'regime_bullish_breakout',
                        'regime': self.current_regime.value,
                        'breakout_level': recent_high
                    }
                )
                signals.append(signal)
                
            elif current_price < recent_low * 0.999:
                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=SignalType.SELL,
                    confidence=min(0.80 + self.regime_confidence * 0.1, 0.90),
                    price=current_price,
                    timeframe=timeframe,
                    strength=0.8,
                    stop_loss=recent_low + (atr * 0.5),
                    take_profit=current_price - (atr * 4),
                    metadata={
                        'signal_reason': 'regime_bearish_breakout',
                        'regime': self.current_regime.value,
                        'breakout_level': recent_low
                    }
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Breakout signal generation failed: {e}", exc_info=True)
            return []
    
    def _generate_volatility_signals(self, data: pd.DataFrame, symbol: str, timeframe: str, vol_type: str) -> List[Signal]:
        """Generate volatility-based signals"""
        signals = []
        try:
            current_price = data['Close'].iloc[-1]
            close = data['Close']
            atr = self._calculate_atr(data)
            
            if len(close) < 5: return signals # Not enough data for momentum
            momentum = (current_price - close.iloc[-5]) / close.iloc[-5]
            
            if vol_type == 'high':
                if momentum > 0.003:
                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name=self.strategy_name,
                        signal_type=SignalType.BUY,
                        confidence=min(0.65 + self.regime_confidence * 0.1, 0.80),
                        price=current_price,
                        timeframe=timeframe,
                        strength=0.65,
                        stop_loss=current_price - (atr * 3),
                        take_profit=current_price + (atr * 5),
                        metadata={
                            'signal_reason': 'regime_high_vol_momentum_buy',
                            'regime': self.current_regime.value,
                            'momentum': momentum
                        }
                    )
                    signals.append(signal)
                elif momentum < -0.003:
                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name=self.strategy_name,
                        signal_type=SignalType.SELL,
                        confidence=min(0.65 + self.regime_confidence * 0.1, 0.80),
                        price=current_price,
                        timeframe=timeframe,
                        strength=0.65,
                        stop_loss=current_price + (atr * 3),
                        take_profit=current_price - (atr * 5),
                        metadata={
                            'signal_reason': 'regime_high_vol_momentum_sell',
                            'regime': self.current_regime.value,
                            'momentum': momentum
                        }
                    )
                    signals.append(signal)
            else:
                sma_10 = close.rolling(min(10, len(close))).mean().iloc[-1]
                if not pd.isna(sma_10):
                    if current_price > sma_10 * 1.002:
                        signal = Signal(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            strategy_name=self.strategy_name,
                            signal_type=SignalType.BUY,
                            confidence=min(0.60 + self.regime_confidence * 0.1, 0.75),
                            price=current_price,
                            timeframe=timeframe,
                            strength=0.6,
                            stop_loss=current_price - (atr * 1),
                            take_profit=current_price + (atr * 2),
                            metadata={
                                'signal_reason': 'regime_low_vol_range_buy',
                                'regime': self.current_regime.value,
                                'sma_position': 'above'
                            }
                        )
                        signals.append(signal)
                    elif current_price < sma_10 * 0.998:
                        signal = Signal(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            strategy_name=self.strategy_name,
                            signal_type=SignalType.SELL,
                            confidence=min(0.60 + self.regime_confidence * 0.1, 0.75),
                            price=current_price,
                            timeframe=timeframe,
                            strength=0.6,
                            stop_loss=current_price + (atr * 1),
                            take_profit=current_price - (atr * 2),
                            metadata={
                                'signal_reason': 'regime_low_vol_range_sell',
                                'regime': self.current_regime.value,
                                'sma_position': 'below'
                            }
                        )
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Volatility signal generation failed: {e}", exc_info=True)
            return []
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(data) < period + 1:
                return data['Close'].iloc[-1] * 0.01 # Fallback if not enough data
            
            tr1 = data['High'] - data['Low']
            tr2 = abs(data['High'] - data['Close'].shift())
            tr3 = abs(data['Low'] - data['Close'].shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).dropna() # dropna added
            
            if true_range.empty: # Handle case where true_range is empty after dropna
                return data['Close'].iloc[-1] * 0.01

            atr = true_range.rolling(min(period, len(true_range))).mean().iloc[-1] # Use min(period, len(true_range))
            
            return float(atr) if not pd.isna(atr) else data['Close'].iloc[-1] * 0.01
        except Exception as e:
            self.logger.error(f"ATR calculation failed: {e}", exc_info=True)
            return data['Close'].iloc[-1] * 0.01
    
    def _cleanup_memory(self):
        """Clean up memory for 8GB RAM optimization"""
        try:
            import gc
            
            if len(self.regime_history) > self.max_regime_history:
                self.regime_history = self.regime_history[-self.max_regime_history//2:]
            
            if len(self.component_signal_history) > self.max_history:
                self.component_signal_history = self.component_signal_history[-self.max_history//2:]
            
            for regime in list(self.regime_performance.keys()):
                for strategy in list(self.regime_performance[regime].keys()):
                    if len(self.regime_performance[regime][strategy]) > 50:
                        self.regime_performance[regime][strategy] = self.regime_performance[regime][strategy][-25:]
            
            gc.collect()
            
            self.logger.info(f"Memory cleanup completed (prediction #{self.prediction_count})")
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {str(e)}", exc_info=True)

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the Regime Detection Fusion Strategy,
        including its parameters, current regime state, and overall performance.
        """
        # Get overall trading performance from AbstractStrategy base class
        base_trading_performance = self.get_performance_summary()
        
        # Get detailed regime statistics
        regime_stats = self.get_regime_statistics()

        info = {
            'name': 'Regime Detection Fusion Strategy',
            'version': '2.0.0',
            'type': 'Fusion', # Categorized as Fusion
            'description': 'Detects market regimes and adapts signal fusion based on current market conditions.',
            'parameters': {
                'min_signals': self.min_signals,
                'min_confidence': self.min_confidence,
                'lookback_period': self.lookback_period,
                'trend_threshold': self.trend_threshold,
                'volatility_window': self.volatility_window,
                'volatility_threshold': self.volatility_threshold,
                'breakout_threshold': self.breakout_threshold,
                'max_regime_history': self.max_regime_history,
                'max_history_component_signals': self.max_history
            },
            'current_regime_state': {
                'regime_type': self.current_regime.value,
                'confidence': self.regime_confidence,
                'history_length': len(self.regime_history)
            },
            'regime_detection_statistics': regime_stats, # Includes component strategy performance by regime
            'overall_trading_performance': { # Performance of the signals generated by this strategy
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
            'min_signals': 2,
            'min_confidence': 0.55,
            'lookback_period': 30,
            'trend_threshold': 0.015,
            'volatility_window': 15,
            'volatility_threshold': 0.012,
            'breakout_threshold': 1.5,
            'max_regime_history': 100,
            'max_history': 400,
            'memory_cleanup_interval': 15
        }
    }
    
    # Mock MT5 manager for testing
    class MockMT5Manager:
        def get_historical_data(self, symbol, timeframe, bars):
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            dates = pd.date_range(start=datetime.now() - timedelta(days=5), 
                                 end=datetime.now(), freq='15Min')[:bars]
            
            # Generate sample OHLCV data with different regimes
            np.random.seed(42)
            
            # Create trending data
            trend = np.linspace(1950, 1980, len(dates))
            noise = np.cumsum(np.random.randn(len(dates)) * 1.5)
            close_prices = trend + noise
            
            data = pd.DataFrame({
                'Open': close_prices + np.random.randn(len(dates)) * 0.5,
                'High': close_prices + np.abs(np.random.randn(len(dates)) * 3),
                'Low': close_prices - np.abs(np.random.randn(len(dates)) * 3),
                'Close': close_prices,
                'Volume': np.random.randint(100, 1000, len(dates))
            }, index=dates)
            
            # Ensure High >= Close >= Low (already in original data, just for clarity)
            data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
            data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
            
            return data
    
    # Create strategy instance
    mock_mt5 = MockMT5Manager()
    strategy = RegimeDetection(test_config, mock_mt5, database=None)
    
    # Output header matching other strategy files
    print("============================================================")
    print("TESTING MODIFIED REGIME DETECTION STRATEGY")
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
            print(f"     Regime: {signal.metadata.get('regime', 'N/A')}")
    
    # 2. Testing analysis method
    print("\n2. Testing analysis method:")
    mock_data = mock_mt5.get_historical_data("XAUUSDm", "M15", 90)
    analysis_results = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis results keys: {analysis_results.keys()}")
    print(f"   Current Regime Detected: {analysis_results.get('current_regime_detected', 'N/A')}")
    print(f"   Regime History Length: {analysis_results.get('regime_detection_statistics', {}).get('regime_history_length', 'N/A')}")

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
    print(f"   Current Regime State: {info['current_regime_state']}")
    print(f"   Regime Detection Statistics: {info['regime_detection_statistics']}")
    print(f"   Overall Trading Performance:")
    print(f"     Total Signals Generated: {info['overall_trading_performance']['total_signals_generated']}")
    print(f"     Win Rate: {info['overall_trading_performance']['win_rate']:.2%}")
    print(f"     Profit Factor: {info['overall_trading_performance']['profit_factor']:.2f}")

    # Footer matching other strategy files
    print("\n============================================================")
    print("REGIME DETECTION STRATEGY TEST COMPLETED!")
    print("============================================================")