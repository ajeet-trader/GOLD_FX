
"""
Momentum Divergence Strategy - Advanced Divergence Detection
===========================================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-08-15

Advanced momentum divergence strategy for XAUUSD trading:
- Detects regular and hidden divergences using RSI or MACD
- Multi-timeframe analysis for intraday and short-term signals
- Supports both trend reversal and continuation setups
- Configurable oscillator and divergence parameters
- Designed to generate 5–10 valid signals daily

Features:
- Regular bullish/bearish divergence detection
- Hidden bullish/bearish divergence detection
- RSI and MACD oscillator support
- Multi-timeframe confirmation
- Dynamic signal validation
- Cooldown mechanism to prevent signal spam

Dependencies:
    - pandas
    - numpy
    - datetime
    - typing
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade

# Import CLI args for mode selection
try:
    from src.utils.cli_args import parse_mode, print_mode_banner
except Exception:
    def parse_mode(*_args, **_kwargs): # type: ignore
        return 'mock'
    def print_mode_banner(_mode): # type: ignore
        pass


class DivergenceType(Enum):
    """Divergence type enumeration"""
    REGULAR_BULLISH = "regular_bullish"
    REGULAR_BEARISH = "regular_bearish"
    HIDDEN_BULLISH = "hidden_bullish"
    HIDDEN_BEARISH = "hidden_bearish"


@dataclass
class DivergenceInfo:
    """Divergence data structure"""
    div_type: DivergenceType
    price_point: float
    osc_value: float
    index: int
    confidence: float
    timeframe: str
    momentum_slope: float


class MomentumDivergenceStrategy(AbstractStrategy):
    """
    Advanced Momentum Divergence Strategy
    
    Detects divergences between price action and oscillators (RSI or MACD) that may
    signal trend reversals or continuations. Supports both regular and hidden divergences
    for bullish and bearish setups.
    
    Signal Generation:
    - Regular Bullish: Price lower low, oscillator higher low
    - Regular Bearish: Price higher high, oscillator lower high
    - Hidden Bullish: Price higher low, oscillator lower low
    - Hidden Bearish: Price lower high, oscillator higher high
    
    Example:
        >>> strategy = MomentumDivergenceStrategy(config, mt5_manager)
        >>> signals = strategy.generate_signal("XAUUSDm", "M15")
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """
        Initialize Momentum Divergence strategy
        
        Args:
            config: Strategy configuration
            mt5_manager: MT5 connection manager (optional)
            database: Database manager (optional)
        """
        super().__init__(config, mt5_manager, database)
        
        # Determine mode (CLI overrides config)
        cfg_mode = (self.config.get('parameters', {}) or {}).get('mode') or 'mock'
        self.mode = parse_mode() or cfg_mode
        print_mode_banner(self.mode)
        
        # Create appropriate MT5 manager based on mode
        if self.mode == 'live' and mt5_manager is None:
            try:
                from src.core.mt5_manager import MT5Manager
                live_mgr = MT5Manager()
                if hasattr(live_mgr, 'connect') and live_mgr.connect():
                    self.mt5_manager = live_mgr
                    print("✅ Connected to live MT5")
                else:
                    print("⚠️  Failed to connect to live MT5, falling back to mock data")
                    self.mt5_manager = self._create_mock_mt5()
                    self.mode = 'mock'
            except ImportError:
                print("⚠️  MT5Manager not available, using mock data")
                self.mt5_manager = self._create_mock_mt5()
                self.mode = 'mock'
        elif self.mode == 'mock' or mt5_manager is None:
            self.mt5_manager = self._create_mock_mt5()
        else:
            self.mt5_manager = mt5_manager

        # Strategy parameters
        self.lookback_period = config.get('parameters', {}).get('lookback_period', 200)
        self.oscillator = config.get('parameters', {}).get('oscillator', 'RSI').upper()
        self.rsi_period = config.get('parameters', {}).get('rsi_period', 14)
        self.macd_fast = config.get('parameters', {}).get('macd_fast', 12)
        self.macd_slow = config.get('parameters', {}).get('macd_slow', 26)
        self.macd_signal = config.get('parameters', {}).get('macd_signal', 9)
        self.divergence_tolerance = config.get('parameters', {}).get('divergence_tolerance', 2)
        self.confidence_threshold = config.get('parameters', {}).get('confidence_threshold', 0.65)
        self.cooldown_bars = config.get('parameters', {}).get('cooldown_bars', 3)
        
        # Signal tracking
        self.last_signal_time = None
        self.signal_cooldown = timedelta(minutes=15 * self.cooldown_bars)
        
        # Logger (inherited from parent)
        self.logger.info(f"Initialized MomentumDivergenceStrategy with oscillator: {self.oscillator}")
    
    def _create_mock_mt5(self):
        """Create mock MT5 manager with mode-specific data"""
        class MockMT5Manager:
            def __init__(self, mode):
                self.mode = mode
                
            def get_historical_data(self, symbol, timeframe, bars):
                # Generate sample data with synthetic divergences and different seeds
                dates = pd.date_range(start=datetime.now() - timedelta(days=10), 
                                     end=datetime.now(), freq='15Min')[:bars]
                
                np.random.seed(42 if self.mode == 'mock' else 123)
                base_price = 1950 if self.mode == 'mock' else 1975
                prices = []
                for i in range(len(dates)):
                    if i % 50 < 10:  # Create lower lows for bullish divergence
                        base_price -= 2
                    elif i % 50 < 20:  # Create higher lows for hidden bullish
                        base_price += 1
                    elif i % 50 < 30:  # Create higher highs for bearish divergence
                        base_price += 2
                    elif i % 50 < 40:  # Create lower highs for hidden bearish
                        base_price -= 1
                    else:
                        base_price += np.random.randn() * 0.5
                    prices.append(base_price)
                
                data = pd.DataFrame({
                    'Open': np.array(prices) + np.random.randn(len(dates)) * 0.5,
                    'High': np.array(prices) + np.abs(np.random.randn(len(dates)) * 3),
                    'Low': np.array(prices) - np.abs(np.random.randn(len(dates)) * 3),
                    'Close': prices,
                    'Volume': np.random.randint(100, 1000, len(dates))
                }, index=dates)
                
                return data
        
        return MockMT5Manager(self.mode)

    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            self.logger.error(f"RSI calculation failed: {str(e)}")
            return pd.Series()
    
    def _calculate_macd(self, data: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.Series:
        """Calculate MACD histogram"""
        try:
            ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
            ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            macd_hist = macd_line - signal_line
            return macd_hist
        except Exception as e:
            self.logger.error(f"MACD calculation failed: {str(e)}")
            return pd.Series()
    
    def _find_pivot_points(self, data: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
        """Find swing highs and lows in price data"""
        highs = []
        lows = []
        for i in range(window, len(data) - window):
            if (data['High'].iloc[i] >= data['High'].iloc[i-window:i].max() and
                data['High'].iloc[i] >= data['High'].iloc[i+1:i+window+1].max()):
                highs.append(i)
            if (data['Low'].iloc[i] <= data['Low'].iloc[i-window:i].min() and
                data['Low'].iloc[i] <= data['Low'].iloc[i+1:i+window+1].min()):
                lows.append(i)
        return highs, lows
    
    def _detect_divergences(self, data: pd.DataFrame, oscillator: pd.Series, 
                           highs: List[int], lows: List[int]) -> List[DivergenceInfo]:
        """Detect regular and hidden divergences"""
        divergences = []
        try:
            # Calculate momentum slope (price and oscillator)
            price_slope = data['Close'].diff(5).iloc[-1] / 5
            osc_slope = oscillator.diff(5).iloc[-1] / 5 if not oscillator.empty else 0
            
            # Regular and hidden divergences for lows
            for i in range(1, len(lows)):
                curr_idx, prev_idx = lows[i], lows[i-1]
                if curr_idx - prev_idx > self.divergence_tolerance:
                    continue
                
                curr_price, prev_price = data['Low'].iloc[curr_idx], data['Low'].iloc[prev_idx]
                curr_osc, prev_osc = oscillator.iloc[curr_idx], oscillator.iloc[prev_idx]
                
                # Regular Bullish: Price lower low, oscillator higher low
                if curr_price < prev_price and curr_osc > prev_osc:
                    confidence = self._calculate_divergence_confidence(data, curr_idx, 'bullish')
                    divergences.append(DivergenceInfo(
                        div_type=DivergenceType.REGULAR_BULLISH,
                        price_point=curr_price,
                        osc_value=curr_osc,
                        index=curr_idx,
                        confidence=confidence,
                        timeframe=data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                        momentum_slope=osc_slope
                    ))
                
                # Hidden Bullish: Price higher low, oscillator lower low
                elif curr_price > prev_price and curr_osc < prev_osc:
                    confidence = self._calculate_divergence_confidence(data, curr_idx, 'bullish')
                    divergences.append(DivergenceInfo(
                        div_type=DivergenceType.HIDDEN_BULLISH,
                        price_point=curr_price,
                        osc_value=curr_osc,
                        index=curr_idx,
                        confidence=confidence,
                        timeframe=data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                        momentum_slope=osc_slope
                    ))
            
            # Regular and hidden divergences for highs
            for i in range(1, len(highs)):
                curr_idx, prev_idx = highs[i], highs[i-1]
                if curr_idx - prev_idx > self.divergence_tolerance:
                    continue
                
                curr_price, prev_price = data['High'].iloc[curr_idx], data['High'].iloc[prev_idx]
                curr_osc, prev_osc = oscillator.iloc[curr_idx], oscillator.iloc[prev_idx]
                
                # Regular Bearish: Price higher high, oscillator lower high
                if curr_price > prev_price and curr_osc < prev_osc:
                    confidence = self._calculate_divergence_confidence(data, curr_idx, 'bearish')
                    divergences.append(DivergenceInfo(
                        div_type=DivergenceType.REGULAR_BEARISH,
                        price_point=curr_price,
                        osc_value=curr_osc,
                        index=curr_idx,
                        confidence=confidence,
                        timeframe=data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                        momentum_slope=osc_slope
                    ))
                
                # Hidden Bearish: Price lower high, oscillator higher high
                elif curr_price < prev_price and curr_osc > prev_osc:
                    confidence = self._calculate_divergence_confidence(data, curr_idx, 'bearish')
                    divergences.append(DivergenceInfo(
                        div_type=DivergenceType.HIDDEN_BEARISH,
                        price_point=curr_price,
                        osc_value=curr_osc,
                        index=curr_idx,
                        confidence=confidence,
                        timeframe=data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                        momentum_slope=osc_slope
                    ))
        
        except Exception as e:
            self.logger.error(f"Divergence detection failed: {str(e)}")
        
        return divergences
    
    def _calculate_divergence_confidence(self, data: pd.DataFrame, idx: int, direction: str) -> float:
        """Calculate confidence score for a divergence"""
        try:
            confidence = self.confidence_threshold
            # Adjust confidence based on momentum and trend
            price_slope = data['Close'].diff(5).iloc[idx] / 5 if idx < len(data) else 0
            if direction == 'bullish' and price_slope > 0:
                confidence += 0.1  # Momentum confirmation
            elif direction == 'bearish' and price_slope < 0:
                confidence += 0.1
            return min(confidence, 0.95)
        except:
            return self.confidence_threshold
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """
        Generate divergence-based trading signals
        
        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            List of trading signals
        """
        signals = []
        try:
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available")
                return []
            
            # Check cooldown
            current_time = datetime.now()
            if (self.last_signal_time and 
                current_time - self.last_signal_time < self.signal_cooldown):
                self.logger.debug("Signal generation skipped due to cooldown")
                return []
            
            # Fetch market data
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_period)
            if data is None or len(data) < self.rsi_period + self.divergence_tolerance:
                self.logger.warning(f"Insufficient data: {len(data) if data is not None else 0} bars")
                return []
            
            # Calculate oscillator
            oscillator = pd.Series()
            if self.oscillator == 'RSI':
                oscillator = self._calculate_rsi(data, self.rsi_period)
            elif self.oscillator == 'MACD':
                oscillator = self._calculate_macd(data, self.macd_fast, self.macd_slow, self.macd_signal)
            
            if oscillator.empty:
                self.logger.warning(f"No valid {self.oscillator} data")
                return []
            
            # Find pivot points
            highs, lows = self._find_pivot_points(data)
            if len(highs) < 2 and len(lows) < 2:
                self.logger.debug("Insufficient pivot points for divergence detection")
                return []
            
            # Detect divergences
            divergences = self._detect_divergences(data, oscillator, highs, lows)
            
            # Generate signals from divergences
            for div in divergences:
                signal_type = SignalType.BUY if 'bullish' in div.div_type.value else SignalType.SELL
                current_price = data['Close'].iloc[-1]
                
                # Calculate stop loss and take profit
                atr = (data['High'] - data['Low']).rolling(window=14).mean().iloc[-1]
                stop_loss = current_price - atr * 1.5 if signal_type == SignalType.BUY else current_price + atr * 1.5
                take_profit = current_price + atr * 3 if signal_type == SignalType.BUY else current_price - atr * 3
                
                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=signal_type,
                    confidence=div.confidence,
                    price=current_price,
                    timeframe=timeframe,
                    rsi=oscillator.iloc[-1] if self.oscillator == 'RSI' else None,
                    macd=oscillator.iloc[-1] if self.oscillator == 'MACD' else None,
                    atr=atr,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'divergence_type': div.div_type.value,
                        'oscillator_value': div.osc_value,
                        'momentum_slope': div.momentum_slope
                    }
                )
                
                if self.validate_signal(signal):
                    signals.append(signal)
                    self.last_signal_time = current_time
            
            self.logger.info(f"Generated {len(signals)} signals for {symbol} on {timeframe}")
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            return []
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze market data for divergences and return analysis results
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            Dictionary with analysis results
        """
        try:
            analysis = {
                'oscillator': self.oscillator,
                'divergences_detected': [],
                'recent_rsi_values': None,
                'recent_macd_hist': None,
                'trend_context': 'sideways'
            }
            
            if data is None or len(data) < self.rsi_period + self.divergence_tolerance:
                self.logger.warning(f"Insufficient data for analysis: {len(data) if data is not None else 0} bars")
                return analysis
            
            # Calculate oscillator
            oscillator = pd.Series()
            if self.oscillator == 'RSI':
                oscillator = self._calculate_rsi(data, self.rsi_period)
                analysis['recent_rsi_values'] = oscillator[-10:].tolist()
            elif self.oscillator == 'MACD':
                oscillator = self._calculate_macd(data, self.macd_fast, self.macd_slow, self.macd_signal)
                analysis['recent_macd_hist'] = oscillator[-10:].tolist()
            
            # Determine trend context
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
            analysis['trend_context'] = 'up' if sma_20 > sma_50 else 'down' if sma_20 < sma_50 else 'sideways'
            
            # Find pivot points and divergences
            highs, lows = self._find_pivot_points(data)
            divergences = self._detect_divergences(data, oscillator, highs, lows)
            
            analysis['divergences_detected'] = [
                {
                    'type': div.div_type.value,
                    'price_point': div.price_point,
                    'osc_value': div.osc_value
                } for div in divergences
            ]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return analysis
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and configuration
        
        Returns:
            Dictionary with strategy details
        """
        return {
            'name': self.strategy_name,
            'type': 'Technical',
            'description': 'Detects regular and hidden divergences using RSI or MACD for trend reversal and continuation setups',
            'parameters': {
                'lookback_period': self.lookback_period,
                'oscillator': self.oscillator,
                'rsi_period': self.rsi_period,
                'macd_fast': self.macd_fast,
                'macd_slow': self.macd_slow,
                'macd_signal': self.macd_signal,
                'divergence_tolerance': self.divergence_tolerance,
                'confidence_threshold': self.confidence_threshold,
                'cooldown_bars': self.cooldown_bars
            },
            'performance': {
                'success_rate': self.performance.win_rate,
                'profit_factor': self.performance.profit_factor
            }
        }
    
    def _create_empty_performance_metrics(self):
        """Create empty performance metrics for direct script runs"""
        from src.core.base import StrategyPerformance
        return StrategyPerformance(
            strategy_name=self.strategy_name,
            win_rate=0.0,
            profit_factor=0.0
        )


if __name__ == "__main__":
    """Test the Momentum Divergence strategy"""
    
    # Test configuration
    test_config = {
        'parameters': {
            'lookback_period': 200,
            'oscillator': 'RSI',
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'divergence_tolerance': 2,
            'confidence_threshold': 0.65,
            'cooldown_bars': 3,
            'mode': 'mock' # Added mode parameter
        }
    }
    
    # Create strategy instance
    strategy = MomentumDivergenceStrategy(test_config, mt5_manager=None) # Pass mt5_manager=None to trigger internal mock creation
    
    print("============================================================")
    print("TESTING MOMENTUM DIVERGENCE STRATEGY")
    print("============================================================")
    print(f"Running in {strategy.mode.upper()} mode") # Print mode
    
    # Test signal generation
    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
        print(f"     Divergence: {signal.metadata.get('divergence_type', 'Unknown')}")
        print(f"     Oscillator Value: {signal.metadata.get('oscillator_value', 0):.2f}")
    
    # Test analysis
    print("\n2. Testing analysis method:")
    mock_data = strategy.mt5_manager.get_historical_data("XAUUSDm", "M15", 200)
    analysis_results = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis results keys: {analysis_results.keys()}")
    print(f"   Detected divergences: {len(analysis_results['divergences_detected'])}")
    for div in analysis_results['divergences_detected'][:3]:  # Show first 3
        print(f"     - {div['type']}: Price {div['price_point']:.2f}, "
              f"Oscillator {div['osc_value']:.2f}")
    
    # Test performance summary
    print("\n3. Testing performance tracking:")
    summary = strategy.get_performance_summary()
    print(f"   {summary}")
    
    # Test strategy info
    print("\n4. Strategy Information:")
    strategy_info = strategy.get_strategy_info()
    print(f"   Name: {strategy_info['name']}")
    print(f"   Type: {strategy_info['type']}")
    print(f"   Description: {strategy_info['description']}")
    print(f"   Parameters:")
    for param, value in strategy_info['parameters'].items():
        print(f"     {param}: {value}")
    print(f"   Performance Summary:")
    print(f"     Success Rate: {strategy_info['performance']['success_rate']:.2%}")
    print(f"     Profit Factor: {strategy_info['performance']['profit_factor']:.2f}")
    
    print("\n============================================================")
    print("MOMENTUM DIVERGENCE STRATEGY TEST COMPLETED!")
    print("============================================================")
