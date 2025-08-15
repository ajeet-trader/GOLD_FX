"""
Fibonacci Advanced Strategy - Advanced Technical Analysis
========================================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-08-15

Advanced Fibonacci-based strategy for XAUUSD trading:
- Multi-timeframe Fibonacci retracements and extensions
- Confluence cluster detection
- Dynamic swing point analysis
- Trend confirmation
- High-probability reversal and continuation signals

Features:
- Fibonacci retracement levels (0.236, 0.382, 0.5, 0.618, 0.786)
- Fibonacci extension levels (1.272, 1.618, 2.0)
- Cluster detection for high-probability zones
- Multi-timeframe confluence analysis
- Dynamic stop-loss and take-profit calculation

Dependencies:
    - pandas
    - numpy
    - datetime
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade


@dataclass
class FibonacciLevel:
    """Fibonacci level data structure"""
    price: float
    level: float  # Fibonacci ratio (e.g., 0.618, 1.272)
    type: str  # "retracement" or "extension"
    swing_ref: Tuple[float, float]  # Reference swing high/low
    timeframe: str
    confidence: float


class FibonacciAdvancedStrategy(AbstractStrategy):
    """
    Advanced Fibonacci Strategy
    
    Uses advanced Fibonacci tools (retracements, extensions, time projections, 
    and confluence clusters) to detect high-probability reversal and continuation zones.
    Generates signals when price approaches, reacts to, or breaks through these zones.
    
    Signal Generation:
    - BUY: Price retraces to 0.5–0.618 in uptrend with bullish reaction
    - SELL: Price retraces to 0.5–0.618 in downtrend with bearish reaction
    - BUY/SELL: Breakout through extension levels (1.272, 1.618)
    - Multi-timeframe confluence increases signal confidence
    
    Example:
        >>> strategy = FibonacciAdvancedStrategy(config, mt5_manager)
        >>> signals = strategy.generate_signal("XAUUSDm", "M15")
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """
        Initialize Fibonacci Advanced Strategy
        
        Args:
            config: Strategy configuration
            mt5_manager: MT5 connection manager (optional)
            database: Database manager (optional)
        """
        super().__init__(config, mt5_manager, database)
        
        # Fibonacci parameters
        self.lookback_period = self.config.get('parameters', {}).get('lookback_period', 200)
        self.fib_levels = self.config.get('parameters', {}).get('fib_levels', 
                            [0.236, 0.382, 0.5, 0.618, 0.786, 1.272, 1.618, 2.0])
        self.cluster_tolerance = self.config.get('parameters', {}).get('cluster_tolerance', 0.003)  # 0.3%
        self.confidence_threshold = self.config.get('parameters', {}).get('confidence_threshold', 0.65)
        self.multi_timeframe = self.config.get('parameters', {}).get('multi_timeframe', True)
        self.cooldown_bars = self.config.get('parameters', {}).get('cooldown_bars', 3)
        
        # Strategy parameters
        self.primary_timeframe = config.get('parameters', {}).get('timeframe_primary', 'M15')
        self.secondary_timeframe = config.get('parameters', {}).get('timeframe_secondary', 'H1')
        self.min_cluster_size = 3  # Minimum number of levels in a cluster
        self.max_signals_per_hour = 3  # Maximum signals per hour
        self.last_signal_time = None
        
        # Logger is set up by parent class
        self.logger.info(f"Initialized FibonacciAdvancedStrategy with lookback={self.lookback_period}, "
                        f"cluster_tolerance={self.cluster_tolerance*100:.1f}%")
    
    def _find_swing_points(self, data: pd.DataFrame, window: int = 5) -> List[Tuple[int, float, str]]:
        """
        Find swing highs and lows in price data
        
        Args:
            data: OHLCV DataFrame
            window: Lookback window for swing detection
            
        Returns:
            List of (index, price, type) tuples for swing points
        """
        swings = []
        highs = data['High'].rolling(window=window, center=True).max()
        lows = data['Low'].rolling(window=window, center=True).min()
        
        for i in range(window, len(data) - window):
            if data['High'].iloc[i] == highs.iloc[i]:
                swings.append((i, data['High'].iloc[i], 'high'))
            if data['Low'].iloc[i] == lows.iloc[i]:
                swings.append((i, data['Low'].iloc[i], 'low'))
        
        return sorted(swings, key=lambda x: x[0])
    
    def _calculate_fibonacci_levels(self, swing_high: float, swing_low: float, 
                                  timeframe: str) -> List[FibonacciLevel]:
        """
        Calculate Fibonacci retracement and extension levels
        
        Args:
            swing_high: Swing high price
            swing_low: Swing low price
            timeframe: Analysis timeframe
            
        Returns:
            List of FibonacciLevel objects
        """
        fib_range = swing_high - swing_low
        levels = []
        
        # Retracement levels
        for level in [l for l in self.fib_levels if l <= 1.0]:
            price = swing_high - (fib_range * level)
            levels.append(FibonacciLevel(
                price=price,
                level=level,
                type='retracement',
                swing_ref=(swing_high, swing_low),
                timeframe=timeframe,
                confidence=0.5
            ))
        
        # Extension levels
        for level in [l for l in self.fib_levels if l > 1.0]:
            price = swing_high + (fib_range * (level - 1.0))
            levels.append(FibonacciLevel(
                price=price,
                level=level,
                type='extension',
                swing_ref=(swing_high, swing_low),
                timeframe=timeframe,
                confidence=0.5
            ))
        
        return levels
    
    def _find_fibonacci_clusters(self, levels: List[FibonacciLevel], 
                               current_price: float) -> List[Tuple[float, float, int]]:
        """
        Find price zones where multiple Fibonacci levels converge
        
        Args:
            levels: List of FibonacciLevel objects
            current_price: Current market price
            
        Returns:
            List of (center_price, tolerance_range, level_count) tuples
        """
        clusters = []
        price_points = sorted([level.price for level in levels])
        
        for i, price in enumerate(price_points):
            nearby_levels = [p for p in price_points if 
                          abs(p - price) <= price * self.cluster_tolerance]
            if len(nearby_levels) >= self.min_cluster_size:
                center = sum(nearby_levels) / len(nearby_levels)
                tolerance = center * self.cluster_tolerance
                clusters.append((center, tolerance, len(nearby_levels)))
        
        return sorted(clusters, key=lambda x: abs(current_price - x[0]))
    
    def _get_trend_direction(self, data: pd.DataFrame) -> str:
        """
        Determine trend direction using moving averages
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Trend direction ("up", "down", "sideways")
        """
        try:
            short_ma = data['Close'].rolling(window=20).mean().iloc[-1]
            long_ma = data['Close'].rolling(window=50).mean().iloc[-1]
            
            if short_ma > long_ma * 1.005:
                return "up"
            elif short_ma < long_ma * 0.995:
                return "down"
            else:
                return "sideways"
        except:
            return "sideways"
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """
        Generate Fibonacci-based trading signals
        
        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            List of trading signals
        """
        try:
            # Check cooldown
            current_time = datetime.now()
            if self.last_signal_time and (
                current_time - self.last_signal_time < timedelta(minutes=60/self.max_signals_per_hour)):
                return []
            
            # Get market data
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available")
                return []
                
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_period)
            if data is None or len(data) < 50:
                self.logger.warning(f"Insufficient data: {len(data) if data is not None else 0} bars")
                return []
            
            # Get secondary timeframe data if multi-timeframe enabled
            secondary_data = None
            if self.multi_timeframe:
                secondary_data = self.mt5_manager.get_historical_data(
                    symbol, self.secondary_timeframe, self.lookback_period)
            
            # Find swing points
            swings = self._find_swing_points(data)
            if len(swings) < 4:
                self.logger.warning(f"Insufficient swing points: {len(swings)}")
                return []
            
            # Calculate Fibonacci levels
            fib_levels = []
            for i in range(len(swings) - 1):
                current_swing = swings[i]
                next_swing = swings[i + 1]
                if current_swing[2] == 'high' and next_swing[2] == 'low':
                    fib_levels.extend(self._calculate_fibonacci_levels(
                        current_swing[1], next_swing[1], timeframe))
                elif current_swing[2] == 'low' and next_swing[2] == 'high':
                    fib_levels.extend(self._calculate_fibonacci_levels(
                        next_swing[1], current_swing[1], timeframe))
            
            # Add secondary timeframe levels
            if secondary_data is not None and len(secondary_data) >= 50:
                secondary_swings = self._find_swing_points(secondary_data)
                for i in range(len(secondary_swings) - 1):
                    current_swing = secondary_swings[i]
                    next_swing = secondary_swings[i + 1]
                    if current_swing[2] == 'high' and next_swing[2] == 'low':
                        fib_levels.extend(self._calculate_fibonacci_levels(
                            current_swing[1], next_swing[1], self.secondary_timeframe))
                    elif current_swing[2] == 'low' and next_swing[2] == 'high':
                        fib_levels.extend(self._calculate_fibonacci_levels(
                            next_swing[1], current_swing[1], self.secondary_timeframe))
            
            # Find clusters
            current_price = data['Close'].iloc[-1]
            clusters = self._find_fibonacci_clusters(fib_levels, current_price)
            if not clusters:
                return []
            
            # Get trend direction
            trend = self._get_trend_direction(data)
            
            signals = []
            for cluster_center, tolerance, level_count in clusters[:2]:  # Check top 2 clusters
                # Calculate confidence based on cluster size and timeframe confluence
                base_confidence = min(0.9, self.confidence_threshold + (level_count - self.min_cluster_size) * 0.1)
                if secondary_data is not None:
                    base_confidence *= 1.2  # Boost for multi-timeframe confluence
                
                # Check for retracement signals
                if abs(current_price - cluster_center) <= tolerance:
                    if trend == "up" and 0.5 <= fib_levels[0].level <= 0.618:
                        signal_type = SignalType.BUY
                        confidence = base_confidence * 1.1
                    elif trend == "down" and 0.5 <= fib_levels[0].level <= 0.618:
                        signal_type = SignalType.SELL
                        confidence = base_confidence * 1.1
                    else:
                        continue
                
                # Check for extension breakout signals
                elif current_price > cluster_center and fib_levels[0].type == "extension":
                    signal_type = SignalType.BUY
                    confidence = base_confidence
                elif current_price < cluster_center and fib_levels[0].type == "extension":
                    signal_type = SignalType.SELL
                    confidence = base_confidence
                else:
                    continue
                
                # Calculate stop loss and take profit
                fib_range = abs(fib_levels[0].swing_ref[0] - fib_levels[0].swing_ref[1])
                if signal_type == SignalType.BUY:
                    stop_loss = cluster_center - fib_range * 0.2
                    take_profit = cluster_center + fib_range * 0.4
                else:
                    stop_loss = cluster_center + fib_range * 0.2
                    take_profit = cluster_center - fib_range * 0.4
                
                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name="FibonacciAdvanced",
                    signal_type=signal_type,
                    confidence=min(confidence, 0.95),
                    price=cluster_center,
                    timeframe=timeframe,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'cluster_center': cluster_center,
                        'level_count': level_count,
                        'trend': trend,
                        'fib_type': fib_levels[0].type
                    }
                )
                
                if self.validate_signal(signal):
                    signals.append(signal)
                    self.last_signal_time = current_time
            
            return signals[:self.max_signals_per_hour]
        
        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            return []
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze market data for Fibonacci patterns
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Find swing points
            swings = self._find_swing_points(data)
            fib_levels = []
            for i in range(len(swings) - 1):
                current_swing = swings[i]
                next_swing = swings[i + 1]
                if current_swing[2] == 'high' and next_swing[2] == 'low':
                    fib_levels.extend(self._calculate_fibonacci_levels(
                        current_swing[1], next_swing[1], timeframe))
                elif current_swing[2] == 'low' and next_swing[2] == 'high':
                    fib_levels.extend(self._calculate_fibonacci_levels(
                        next_swing[1], current_swing[1], timeframe))
            
            # Find clusters
            current_price = data['Close'].iloc[-1]
            clusters = self._find_fibonacci_clusters(fib_levels, current_price)
            
            # Get trend
            trend = self._get_trend_direction(data)
            
            return {
                'recent_swings': [
                    {'high': s[1] if s[2] == 'high' else None, 
                     'low': s[1] if s[2] == 'low' else None}
                    for s in swings[-5:]
                ],
                'retracement_levels': [l.price for l in fib_levels if l.type == 'retracement'],
                'extension_levels': [l.price for l in fib_levels if l.type == 'extension'],
                'clusters': [c[0] for c in clusters],
                'current_price': current_price,
                'trend_direction': trend
            }
        
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {}
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and parameters
        
        Returns:
            Strategy information dictionary
        """
        return {
            'name': 'FibonacciAdvanced',
            'type': 'Technical',
            'description': 'Advanced Fibonacci strategy using retracements, extensions, and clusters',
            'parameters': {
                'lookback_period': self.lookback_period,
                'fib_levels': self.fib_levels,
                'cluster_tolerance': self.cluster_tolerance,
                'confidence_threshold': self.confidence_threshold,
                'multi_timeframe': self.multi_timeframe,
                'cooldown_bars': self.cooldown_bars
            },
            'performance': {
                'success_rate': self.performance.win_rate,
                'profit_factor': self.performance.profit_factor
            }
        }


if __name__ == "__main__":
    """Test the Fibonacci Advanced Strategy"""
    
    # Test configuration
    test_config = {
        'parameters': {
            'lookback_period': 200,
            'fib_levels': [0.236, 0.382, 0.5, 0.618, 0.786, 1.272, 1.618, 2.0],
            'cluster_tolerance': 0.003,
            'confidence_threshold': 0.65,
            'multi_timeframe': True,
            'cooldown_bars': 3,
            'timeframe_primary': 'M15',
            'timeframe_secondary': 'H1'
        }
    }
    
    # Mock MT5 manager for testing
    class MockMT5Manager:
        def get_historical_data(self, symbol, timeframe, bars):
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=10), 
                end=datetime.now(), 
                freq='15Min' if timeframe == 'M15' else 'H'
            )[:bars]
            
            np.random.seed(42)
            base_price = 1950
            prices = []
            for i in range(len(dates)):
                if i % 50 < 10:
                    price = base_price + 20 * np.sin(i/10) + np.random.normal(0, 2)
                elif i % 50 < 20:
                    price = base_price - 15 * np.sin(i/10) + np.random.normal(0, 2)
                else:
                    price = base_price + 10 * np.sin(i/10) + np.random.normal(0, 2)
                prices.append(price)
            
            data = pd.DataFrame({
                'Open': prices + np.random.normal(0, 0.5, len(dates)),
                'High': np.array(prices) + np.abs(np.random.normal(2, 1, len(dates))),
                'Low': np.array(prices) - np.abs(np.random.normal(2, 1, len(dates))),
                'Close': prices,
                'Volume': np.random.randint(100, 1000, len(dates))
            }, index=dates)
            
            return data
    
    # Create strategy instance
    mock_mt5 = MockMT5Manager()
    strategy = FibonacciAdvancedStrategy(test_config, mock_mt5)
    
    print("============================================================")
    print("TESTING FIBONACCI ADVANCED STRATEGY")
    print("============================================================")
    
    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
        print(f"     Cluster Center: {signal.metadata.get('cluster_center', 'N/A'):.2f}")
        print(f"     Trend: {signal.metadata.get('trend', 'N/A')}")
    
    print("\n2. Testing analysis method:")
    mock_data = mock_mt5.get_historical_data("XAUUSDm", "M15", 200)
    analysis_results = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis results keys: {list(analysis_results.keys())}")
    print(f"   Recent swings: {len(analysis_results.get('recent_swings', []))}")
    print(f"   Retracement levels: {len(analysis_results.get('retracement_levels', []))}")
    print(f"   Extension levels: {len(analysis_results.get('extension_levels', []))}")
    print(f"   Clusters: {len(analysis_results.get('clusters', []))}")
    print(f"   Trend direction: {analysis_results.get('trend_direction', 'N/A')}")
    
    print("\n3. Testing performance tracking:")
    summary = strategy.get_performance_summary()
    print(f"   {summary}")
    
    print("\n4. Strategy Information:")
    strategy_info = strategy.get_strategy_info()
    print(f"   Name: {strategy_info['name']}")
    print(f"   Type: {strategy_info['type']}")
    print(f"   Description: {strategy_info['description']}")
    print(f"   Parameters:")
    for key, value in strategy_info['parameters'].items():
        print(f"     {key}: {value}")
    print(f"   Performance Summary:")
    print(f"     Success Rate: {strategy_info['performance']['success_rate']:.2%}")
    print(f"     Profit Factor: {strategy_info['performance']['profit_factor']:.2f}")
    
    print("\n============================================================")
    print("FIBONACCI ADVANCED STRATEGY TEST COMPLETED!")
    print("============================================================")
