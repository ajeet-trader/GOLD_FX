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

# Import CLI args for mode selection
try:
    from src.utils.cli_args import parse_mode, print_mode_banner
except Exception:
    def parse_mode(*_args, **_kwargs): # type: ignore
        return 'mock'
    def print_mode_banner(_mode): # type: ignore
        pass


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
    
    def _create_mock_mt5(self):
        """Create mock MT5 manager with mode-specific data"""
        class MockMT5Manager:
            def __init__(self, mode):
                self.mode = mode
                
            def get_historical_data(self, symbol, timeframe, bars):
                dates = pd.date_range(
                    start=datetime.now() - timedelta(days=10), 
                    end=datetime.now(), 
                    freq='15Min' if timeframe == 'M15' else 'h'
                )[:bars]
                
                np.random.seed(42 if self.mode == 'mock' else 123)
                base_price = 1950 if self.mode == 'mock' else 1975
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
        
        return MockMT5Manager(self.mode)

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
        used_points = set()
        
        for i, price in enumerate(price_points):
            if price in used_points:
                continue
                
            nearby_levels = [p for p in price_points if 
                          abs(p - price) <= price * self.cluster_tolerance]
            
            if len(nearby_levels) >= self.min_cluster_size:
                center = sum(nearby_levels) / len(nearby_levels)
                tolerance = center * self.cluster_tolerance
                clusters.append((center, tolerance, len(nearby_levels)))
                
                # Mark these points as used to avoid duplicate clusters
                for point in nearby_levels:
                    used_points.add(point)
        
        # Sort by proximity to current price and return top 5 clusters
        clusters = sorted(clusters, key=lambda x: abs(current_price - x[0]))
        return clusters[:5]
    
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
            
            # Get recent price action for signal generation
            recent_high = data['High'].tail(10).max()
            recent_low = data['Low'].tail(10).min()
            price_range = recent_high - recent_low
            
            for cluster_center, tolerance, level_count in clusters:  # Check all clusters
                # Calculate confidence based on cluster size and proximity to current price
                distance_factor = 1.0 - min(abs(current_price - cluster_center) / (price_range + 1), 0.5)
                base_confidence = min(0.9, 0.6 + (level_count - self.min_cluster_size) * 0.05 + distance_factor * 0.2)
                
                if secondary_data is not None:
                    base_confidence *= 1.1  # Boost for multi-timeframe confluence
                
                signal_type = None
                confidence = base_confidence
                signal_reason = ""
                
                # Check for various signal conditions
                price_to_cluster_distance = abs(current_price - cluster_center)
                
                # 1. Price near cluster (retracement/support/resistance)
                if price_to_cluster_distance <= tolerance * 2:
                    if trend == "up" and current_price > cluster_center:
                        signal_type = SignalType.BUY
                        confidence *= 1.1
                        signal_reason = "bullish_retracement"
                    elif trend == "down" and current_price < cluster_center:
                        signal_type = SignalType.SELL
                        confidence *= 1.1
                        signal_reason = "bearish_retracement"
                    elif trend == "sideways":
                        # In sideways market, trade bounces off levels
                        if current_price < cluster_center:
                            signal_type = SignalType.BUY
                            signal_reason = "support_bounce"
                        else:
                            signal_type = SignalType.SELL
                            signal_reason = "resistance_rejection"
                
                # 2. Price approaching cluster from distance
                elif price_to_cluster_distance <= tolerance * 5:
                    if trend == "up" and current_price < cluster_center:
                        signal_type = SignalType.BUY
                        confidence *= 0.9
                        signal_reason = "approaching_resistance"
                    elif trend == "down" and current_price > cluster_center:
                        signal_type = SignalType.SELL
                        confidence *= 0.9
                        signal_reason = "approaching_support"
                
                # 3. Breakout signals
                elif current_price > cluster_center and trend in ["up", "sideways"]:
                    signal_type = SignalType.BUY
                    confidence *= 0.8
                    signal_reason = "breakout_above"
                elif current_price < cluster_center and trend in ["down", "sideways"]:
                    signal_type = SignalType.SELL
                    confidence *= 0.8
                    signal_reason = "breakdown_below"
                
                if signal_type and confidence >= 0.5:  # Lower threshold for more signals
                    # Calculate stop loss and take profit based on recent price action
                    atr_estimate = price_range * 0.3  # Rough ATR estimate
                    
                    if signal_type == SignalType.BUY:
                        stop_loss = current_price - atr_estimate * 1.5
                        take_profit = current_price + atr_estimate * 2.0
                    else:
                        stop_loss = current_price + atr_estimate * 1.5
                        take_profit = current_price - atr_estimate * 2.0
                    
                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name="FibonacciAdvanced",
                        signal_type=signal_type,
                        confidence=min(confidence, 0.95),
                        price=current_price,  # Use current price instead of cluster center
                        timeframe=timeframe,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'cluster_center': cluster_center,
                            'level_count': level_count,
                            'trend': trend,
                            'signal_reason': signal_reason,
                            'distance_to_cluster': price_to_cluster_distance,
                            'price_range': price_range
                        }
                    )
                    
                    signals.append(signal)
                    
                    # Limit signals to prevent overtrading
                    if len(signals) >= 8:
                        break
            
            # Filter and validate signals
            valid_signals = []
            for signal in signals:
                if signal.confidence >= 0.5:  # Lower validation threshold
                    valid_signals.append(signal)
                    
            if valid_signals:
                self.last_signal_time = current_time
            
            return valid_signals[:self.max_signals_per_hour]
        
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
            'timeframe_secondary': 'H1',
            'mode': 'mock' # Added mode parameter to test config
        }
    }
    
    # Create strategy instance
    strategy = FibonacciAdvancedStrategy(test_config, mt5_manager=None, database=None) # Pass mt5_manager=None to trigger internal mock creation
    
    print("============================================================")
    print("TESTING FIBONACCI ADVANCED STRATEGY")
    print("============================================================")
    print(f"Running in {strategy.mode.upper()} mode") # Print mode
    
    print("\n1. Testing signal generation (Multiple runs to simulate daily signals):")
    all_signals = []
    
    # Simulate multiple runs throughout the day with different random seeds
    for run in range(8):  # Simulate 8 different time periods
        # Use strategy's internal mock MT5 manager for consistent data
        mock_data_run = strategy.mt5_manager.get_historical_data("XAUUSDm", "M15", 200)
        
        # Override random seed for price generation consistency in mock
        original_seed = np.random.get_state()
        np.random.seed(42 + run)
        
        # Note: FibonacciStrategy relies on getting fresh data in generate_signal,
        # so this loop will always regenerate signals based on slightly varied mock data.
        # For a truly isolated test per run, you'd feed pre-generated data into generate_signal.
        
        strategy.last_signal_time = None  # Reset cooldown for testing
        
        signals = strategy.generate_signal("XAUUSDm", "M15")
        if signals:
            all_signals.extend(signals)
            print(f"   Run {run+1}: Generated {len(signals)} signals")
            for signal in signals:
                print(f"     - {signal.signal_type.value} at {signal.price:.2f}, "
                      f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
                print(f"       Reason: {signal.metadata.get('signal_reason', 'N/A')}")
                print(f"       SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f}")
                print(f"       Risk/Reward: {abs(signal.take_profit - signal.price) / abs(signal.price - signal.stop_loss):.2f}")
        else:
            print(f"   Run {run+1}: No signals generated")
        
        np.random.set_state(original_seed) # Restore original random state
    
    print(f"\n   TOTAL DAILY SIGNALS: {len(all_signals)}")
    print(f"   Signal Distribution:")
    buy_signals = [s for s in all_signals if s.signal_type == SignalType.BUY]
    sell_signals = [s for s in all_signals if s.signal_type == SignalType.SELL]
    print(f"     BUY signals: {len(buy_signals)}")
    print(f"     SELL signals: {len(sell_signals)}")
    
    if all_signals:
        avg_confidence = sum(s.confidence for s in all_signals) / len(all_signals)
        print(f"     Average confidence: {avg_confidence:.3f}")
        
        grade_counts = {}
        for signal in all_signals:
            grade = signal.grade.value
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        print(f"     Grade distribution: {grade_counts}")
    
    print("\n2. Testing analysis method:")
    mock_data = strategy.mt5_manager.get_historical_data("XAUUSDm", "M15", 200)
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
