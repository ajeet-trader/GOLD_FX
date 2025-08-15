"""
Market Structure Strategy - Smart Money Concepts
==============================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-08-15

Advanced Market Structure Strategy for XAUUSD trading:
- Identifies Higher Highs (HH), Higher Lows (HL), Lower Highs (LH), and Lower Lows (LL)
- Detects Break of Structure (BOS) and Change of Character (CHoCH) events
- Generates signals based on BOS with retest and CHoCH with pullback
- Tuned for 5–10 daily signals using flexible swing detection

Features:
- Multi-timeframe structure analysis
- BOS and CHoCH event detection
- Retest confirmation for continuation signals
- Pullback confirmation for reversal signals
- Dynamic stop-loss and take-profit based on swing levels

Dependencies:
    - pandas
    - numpy
    - datetime
    - typing
    - dataclasses
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
from enum import Enum

from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade

class MarketTrend(Enum):
    """Market trend enumeration"""
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGE = "RANGE"

@dataclass
class SwingPoint:
    """Swing point data structure"""
    type: str  # "high" or "low"
    price: float
    index: int
    timestamp: datetime

@dataclass
class StructureEvent:
    """Market structure event (BOS or CHoCH)"""
    type: str  # "BOS" or "CHoCH"
    direction: str  # "up" or "down" for BOS, "up_to_down" or "down_to_up" for CHoCH
    level: float
    bar_index: int
    timestamp: datetime

class MarketStructureStrategy(AbstractStrategy):
    """
    Advanced Market Structure Strategy

    Detects trend via higher highs (HH), higher lows (HL), lower highs (LH), and lower lows (LL).
    Identifies key Smart Money Concepts events: Break of Structure (BOS) and Change of Character (CHoCH).
    Generates signals when BOS or CHoCH aligns with momentum or retest confirmations.

    Signal Generation:
    - Continuation: BOS + retest within retest_window → trade with trend
    - Reversal: CHoCH + small pullback → trade opposite
    - Tuned for 5–10 signals/day with small cooldown and flexible swing detection

    Example:
        >>> strategy = MarketStructureStrategy(config, mt5_manager)
        >>> signals = strategy.generate_signal("XAUUSDm", "M15")
    """

    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """
        Initialize Market Structure strategy

        Args:
            config: Strategy configuration
            mt5_manager: MT5 connection manager (optional)
            database: Database manager (optional)
        """
        super().__init__(config, mt5_manager, database)

        # Strategy parameters from config
        self.lookback_bars = config.get('parameters', {}).get('lookback_bars', 200)
        self.swing_window = config.get('parameters', {}).get('swing_window', 5)
        self.retest_window = config.get('parameters', {}).get('retest_window', 3)
        self.confidence_threshold = config.get('parameters', {}).get('confidence_threshold', 0.65)
        self.cooldown_bars = config.get('parameters', {}).get('cooldown_bars', 3)
        self.swing_tolerance = config.get('parameters', {}).get('swing_tolerance', 0.002)  # 0.2% price tolerance

        # Strategy state
        self.last_signal_bar = -self.cooldown_bars - 1  # Allow immediate signals
        self.last_bos = None
        self.last_choch = None
        self.recent_swings = []
        self.current_trend = MarketTrend.RANGE

        self.logger.info(f"MarketStructureStrategy initialized with lookback={self.lookback_bars}, "
                        f"swing_window={self.swing_window}, confidence_threshold={self.confidence_threshold}")

    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """
        Generate market structure-based trading signals

        Args:
            symbol: Trading symbol (e.g., "XAUUSDm")
            timeframe: Analysis timeframe (e.g., "M15")

        Returns:
            List of trading signals
        """
        try:
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available")
                return []

            # Get historical data
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_bars)
            if data is None or len(data) < self.swing_window * 2 + 1:
                self.logger.warning(f"Insufficient data: {len(data) if data is not None else 0} bars")
                return []

            # Find swing points
            swings = self._find_swing_points(data)
            if len(swings) < 4:
                self.logger.debug(f"Insufficient swing points: {len(swings)}")
                return []

            # Update current trend
            self.current_trend = self._determine_trend(swings)

            # Detect structure events
            bos_event = self._detect_bos(data, swings)
            choch_event = self._detect_choch(data, swings)

            signals = []
            current_bar = len(data) - 1

            # Check if we're in cooldown period
            if current_bar - self.last_signal_bar <= self.cooldown_bars:
                self.logger.debug("In cooldown period, skipping signal generation")
                return []

            # BOS signal (Continuation)
            if bos_event and bos_event.bar_index >= len(data) - self.retest_window:
                signal = self._create_bos_signal(bos_event, data, symbol, timeframe, swings)
                if signal and self.validate_signal(signal):
                    signals.append(signal)
                    self.last_signal_bar = current_bar

            # CHoCH signal (Reversal)
            if choch_event and choch_event.bar_index >= len(data) - self.retest_window:
                signal = self._create_choch_signal(choch_event, data, symbol, timeframe, swings)
                if signal and self.validate_signal(signal):
                    signals.append(signal)
                    self.last_signal_bar = current_bar

            # Update state
            self.last_bos = bos_event
            self.last_choch = choch_event
            self.recent_swings = swings[-5:]  # Keep last 5 swings

            if signals:
                self.logger.info(f"Generated {len(signals)} signals for {symbol} on {timeframe}")
                for signal in signals:
                    self.database_manager.store_signal({
                        'symbol': signal.symbol,
                        'strategy': signal.strategy_name,
                        'signal_type': signal.signal_type.value,
                        'confidence': signal.confidence,
                        'price': signal.price,
                        'timeframe': signal.timeframe,
                        'strength': signal.strength,
                        'quality_grade': signal.grade.value if signal.grade else None,
                        'metadata': signal.metadata
                    })

            return signals

        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            return []

    def _find_swing_points(self, data: pd.DataFrame) -> List[SwingPoint]:
        """
        Identify swing highs and lows in the data

        Args:
            data: OHLCV DataFrame

        Returns:
            List of SwingPoint objects
        """
        swings = []
        for i in range(self.swing_window, len(data) - self.swing_window):
            high = data['High'].iloc[i]
            low = data['Low'].iloc[i]
            timestamp = data.index[i]

            # Check for swing high
            if high == max(data['High'].iloc[i - self.swing_window:i + self.swing_window + 1]):
                swings.append(SwingPoint(
                    type="high",
                    price=high,
                    index=i,
                    timestamp=timestamp
                ))

            # Check for swing low
            if low == min(data['Low'].iloc[i - self.swing_window:i + self.swing_window + 1]):
                swings.append(SwingPoint(
                    type="low",
                    price=low,
                    index=i,
                    timestamp=timestamp
                ))

        # Filter swings within tolerance
        filtered_swings = []
        for i, swing in enumerate(swings):
            too_close = False
            for prev_swing in filtered_swings[-2:]:
                if (abs(swing.price - prev_swing.price) / swing.price) < self.swing_tolerance:
                    too_close = True
                    break
            if not too_close:
                filtered_swings.append(swing)

        return filtered_swings

    def _determine_trend(self, swings: List[SwingPoint]) -> MarketTrend:
        """
        Determine current market trend based on swing points

        Args:
            swings: List of SwingPoint objects

        Returns:
            MarketTrend enum
        """
        if len(swings) < 4:
            return MarketTrend.RANGE

        recent_highs = [s for s in swings[-4:] if s.type == "high"]
        recent_lows = [s for s in swings[-4:] if s.type == "low"]

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return MarketTrend.RANGE

        # Check for Higher Highs and Higher Lows (Uptrend)
        if (recent_highs[-1].price > recent_highs[-2].price and
            recent_lows[-1].price > recent_lows[-2].price):
            return MarketTrend.UPTREND

        # Check for Lower Highs and Lower Lows (Downtrend)
        if (recent_highs[-1].price < recent_highs[-2].price and
            recent_lows[-1].price < recent_lows[-2].price):
            return MarketTrend.DOWNTREND

        return MarketTrend.RANGE

    def _detect_bos(self, data: pd.DataFrame, swings: List[SwingPoint]) -> Optional[StructureEvent]:
        """
        Detect Break of Structure events

        Args:
            data: OHLCV DataFrame
            swings: List of SwingPoint objects

        Returns:
            StructureEvent if BOS detected, None otherwise
        """
        if len(swings) < 2:
            return None

        current_price = data['Close'].iloc[-1]
        current_idx = len(data) - 1
        timestamp = data.index[-1]

        last_high = max((s for s in swings if s.type == "high"), key=lambda x: x.index, default=None)
        last_low = max((s for s in swings if s.type == "low"), key=lambda x: x.index, default=None)

        if not last_high or not last_low:
            return None

        # Uptrend BOS: Break above last swing high
        if (self.current_trend == MarketTrend.UPTREND and
            current_price > last_high.price and
            last_high.index < current_idx - 1):
            # Check for retest within window
            recent_lows = data['Low'].iloc[-self.retest_window:]
            if any(recent_lows <= last_high.price * (1 + self.swing_tolerance)):
                return StructureEvent(
                    type="BOS",
                    direction="up",
                    level=last_high.price,
                    bar_index=current_idx,
                    timestamp=timestamp
                )

        # Downtrend BOS: Break below last swing low
        if (self.current_trend == MarketTrend.DOWNTREND and
            current_price < last_low.price and
            last_low.index < current_idx - 1):
            # Check for retest within window
            recent_highs = data['High'].iloc[-self.retest_window:]
            if any(recent_highs >= last_low.price * (1 - self.swing_tolerance)):
                return StructureEvent(
                    type="BOS",
                    direction="down",
                    level=last_low.price,
                    bar_index=current_idx,
                    timestamp=timestamp
                )

        return None

    def _detect_choch(self, data: pd.DataFrame, swings: List[SwingPoint]) -> Optional[StructureEvent]:
        """
        Detect Change of Character events

        Args:
            data: OHLCV DataFrame
            swings: List of SwingPoint objects

        Returns:
            StructureEvent if CHoCH detected, None otherwise
        """
        if len(swings) < 4:
            return None

        current_price = data['Close'].iloc[-1]
        current_idx = len(data) - 1
        timestamp = data.index[-1]

        last_high = max((s for s in swings if s.type == "high"), key=lambda x: x.index, default=None)
        last_low = max((s for s in swings if s.type == "low"), key=lambda x: x.index, default=None)

        if not last_high or not last_low:
            return None

        # Uptrend to Downtrend CHoCH: Break below last swing low in uptrend
        if (self.current_trend == MarketTrend.UPTREND and
            current_price < last_low.price and
            last_low.index < current_idx - 1):
            # Check for pullback confirmation
            recent_highs = data['High'].iloc[-self.retest_window:]
            if any(recent_highs >= last_low.price * (1 + self.swing_tolerance * 0.5)):
                return StructureEvent(
                    type="CHoCH",
                    direction="up_to_down",
                    level=last_low.price,
                    bar_index=current_idx,
                    timestamp=timestamp
                )

        # Downtrend to Uptrend CHoCH: Break above last swing high in downtrend
        if (self.current_trend == MarketTrend.DOWNTREND and
            current_price > last_high.price and
            last_high.index < current_idx - 1):
            # Check for pullback confirmation
            recent_lows = data['Low'].iloc[-self.retest_window:]
            if any(recent_lows <= last_high.price * (1 - self.swing_tolerance * 0.5)):
                return StructureEvent(
                    type="CHoCH",
                    direction="down_to_up",
                    level=last_high.price,
                    bar_index=current_idx,
                    timestamp=timestamp
                )

        return None

    def _create_bos_signal(self, event: StructureEvent, data: pd.DataFrame, symbol: str,
                          timeframe: str, swings: List[SwingPoint]) -> Optional[Signal]:
        """
        Create a BOS continuation signal

        Args:
            event: BOS StructureEvent
            data: OHLCV DataFrame
            symbol: Trading symbol
            timeframe: Timeframe
            swings: List of SwingPoint objects

        Returns:
            Signal object or None
        """
        try:
            current_price = data['Close'].iloc[-1]
            timestamp = data.index[-1]

            # Calculate stop loss and take profit
            recent_lows = [s.price for s in swings[-4:] if s.type == "low"]
            recent_highs = [s.price for s in swings[-4:] if s.type == "high"]

            if event.direction == "up":
                signal_type = SignalType.BUY
                stop_loss = min(recent_lows) if recent_lows else current_price * 0.995
                take_profit = current_price + (current_price - stop_loss) * 2  # 2:1 RR
                confidence = self.confidence_threshold + 0.1  # Slightly higher for continuation
            else:
                signal_type = SignalType.SELL
                stop_loss = max(recent_highs) if recent_highs else current_price * 1.005
                take_profit = current_price - (stop_loss - current_price) * 2  # 2:1 RR
                confidence = self.confidence_threshold + 0.1

            signal = Signal(
                timestamp=timestamp,
                symbol=symbol,
                strategy_name=self.strategy_name,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                timeframe=timeframe,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strength=0.8,
                metadata={
                    'event_type': 'BOS',
                    'event_level': event.level,
                    'trend': self.current_trend.value
                }
            )

            return signal

        except Exception as e:
            self.logger.error(f"BOS signal creation failed: {str(e)}")
            return None

    def _create_choch_signal(self, event: StructureEvent, data: pd.DataFrame, symbol: str,
                            timeframe: str, swings: List[SwingPoint]) -> Optional[Signal]:
        """
        Create a CHoCH reversal signal

        Args:
            event: CHoCH StructureEvent
            data: OHLCV DataFrame
            symbol: Trading symbol
            timeframe: Timeframe
            swings: List of SwingPoint objects

        Returns:
            Signal object or None
        """
        try:
            current_price = data['Close'].iloc[-1]
            timestamp = data.index[-1]

            # Calculate stop loss and take profit
            recent_lows = [s.price for s in swings[-4:] if s.type == "low"]
            recent_highs = [s.price for s in swings[-4:] if s.type == "high"]

            if event.direction == "down_to_up":
                signal_type = SignalType.BUY
                stop_loss = min(recent_lows) if recent_lows else current_price * 0.995
                take_profit = current_price + (current_price - stop_loss) * 2  # 2:1 RR
                confidence = self.confidence_threshold
            else:  # up_to_down
                signal_type = SignalType.SELL
                stop_loss = max(recent_highs) if recent_highs else current_price * 1.005
                take_profit = current_price - (stop_loss - current_price) * 2  # 2:1 RR
                confidence = self.confidence_threshold

            signal = Signal(
                timestamp=timestamp,
                symbol=symbol,
                strategy_name=self.strategy_name,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                timeframe=timeframe,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strength=0.7,
                metadata={
                    'event_type': 'CHoCH',
                    'event_level': event.level,
                    'trend': self.current_trend.value
                }
            )

            return signal

        except Exception as e:
            self.logger.error(f"CHoCH signal creation failed: {str(e)}")
            return None

    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze market structure

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dictionary with trend and structure event information
        """
        try:
            swings = self._find_swing_points(data)
            trend = self._determine_trend(swings)
            bos = self._detect_bos(data, swings)
            choch = self._detect_choch(data, swings)

            analysis = {
                'trend': trend.value,
                'last_bos': {
                    'type': bos.direction,
                    'level': bos.level,
                    'bar_index': bos.bar_index
                } if bos else None,
                'last_choch': {
                    'from': 'up' if bos and bos.direction == 'down_to_up' else 'down',
                    'to': 'down' if bos and bos.direction == 'down_to_up' else 'up',
                    'level': choch.level,
                    'bar_index': choch.bar_index
                } if choch else None,
                'recent_swings': [
                    {
                        'type': s.type,
                        'price': s.price,
                        'index': s.index
                    } for s in swings[-5:]
                ]
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {}

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and configuration

        Returns:
            Dictionary containing strategy details
        """
        try:
            return {
                'name': self.strategy_name,
                'type': 'SMC',
                'description': ('Detects market structure via HH/HL/LH/LL patterns, '
                              'generating signals on BOS and CHoCH events with retest/pullback confirmation.'),
                'parameters': {
                    'lookback_bars': self.lookback_bars,
                    'swing_window': self.swing_window,
                    'retest_window': self.retest_window,
                    'confidence_threshold': self.confidence_threshold,
                    'cooldown_bars': self.cooldown_bars,
                    'swing_tolerance': self.swing_tolerance
                },
                'performance': {
                    'success_rate': self.performance.win_rate,
                    'profit_factor': self.performance.profit_factor
                }
            }
        except Exception as e:
            self.logger.error(f"Strategy info generation failed: {str(e)}")
            return {}

    def _create_empty_performance_metrics(self):
        """Create empty performance metrics for direct script runs"""
        from src.core.base import StrategyPerformance
        return StrategyPerformance(
            strategy_name=self.strategy_name,
            win_rate=0.0,
            profit_factor=0.0
        )

if __name__ == "__main__":
    """Test the Market Structure strategy"""

    # Test configuration
    test_config = {
        'parameters': {
            'lookback_bars': 200,
            'swing_window': 5,
            'retest_window': 3,
            'confidence_threshold': 0.65,
            'cooldown_bars': 3,
            'swing_tolerance': 0.002
        }
    }

    # Mock MT5 manager for testing
    class MockMT5Manager:
        def get_historical_data(self, symbol, timeframe, bars):
            dates = pd.date_range(start=datetime.now() - timedelta(days=10),
                                 end=datetime.now(), freq='15Min')[:bars]

            np.random.seed(42)
            base_price = 1950
            prices = []
            for i in range(len(dates)):
                if i < 50:
                    base_price += 2  # Uptrend (HH/HL)
                elif i < 100:
                    base_price -= 1.5  # Downtrend (LH/LL)
                elif i < 150:
                    base_price += 1.8  # Uptrend (HH/HL)
                else:
                    base_price -= 1.2  # Downtrend (LH/LL)
                prices.append(base_price + np.random.normal(0, 0.5))

            data = pd.DataFrame({
                'Open': np.array(prices) + np.random.normal(0, 0.5, len(dates)),
                'High': np.array(prices) + np.abs(np.random.normal(2, 1, len(dates))),
                'Low': np.array(prices) - np.abs(np.random.normal(2, 1, len(dates))),
                'Close': prices,
                'Volume': np.random.randint(500, 1500, len(dates))
            }, index=dates)

            return data

    # Create strategy instance
    mock_mt5 = MockMT5Manager()
    strategy = MarketStructureStrategy(test_config, mock_mt5)

    print("="*60)
    print("TESTING MARKET STRUCTURE STRATEGY")
    print("="*60)

    # Test signal generation
    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
        print(f"     Event: {signal.metadata.get('event_type', 'Unknown')}")
        print(f"     Trend: {signal.metadata.get('trend', 'Unknown')}")

    # Test analysis
    print("\n2. Testing analysis method:")
    mock_data = mock_mt5.get_historical_data("XAUUSDm", "M15", 200)
    analysis = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis keys: {list(analysis.keys())}")
    print(f"   Current Trend: {analysis.get('trend', 'N/A')}")
    if 'last_bos' in analysis and analysis['last_bos']:
        print(f"   Last BOS: {analysis['last_bos']['type']} at {analysis['last_bos']['level']:.2f}")
    if 'last_choch' in analysis and analysis['last_choch']:
        print(f"   Last CHoCH: {analysis['last_choch']['from']} to {analysis['last_choch']['to']}")

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

    print("\n" + "="*60)
    print("MARKET STRUCTURE STRATEGY TEST COMPLETED!")
    print("="*60)
