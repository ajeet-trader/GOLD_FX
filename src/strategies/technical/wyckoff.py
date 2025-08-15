"""
Wyckoff Strategy - Advanced Market Phase Analysis
================================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-08-15

Implements the Wyckoff Method to identify market phases (Accumulation, Distribution, 
Re-accumulation, Redistribution) and key events such as Springs, Upthrusts, Signs of Strength (SOS), 
and Signs of Weakness (SOW). Generates signals based on phase transitions and key event confirmations.

Features:
- Detection of Wyckoff accumulation and distribution phases
- Identification of Springs, Upthrusts, SOS, and SOW events
- Multi-timeframe phase analysis
- Dynamic range detection
- Signal generation for both range-bound and breakout trades

Dependencies:
    - pandas
    - numpy
    - datetime
    - typing
    - dataclasses
    - enum
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

class WyckoffPhase(Enum):
    """Wyckoff phase enumeration"""
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    RE_ACCUMULATION = "RE_ACCUMULATION"
    RE_DISTRIBUTION = "RE_DISTRIBUTION"
    UNKNOWN = "UNKNOWN"

class WyckoffEvent(Enum):
    """Wyckoff event enumeration"""
    SPRING = "SPRING"
    UPTHRUST = "UPTHRUST"
    SOS = "SIGN_OF_STRENGTH"
    SOW = "SIGN_OF_WEAKNESS"
    NONE = "NONE"

@dataclass
class WyckoffStructure:
    """Wyckoff structure data structure"""
    phase: WyckoffPhase
    range_high: float
    range_low: float
    event: WyckoffEvent
    confidence: float
    timestamp: datetime
    bar_index: int
    trend_context: str  # "Uptrend", "Downtrend", "Sideways"

class WyckoffStrategy(AbstractStrategy):
    """
    Wyckoff Method Strategy
    
    This strategy implements the Wyckoff Method for XAUUSD trading:
    - Detects accumulation and distribution phases
    - Identifies key Wyckoff events (Springs, Upthrusts, SOS, SOW)
    - Generates signals for both range-bound events and breakouts
    - Supports multi-timeframe analysis for trend context
    - Designed to produce 5–10 valid signals daily through minor and major structures
    
    Signal Generation:
    - BUY: Spring in accumulation, SOS breakout, re-accumulation confirmation
    - SELL: Upthrust in distribution, SOW breakdown, re-distribution confirmation
    - HOLD: No clear phase or event detected
    
    Example:
        >>> strategy = WyckoffStrategy(config, mt5_manager)
        >>> signals = strategy.generate_signal("XAUUSDm", "M15")
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """
        Initialize Wyckoff strategy
        
        Args:
            config: Strategy configuration
            mt5_manager: MT5 connection manager (optional)
            database: Database manager (optional)
        """
        super().__init__(config, mt5_manager, database)
        
        # Wyckoff parameters
        self.lookback_period = config.get('parameters', {}).get('lookback_period', 150)
        self.swing_detection_window = config.get('parameters', {}).get('swing_detection_window', 10)
        self.phase_confirmation_bars = config.get('parameters', {}).get('phase_confirmation_bars', 3)
        self.confidence_threshold = config.get('parameters', {}).get('confidence_threshold', 0.65)
        self.min_strength = config.get('parameters', {}).get('min_strength', 0.5)
        self.cooldown_bars = config.get('parameters', {}).get('cooldown_bars', 3)
        
        # Internal state
        self.last_signal_bar = -self.cooldown_bars
        self.detected_structures = []
        
        self.logger.info(f"WyckoffStrategy initialized with lookback={self.lookback_period}, "
                        f"confidence_threshold={self.confidence_threshold}")
    
    def _detect_swings(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Detect swing highs and lows"""
        highs = []
        lows = []
        window = self.swing_detection_window
        
        for i in range(window, len(data) - window):
            if (data['High'].iloc[i] == data['High'].iloc[i-window:i+window+1].max() and
                data['High'].iloc[i] > data['Close'].iloc[i-1]):
                highs.append(i)
            if (data['Low'].iloc[i] == data['Low'].iloc[i-window:i+window+1].min() and
                data['Low'].iloc[i] < data['Close'].iloc[i-1]):
                lows.append(i)
        
        return highs, lows
    
    def _identify_phase(self, data: pd.DataFrame, highs: List[int], lows: List[int]) -> WyckoffPhase:
        """Identify Wyckoff phase based on price action"""
        try:
            if len(highs) < 2 or len(lows) < 2:
                return WyckoffPhase.UNKNOWN
                
            recent_highs = data['High'].iloc[highs[-2:]]
            recent_lows = data['Low'].iloc[lows[-2:]]
            
            # Calculate range characteristics
            range_high = recent_highs.max()
            range_low = recent_lows.min()
            range_width = range_high - range_low
            atr = (data['High'] - data['Low']).mean()
            
            # Determine trend context
            trend_context = self._get_trend_context(data)
            
            # Phase detection logic
            if range_width < atr * 3:  # Tight range
                if trend_context == "Downtrend":
                    return WyckoffPhase.ACCUMULATION
                elif trend_context == "Uptrend":
                    return WyckoffPhase.DISTRIBUTION
                elif trend_context == "Sideways":
                    if data['Close'].iloc[-self.phase_confirmation_bars:].mean() > data['Close'].iloc[-self.phase_confirmation_bars*2:-self.phase_confirmation_bars].mean():
                        return WyckoffPhase.RE_ACCUMULATION
                    else:
                        return WyckoffPhase.RE_DISTRIBUTION
            return WyckoffPhase.UNKNOWN
        except Exception as e:
            self.logger.error(f"Phase identification failed: {str(e)}")
            return WyckoffPhase.UNKNOWN
    
    def _get_trend_context(self, data: pd.DataFrame) -> str:
        """Determine the broader trend context"""
        try:
            sma_fast = data['Close'].rolling(window=20).mean()
            sma_slow = data['Close'].rolling(window=50).mean()
            
            if sma_fast.iloc[-1] > sma_slow.iloc[-1] and sma_fast.iloc[-2] > sma_slow.iloc[-2]:
                return "Uptrend"
            elif sma_fast.iloc[-1] < sma_slow.iloc[-1] and sma_fast.iloc[-2] < sma_slow.iloc[-2]:
                return "Downtrend"
            else:
                return "Sideways"
        except:
            return "Sideways"
    
    def _detect_event(self, data: pd.DataFrame, phase: WyckoffPhase, highs: List[int], lows: List[int]) -> Tuple[WyckoffEvent, float, int]:
        """Detect Wyckoff events (Spring, Upthrust, SOS, SOW)"""
        try:
            if len(data) < self.phase_confirmation_bars:
                return WyckoffEvent.NONE, 0.0, -1
                
            range_high = data['High'].iloc[highs[-2:]].max()
            range_low = data['Low'].iloc[lows[-2:]].min()
            current_price = data['Close'].iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].iloc[-self.phase_confirmation_bars:].mean()
            
            # Spring detection (false breakdown in accumulation)
            if phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.RE_ACCUMULATION]:
                if (data['Low'].iloc[-1] < range_low and 
                    data['Close'].iloc[-1] > range_low and
                    current_volume > avg_volume * 1.2):
                    confidence = min(0.9, self.confidence_threshold + 0.25)
                    return WyckoffEvent.SPRING, confidence, len(data) - 1
            
            # Upthrust detection (false breakout in distribution)
            elif phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.RE_DISTRIBUTION]:
                if (data['High'].iloc[-1] > range_high and 
                    data['Close'].iloc[-1] < range_high and
                    current_volume > avg_volume * 1.2):
                    confidence = min(0.9, self.confidence_threshold + 0.25)
                    return WyckoffEvent.UPTHRUST, confidence, len(data) - 1
            
            # SOS (Sign of Strength - breakout from accumulation)
            elif phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.RE_ACCUMULATION]:
                if (current_price > range_high and
                    data['Close'].iloc[-self.phase_confirmation_bars:].mean() > range_high and
                    current_volume > avg_volume * 1.5):
                    confidence = min(0.95, self.confidence_threshold + 0.3)
                    return WyckoffEvent.SOS, confidence, len(data) - 1
            
            # SOW (Sign of Weakness - breakdown from distribution)
            elif phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.RE_DISTRIBUTION]:
                if (current_price < range_low and
                    data['Close'].iloc[-self.phase_confirmation_bars:].mean() < range_low and
                    current_volume > avg_volume * 1.5):
                    confidence = min(0.95, self.confidence_threshold + 0.3)
                    return WyckoffEvent.SOW, confidence, len(data) - 1
            
            return WyckoffEvent.NONE, 0.0, -1
        except Exception as e:
            self.logger.error(f"Event detection failed: {str(e)}")
            return WyckoffEvent.NONE, 0.0, -1
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """
        Generate Wyckoff-based trading signals
        
        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            List of trading signals
        """
        try:
            # Check cooldown
            current_bar = self.mt5_manager.get_historical_data(symbol, timeframe, 1).index[-1]
            if hasattr(self, 'last_signal_time') and (current_bar - self.last_signal_time).total_seconds() / 60 < self.cooldown_bars * 15:
                return []
                
            # Get market data
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available")
                return []
                
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_period)
            if data is None or len(data) < self.swing_detection_window * 2:
                self.logger.warning(f"Insufficient data for Wyckoff analysis: {len(data) if data is not None else 0} bars")
                return []
            
            # Detect swings and phase
            highs, lows = self._detect_swings(data)
            phase = self._identify_phase(data, highs, lows)
            event, confidence, bar_index = self._detect_event(data, phase, highs, lows)
            
            signals = []
            if event != WyckoffEvent.NONE and confidence >= self.confidence_threshold:
                # Calculate trade parameters
                range_high = data['High'].iloc[highs[-2:]].max()
                range_low = data['Low'].iloc[lows[-2:]].min()
                current_price = data['Close'].iloc[-1]
                atr = (data['High'] - data['Low']).mean()
                
                # Determine signal type and levels
                signal_type = SignalType.HOLD
                stop_loss = None
                take_profit = None
                
                if event == WyckoffEvent.SPRING:
                    signal_type = SignalType.BUY
                    stop_loss = range_low - atr * 0.5
                    take_profit = range_high + atr * 1.0
                elif event == WyckoffEvent.UPTHRUST:
                    signal_type = SignalType.SELL
                    stop_loss = range_high + atr * 0.5
                    take_profit = range_low - atr * 1.0
                elif event == WyckoffEvent.SOS:
                    signal_type = SignalType.BUY
                    stop_loss = range_low - atr * 0.5
                    take_profit = current_price + atr * 2.0
                elif event == WyckoffEvent.SOW:
                    signal_type = SignalType.SELL
                    stop_loss = range_high + atr * 0.5
                    take_profit = current_price - atr * 2.0
                
                if signal_type != SignalType.HOLD:
                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name="wyckoff",
                        signal_type=signal_type,
                        confidence=confidence,
                        price=current_price,
                        timeframe=timeframe,
                        strength=max(self.min_strength, confidence - 0.1),
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'phase': phase.value,
                            'event': event.value,
                            'range_high': range_high,
                            'range_low': range_low,
                            'trend_context': self._get_trend_context(data)
                        }
                    )
                    
                    if self.validate_signal(signal):
                        signals.append(signal)
                        self.last_signal_time = current_bar
                        self.detected_structures.append(WyckoffStructure(
                            phase=phase,
                            range_high=range_high,
                            range_low=range_low,
                            event=event,
                            confidence=confidence,
                            timestamp=datetime.now(),
                            bar_index=bar_index,
                            trend_context=self._get_trend_context(data)
                        ))
                        self.logger.info(f"Generated {signal_type.value} signal: {event.value} in {phase.value}, "
                                       f"Confidence: {confidence:.2f}")
            
            return signals
        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            return []
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze market data for Wyckoff structures
        
        Args:
            data: OHLCV data
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            Dictionary containing Wyckoff analysis results
        """
        try:
            highs, lows = self._detect_swings(data)
            phase = self._identify_phase(data, highs, lows)
            event, confidence, bar_index = self._detect_event(data, phase, highs, lows)
            
            range_high = data['High'].iloc[highs[-2:]].max() if highs else data['High'].max()
            range_low = data['Low'].iloc[lows[-2:]].min() if lows else data['Low'].min()
            
            detected_events = []
            if event != WyckoffEvent.NONE:
                detected_events.append({
                    'event': event.value,
                    'bar_index': bar_index,
                    'confidence': confidence
                })
            
            return {
                'current_phase': phase.value,
                'detected_events': detected_events,
                'range_high': range_high,
                'range_low': range_low,
                'trend_context': self._get_trend_context(data),
                'analysis_time': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {
                'current_phase': WyckoffPhase.UNKNOWN.value,
                'detected_events': [],
                'range_high': 0.0,
                'range_low': 0.0,
                'trend_context': "Unknown",
                'error': str(e)
            }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and parameters
        
        Returns:
            Dictionary containing strategy details
        """
        try:
            return {
                'name': 'Wyckoff Strategy',
                'type': 'Technical',
                'description': 'Implements Wyckoff Method for phase and event detection',
                'parameters': {
                    'lookback_period': self.lookback_period,
                    'swing_detection_window': self.swing_detection_window,
                    'phase_confirmation_bars': self.phase_confirmation_bars,
                    'confidence_threshold': self.confidence_threshold,
                    'min_strength': self.min_strength,
                    'cooldown_bars': self.cooldown_bars
                },
                'performance': {
                    'success_rate': self.performance.win_rate,
                    'profit_factor': self.performance.profit_factor
                },
                'events_supported': [e.value for e in WyckoffEvent if e != WyckoffEvent.NONE],
                'phases_supported': [p.value for p in WyckoffPhase if p != WyckoffPhase.UNKNOWN]
            }
        except Exception as e:
            self.logger.error(f"Strategy info generation failed: {str(e)}")
            return {'error': str(e)}

    # Helper for get_strategy_info to handle uninitialized performance
    def _create_empty_performance_metrics(self):
        from src.core.base import StrategyPerformance
        return StrategyPerformance(
            strategy_name="wyckoff",
            win_rate=0.0,
            profit_factor=0.0
        )

if __name__ == "__main__":
    """Test the Wyckoff strategy"""
    
    # Test configuration
    test_config = {
        'parameters': {
            'lookback_period': 150,
            'swing_detection_window': 10,
            'phase_confirmation_bars': 3,
            'confidence_threshold': 0.65,
            'min_strength': 0.5,
            'cooldown_bars': 3
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
            
            # Create synthetic data with clear Wyckoff patterns
            for i in range(bars):
                if i < bars//3:  # Downtrend to accumulation
                    base_price -= 0.5
                    prices.append(base_price + np.random.normal(0, 1.5))
                elif i < 2*bars//3:  # Range-bound accumulation
                    prices.append(base_price + np.random.normal(0, 2))
                else:  # Breakout (SOS)
                    base_price += 0.7
                    prices.append(base_price + np.random.normal(0, 1.5))
            
            data = pd.DataFrame({
                'Open': prices + np.random.normal(0, 0.5, bars),
                'High': np.array(prices) + np.abs(np.random.normal(2, 1, bars)),
                'Low': np.array(prices) - np.abs(np.random.normal(2, 1, bars)),
                'Close': prices,
                'Volume': np.random.randint(500, 1500, bars)
            }, index=dates)
            
            return data
    
    # Create strategy instance
    mock_mt5 = MockMT5Manager()
    strategy = WyckoffStrategy(test_config, mock_mt5)
    
    print("="*60)
    print("TESTING WYCKOFF STRATEGY")
    print("="*60)
    
    # Test signal generation
    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
        print(f"     Event: {signal.metadata.get('event', 'Unknown')}")
        print(f"     Phase: {signal.metadata.get('phase', 'Unknown')}")
    
    # Test analysis
    print("\n2. Testing analysis method:")
    mock_data = mock_mt5.get_historical_data("XAUUSDm", "M15", 150)
    analysis = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis keys: {list(analysis.keys())}")
    print(f"   Current Phase: {analysis['current_phase']}")
    print(f"   Detected Events: {len(analysis['detected_events'])}")
    if analysis['detected_events']:
        print(f"   Latest Event: {analysis['detected_events'][-1]['event']}")
    
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
    print(f"   Events Supported: {', '.join(strategy_info['events_supported'])}")
    print(f"   Phases Supported: {', '.join(strategy_info['phases_supported'])}")
    print(f"   Lookback Period: {strategy_info['parameters']['lookback_period']}")
    print(f"   Confidence Threshold: {strategy_info['parameters']['confidence_threshold']:.2f}")
    print(f"   Performance Summary:")
    print(f"     Success Rate: {strategy_info['performance']['success_rate']:.2%}")
    print(f"     Profit Factor: {strategy_info['performance']['profit_factor']:.2f}")
    
    print("\n" + "="*60)
    print("WYCKOFF STRATEGY TEST COMPLETED!")
    print("="*60)
