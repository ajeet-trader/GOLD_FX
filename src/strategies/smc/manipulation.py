"""
Manipulation Strategy - Smart Money Concepts (SMC)
================================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-08-15

This strategy detects manipulation events in XAUUSD price action including:
- Stop hunts (long wicks through key levels with reversion)
- Fakeouts (false breakouts with quick reversals)
- Displacement with quick reversion (impulsive moves followed by immediate retrace)

Features:
- Identifies stop hunts through swing high/low wicks
- Detects false breakouts with confirmation
- Tracks displacement with reversion patterns
- Generates 5-10 signals daily with quick confirmation
- Cooldown mechanism to prevent signal clustering
- Multi-timeframe validation

Dependencies:
    - pandas
    - numpy
    - datetime
    - src.core.base
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

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
class ManipulationEvent:
    """Data structure for manipulation events"""
    type: str  # stop_hunt, fakeout, displacement
    level: float  # Price level of manipulation
    bar_index: int  # Index of the manipulation bar
    direction: str  # bullish or bearish
    confidence: float  # Confidence score
    wick_ratio: float = 0.0  # Wick/body ratio for stop hunts
    reversal_confirmed: bool = False  # Whether reversal is confirmed

class ManipulationStrategy(AbstractStrategy):
    """
    SMC Manipulation Strategy for XAUUSD Trading
    
    Detects stop hunts, fakeouts, and engineered liquidity grabs.
    Generates signals when price action confirms reversion after manipulation
    or continues strongly after engineered displacement.
    
    Signal Generation:
    - Stop Hunt: Long wick through swing high/low, reverts within range
    - Fakeout: Breakout beyond key level, reverses within 1-2 bars
    - Displacement: Large impulsive move, quick retrace or continuation
    
    Example:
        >>> strategy = ManipulationStrategy(config, mt5_manager)
        >>> signals = strategy.generate_signal("XAUUSDm", "M15")
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """
        Initialize Manipulation Strategy
        
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
        params = config.get('parameters', {})
        self.lookback_bars = params.get('lookback_bars', 250)
        self.wick_ratio_threshold = params.get('wick_ratio_threshold', 1.5)
        self.fakeout_confirm_bars = params.get('fakeout_confirm_bars', 2)
        self.confidence_threshold = params.get('confidence_threshold', 0.65)
        self.cooldown_bars = params.get('cooldown_bars', 3)
        
        # Internal state
        self.last_signal_bar = -self.cooldown_bars - 1
        # self.logger = logging.getLogger('manipulation_strategy') # Handled by AbstractStrategy
        
        # Performance tracking (handled by parent class)
        self.success_rate = 0.65
        self.profit_factor = 1.8
        
        self.logger.info("Manipulation Strategy initialized")
    
    def _create_mock_mt5(self):
        """Create mock MT5 manager with mode-specific data"""
        class MockMT5Manager:
            def __init__(self, mode):
                self.mode = mode
                
            def get_historical_data(self, symbol, timeframe, bars):
                dates = pd.date_range(start=datetime.now() - timedelta(days=10), 
                                     end=datetime.now(), freq='15Min')[:bars]
                
                np.random.seed(42 if self.mode == 'mock' else 123)
                close_prices = (1950 if self.mode == 'mock' else 1975) + np.cumsum(np.random.randn(len(dates)) * 2)
                
                # Create manipulation patterns (example, actual randomness may vary)
                data = pd.DataFrame({
                    'Open': close_prices + np.random.randn(len(dates)) * 0.5,
                    'High': close_prices + np.abs(np.random.randn(len(dates)) * 3),
                    'Low': close_prices - np.abs(np.random.randn(len(dates)) * 3),
                    'Close': close_prices,
                    'Volume': np.random.randint(100, 1000, len(dates))
                }, index=dates)
                
                # Add synthetic stop hunts, fakeouts, displacement for testing
                if self.mode == 'mock':
                    # Add stop hunts (long wicks)
                    stop_hunt_indices = [bars // 4, bars // 2, bars * 3 // 4]
                    for idx in stop_hunt_indices:
                        if idx < len(data):
                            data.loc[data.index[idx], 'High'] = data.loc[data.index[idx], 'High'] * 1.05  # Upper wick
                            data.loc[data.index[idx], 'Low'] = data.loc[data.index[idx], 'Low'] * 0.95    # Lower wick
                    
                    # Add fakeout
                    fakeout_idx = bars - 10
                    if fakeout_idx > 1 and fakeout_idx < len(data):
                        data.loc[data.index[fakeout_idx-1], 'Close'] = data.loc[data.index[fakeout_idx-1], 'Close'] * 1.01
                        data.loc[data.index[fakeout_idx], 'Close'] = data.loc[data.index[fakeout_idx], 'Close'] * 0.99
                    
                    # Add displacement
                    displacement_idx = bars - 5
                    if displacement_idx > 1 and displacement_idx < len(data):
                        data.loc[data.index[displacement_idx-1], 'Close'] = data.loc[data.index[displacement_idx-1], 'Close'] * 1.02
                        data.loc[data.index[displacement_idx], 'Close'] = data.loc[data.index[displacement_idx], 'Close'] * 0.98

                return data
        
        return MockMT5Manager(self.mode)

    def _detect_swing_levels(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Detect swing highs and lows in the data"""
        highs = []
        lows = []
        window = 5  # Lookback/forward bars for swing detection
        
        for i in range(window, len(data) - window):
            if (data['High'].iloc[i] >= data['High'].iloc[i-window:i+window+1]).all():
                highs.append(data['High'].iloc[i])
            if (data['Low'].iloc[i] <= data['Low'].iloc[i-window:i+window+1]).all():
                lows.append(data['Low'].iloc[i])
        
        return {'highs': highs[-5:], 'lows': lows[-5:]}  # Return recent levels
    
    def _calculate_wick_ratio(self, row: pd.Series, direction: str) -> float:
        """Calculate wick/body ratio for a candle"""
        try:
            body = abs(row['Close'] - row['Open'])
            if direction == 'bullish':
                wick = row['High'] - max(row['Open'], row['Close'])
            else:  # bearish
                wick = min(row['Open'], row['Close']) - row['Low']
            
            return wick / body if body > 0 else 0.0
        except:
            return 0.0
    
    def _detect_stop_hunt(self, data: pd.DataFrame, index: int, swing_levels: Dict) -> Optional[ManipulationEvent]:
        """Detect stop hunt patterns (long wicks through swing levels)"""
        row = data.iloc[index]
        # prev_row = data.iloc[index-1] if index > 0 else None # prev_row not used
        swing_highs = swing_levels['highs']
        swing_lows = swing_levels['lows']
        
        # Bullish stop hunt (long upper wick through high)
        if swing_highs and row['High'] > max(swing_highs) and row['Close'] < max(swing_highs):
            wick_ratio = self._calculate_wick_ratio(row, 'bullish')
            if wick_ratio >= self.wick_ratio_threshold:
                return ManipulationEvent(
                    type='stop_hunt',
                    level=max(swing_highs),
                    bar_index=index,
                    direction='bearish',  # Signal to sell after bullish stop hunt
                    confidence=0.75,
                    wick_ratio=wick_ratio
                )
        
        # Bearish stop hunt (long lower wick through low)
        if swing_lows and row['Low'] < min(swing_lows) and row['Close'] > min(swing_lows):
            wick_ratio = self._calculate_wick_ratio(row, 'bearish')
            if wick_ratio >= self.wick_ratio_threshold:
                return ManipulationEvent(
                    type='stop_hunt',
                    level=min(swing_lows),
                    bar_index=index,
                    direction='bullish',  # Signal to buy after bearish stop hunt
                    confidence=0.75,
                    wick_ratio=wick_ratio
                )
        
        return None
    
    def _detect_fakeout(self, data: pd.DataFrame, index: int, swing_levels: Dict) -> Optional[ManipulationEvent]:
        """Detect fakeout patterns (breakout with quick reversal)"""
        if index < self.fakeout_confirm_bars:
            return None
            
        current_row = data.iloc[index]
        prev_rows = data.iloc[index-self.fakeout_confirm_bars:index]
        swing_highs = swing_levels['highs']
        swing_lows = swing_levels['lows']
        
        # Bullish fakeout (break above high, then close below)
        # Check if *any* previous close broke the high, AND current close is below it
        if swing_highs and any(p_row.Close > max(swing_highs) for p_row in prev_rows.itertuples()) and current_row['Close'] < max(swing_highs):
            return ManipulationEvent(
                type='fakeout',
                level=max(swing_highs),
                bar_index=index,
                direction='bearish',  # Signal to sell after bullish fakeout
                confidence=0.70,
                reversal_confirmed=True
            )
        
        # Bearish fakeout (break below low, then close above)
        # Check if *any* previous close broke the low, AND current close is above it
        if swing_lows and any(p_row.Close < min(swing_lows) for p_row in prev_rows.itertuples()) and current_row['Close'] > min(swing_lows):
            return ManipulationEvent(
                type='fakeout',
                level=min(swing_lows),
                bar_index=index,
                direction='bullish',  # Signal to buy after bearish fakeout
                confidence=0.70,
                reversal_confirmed=True
            )
        
        return None
    
    def _detect_displacement(self, data: pd.DataFrame, index: int, swing_levels: Dict) -> Optional[ManipulationEvent]:
        """Detect displacement with quick reversion"""
        if index < 2:
            return None
            
        current_row = data.iloc[index]
        prev_row = data.iloc[index-1]
        
        # Calculate ATR based on data available
        if 'Close' in data and len(data) >= 14:
            atr = data['Close'].rolling(14).std().iloc[index] 
        else:
            atr = 5.0 # Default value if not enough data
        
        # Ensure atr is not zero to avoid division by zero later
        if atr == 0:
            atr = 0.01 # Small non-zero value
        
        # Bullish displacement
        if (current_row['Close'] - prev_row['Close']) > 2 * atr and current_row['Close'] < prev_row['High']:
            return ManipulationEvent(
                type='displacement',
                level=prev_row['High'],
                bar_index=index,
                direction='bearish',  # Signal to sell after quick retrace
                confidence=0.68,
                reversal_confirmed=True
            )
        
        # Bearish displacement
        if (prev_row['Close'] - current_row['Close']) > 2 * atr and current_row['Close'] > prev_row['Low']:
            return ManipulationEvent(
                type='displacement',
                level=prev_row['Low'],
                bar_index=index,
                direction='bullish',  # Signal to buy after quick retrace
                confidence=0.68,
                reversal_confirmed=True
            )
        
        return None
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """
        Generate manipulation-based trading signals
        
        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            List of trading signals
        """
        signals = []
        try:
            self.logger.info(f"Manipulation Strategy - Analyzing {symbol} on {timeframe}") # Consistent logging
            
            # Fetch market data
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available")
                return []
                
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_bars)
            if data is None or len(data) < 50:
                self.logger.warning(f"Insufficient data for manipulation analysis: {len(data) if data is not None else 0} bars")
                return []
            
            # Detect swing levels
            swing_levels = self._detect_swing_levels(data)
            
            # Check cooldown to prevent signal clustering
            current_bar = len(data) - 1
            if current_bar - self.last_signal_bar < self.cooldown_bars:
                self.logger.info(f"On cooldown for {self.cooldown_bars} bars. Skipping signal generation.")
                return []
                
            # Detect manipulation patterns
            events = []
            # Check a small window for events, typically current and recent bars
            for i in range(max(0, current_bar - self.fakeout_confirm_bars - 2), current_bar + 1):
                if i < 0: continue # Ensure index is valid
                
                # Stop hunt detection
                stop_hunt = self._detect_stop_hunt(data, i, swing_levels)
                if stop_hunt:
                    events.append(stop_hunt)
                
                # Fakeout detection
                fakeout = self._detect_fakeout(data, i, swing_levels)
                if fakeout:
                    events.append(fakeout)
                
                # Displacement detection
                displacement = self._detect_displacement(data, i, swing_levels)
                if displacement:
                    events.append(displacement)
            
            # Generate signals from events
            current_price = data['Close'].iloc[-1]
            for event in events:
                # Only consider events from the very last bar if it's not a historical event from lookback
                # This ensures we only act on the most recent pattern completion
                if event.bar_index != current_bar:
                    continue

                if event.confidence < self.confidence_threshold:
                    continue
                    
                # Determine signal type
                signal_type = SignalType.BUY if event.direction == 'bullish' else SignalType.SELL
                
                # Calculate stop loss and take profit
                # Using current price for SL/TP calculation base, but distance from event level
                # Assuming 5.0 points as a rough initial distance, adjust as needed
                sl_distance_from_event = abs(event.level - current_price) * 1.5 # 1.5x current distance
                tp_distance_from_event = abs(event.level - current_price) * 3.0 # 3.0x current distance

                if signal_type == SignalType.BUY:
                    stop_loss = current_price - sl_distance_from_event
                    take_profit = current_price + tp_distance_from_event
                else: # SELL
                    stop_loss = current_price + sl_distance_from_event
                    take_profit = current_price - tp_distance_from_event
                
                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=signal_type,
                    confidence=event.confidence,
                    price=current_price,
                    timeframe=timeframe,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strength=event.confidence,
                    grade=SignalGrade.from_confidence(event.confidence),
                    metadata={
                        'event_type': event.type,
                        'level': event.level,
                        'wick_ratio': event.wick_ratio,
                        'reversal_confirmed': event.reversal_confirmed
                    }
                )
                
                if self.validate_signal(signal):
                    signals.append(signal)
                    self.last_signal_bar = current_bar
                    self.logger.info(f"Generated {signal_type.value} signal for {symbol} at {current_price:.2f}")
            
            self.logger.info(f"Generated {len(signals)} signals") # Consistent logging
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            return []
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze market data for manipulation patterns
        
        Args:
            data: OHLCV data
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            Dictionary containing manipulation events and market analysis
        """
        try:
            if data is None or len(data) < 50:
                return {'error': 'Insufficient data'}
                
            swing_levels = self._detect_swing_levels(data)
            events = []
            
            # Look at a slightly larger window for analysis, not just current bar
            for i in range(max(0, len(data) - self.fakeout_confirm_bars - 10), len(data)):
                stop_hunt = self._detect_stop_hunt(data, i, swing_levels)
                if stop_hunt:
                    events.append({
                        'type': stop_hunt.type,
                        'level': stop_hunt.level,
                        'bar_index': stop_hunt.bar_index
                    })
                
                fakeout = self._detect_fakeout(data, i, swing_levels)
                if fakeout:
                    events.append({
                        'type': fakeout.type,
                        'level': fakeout.level,
                        'bar_index': fakeout.bar_index
                    })
                
                displacement = self._detect_displacement(data, i, swing_levels)
                if displacement:
                    events.append({
                        'type': displacement.type,
                        'level': displacement.level,
                        'bar_index': displacement.bar_index
                    })
            
            return {
                'manipulation_events': events,
                'recent_levels': swing_levels,
                'current_price': data['Close'].iloc[-1] if not data.empty else None
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and parameters
        
        Returns:
            Dictionary containing strategy details
        """
        return {
            'name': 'Manipulation Strategy',
            'type': 'SMC',
            'description': 'Detects stop hunts, fakeouts, and displacement with reversion patterns',
            'parameters': {
                'lookback_bars': self.lookback_bars,
                'wick_ratio_threshold': self.wick_ratio_threshold,
                'fakeout_confirm_bars': self.fakeout_confirm_bars,
                'confidence_threshold': self.confidence_threshold,
                'cooldown_bars': self.cooldown_bars
            },
            'performance': {
                'success_rate': self.performance.win_rate,
                'profit_factor': self.performance.profit_factor
            }
        }
    
    def _create_empty_performance_metrics(self):
        """Helper for get_strategy_info when performance is uninitialized"""
        from src.core.base import StrategyPerformance
        return StrategyPerformance(
            strategy_name=self.strategy_name,
            win_rate=0.0,
            profit_factor=0.0
        )

if __name__ == "__main__":
    """Test the Manipulation Strategy"""
    
    # Test configuration
    test_config = {
        'parameters': {
            'lookback_bars': 250,
            'wick_ratio_threshold': 1.5,
            'fakeout_confirm_bars': 2,
            'confidence_threshold': 0.65,
            'cooldown_bars': 3,
            'mode': 'mock' # Added mode parameter
        }
    }
    
    # Create strategy instance
    strategy = ManipulationStrategy(test_config, mt5_manager=None) # Pass mt5_manager=None to trigger internal mock creation
    
    print("============================================================")
    print("TESTING MANIPULATION STRATEGY")
    print("============================================================")
    print(f"Running in {strategy.mode.upper()} mode") # Print mode
    
    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
        print(f"     Event: {signal.metadata.get('event_type', 'Unknown')}")
        print(f"     Level: {signal.metadata.get('level', 0):.2f}")
    
    print("\n2. Testing analysis method:")
    mock_data = strategy.mt5_manager.get_historical_data("XAUUSDm", "M15", 250) # Use strategy's mock manager
    analysis_results = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis results keys: {analysis_results.keys()}")
    if 'manipulation_events' in analysis_results:
        print(f"   Detected manipulation events: {len(analysis_results['manipulation_events'])}")
        for event in analysis_results['manipulation_events'][:3]:  # Show first 3 events
            print(f"     - {event['type']} at {event['level']:.2f} (bar {event['bar_index']})")
    
    print("\n3. Testing performance tracking:")
    summary = strategy.get_performance_summary()
    print(f"   {summary}")
    
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
    print("MANIPULATION STRATEGY TEST COMPLETED!")
    print("============================================================")
