"""
Market Profile Strategy - Advanced Technical Analysis
===================================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-08-15

Market Profile visualizes price distribution over time, identifying:
- Point of Control (POC): Price with highest time-based TPO count
- Value Area High (VAH) & Low (VAL): Prices covering value_area_pct of TPOs
- Initial Balance (IB): High/low of first ib_period minutes after session open
- Day types: Trend, normal, or neutral

This strategy generates signals by trading:
- Breakouts above VAH/below VAL with momentum confirmation
- Rotations back to POC after failed breakouts
- IB breakouts (above IB High = buy, below IB Low = sell)
- Reversals at VAH/VAL after multiple touches

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
from enum import Enum

from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade


class DayType(Enum):
    """Market Profile day type classification"""
    TREND = "TREND"
    NORMAL = "NORMAL"
    NEUTRAL = "NEUTRAL"


@dataclass
class MarketProfileComponents:
    """Market Profile indicator components"""
    poc: float
    vah: float
    val: float
    ib_high: float
    ib_low: float
    day_type: DayType
    position_vs_value_area: str  # "above", "within", "below"
    tpo_distribution: Dict[float, int]


class MarketProfileStrategy(AbstractStrategy):
    """
    Advanced Market Profile Strategy
    
    This strategy implements comprehensive Market Profile analysis including:
    - TPO-based price distribution
    - Value Area calculation
    - Initial Balance breakouts
    - Day type classification
    - Breakout and rotation signals
    
    Signal Generation:
    - Breakout Buy: Price breaks above VAH or IB High with momentum
    - Breakout Sell: Price breaks below VAL or IB Low with momentum
    - Rotation: Price returns to POC after failed breakout
    - Reversal: Multiple touches of VAH/VAL with rejection
    
    Example:
        >>> strategy = MarketProfileStrategy(config, mt5_manager)
        >>> signals = strategy.generate_signal("XAUUSDm", "M15")
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """
        Initialize Market Profile strategy
        
        Args:
            config: Strategy configuration
            mt5_manager: MT5 connection manager (optional)
            database: Database manager (optional)
        """
        super().__init__(config, mt5_manager, database)
        
        # Market Profile parameters
        self.lookback_period = config.get('parameters', {}).get('lookback_period', 200)
        self.value_area_pct = config.get('parameters', {}).get('value_area_pct', 0.7)
        self.ib_period = config.get('parameters', {}).get('ib_period', 60)  # minutes
        self.confidence_threshold = config.get('parameters', {}).get('confidence_threshold', 0.65)
        self.min_price_distance = config.get('parameters', {}).get('min_price_distance', 0.15)
        self.breakout_buffer = config.get('parameters', {}).get('breakout_buffer', 0.001)  # 0.1%
        
        # Signal filters
        self.min_momentum_bars = 3  # Bars for momentum confirmation
        self.max_signals_per_session = 3  # Max signals per trading session
        self.price_precision = 2  # XAUUSD price precision
        
        # Performance tracking
        self.success_rate = 0.65
        self.profit_factor = 1.8
        
        self.logger.info("Market Profile Strategy initialized")
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """
        Generate Market Profile-based trading signals
        
        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            List of trading signals
        """
        try:
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available")
                return []
                
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_period)
            if data is None or len(data) < 50:
                self.logger.warning(f"Insufficient data for Market Profile: {len(data) if data is not None else 0} bars")
                return []
            
            # Calculate Market Profile components
            profile = self._calculate_market_profile(data)
            if not profile:
                return []
            
            current_price = data['Close'].iloc[-1]
            signals = []
            
            # Breakout signals (VAH/VAL and IB)
            breakout_signals = self._check_breakout_signals(data, profile, symbol, timeframe, current_price)
            signals.extend(breakout_signals)
            
            # Rotation signals to POC
            rotation_signals = self._check_rotation_signals(data, profile, symbol, timeframe, current_price)
            signals.extend(rotation_signals)
            
            # Reversal signals at VAH/VAL
            reversal_signals = self._check_reversal_signals(data, profile, symbol, timeframe, current_price)
            signals.extend(reversal_signals)
            
            # Validate and filter signals
            validated_signals = [signal for signal in signals if self.validate_signal(signal)]
            
            # Limit signals per session
            session = self._get_current_session()
            if len(validated_signals) > self.max_signals_per_session:
                validated_signals = validated_signals[:self.max_signals_per_session]
                self.logger.info(f"Limited signals to {self.max_signals_per_session} for {session} session")
            
            return validated_signals
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            return []
    
    def _calculate_market_profile(self, data: pd.DataFrame) -> Optional[MarketProfileComponents]:
        """
        Calculate Market Profile components
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            MarketProfileComponents or None if calculation fails
        """
        try:
            # Calculate TPO distribution
            price_step = 0.1  # Smallest price increment for TPO
            prices = np.round(data[['Open', 'High', 'Low', 'Close']].stack(), self.price_precision)
            tpo_counts = pd.value_counts(prices).to_dict()
            
            if not tpo_counts:
                return None
                
            # Find POC (price with highest TPO count)
            poc = max(tpo_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate Value Area (VAH/VAL covering value_area_pct of TPOs)
            total_tpos = sum(tpo_counts.values())
            target_tpos = total_tpos * self.value_area_pct
            sorted_prices = sorted(tpo_counts.keys())
            
            cumulative_tpos = 0
            vah = val = poc
            for price in sorted_prices:
                if price >= poc:
                    cumulative_tpos += tpo_counts.get(price, 0)
                    if cumulative_tpos >= target_tpos / 2:
                        vah = price
                        break
            cumulative_tpos = 0
            for price in sorted_prices[::-1]:
                if price <= poc:
                    cumulative_tpos += tpo_counts.get(price, 0)
                    if cumulative_tpos >= target_tpos / 2:
                        val = price
                        break
            
            # Calculate Initial Balance (first ib_period minutes)
            session_start = data.index[-1].replace(hour=0, minute=0, second=0, microsecond=0)
            ib_end = session_start + timedelta(minutes=self.ib_period)
            ib_data = data[(data.index >= session_start) & (data.index <= ib_end)]
            
            ib_high = ib_data['High'].max() if not ib_data.empty else poc
            ib_low = ib_data['Low'].min() if not ib_data.empty else poc
            
            # Classify day type (simplified)
            price_range = data['High'].max() - data['Low'].min()
            avg_range = (data['High'] - data['Low']).mean()
            day_type = DayType.TREND if price_range > 1.5 * avg_range else DayType.NORMAL
            
            # Determine price position vs value area
            current_price = data['Close'].iloc[-1]
            position_vs_value_area = (
                "above" if current_price > vah else
                "below" if current_price < val else
                "within"
            )
            
            return MarketProfileComponents(
                poc=poc,
                vah=vah,
                val=val,
                ib_high=ib_high,
                ib_low=ib_low,
                day_type=day_type,
                position_vs_value_area=position_vs_value_area,
                tpo_distribution=tpo_counts
            )
            
        except Exception as e:
            self.logger.error(f"Market Profile calculation failed: {str(e)}")
            return None
    
    def _check_breakout_signals(self, data: pd.DataFrame, profile: MarketProfileComponents,
                              symbol: str, timeframe: str, current_price: float) -> List[Signal]:
        """Check for breakout signals above VAH or below VAL and IB"""
        signals = []
        try:
            # Calculate momentum
            momentum = self._calculate_momentum(data)
            buffer = self.breakout_buffer * current_price
            
            # VAH breakout (buy)
            if (current_price > profile.vah + buffer and
                profile.position_vs_value_area == "within" and
                momentum > 0):
                signals.append(Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name="market_profile",
                    signal_type=SignalType.BUY,
                    confidence=self.confidence_threshold + 0.1,
                    price=current_price,
                    timeframe=timeframe,
                    stop_loss=profile.vah,
                    take_profit=current_price + 2 * (current_price - profile.vah),
                    metadata={'pattern': 'vah_breakout', 'day_type': profile.day_type.value}
                ))
            
            # VAL breakout (sell)
            if (current_price < profile.val - buffer and
                profile.position_vs_value_area == "within" and
                momentum < 0):
                signals.append(Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name="market_profile",
                    signal_type=SignalType.SELL,
                    confidence=self.confidence_threshold + 0.1,
                    price=current_price,
                    timeframe=timeframe,
                    stop_loss=profile.val,
                    take_profit=current_price - 2 * (profile.val - current_price),
                    metadata={'pattern': 'val_breakout', 'day_type': profile.day_type.value}
                ))
            
            # IB breakout (buy)
            if (current_price > profile.ib_high + buffer and
                momentum > 0):
                signals.append(Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name="market_profile",
                    signal_type=SignalType.BUY,
                    confidence=self.confidence_threshold + 0.05,
                    price=current_price,
                    timeframe=timeframe,
                    stop_loss=profile.ib_high,
                    take_profit=current_price + 2 * (current_price - profile.ib_high),
                    metadata={'pattern': 'ib_breakout', 'day_type': profile.day_type.value}
                ))
            
            # IB breakout (sell)
            if (current_price < profile.ib_low - buffer and
                momentum < 0):
                signals.append(Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name="market_profile",
                    signal_type=SignalType.SELL,
                    confidence=self.confidence_threshold + 0.05,
                    price=current_price,
                    timeframe=timeframe,
                    stop_loss=profile.ib_low,
                    take_profit=current_price - 2 * (profile.ib_low - current_price),
                    metadata={'pattern': 'ib_breakout', 'day_type': profile.day_type.value}
                ))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Breakout signal generation failed: {str(e)}")
            return []
    
    def _check_rotation_signals(self, data: pd.DataFrame, profile: MarketProfileComponents,
                              symbol: str, timeframe: str, current_price: float) -> List[Signal]:
        """Check for rotation signals back to POC after failed breakouts"""
        signals = []
        try:
            buffer = self.breakout_buffer * current_price
            recent_high = data['High'].iloc[-5:].max()
            recent_low = data['Low'].iloc[-5:].min()
            
            # Rotation to POC after failed VAH breakout
            if (recent_high > profile.vah + buffer and
                current_price <= profile.poc + self.min_price_distance and
                current_price > profile.val):
                signals.append(Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name="market_profile",
                    signal_type=SignalType.SELL,
                    confidence=self.confidence_threshold,
                    price=current_price,
                    timeframe=timeframe,
                    stop_loss=profile.vah,
                    take_profit=profile.poc,
                    metadata={'pattern': 'poc_rotation', 'day_type': profile.day_type.value}
                ))
            
            # Rotation to POC after failed VAL breakout
            if (recent_low < profile.val - buffer and
                current_price >= profile.poc - self.min_price_distance and
                current_price < profile.vah):
                signals.append(Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name="market_profile",
                    signal_type=SignalType.BUY,
                    confidence=self.confidence_threshold,
                    price=current_price,
                    timeframe=timeframe,
                    stop_loss=profile.val,
                    take_profit=profile.poc,
                    metadata={'pattern': 'poc_rotation', 'day_type': profile.day_type.value}
                ))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Rotation signal generation failed: {str(e)}")
            return []
    
    def _check_reversal_signals(self, data: pd.DataFrame, profile: MarketProfileComponents,
                               symbol: str, timeframe: str, current_price: float) -> List[Signal]:
        """Check for reversal signals at VAH/VAL after multiple touches"""
        signals = []
        try:
            buffer = self.breakout_buffer * current_price
            recent_prices = data['Close'].iloc[-10:]
            vah_touches = sum(1 for p in recent_prices if abs(p - profile.vah) <= buffer)
            val_touches = sum(1 for p in recent_prices if abs(p - profile.val) <= buffer)
            
            # Reversal at VAH (buy)
            if vah_touches >= 2 and abs(current_price - profile.vah) <= buffer:
                signals.append(Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name="market_profile",
                    signal_type=SignalType.BUY,
                    confidence=self.confidence_threshold + 0.05,
                    price=current_price,
                    timeframe=timeframe,
                    stop_loss=profile.val,
                    take_profit=profile.poc,
                    metadata={'pattern': 'vah_reversal', 'day_type': profile.day_type.value}
                ))
            
            # Reversal at VAL (sell)
            if val_touches >= 2 and abs(current_price - profile.val) <= buffer:
                signals.append(Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name="market_profile",
                    signal_type=SignalType.SELL,
                    confidence=self.confidence_threshold + 0.05,
                    price=current_price,
                    timeframe=timeframe,
                    stop_loss=profile.vah,
                    take_profit=profile.poc,
                    metadata={'pattern': 'val_reversal', 'day_type': profile.day_type.value}
                ))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Reversal signal generation failed: {str(e)}")
            return []
    
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate price momentum for breakout confirmation"""
        try:
            if len(data) < self.min_momentum_bars:
                return 0.0
            recent_prices = data['Close'].iloc[-self.min_momentum_bars:]
            momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            return momentum
        except:
            return 0.0
    
    def _get_current_session(self) -> str:
        """Get current trading session"""
        from src.core.base import TradingSession
        return TradingSession.get_current_session().value
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze market profile for given data
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            Dictionary with Market Profile analysis results
        """
        try:
            profile = self._calculate_market_profile(data)
            if not profile:
                return {}
                
            current_price = data['Close'].iloc[-1] if not data.empty else 0.0
            
            return {
                'poc': profile.poc,
                'vah': profile.vah,
                'val': profile.val,
                'ib_high': profile.ib_high,
                'ib_low': profile.ib_low,
                'day_type': profile.day_type.value,
                'current_price': current_price,
                'position_vs_value_area': profile.position_vs_value_area,
                'tpo_distribution': profile.tpo_distribution
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {}
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and parameters
        
        Returns:
            Dictionary with strategy details
        """
        return {
            'name': 'Market Profile Strategy',
            'type': 'Technical',
            'description': 'Trades breakouts, rotations, and reversals based on Market Profile analysis',
            'parameters': {
                'lookback_period': self.lookback_period,
                'value_area_pct': self.value_area_pct,
                'ib_period': self.ib_period,
                'confidence_threshold': self.confidence_threshold,
                'min_price_distance': self.min_price_distance,
                'breakout_buffer': self.breakout_buffer
            },
            'performance': {
                'success_rate': self.performance.win_rate,
                'profit_factor': self.performance.profit_factor
            }
        }


# Testing function
if __name__ == "__main__":
    """Test the Market Profile strategy"""
    
    # Test configuration
    test_config = {
        'parameters': {
            'lookback_period': 200,
            'value_area_pct': 0.7,
            'ib_period': 60,
            'confidence_threshold': 0.65,
            'min_price_distance': 0.15,
            'breakout_buffer': 0.001
        }
    }
    
    # Mock MT5 manager for testing
    class MockMT5Manager:
        def get_historical_data(self, symbol, timeframe, bars):
            # Generate sample data with clear patterns
            dates = pd.date_range(start=datetime.now() - timedelta(days=10), 
                                end=datetime.now(), freq='15Min')[:bars]
            
            np.random.seed(42)
            base_price = 1950
            prices = []
            
            # Create synthetic data with clear IB and VAH/VAL patterns
            for i in range(len(dates)):
                if i < 24:  # First 6 hours (IB period)
                    price = base_price + np.random.normal(0, 2)  # Tight IB range
                elif i < 96:  # Middle of day (value area)
                    price = base_price + 5 + np.random.normal(0, 3)  # Value area
                else:  # Breakout potential
                    price = base_price + 10 + np.random.normal(0, 5)  # Breakout zone
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
    strategy = MarketProfileStrategy(test_config, mock_mt5)
    
    print("="*60)
    print("TESTING MARKET PROFILE STRATEGY")
    print("="*60)
    
    # Test signal generation
    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
        print(f"     Pattern: {signal.metadata.get('pattern', 'Unknown')}")
    
    # Test analysis
    print("\n2. Testing analysis method:")
    mock_data = mock_mt5.get_historical_data("XAUUSDm", "M15", 200)
    analysis = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis keys: {list(analysis.keys())}")
    if analysis:
        print(f"   POC: {analysis.get('poc', 'N/A'):.2f}")
        print(f"   VAH: {analysis.get('vah', 'N/A'):.2f}")
        print(f"   VAL: {analysis.get('val', 'N/A'):.2f}")
        print(f"   IB High: {analysis.get('ib_high', 'N/A'):.2f}")
        print(f"   IB Low: {analysis.get('ib_low', 'N/A'):.2f}")
        print(f"   Day Type: {analysis.get('day_type', 'N/A')}")
    
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
    print(f"   Parameters: {strategy_info['parameters']}")
    print(f"   Performance Summary:")
    print(f"     Success Rate: {strategy_info['performance']['success_rate']:.2%}")
    print(f"     Profit Factor: {strategy_info['performance']['profit_factor']:.2f}")
    
    print("\n" + "="*60)
    print("MARKET PROFILE STRATEGY TEST COMPLETED!")
    print("="*60)
