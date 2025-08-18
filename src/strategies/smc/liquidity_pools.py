"""
Liquidity Pools Strategy - Smart Money Concepts
==============================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-08-15

Identifies liquidity pools at swing highs/lows, equal highs/lows, and session highs/lows.
Generates trade signals when price sweeps these pools and reverses, or breaks through
and continues in the breakout direction.

Features:
- Swing high/low detection
- Equal highs/lows clustering
- Session high/low identification
- Sweep-reversal setups
- Break-continuation setups
- Approach-based entries
- Dynamic SL/TP calculation

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


class PoolType(Enum):
    """Liquidity pool type enumeration"""
    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"
    EQUAL_HIGHS = "EQUAL_HIGHS"
    EQUAL_LOWS = "EQUAL_LOWS"
    SESSION_HIGH = "SESSION_HIGH"
    SESSION_LOW = "SESSION_LOW"


@dataclass
class LiquidityPool:
    """Liquidity pool data structure"""
    pool_type: PoolType
    level: float
    bar_index: int
    strength: float  # Number of touches or cluster size
    last_interaction: Optional[datetime] = None
    active: bool = True


class LiquidityPoolsStrategy(AbstractStrategy):
    """
    Liquidity Pools Strategy for XAUUSD Trading
    
    Identifies liquidity pools and generates signals based on:
    - Sweep-reversal setups: Price wicks through pool then reverses
    - Break-continuation setups: Strong break through pool with pullback
    - Approach entries: Price approaches pool with momentum
    Tuned to generate 5-10 signals daily through multiple pool types and relaxed tolerances.
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """Initialize Liquidity Pools strategy"""
        super().__init__(config, mt5_manager, database)
        
        # Strategy parameters
        self.lookback_bars = self.config.get('parameters', {}).get('lookback_bars', 300)
        self.equal_highs_tolerance = self.config.get('parameters', {}).get('equal_highs_tolerance', 0.12)
        self.approach_buffer = self.config.get('parameters', {}).get('approach_buffer', 0.2)
        self.confidence_threshold = self.config.get('parameters', {}).get('confidence_threshold', 0.75)  # Increased base confidence
        self.cooldown_bars = self.config.get('parameters', {}).get('cooldown_bars', 3)
        
        # Internal state
        self.active_pools: List[LiquidityPool] = []
        self.recent_sweeps: List[Dict] = []
        self.last_signal_bar: Dict[str, int] = {}  # Track last signal per pool
        self.logger = logging.getLogger('liquidity_pools_strategy')
        
        # Performance tracking (handled by parent class)
        self.success_rate = 0.65
        self.profit_factor = 1.8
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """Generate trading signals based on liquidity pool interactions"""
        signals = []
        try:
            # Get market data
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available")
                return []
                
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_bars)
            if data is None or len(data) < 50:
                self.logger.warning(f"Insufficient data: {len(data) if data is not None else 0} bars")
                return []
            
            # Identify liquidity pools
            self.active_pools = self._identify_liquidity_pools(data)
            
            # Get current price and ATR
            current_price = data['Close'].iloc[-1]
            atr = self._calculate_atr(data)
            
            # Process each pool
            for pool in self.active_pools:
                # Skip if pool is on cooldown
                last_signal_bar = self.last_signal_bar.get(f"{pool.pool_type.value}_{pool.level}", -self.cooldown_bars - 1)
                if len(data) - 1 - last_signal_bar < self.cooldown_bars:
                    continue
                
                # Calculate signal metrics
                signal = self._evaluate_pool_interaction(data, pool, current_price, atr, symbol, timeframe)
                if signal and self.validate_signal(signal):
                    signals.append(signal)
                    self.last_signal_bar[f"{pool.pool_type.value}_{pool.level}"] = len(data) - 1
            
            # Store sweeps for analysis
            self._update_sweeps(data)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            return []
    
    def _identify_liquidity_pools(self, data: pd.DataFrame) -> List[LiquidityPool]:
        """Identify liquidity pools from price data"""
        pools = []
        
        # Swing highs/lows
        swing_pivots = self._find_swing_points(data)
        for pivot in swing_pivots:
            pool_type = PoolType.SWING_HIGH if pivot['type'] == 'high' else PoolType.SWING_LOW
            pools.append(LiquidityPool(
                pool_type=pool_type,
                level=pivot['price'],
                bar_index=pivot['index'],
                strength=pivot['strength']
            ))
        
        # Equal highs/lows
        equal_levels = self._find_equal_levels(data)
        for level in equal_levels:
            pool_type = PoolType.EQUAL_HIGHS if level['type'] == 'high' else PoolType.EQUAL_LOWS
            pools.append(LiquidityPool(
                pool_type=pool_type,
                level=level['price'],
                bar_index=level['index'],
                strength=level['count']
            ))
        
        # Session highs/lows (simplified for testing)
        session_levels = self._find_session_levels(data)
        for level in session_levels:
            pool_type = PoolType.SESSION_HIGH if level['type'] == 'high' else PoolType.SESSION_LOW
            pools.append(LiquidityPool(
                pool_type=pool_type,
                level=level['price'],
                bar_index=level['index'],
                strength=level['strength']
            ))
        
        return pools
    
    def _find_swing_points(self, data: pd.DataFrame) -> List[Dict]:
        """Find swing highs and lows"""
        pivots = []
        window = 5  # Lookback/forward bars for swing detection
        
        for i in range(window, len(data) - window):
            high = data['High'].iloc[i]
            low = data['Low'].iloc[i]
            
            # Swing high
            if all(high > data['High'].iloc[i-window:i]) and all(high > data['High'].iloc[i+1:i+window+1]):
                pivots.append({
                    'type': 'high',
                    'price': high,
                    'index': i,
                    'strength': (high - data['Low'].iloc[i-window:i+window+1].min()) / data['Close'].iloc[i]
                })
            
            # Swing low
            if all(low < data['Low'].iloc[i-window:i]) and all(low < data['Low'].iloc[i+1:i+window+1]):
                pivots.append({
                    'type': 'low',
                    'price': low,
                    'index': i,
                    'strength': (data['High'].iloc[i-window:i+window+1].max() - low) / data['Close'].iloc[i]
                })
        
        return pivots
    
    def _find_equal_levels(self, data: pd.DataFrame) -> List[Dict]:
        """Find clusters of equal highs/lows within tolerance"""
        levels = []
        tolerance = self.equal_highs_tolerance / 100  # Convert to decimal
        
        # Group highs
        highs = data['High'].values
        high_indices = np.arange(len(data))
        
        for i in range(len(highs)):
            cluster = [i]
            cluster_price = highs[i]
            
            for j in range(i + 1, len(highs)):
                if abs(highs[j] - cluster_price) / cluster_price <= tolerance:
                    cluster.append(j)
            
            if len(cluster) >= 3:  # Minimum cluster size
                levels.append({
                    'type': 'high',
                    'price': np.mean([highs[idx] for idx in cluster]),
                    'index': cluster[-1],
                    'count': len(cluster)
                })
        
        # Group lows
        lows = data['Low'].values
        for i in range(len(lows)):
            cluster = [i]
            cluster_price = lows[i]
            
            for j in range(i + 1, len(lows)):
                if abs(lows[j] - cluster_price) / cluster_price <= tolerance:
                    cluster.append(j)
            
            if len(cluster) >= 3:
                levels.append({
                    'type': 'low',
                    'price': np.mean([lows[idx] for idx in cluster]),
                    'index': cluster[-1],
                    'count': len(cluster)
                })
        
        return levels
    
    def _find_session_levels(self, data: pd.DataFrame) -> List[Dict]:
        """Find session high/low levels (simplified)"""
        # Group by day and find high/low
        daily = data.resample('D').agg({
            'High': 'max',
            'Low': 'min'
        }).dropna()
        
        levels = []
        for idx, row in daily.iterrows():
            bar_index = data.index.get_loc(data.index[data.index.date == idx.date()][-1])
            levels.append({
                'type': 'high',
                'price': row['High'],
                'index': bar_index,
                'strength': 1.0
            })
            levels.append({
                'type': 'low',
                'price': row['Low'],
                'index': bar_index,
                'strength': 1.0
            })
        
        return levels
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean().iloc[-1]
    
    def _evaluate_pool_interaction(self, data: pd.DataFrame, pool: LiquidityPool, 
                                 current_price: float, atr: float, 
                                 symbol: str, timeframe: str) -> Optional[Signal]:
        """Evaluate price interaction with a liquidity pool"""
        current_bar = len(data) - 1
        last_high = data['High'].iloc[-1]
        last_low = data['Low'].iloc[-1]
        last_close = data['Close'].iloc[-1]
        
        # Buffer in points
        buffer = atr * (self.approach_buffer / 100)
        pool_level = pool.level
        
        # Calculate stop loss and take profit
        sl_distance = atr * 1.5
        tp_distance = atr * 3.0
        
        # Sweep-reversal setup
        if pool.pool_type in [PoolType.SWING_HIGH, PoolType.EQUAL_HIGHS, PoolType.SESSION_HIGH]:
            if last_high > pool_level and last_close < pool_level:
                # Bearish reversal after sweeping high
                sl = pool_level + sl_distance
                tp = pool_level - tp_distance
                # Fixed confidence calculation: use base confidence + strength bonus
                confidence = min(0.85, self.confidence_threshold + (pool.strength * 0.1))
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    price=last_close,
                    timeframe=timeframe,
                    stop_loss=sl,
                    take_profit=tp,
                    strength=pool.strength,
                    metadata={
                        'pool_type': pool.pool_type.value,
                        'pool_level': pool_level,
                        'setup': 'sweep_reversal'
                    }
                )
            
            # Approach entry (preemptive)
            if pool_level - buffer < current_price < pool_level and last_close > data['Close'].iloc[-2]:
                sl = pool_level + sl_distance
                tp = pool_level - tp_distance
                confidence = min(0.75, (self.confidence_threshold * 0.9) + (pool.strength * 0.05))
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    price=last_close,
                    timeframe=timeframe,
                    stop_loss=sl,
                    take_profit=tp,
                    strength=pool.strength,
                    metadata={
                        'pool_type': pool.pool_type.value,
                        'pool_level': pool_level,
                        'setup': 'approach_entry'
                    }
                )
            
            # Break-continuation setup
            if last_close > pool_level + buffer and data['Close'].iloc[-2] < pool_level:
                sl = pool_level - sl_distance
                tp = pool_level + tp_distance
                confidence = min(0.80, (self.confidence_threshold * 0.95) + (pool.strength * 0.05))
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    price=last_close,
                    timeframe=timeframe,
                    stop_loss=sl,
                    take_profit=tp,
                    strength=pool.strength,
                    metadata={
                        'pool_type': pool.pool_type.value,
                        'pool_level': pool_level,
                        'setup': 'break_continuation'
                    }
                )
        
        # Similar logic for low pools
        if pool.pool_type in [PoolType.SWING_LOW, PoolType.EQUAL_LOWS, PoolType.SESSION_LOW]:
            if last_low < pool_level and last_close > pool_level:
                # Bullish reversal after sweeping low
                sl = pool_level - sl_distance
                tp = pool_level + tp_distance
                confidence = min(0.85, self.confidence_threshold + (pool.strength * 0.1))
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    price=last_close,
                    timeframe=timeframe,
                    stop_loss=sl,
                    take_profit=tp,
                    strength=pool.strength,
                    metadata={
                        'pool_type': pool.pool_type.value,
                        'pool_level': pool_level,
                        'setup': 'sweep_reversal'
                    }
                )
            
            # Approach entry (preemptive)
            if pool_level < current_price < pool_level + buffer and last_close < data['Close'].iloc[-2]:
                sl = pool_level - sl_distance
                tp = pool_level + tp_distance
                confidence = min(0.75, (self.confidence_threshold * 0.9) + (pool.strength * 0.05))
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    price=last_close,
                    timeframe=timeframe,
                    stop_loss=sl,
                    take_profit=tp,
                    strength=pool.strength,
                    metadata={
                        'pool_type': pool.pool_type.value,
                        'pool_level': pool_level,
                        'setup': 'approach_entry'
                    }
                )
            
            # Break-continuation setup
            if last_close < pool_level - buffer and data['Close'].iloc[-2] > pool_level:
                sl = pool_level + sl_distance
                tp = pool_level - tp_distance
                confidence = min(0.80, (self.confidence_threshold * 0.95) + (pool.strength * 0.05))
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    price=last_close,
                    timeframe=timeframe,
                    stop_loss=sl,
                    take_profit=tp,
                    strength=pool.strength,
                    metadata={
                        'pool_type': pool.pool_type.value,
                        'pool_level': pool_level,
                        'setup': 'break_continuation'
                    }
                )
        
        return None
    
    def _update_sweeps(self, data: pd.DataFrame) -> None:
        """Update recent sweeps tracking"""
        current_bar = len(data) - 1
        last_high = data['High'].iloc[-1]
        last_low = data['Low'].iloc[-1]
        
        for pool in self.active_pools:
            if pool.pool_type in [PoolType.SWING_HIGH, PoolType.EQUAL_HIGHS, PoolType.SESSION_HIGH]:
                if last_high > pool.level and data['Close'].iloc[-1] < pool.level:
                    self.recent_sweeps.append({
                        'level': pool.level,
                        'direction': 'up',
                        'bar_index': current_bar
                    })
            elif pool.pool_type in [PoolType.SWING_LOW, PoolType.EQUAL_LOWS, PoolType.SESSION_LOW]:
                if last_low < pool.level and data['Close'].iloc[-1] > pool.level:
                    self.recent_sweeps.append({
                        'level': pool.level,
                        'direction': 'down',
                        'bar_index': current_bar
                    })
        
        # Keep only recent sweeps
        self.recent_sweeps = [sweep for sweep in self.recent_sweeps if current_bar - sweep['bar_index'] < 20]
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze market data and return liquidity pool information"""
        try:
            self.active_pools = self._identify_liquidity_pools(data)
            self._update_sweeps(data)
            
            return {
                'pools': [
                    {
                        'type': pool.pool_type.value,
                        'level': pool.level,
                        'strength': pool.strength,
                        'active': pool.active
                    } for pool in self.active_pools
                ],
                'recent_sweeps': self.recent_sweeps,
                'current_price': data['Close'].iloc[-1] if not data.empty else None
            }
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {}
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information"""
        try:
            return {
                'name': self.strategy_name,
                'type': 'SMC',
                'description': 'Liquidity Pools Strategy identifying swing highs/lows, equal highs/lows, and session levels for sweep-reversal and break-continuation trades',
                'version': '1.0.0',
                'parameters': {
                    'lookback_bars': self.lookback_bars,
                    'equal_highs_tolerance': self.equal_highs_tolerance,
                    'approach_buffer': self.approach_buffer,
                    'confidence_threshold': self.confidence_threshold,
                    'cooldown_bars': self.cooldown_bars
                },
                'performance': {
                    'success_rate': self.performance.win_rate,
                    'profit_factor': self.performance.profit_factor
                }
            }
        except Exception as e:
            self.logger.error(f"Strategy info generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _create_empty_performance_metrics(self):
        """Create empty performance metrics for direct script runs"""
        from src.core.base import StrategyPerformance
        return StrategyPerformance(
            strategy_name=self.strategy_name,
            win_rate=0.0,
            profit_factor=0.0
        )


if __name__ == "__main__":
    """Test the Liquidity Pools strategy"""
    
    # Test configuration
    test_config = {
        'parameters': {
            'lookback_bars': 300,
            'equal_highs_tolerance': 0.12,
            'approach_buffer': 0.2,
            'confidence_threshold': 0.65,
            'cooldown_bars': 3
        }
    }
    
    class MockMT5Manager:
        def get_historical_data(self, symbol, timeframe, bars):
            # Generate sample data with equal highs/lows and sweep events
            dates = pd.date_range(start=datetime.now() - timedelta(days=10), 
                                 end=datetime.now(), freq='15Min')[:bars]
            
            np.random.seed(42)
            base_price = 1950
            prices = []
            
            # Create patterns: equal highs/lows, swings, sweeps
            for i in range(len(dates)):
                if i % 50 < 10:  # Equal highs cluster
                    price = base_price + 10 + np.random.normal(0, 0.5)
                elif i % 50 < 20:  # Equal lows cluster
                    price = base_price - 10 + np.random.normal(0, 0.5)
                elif i % 50 == 25:  # Sweep high
                    price = base_price + 12 + np.random.normal(0, 0.5)
                elif i % 50 == 35:  # Sweep low
                    price = base_price - 12 + np.random.normal(0, 0.5)
                else:
                    price = base_price + np.random.normal(0, 2)
                prices.append(price)
            
            data = pd.DataFrame({
                'Open': prices + np.random.normal(0, 0.5, len(dates)),
                'High': np.array(prices) + np.abs(np.random.normal(2, 1, len(dates))),
                'Low': np.array(prices) - np.abs(np.random.normal(2, 1, len(dates))),
                'Close': prices,
                'Volume': np.random.randint(500, 1500, len(dates))
            }, index=dates)
            
            return data
    
    # Create strategy instance
    mock_mt5 = MockMT5Manager()
    strategy = LiquidityPoolsStrategy(test_config, mock_mt5)
    
    print("============================================================")
    print("TESTING LIQUIDITY POOLS STRATEGY")
    print("============================================================")
    
    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
        print(f"     Pool Type: {signal.metadata.get('pool_type', 'Unknown')}")
        print(f"     Setup: {signal.metadata.get('setup', 'Unknown')}")
    
    print("\n2. Testing analysis method:")
    mock_data = mock_mt5.get_historical_data("XAUUSDm", "M15", 300)
    analysis_results = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis results keys: {list(analysis_results.keys())}")
    if 'pools' in analysis_results:
        print(f"   Detected pools: {len(analysis_results['pools'])}")
        for pool in analysis_results['pools'][:3]:  # Show first 3 pools
            print(f"     - {pool['type']}: {pool['level']:.2f} (Strength: {pool['strength']:.2f})")
    
    print("\n3. Testing performance tracking:")
    summary = strategy.get_performance_summary()
    print(f"   {summary}")
    
    print("\n4. Strategy Information:")
    strategy_info = strategy.get_strategy_info()
    print(f"   Name: {strategy_info['name']}")
    print(f"   Version: {strategy_info['version']}")
    print(f"   Description: {strategy_info['description']}")
    print(f"   Type: {strategy_info['type']}")
    print(f"   Parameters:")
    for key, value in strategy_info['parameters'].items():
        print(f"     {key}: {value}")
    print(f"   Performance Summary:")
    print(f"     Success Rate: {strategy_info['performance']['success_rate']:.2%}")
    print(f"     Profit Factor: {strategy_info['performance']['profit_factor']:.2f}")
    
    print("\n============================================================")
    print("LIQUIDITY POOLS STRATEGY TEST COMPLETED!")
    print("============================================================")
