"""
Order Blocks Strategy - Smart Money Concepts (SMC) - COMPLETE
============================================================
Author: XAUUSD Trading System
Version: 2.0.0
Date: 2025-08-08 (Modified for base.py integration: 2025-08-15)

Advanced Order Block detection and trading for XAUUSD:
- Institutional order block identification
- Fair Value Gap (FVG) detection
- Break of Structure (BOS) analysis
- Change of Character (CHOCH) recognition
- Liquidity sweep detection
- Premium/Discount zones

Order blocks represent areas where institutions have placed large orders,
creating significant supply/demand zones that often act as strong
support/resistance levels.

Key Concepts:
- Bullish Order Block: Last bearish candle before bullish impulse
- Bearish Order Block: Last bullish candle before bearish impulse
- Fair Value Gap: Imbalance in price showing inefficiency
- Mitigation: When price returns to test order block

Dependencies:
    - pandas
    - numpy
    - datetime
"""

import sys
import os
from pathlib import Path

# Add src to path for module resolution when run as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

# Import base classes from src.core.base
from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade, StrategyPerformance


class OrderBlockType(Enum):
    """Order block types"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class MarketStructure(Enum):
    """Market structure states"""
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND" 
    RANGING = "RANGING"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class OrderBlock:
    """Order block data structure"""
    id: str
    block_type: OrderBlockType
    high: float
    low: float
    open: float
    close: float
    timestamp: datetime
    timeframe: str
    
    # Order block characteristics
    strength: float  # 0-1 based on various factors
    volume: Optional[float] = None
    tested: bool = False
    mitigation_count: int = 0
    age_hours: float = 0.0
    
    # Related structures
    fair_value_gap: Optional[Dict] = None
    liquidity_sweep: bool = False
    break_of_structure: bool = False
    
    def __post_init__(self):
        """Calculate derived properties"""
        self.age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600
        
        # Calculate order block range
        self.range = self.high - self.low
        self.mid_point = (self.high + self.low) / 2
        
        # Premium/Discount levels (Fibonacci-based)
        self.premium_zone = self.low + (self.range * 0.618)  # 61.8% level
        self.discount_zone = self.low + (self.range * 0.382)  # 38.2% level
        self.equilibrium = self.mid_point


@dataclass  
class FairValueGap:
    """Fair Value Gap data structure"""
    id: str
    gap_type: OrderBlockType  # Direction of the gap
    top: float
    bottom: float
    timestamp: datetime
    timeframe: str
    
    # Gap characteristics
    size: float
    filled: bool = False
    partial_fill: float = 0.0  # Percentage filled
    
    def __post_init__(self):
        """Calculate gap properties"""
        self.size = self.top - self.bottom
        self.mid_point = (self.top + self.bottom) / 2


class OrderBlocksStrategy(AbstractStrategy): # Inherit from AbstractStrategy
    """
    Advanced Order Blocks Strategy using Smart Money Concepts
    
    This strategy identifies and trades institutional order blocks by:
    - Detecting order blocks after significant price moves
    - Identifying Fair Value Gaps (FVGs) 
    - Recognizing Break of Structure (BOS) and Change of Character (CHOCH)
    - Trading retest/mitigation opportunities
    - Managing risk with proper stop losses
    """
    
    # Modified __init__ signature to match AbstractStrategy
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """Initialize Order Blocks strategy"""
        super().__init__(config, mt5_manager, database) # Call parent __init__
        
        # Strategy parameters - use self.config from AbstractStrategy
        self.min_confidence = self.config.get('parameters', {}).get('confidence_threshold', 0.70)
        self.swing_length = self.config.get('parameters', {}).get('swing_length', 10)
        self.min_ob_strength = self.config.get('parameters', {}).get('order_block_min_strength', 2.0)
        self.fvg_min_size = self.config.get('parameters', {}).get('fvg_min_size', 0.5)
        self.liquidity_sweep_tolerance = self.config.get('parameters', {}).get('liquidity_sweep_tolerance', 1.2)
        
        # Timeframe settings
        # Access timeframes from the 'timeframes' key in the config dict
        self.structure_tf = self.config.get('timeframes', {}).get('structure', 'H4')
        self.intermediate_tf = self.config.get('timeframes', {}).get('intermediate', 'H1') 
        self.entry_tf = self.config.get('timeframes', {}).get('entry', 'M15')
        self.execution_tf = self.config.get('timeframes', {}).get('execution', 'M5')
        
        # Order block storage
        self.active_order_blocks = []
        self.active_fvgs = []
        self.market_structure = MarketStructure.UNCERTAIN
        
        # Performance tracking is now handled by AbstractStrategy base class
        # self.success_rate = 0.78
        # self.profit_factor = 2.5
        
        # Logger is now handled by parent class
        # self.logger = logging.getLogger('order_blocks_strategy')
    
    # Renamed from generate_signals to generate_signal to match AbstractStrategy
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """Generate Order Block trading signals"""
        signals = []
        try:
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available for signal generation.")
                return []

            # Get market data for multiple timeframes
            data = self.mt5_manager.get_historical_data(symbol, timeframe, 500)
            if data is None or len(data) < 100:
                self.logger.warning(f"Insufficient data for Order Blocks analysis: {len(data) if data is not None else 0}")
                return []
            
            # Update market structure
            self._update_market_structure(data, symbol, timeframe)
            
            # Find and update order blocks
            self._identify_order_blocks(data, symbol, timeframe)
            
            # Find Fair Value Gaps
            self._identify_fair_value_gaps(data, symbol, timeframe)
            
            # Generate raw signals from internal logic
            raw_signals = []
            raw_signals.extend(self._generate_order_block_signals(data, symbol, timeframe))
            raw_signals.extend(self._generate_fvg_signals(data, symbol, timeframe))
            raw_signals.extend(self._generate_bos_signals(data, symbol, timeframe))
            
            # Filter and validate signals using internal logic first
            filtered_signals_internal = self._validate_signals(raw_signals, data)
            
            # Clean up old order blocks and FVGs
            self._cleanup_old_structures()
            
            self.logger.info(f"Order Blocks generated {len(filtered_signals_internal)} signals from {len(raw_signals)} candidates")
            
            # Final validation using base class method
            validated_signals_final = []
            for signal in filtered_signals_internal:
                if self.validate_signal(signal):
                    validated_signals_final.append(signal)

            return validated_signals_final
            
        except Exception as e:
            self.logger.error(f"Order Blocks signal generation failed: {str(e)}", exc_info=True)
            return []
    
    # New method: analyze, required by AbstractStrategy
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Performs detailed Order Block and SMC analysis without generating executable signals.
        
        Args:
            data: Historical price data.
            symbol: Trading symbol.
            timeframe: Analysis timeframe.
            
        Returns:
            Dictionary containing detailed analysis results.
        """
        try:
            if data is None or len(data) < 100:
                return {
                    'status': 'Insufficient data for analysis',
                    'required_bars': 100,
                    'available_bars': len(data) if data is not None else 0
                }
            
            self._update_market_structure(data, symbol, timeframe)
            self._identify_order_blocks(data, symbol, timeframe)
            self._identify_fair_value_gaps(data, symbol, timeframe)

            analysis_output = {
                'strategy': self.strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_time': datetime.now().isoformat(),
                'data_points': len(data),
                'current_market_structure': self.market_structure.value,
                'active_order_blocks_count': len(self.active_order_blocks),
                'active_fair_value_gaps_count': len(self.active_fvgs),
                'recent_order_blocks': [],
                'recent_fair_value_gaps': []
            }

            for ob in self.active_order_blocks[-5:]: # Show up to 5 most recent OBs
                analysis_output['recent_order_blocks'].append({
                    'id': ob.id,
                    'type': ob.block_type.value,
                    'range': (ob.low, ob.high),
                    'timestamp': ob.timestamp.isoformat(),
                    'strength': round(ob.strength, 2),
                    'tested': ob.tested,
                    'age_hours': round(ob.age_hours, 2)
                })

            for fvg in self.active_fvgs[-5:]: # Show up to 5 most recent FVGs
                analysis_output['recent_fair_value_gaps'].append({
                    'id': fvg.id,
                    'type': fvg.gap_type.value,
                    'range': (fvg.bottom, fvg.top),
                    'size': round(fvg.size, 2),
                    'timestamp': fvg.timestamp.isoformat(),
                    'filled': fvg.filled
                })
            
            self._cleanup_old_structures() # Clean up after analysis
            
            return analysis_output

        except Exception as e:
            self.logger.error(f"Error during Order Blocks analysis method: {str(e)}", exc_info=True)
            return {'error': str(e)}


    def _update_market_structure(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Update current market structure analysis"""
        try:
            if len(data) < 50:
                return
            
            # Calculate swing highs and lows
            swing_highs, swing_lows = self._find_swing_points(data)
            
            if len(swing_highs) < 2 or len(swing_lows) < 2:
                self.market_structure = MarketStructure.UNCERTAIN
                return
            
            # Analyze recent swing structure
            recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
            recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
            
            # Determine trend based on higher highs/higher lows or lower highs/lower lows
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                higher_highs = all(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs)))
                higher_lows = all(recent_lows[i] > recent_lows[i-1] for i in range(1, len(recent_lows)))
                
                lower_highs = all(recent_highs[i] < recent_highs[i-1] for i in range(1, len(recent_highs)))
                lower_lows = all(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows)))
                
                if higher_highs and higher_lows:
                    self.market_structure = MarketStructure.UPTREND
                elif lower_highs and lower_lows:
                    self.market_structure = MarketStructure.DOWNTREND
                else:
                    self.market_structure = MarketStructure.RANGING
            else:
                self.market_structure = MarketStructure.UNCERTAIN
                
        except Exception as e:
            self.logger.error(f"Market structure update failed: {str(e)}", exc_info=True)
            self.market_structure = MarketStructure.UNCERTAIN
    
    def _find_swing_points(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Find swing highs and lows"""
        try:
            highs = []
            lows = []
            
            for i in range(self.swing_length, len(data) - self.swing_length):
                # Check for swing high
                is_swing_high = True
                current_high = data['High'].iloc[i]
                
                for j in range(i - self.swing_length, i + self.swing_length + 1):
                    if j != i and data['High'].iloc[j] >= current_high:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    highs.append(current_high)
                
                # Check for swing low
                is_swing_low = True
                current_low = data['Low'].iloc[i]
                
                for j in range(i - self.swing_length, i + self.swing_length + 1):
                    if j != i and data['Low'].iloc[j] <= current_low:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    lows.append(current_low)
            
            return highs, lows
            
        except Exception as e:
            self.logger.error(f"Swing point detection failed: {str(e)}", exc_info=True)
            return [], []
    
    def _identify_order_blocks(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Identify institutional order blocks"""
        try:
            # Find significant moves (impulses)
            impulses = self._find_impulse_moves(data)
            
            for impulse in impulses:
                # Find the order block (last opposite candle before impulse)
                ob_candle_idx = impulse['start_idx'] - 1
                
                if ob_candle_idx < 0 or ob_candle_idx >= len(data):
                    continue
                
                ob_candle = data.iloc[ob_candle_idx]
                
                # Determine order block type
                if impulse['direction'] == 'bullish':
                    # Bullish order block - last bearish candle before bullish move
                    if ob_candle['Close'] < ob_candle['Open']:  # Bearish candle
                        block_type = OrderBlockType.BULLISH
                        high = ob_candle['High']
                        low = ob_candle['Low']
                    else:
                        continue
                else:
                    # Bearish order block - last bullish candle before bearish move  
                    if ob_candle['Close'] > ob_candle['Open']:  # Bullish candle
                        block_type = OrderBlockType.BEARISH
                        high = ob_candle['High']
                        low = ob_candle['Low']
                    else:
                        continue
                
                # Calculate order block strength
                strength = self._calculate_ob_strength(impulse, ob_candle, data)
                
                if strength >= self.min_ob_strength:
                    # Create order block
                    order_block = OrderBlock(
                        id=f"OB_{symbol}_{timeframe}_{ob_candle_idx}_{datetime.now().strftime('%H%M%S')}",
                        block_type=block_type,
                        high=high,
                        low=low,
                        open=ob_candle['Open'],
                        close=ob_candle['Close'],
                        timestamp=data.index[ob_candle_idx],
                        timeframe=timeframe,
                        strength=strength,
                        volume=ob_candle.get('Volume', 0),
                        break_of_structure=impulse.get('bos', False)
                    )
                    
                    # Add to active order blocks
                    self.active_order_blocks.append(order_block)
                    
                    self.logger.debug(f"Identified order block: {block_type.value} at {high}-{low} with strength {strength}")
                    
        except Exception as e:
            self.logger.error(f"Order block identification failed: {str(e)}", exc_info=True)
    
    def _find_impulse_moves(self, data: pd.DataFrame) -> List[Dict]:
        """Find significant impulse moves in price"""
        try:
            impulses = []
            atr = self._calculate_atr(data, 14)
            
            if atr is None or atr == 0:
                return impulses
            
            min_impulse_size = atr * 2  # Minimum impulse must be 2x ATR
            
            for i in range(20, len(data) - 5):
                current_bar = data.iloc[i]
                
                # Look for bullish impulse
                bullish_move = 0
                bullish_bars = 0
                for j in range(i, min(i + 10, len(data))):
                    if data.iloc[j]['Close'] > data.iloc[j]['Open']:
                        bullish_move += (data.iloc[j]['Close'] - data.iloc[j]['Open'])
                        bullish_bars += 1
                
                # Look for bearish impulse
                bearish_move = 0
                bearish_bars = 0
                for j in range(i, min(i + 10, len(data))):
                    if data.iloc[j]['Close'] < data.iloc[j]['Open']:
                        bearish_move += (data.iloc[j]['Open'] - data.iloc[j]['Close'])
                        bearish_bars += 1
                
                # Check if we have a valid impulse
                if bullish_move >= min_impulse_size and bullish_bars >= 3:
                    impulses.append({
                        'direction': 'bullish',
                        'start_idx': i,
                        'end_idx': min(i + 10, len(data) - 1),
                        'size': bullish_move,
                        'strength': bullish_move / atr,
                        'bars_count': bullish_bars,
                        'bos': self._check_break_of_structure(data, i, 'bullish')
                    })
                
                elif bearish_move >= min_impulse_size and bearish_bars >= 3:
                    impulses.append({
                        'direction': 'bearish',
                        'start_idx': i,
                        'end_idx': min(i + 10, len(data) - 1),
                        'size': bearish_move,
                        'strength': bearish_move / atr,
                        'bars_count': bearish_bars,
                        'bos': self._check_break_of_structure(data, i, 'bearish')
                    })
            
            return impulses
            
        except Exception as e:
            self.logger.error(f"Impulse move detection failed: {str(e)}", exc_info=True)
            return []
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate Average True Range"""
        try:
            if len(data) < period + 1:
                return None
            
            tr_values = []
            for i in range(1, len(data)):
                high = data['High'].iloc[i]
                low = data['Low'].iloc[i]
                prev_close = data['Close'].iloc[i-1]
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                tr_values.append(tr)
            
            if len(tr_values) >= period:
                return sum(tr_values[-period:]) / period
            else:
                # Handle cases where data length is less than period, return average TR
                return sum(tr_values) / len(tr_values) if tr_values else 0.0
                
        except Exception as e:
            self.logger.error(f"ATR calculation failed: {str(e)}", exc_info=True)
            return None
    
    def _check_break_of_structure(self, data: pd.DataFrame, start_idx: int, direction: str) -> bool:
        """Check if the move represents a break of structure"""
        try:
            if start_idx < 20 or start_idx >= len(data) - 5:
                return False
            
            # Get recent swing highs and lows
            recent_data = data.iloc[max(0, start_idx - 20):start_idx]
            recent_highs = recent_data['High'].rolling(window=5).max()
            recent_lows = recent_data['Low'].rolling(window=5).min()
            
            if direction == 'bullish':
                # Check if we broke above recent high
                resistance_level = recent_highs.max()
                current_high = data['High'].iloc[start_idx:start_idx + 5].max()
                return current_high > resistance_level
            else:
                # Check if we broke below recent low
                support_level = recent_lows.min()
                current_low = data['Low'].iloc[start_idx:start_idx + 5].min()
                return current_low < support_level
                
        except Exception as e:
            self.logger.error(f"BOS check failed: {str(e)}", exc_info=True)
            return False
    
    def _calculate_ob_strength(self, impulse: Dict, ob_candle: pd.Series, data: pd.DataFrame) -> float:
        """Calculate order block strength based on multiple factors"""
        try:
            strength = 0.0
            
            # Factor 1: Impulse strength (30% weight)
            impulse_strength = min(impulse['strength'] / 5.0, 1.0)  # Normalize to 0-1
            strength += impulse_strength * 0.3
            
            # Factor 2: Volume (20% weight)
            if 'Volume' in ob_candle and ob_candle['Volume'] > 0:
                avg_volume = data['Volume'].tail(20).mean()
                volume_ratio = min(ob_candle['Volume'] / avg_volume, 3.0) / 3.0 if avg_volume > 0 else 0.1
                strength += volume_ratio * 0.2
            else:
                strength += 0.1  # Default if no volume data
            
            # Factor 3: Break of Structure (25% weight)
            if impulse.get('bos', False):
                strength += 0.25
            
            # Factor 4: Order block size relative to ATR (15% weight)
            ob_range = ob_candle['High'] - ob_candle['Low']
            atr = self._calculate_atr(data, 14)
            if atr and atr > 0:
                size_ratio = min(ob_range / atr, 2.0) / 2.0
                strength += size_ratio * 0.15
            
            # Factor 5: Rejection wicks (10% weight)
            body_size = abs(ob_candle['Close'] - ob_candle['Open'])
            total_range = ob_candle['High'] - ob_candle['Low']
            if total_range > 0:
                wick_ratio = (total_range - body_size) / total_range
                strength += min(wick_ratio * 2, 1.0) * 0.1
            
            return min(strength, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"OB strength calculation failed: {str(e)}", exc_info=True)
            return 0.5  # Default strength
    
    def _identify_fair_value_gaps(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Identify Fair Value Gaps (FVGs)"""
        try:
            for i in range(2, len(data)):
                # Check for bullish FVG (gap up)
                if (data['Low'].iloc[i] > data['High'].iloc[i-2] and 
                    data['Low'].iloc[i-1] > data['High'].iloc[i-2]):
                    
                    gap_bottom = data['High'].iloc[i-2]
                    gap_top = data['Low'].iloc[i]
                    gap_size = gap_top - gap_bottom
                    
                    # Check if gap is significant
                    atr = self._calculate_atr(data, 14)
                    if atr and gap_size >= (atr * self.fvg_min_size):
                        fvg = FairValueGap(
                            id=f"FVG_{symbol}_{timeframe}_{i}_BULL",
                            gap_type=OrderBlockType.BULLISH,
                            top=gap_top,
                            bottom=gap_bottom,
                            timestamp=data.index[i],
                            timeframe=timeframe,
                            size=gap_size
                        )
                        self.active_fvgs.append(fvg)
                
                # Check for bearish FVG (gap down)
                elif (data['High'].iloc[i] < data['Low'].iloc[i-2] and 
                      data['High'].iloc[i-1] < data['Low'].iloc[i-2]):
                    
                    gap_top = data['Low'].iloc[i-2]
                    gap_bottom = data['High'].iloc[i]
                    gap_size = gap_top - gap_bottom
                    
                    # Check if gap is significant
                    atr = self._calculate_atr(data, 14)
                    if atr and gap_size >= (atr * self.fvg_min_size):
                        fvg = FairValueGap(
                            id=f"FVG_{symbol}_{timeframe}_{i}_BEAR",
                            gap_type=OrderBlockType.BEARISH,
                            top=gap_top,
                            bottom=gap_bottom,
                            timestamp=data.index[i],
                            timeframe=timeframe,
                            size=gap_size
                        )
                        self.active_fvgs.append(fvg)
                        
        except Exception as e:
            self.logger.error(f"FVG identification failed: {str(e)}", exc_info=True)
    
    def _generate_order_block_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Generate signals based on order block retests"""
        signals = []
        
        try:
            current_price = data['Close'].iloc[-1]
            current_time = data.index[-1]
            
            for ob in self.active_order_blocks:
                if ob.tested or ob.age_hours > 48:  # Skip old or tested OBs
                    continue
                
                # Check if price is retesting the order block
                retest_signal = self._check_ob_retest(ob, current_price, data.tail(5))
                
                if retest_signal:
                    # Determine signal type based on OB type and market structure
                    if ob.block_type == OrderBlockType.BULLISH and retest_signal['type'] == 'demand':
                        signal_type = SignalType.BUY
                        stop_loss = ob.low - (self._calculate_atr(data, 14) or 10)
                        take_profit = current_price + ((current_price - stop_loss) * 2)
                        
                    elif ob.block_type == OrderBlockType.BEARISH and retest_signal['type'] == 'supply':
                        signal_type = SignalType.SELL
                        stop_loss = ob.high + (self._calculate_atr(data, 14) or 10)
                        take_profit = current_price - ((stop_loss - current_price) * 2)
                    else:
                        continue
                    
                    # Calculate confidence based on multiple factors
                    confidence = self._calculate_ob_signal_confidence(ob, retest_signal, data)
                    
                    if confidence >= self.min_confidence:
                        # Grade is now automatically determined by Signal's __post_init__
                        # grade = self._determine_signal_grade(confidence, ob.strength)
                        
                        signal = Signal(
                            timestamp=current_time,
                            symbol=symbol,
                            strategy_name=self.strategy_name, # Use self.strategy_name from base class
                            signal_type=signal_type,
                            confidence=confidence,
                            price=current_price,
                            timeframe=timeframe,
                            strength=ob.strength,
                            # grade=grade, # Removed, as it's automatically calculated
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            metadata={
                                'order_block_id': ob.id,
                                'order_block_type': ob.block_type.value,
                                'retest_type': retest_signal['type'],
                                'market_structure': self.market_structure.value,
                                'ob_age_hours': ob.age_hours,
                                'break_of_structure': ob.break_of_structure
                            }
                        )
                        
                        signals.append(signal)
                        ob.tested = True  # Mark as tested
                        
        except Exception as e:
            self.logger.error(f"OB signal generation failed: {str(e)}", exc_info=True)
        
        return signals
    
    def _check_ob_retest(self, ob: OrderBlock, current_price: float, recent_data: pd.DataFrame) -> Optional[Dict]:
        """Check if current price action represents an order block retest"""
        try:
            # Check if price is within the order block zone
            if not (ob.low <= current_price <= ob.high):
                return None
            
            # For bullish OB, look for demand (buying pressure)
            if ob.block_type == OrderBlockType.BULLISH:
                # Check for rejection from the zone
                lowest_in_zone = recent_data['Low'].min()
                if lowest_in_zone <= ob.discount_zone and current_price > ob.equilibrium:
                    return {'type': 'demand', 'strength': 0.8}
                elif current_price >= ob.discount_zone:
                    return {'type': 'demand', 'strength': 0.6}
            
            # For bearish OB, look for supply (selling pressure)
            else:
                # Check for rejection from the zone
                highest_in_zone = recent_data['High'].max()
                if highest_in_zone >= ob.premium_zone and current_price < ob.equilibrium:
                    return {'type': 'supply', 'strength': 0.8}
                elif current_price <= ob.premium_zone:
                    return {'type': 'supply', 'strength': 0.6}
            
            return None
            
        except Exception as e:
            self.logger.error(f"OB retest check failed: {str(e)}", exc_info=True)
            return None
    
    def _calculate_ob_signal_confidence(self, ob: OrderBlock, retest_signal: Dict, data: pd.DataFrame) -> float:
        """Calculate confidence for order block signal"""
        try:
            confidence = 0.5  # Base confidence
            
            # Factor 1: Order block strength (30%)
            confidence += ob.strength * 0.3
            
            # Factor 2: Retest strength (25%)
            confidence += retest_signal.get('strength', 0.5) * 0.25
            
            # Factor 3: Market structure alignment (20%)
            if ((ob.block_type == OrderBlockType.BULLISH and self.market_structure == MarketStructure.UPTREND) or
                (ob.block_type == OrderBlockType.BEARISH and self.market_structure == MarketStructure.DOWNTREND)):
                confidence += 0.2
            elif self.market_structure == MarketStructure.RANGING:
                confidence += 0.1
            
            # Factor 4: Order block freshness (15%)
            if ob.age_hours < 4:
                confidence += 0.15
            elif ob.age_hours < 12:
                confidence += 0.10
            elif ob.age_hours < 24:
                confidence += 0.05
            
            # Factor 5: Break of structure (10%)
            if ob.break_of_structure:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"OB signal confidence calculation failed: {str(e)}", exc_info=True)
            return 0.5
    
    def _generate_fvg_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Generate signals based on Fair Value Gap fills"""
        signals = []
        
        try:
            current_price = data['Close'].iloc[-1]
            current_time = data.index[-1]
            
            for fvg in self.active_fvgs:
                if fvg.filled:
                    continue
                
                # Check if price is approaching the FVG
                if fvg.gap_type == OrderBlockType.BULLISH:
                    # Bullish FVG - expect price to go up and fill the gap
                    if current_price <= fvg.bottom * 1.001:  # Within 0.1% of gap
                        signal_type = SignalType.BUY
                        stop_loss = fvg.bottom - (self._calculate_atr(data, 14) or 10)
                        take_profit = fvg.top
                        
                        confidence = self._calculate_fvg_confidence(fvg, current_price, data)
                        
                        if confidence >= self.min_confidence:
                            # grade = self._determine_signal_grade(confidence, 0.7) # Removed
                            
                            signal = Signal(
                                timestamp=current_time,
                                symbol=symbol,
                                strategy_name=self.strategy_name, # Use self.strategy_name
                                signal_type=signal_type,
                                confidence=confidence,
                                price=current_price,
                                timeframe=timeframe,
                                strength=0.7,
                                # grade=grade, # Removed
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                metadata={
                                    'fvg_id': fvg.id,
                                    'fvg_type': fvg.gap_type.value,
                                    'gap_size': fvg.size,
                                    'gap_top': fvg.top,
                                    'gap_bottom': fvg.bottom
                                }
                            )
                            signals.append(signal)
                
                else:  # Bearish FVG
                    if current_price >= fvg.top * 0.999:  # Within 0.1% of gap
                        signal_type = SignalType.SELL
                        stop_loss = fvg.top + (self._calculate_atr(data, 14) or 10)
                        take_profit = fvg.bottom
                        
                        confidence = self._calculate_fvg_confidence(fvg, current_price, data)
                        
                        if confidence >= self.min_confidence:
                            # grade = self._determine_signal_grade(confidence, 0.7) # Removed
                            
                            signal = Signal(
                                timestamp=current_time,
                                symbol=symbol,
                                strategy_name=self.strategy_name, # Use self.strategy_name
                                signal_type=signal_type,
                                confidence=confidence,
                                price=current_price,
                                timeframe=timeframe,
                                strength=0.7,
                                # grade=grade, # Removed
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                metadata={
                                    'fvg_id': fvg.id,
                                    'fvg_type': fvg.gap_type.value,
                                    'gap_size': fvg.size,
                                    'gap_top': fvg.top,
                                    'gap_bottom': fvg.bottom
                                }
                            )
                            signals.append(signal)
                            
        except Exception as e:
            self.logger.error(f"FVG signal generation failed: {str(e)}", exc_info=True)
        
        return signals
    
    def _calculate_fvg_confidence(self, fvg: FairValueGap, current_price: float, data: pd.DataFrame) -> float:
        """Calculate confidence for FVG fill signal"""
        try:
            confidence = 0.6  # Base confidence for FVG
            
            # Factor 1: Gap size significance (30%)
            atr = self._calculate_atr(data, 14)
            if atr and atr > 0:
                size_significance = min(fvg.size / atr, 3.0) / 3.0
                confidence += size_significance * 0.3
            
            # Factor 2: Market structure alignment (25%)
            if ((fvg.gap_type == OrderBlockType.BULLISH and self.market_structure == MarketStructure.UPTREND) or
                (fvg.gap_type == OrderBlockType.BEARISH and self.market_structure == MarketStructure.DOWNTREND)):
                confidence += 0.25
            elif self.market_structure == MarketStructure.RANGING:
                confidence += 0.15
            
            # Factor 3: Distance from gap (20%)
            if fvg.size == 0: # Avoid division by zero
                distance = 1.0 # Max distance if no size
            elif fvg.gap_type == OrderBlockType.BULLISH:
                distance = abs(current_price - fvg.bottom) / fvg.size
            else:
                distance = abs(current_price - fvg.top) / fvg.size
            
            distance_score = max(0, 1 - distance) * 0.2
            confidence += distance_score
            
            # Factor 4: Volume context (15%)
            if 'Volume' in data.columns and not data['Volume'].empty:
                recent_volume = data['Volume'].tail(5).mean()
                avg_volume = data['Volume'].tail(20).mean()
                if avg_volume > 0:
                    volume_ratio = min(recent_volume / avg_volume, 2.0) / 2.0
                    confidence += volume_ratio * 0.15
            else:
                confidence += 0.05 # Small credit if no volume data
            
            # Factor 5: Age of FVG (10%)
            age_hours = (datetime.now() - fvg.timestamp).total_seconds() / 3600
            if age_hours < 12:
                confidence += 0.1
            elif age_hours < 24:
                confidence += 0.05
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"FVG confidence calculation failed: {str(e)}", exc_info=True)
            return 0.6
    
    def _generate_bos_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Generate signals based on Break of Structure"""
        signals = []
        
        try:
            current_price = data['Close'].iloc[-1]
            current_time = data.index[-1]
            
            # Check for recent break of structure
            bos_detection = self._detect_recent_bos(data)
            
            if bos_detection:
                # Generate signal in the direction of the break
                if bos_detection['direction'] == 'bullish':
                    signal_type = SignalType.BUY
                    stop_loss = bos_detection['break_level'] - (self._calculate_atr(data, 14) or 10)
                    take_profit = current_price + ((current_price - stop_loss) * 2)
                    
                else:  # bearish
                    signal_type = SignalType.SELL
                    stop_loss = bos_detection['break_level'] + (self._calculate_atr(data, 14) or 10)
                    take_profit = current_price - ((stop_loss - current_price) * 2)
                
                confidence = self._calculate_bos_confidence(bos_detection, data)
                
                if confidence >= self.min_confidence:
                    # grade = self._determine_signal_grade(confidence, bos_detection['strength']) # Removed
                    
                    signal = Signal(
                        timestamp=current_time,
                        symbol=symbol,
                        strategy_name=self.strategy_name, # Use self.strategy_name
                        signal_type=signal_type,
                        confidence=confidence,
                        price=current_price,
                        timeframe=timeframe,
                        strength=bos_detection['strength'],
                        # grade=grade, # Removed
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'bos_direction': bos_detection['direction'],
                            'break_level': bos_detection['break_level'],
                            'bos_strength': bos_detection['strength'],
                            'volume_surge': bos_detection.get('volume_surge', False)
                        }
                    )
                    signals.append(signal)
                    
        except Exception as e:
            self.logger.error(f"BOS signal generation failed: {str(e)}", exc_info=True)
        
        return signals
    
    def _detect_recent_bos(self, data: pd.DataFrame) -> Optional[Dict]:
        """Detect recent break of structure"""
        try:
            if len(data) < 20:
                return None
            
            # Get recent highs and lows
            recent_high = data['High'].tail(10).max()
            recent_low = data['Low'].tail(10).min()
            
            # Get previous structure levels (look further back)
            prev_high = data['High'].iloc[-20:-10].max()
            prev_low = data['Low'].iloc[-20:-10].min()
            
            # Check for bullish BOS
            if recent_high > prev_high:
                # Confirm with volume if available
                volume_surge = False
                if 'Volume' in data.columns and not data['Volume'].empty:
                    recent_volume = data['Volume'].tail(5).mean()
                    avg_volume = data['Volume'].tail(20).mean()
                    volume_surge = recent_volume > avg_volume * 1.5 if avg_volume > 0 else False
                
                return {
                    'direction': 'bullish',
                    'break_level': prev_high,
                    'strength': min((recent_high - prev_high) / prev_high * 100, 1.0) if prev_high > 0 else 0.0,
                    'volume_surge': volume_surge
                }
            
            # Check for bearish BOS
            elif recent_low < prev_low:
                volume_surge = False
                if 'Volume' in data.columns and not data['Volume'].empty:
                    recent_volume = data['Volume'].tail(5).mean()
                    avg_volume = data['Volume'].tail(20).mean()
                    volume_surge = recent_volume > avg_volume * 1.5 if avg_volume > 0 else False
                
                return {
                    'direction': 'bearish',
                    'break_level': prev_low,
                    'strength': min((prev_low - recent_low) / prev_low * 100, 1.0) if prev_low > 0 else 0.0,
                    'volume_surge': volume_surge
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"BOS detection failed: {str(e)}", exc_info=True)
            return None
    
    def _calculate_bos_confidence(self, bos_detection: Dict, data: pd.DataFrame) -> float:
        """Calculate confidence for BOS signal"""
        try:
            confidence = 0.6  # Base confidence
            
            # Factor 1: Break strength (35%)
            confidence += bos_detection['strength'] * 0.35
            
            # Factor 2: Volume confirmation (25%)
            if bos_detection.get('volume_surge', False):
                confidence += 0.25
            else:
                confidence += 0.1  # Some credit for any volume data
            
            # Factor 3: Market structure context (20%)
            if ((bos_detection['direction'] == 'bullish' and self.market_structure == MarketStructure.UPTREND) or
                (bos_detection['direction'] == 'bearish' and self.market_structure == MarketStructure.DOWNTREND)):
                confidence += 0.2
            elif self.market_structure == MarketStructure.RANGING:
                confidence += 0.15  # BOS in ranging market can be significant
            
            # Factor 4: Timing (20%)
            # Recent BOS is more reliable - current logic doesn't explicitly factor timing.
            # Assuming if a BOS is detected, it's recent enough to add confidence.
            confidence += 0.2
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"BOS confidence calculation failed: {str(e)}", exc_info=True)
            return 0.6
    
    # Removed _determine_signal_grade as it's now handled by base.Signal's __post_init__
    # def _determine_signal_grade(self, confidence: float, strength: float) -> SignalGrade:
    #     """Determine signal grade based on confidence and strength"""
    #     try:
    #         combined_score = (confidence * 0.7) + (strength * 0.3)
            
    #         if combined_score >= 0.85:
    #             return SignalGrade.A
    #         elif combined_score >= 0.75:
    #             return SignalGrade.B
    #         elif combined_score >= 0.65:
    #             return SignalGrade.C
    #         else:
    #             return SignalGrade.D
                
    #     except Exception as e:
    #         self.logger.error(f"Signal grade determination failed: {str(e)}")
    #         return SignalGrade.C
    
    def _validate_signals(self, signals: List[Signal], data: pd.DataFrame) -> List[Signal]:
        """Validate and filter signals (Order Blocks Strategy's specific validation)"""
        validated_signals = []
        
        try:
            for signal in signals:
                # Check signal quality
                if signal.confidence < self.min_confidence:
                    continue
                
                # Check if we have too many signals of the same type
                same_type_signals = [s for s in validated_signals if s.signal_type == signal.signal_type]
                if len(same_type_signals) >= 2:  # Limit to 2 signals per type
                    continue
                
                # Check risk-reward ratio
                if signal.stop_loss and signal.take_profit:
                    if signal.signal_type == SignalType.BUY:
                        risk = signal.price - signal.stop_loss
                        reward = signal.take_profit - signal.price
                    else:
                        risk = signal.stop_loss - signal.price
                        reward = signal.price - signal.take_profit
                    
                    if risk > 0 and (reward / risk) >= 1.5:  # Minimum 1.5:1 RR
                        validated_signals.append(signal)
                else: # If SL/TP not set, skip RR check but still consider valid
                    validated_signals.append(signal)
                    
        except Exception as e:
            self.logger.error(f"Order Blocks internal signal validation failed: {str(e)}", exc_info=True)
            # If internal validation fails, return original signals to allow base class to try
            return signals  
        
        return validated_signals
    
    def _cleanup_old_structures(self) -> None:
        """Clean up old order blocks and FVGs"""
        try:
            current_time = datetime.now()
            
            # Remove old order blocks (older than 48 hours)
            self.active_order_blocks = [
                ob for ob in self.active_order_blocks
                if (current_time - ob.timestamp).total_seconds() / 3600 <= 48
            ]
            
            # Remove old FVGs (older than 24 hours or filled)
            self.active_fvgs = [
                fvg for fvg in self.active_fvgs
                if ((current_time - fvg.timestamp).total_seconds() / 3600 <= 24 and not fvg.filled)
            ]
            
        except Exception as e:
            self.logger.error(f"Structure cleanup failed: {str(e)}", exc_info=True)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and statistics"""
        # Get performance summary from base class
        base_performance = self.get_performance_summary()

        return {
            'name': 'Order Blocks Strategy',
            'version': '2.0.0',
            'type': 'Smart Money Concepts',
            'timeframes': [self.structure_tf, self.intermediate_tf, self.entry_tf, self.execution_tf],
            'active_order_blocks': len(self.active_order_blocks),
            'active_fvgs': len(self.active_fvgs),
            'market_structure': self.market_structure.value,
            'min_confidence': self.min_confidence,
            'performance': { # This structure matches the original request
                'success_rate': base_performance['win_rate'],
                'profit_factor': base_performance['profit_factor']
            },
            'parameters': { # Parameters pulled from config.parameters directly
                'swing_length': self.swing_length,
                'min_ob_strength': self.min_ob_strength,
                'fvg_min_size': self.fvg_min_size,
                'liquidity_sweep_tolerance': self.liquidity_sweep_tolerance
            }
        }


# Testing function
if __name__ == "__main__":
    """Test the Order Blocks strategy"""
    
    # Test configuration
    test_config = {
        'parameters': {
            'confidence_threshold': 0.70,
            'swing_length': 10,
            'order_block_min_strength': 2.0,
            'fvg_min_size': 0.5,
            'liquidity_sweep_tolerance': 1.2
        },
        'timeframes': {
            'structure': 'H4',
            'intermediate': 'H1',
            'entry': 'M15',
            'execution': 'M5'
        }
    }
    
    # Mock MT5 manager for testing
    class MockMT5Manager:
        def get_historical_data(self, symbol, timeframe, bars):
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            dates = pd.date_range(start=datetime.now() - timedelta(days=10), 
                                 end=datetime.now(), freq='15Min')
            
            # Generate sample OHLCV data
            np.random.seed(42)
            close_prices = 1950 + np.cumsum(np.random.randn(len(dates)) * 2)
            
            data = pd.DataFrame({
                'Open': close_prices + np.random.randn(len(dates)) * 0.5,
                'High': close_prices + np.abs(np.random.randn(len(dates)) * 3),
                'Low': close_prices - np.abs(np.random.randn(len(dates)) * 3),
                'Close': close_prices,
                'Volume': np.random.randint(100, 1000, len(dates))
            }, index=dates)
            
            # Ensure High >= Close >= Low
            data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
            data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
            
            return data
    
    # Create strategy instance
    mock_mt5 = MockMT5Manager()
    strategy = OrderBlocksStrategy(test_config, mock_mt5, database=None)
    
    # Output header matching other strategy files
    print("============================================================")
    print("TESTING MODIFIED ORDER BLOCKS STRATEGY")
    print("============================================================")

    # 1. Testing signal generation
    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15") # Renamed method call
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - Signal: {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.2f}, Grade: {signal.grade.value}")
        if signal.metadata:
            print(f"     Type: {signal.metadata.get('order_block_type', signal.metadata.get('fvg_type', signal.metadata.get('bos_direction', 'N/A')))}")
            print(f"     Market Structure: {signal.metadata.get('market_structure', 'N/A')}")
    
    # 2. Testing analysis method
    print("\n2. Testing analysis method:")
    mock_data = mock_mt5.get_historical_data("XAUUSDm", "M15", 200)
    analysis_results = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis results keys: {analysis_results.keys()}")
    print(f"   Current Market Structure: {analysis_results.get('current_market_structure', 'N/A')}")
    print(f"   Active Order Blocks: {analysis_results.get('active_order_blocks_count', 0)}")
    print(f"   Active FVGs: {analysis_results.get('active_fair_value_gaps_count', 0)}")
    if analysis_results.get('recent_order_blocks'):
        print("   Recent Order Block Example:", analysis_results['recent_order_blocks'][0])
    
    # 3. Testing performance tracking
    print("\n3. Testing performance tracking:")
    summary = strategy.get_performance_summary()
    print(f"   {summary}")
    
    # 4. Strategy Information
    print("\n4. Strategy Information:")
    strategy_info = strategy.get_strategy_info()
    print(f"   Name: {strategy_info['name']}")
    print(f"   Version: {strategy_info['version']}")
    print(f"   Type: {strategy_info['type']}")
    print(f"   Timeframes: {', '.join(strategy_info['timeframes'])}")
    print(f"   Min Confidence: {strategy_info['min_confidence']:.2f}")
    print(f"   Performance:")
    print(f"     Success Rate: {strategy_info['performance']['success_rate']:.2%}")
    print(f"     Profit Factor: {strategy_info['performance']['profit_factor']:.2f}")
    print(f"   Parameters:")
    for param, value in strategy_info['parameters'].items():
        print(f"     - {param}: {value}")

    # Footer matching other strategy files
    print("\n============================================================")
    print("ORDER BLOCKS STRATEGY TEST COMPLETED!")
    print("============================================================")