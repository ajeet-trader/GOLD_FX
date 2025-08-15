"""
Ichimoku Cloud Strategy - Advanced Technical Analysis
===================================================
Author: XAUUSD Trading System
Version: 3.0.0
Date: 2025-08-08 (Modified: 2025-01-15)

Advanced Ichimoku Kinko Hyo implementation for XAUUSD trading:
- Multi-timeframe analysis
- Cloud analysis and projections
- Kumo breakouts and reversals
- Chikou span confirmations
- Dynamic support/resistance

The Ichimoku system provides a complete trading framework with:
- Tenkan-sen (Conversion Line): 9-period average
- Kijun-sen (Base Line): 26-period average  
- Senkou Span A (Leading Span A): Cloud boundary
- Senkou Span B (Leading Span B): Cloud boundary
- Chikou Span (Lagging Span): Price displaced 26 periods back

Dependencies:
    - pandas
    - numpy
    - datetime
"""

import sys
import os
from pathlib import Path

# Add src to path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass

# Import from base module instead of local definitions
from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade
print("sys.path:", sys.path)

@dataclass
class IchimokuComponents:
    """Ichimoku indicator components"""
    tenkan_sen: float
    kijun_sen: float
    senkou_span_a: float
    senkou_span_b: float
    chikou_span: float
    cloud_top: float
    cloud_bottom: float
    price_vs_cloud: str  # "above", "below", "inside"
    cloud_color: str     # "bullish", "bearish"
    cloud_thickness: float


class IchimokuStrategy(AbstractStrategy):
    """
    Advanced Ichimoku Cloud Strategy
    
    This strategy implements a comprehensive Ichimoku analysis including:
    - Traditional Ichimoku signals
    - Cloud analysis and breakouts
    - Multi-timeframe confirmations
    - Dynamic support/resistance
    - Kumo twist predictions
    
    Signal Generation:
    - Strong Buy: Price above bullish cloud + Tenkan > Kijun + Chikou clear
    - Strong Sell: Price below bearish cloud + Tenkan < Kijun + Chikou clear
    - Breakout signals: Cloud breakouts with momentum
    - Reversal signals: Bounces from cloud edges
    
    Example:
        >>> strategy = IchimokuStrategy(config, mt5_manager)
        >>> signals = strategy.generate_signal("XAUUSDm", "M15")
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """
        Initialize Ichimoku strategy
        
        Args:
            config: Strategy configuration
            mt5_manager: MT5 connection manager (optional)
            database: Database manager (optional)
        """
        # Call parent class initialization
        super().__init__(config, mt5_manager, database)
        
        # Ichimoku parameters
        self.tenkan_period = 9
        self.kijun_period = 26
        self.senkou_span_b_period = 52
        self.displacement = 26
        
        # Strategy parameters (these can override parent class defaults)
        self.min_confidence = config.get('parameters', {}).get('confidence_threshold', 0.65)
        self.lookback_period = config.get('parameters', {}).get('lookback_period', 200)
        self.primary_timeframe = config.get('parameters', {}).get('timeframe_primary', 'M15')
        self.secondary_timeframe = config.get('parameters', {}).get('timeframe_secondary', 'H1')
        
        # Signal filters
        self.min_cloud_thickness = 5.0  # Minimum cloud thickness in points
        self.min_momentum_bars = 3      # Minimum bars for momentum confirmation
        self.max_signals_per_hour = 2   # Maximum signals per hour
        
        # Performance tracking (now handled by parent class)
        # self.recent_signals = []  # Remove - parent class handles this
        self.success_rate = 0.65
        self.profit_factor = 1.8
        
        # Logger is already set up by parent class
        # self.logger = logging.getLogger('ichimoku_strategy')
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """
        Generate Ichimoku-based trading signals
        
        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            List of trading signals
        """
        try:
            # Get market data
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available")
                return []
                
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_period)
            if data is None or len(data) < self.senkou_span_b_period + self.displacement:
                self.logger.warning(f"Insufficient data for Ichimoku analysis: {len(data) if data is not None else 0} bars")
                return []
            
            # Calculate Ichimoku components
            ichimoku_data = self._calculate_ichimoku(data)
            if ichimoku_data is None:
                return []
            
            # Get current market state
            current_components = self._get_current_ichimoku_state(ichimoku_data)
            if not current_components:
                return []
            
            # Generate signals based on Ichimoku analysis
            signals = []
            
            # Check for cloud breakout
            breakout_signal = self._check_cloud_breakout(ichimoku_data, current_components, symbol, timeframe)
            if breakout_signal:
                signals.append(breakout_signal)
            
            # Check for TK cross
            tk_signal = self._check_tk_cross(ichimoku_data, current_components, symbol, timeframe)
            if tk_signal:
                signals.append(tk_signal)
            
            # Check for Chikou span confirmation
            chikou_signal = self._check_chikou_span(ichimoku_data, current_components, symbol, timeframe)
            if chikou_signal:
                signals.append(chikou_signal)
            
            # Multi-timeframe confirmation if available
            if self.secondary_timeframe and timeframe != self.secondary_timeframe:
                htf_signals = self._get_higher_timeframe_confirmation(symbol, self.secondary_timeframe)
                signals.extend(htf_signals)
            
            # Validate all signals using parent class validation
            validated_signals = []
            for signal in signals:
                if self.validate_signal(signal):
                    validated_signals.append(signal)
                    # Update performance tracking (handled by parent)
                    self.signal_history.append(signal)
            
            self.logger.info(f"Ichimoku generated {len(validated_signals)} valid signals out of {len(signals)} total")
            
            return validated_signals
            
        except Exception as e:
            self.logger.error(f"Error generating Ichimoku signals: {str(e)}")
            return []
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Perform detailed Ichimoku analysis without generating signals
        
        Args:
            data: Historical price data
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            if data is None or len(data) < self.senkou_span_b_period:
                return {
                    'error': 'Insufficient data for analysis',
                    'required_bars': self.senkou_span_b_period,
                    'available_bars': len(data) if data is not None else 0
                }
            
            # Calculate Ichimoku components
            ichimoku_data = self._calculate_ichimoku(data)
            if ichimoku_data is None:
                return {'error': 'Failed to calculate Ichimoku indicators'}
            
            # Get current state
            current = self._get_current_ichimoku_state(ichimoku_data)
            
            # Perform analysis
            analysis = {
                'strategy': self.strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_time': datetime.now().isoformat(),
                'data_points': len(data),
                
                'current_values': {
                    'tenkan_sen': current.tenkan_sen,
                    'kijun_sen': current.kijun_sen,
                    'senkou_span_a': current.senkou_span_a,
                    'senkou_span_b': current.senkou_span_b,
                    'chikou_span': current.chikou_span,
                    'cloud_top': current.cloud_top,
                    'cloud_bottom': current.cloud_bottom
                },
                
                'market_position': {
                    'price_vs_cloud': current.price_vs_cloud,
                    'cloud_color': current.cloud_color,
                    'cloud_thickness': current.cloud_thickness,
                    'tk_relationship': 'bullish' if current.tenkan_sen > current.kijun_sen else 'bearish'
                },
                
                'signals': {
                    'cloud_breakout': self._detect_cloud_breakout_potential(ichimoku_data),
                    'tk_cross': self._detect_tk_cross_potential(ichimoku_data),
                    'kumo_twist': self._detect_kumo_twist(ichimoku_data),
                    'chikou_confirmation': self._check_chikou_confirmation(ichimoku_data)
                },
                
                'support_resistance': {
                    'immediate_support': current.cloud_bottom if current.price_vs_cloud == 'above' else current.cloud_top,
                    'immediate_resistance': current.cloud_top if current.price_vs_cloud == 'below' else current.cloud_bottom,
                    'kijun_level': current.kijun_sen,
                    'tenkan_level': current.tenkan_sen
                },
                
                'trend_strength': self._calculate_trend_strength(ichimoku_data, current),
                'recommendation': self._generate_recommendation(current)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in Ichimoku analysis: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_ichimoku(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate all Ichimoku components"""
        try:
            df = data.copy()
            
            # Tenkan-sen (Conversion Line)
            high_9 = df['High'].rolling(window=self.tenkan_period).max()
            low_9 = df['Low'].rolling(window=self.tenkan_period).min()
            df['tenkan_sen'] = (high_9 + low_9) / 2
            
            # Kijun-sen (Base Line)
            high_26 = df['High'].rolling(window=self.kijun_period).max()
            low_26 = df['Low'].rolling(window=self.kijun_period).min()
            df['kijun_sen'] = (high_26 + low_26) / 2
            
            # Senkou Span A (Leading Span A)
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(self.displacement)
            
            # Senkou Span B (Leading Span B)
            high_52 = df['High'].rolling(window=self.senkou_span_b_period).max()
            low_52 = df['Low'].rolling(window=self.senkou_span_b_period).min()
            df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(self.displacement)
            
            # Chikou Span (Lagging Span)
            df['chikou_span'] = df['Close'].shift(-self.displacement)
            
            # Cloud calculations
            df['cloud_top'] = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
            df['cloud_bottom'] = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
            df['cloud_thickness'] = df['cloud_top'] - df['cloud_bottom']
            df['cloud_color'] = np.where(df['senkou_span_a'] > df['senkou_span_b'], 'bullish', 'bearish')
            
            # Price position relative to cloud
            df['price_vs_cloud'] = np.where(
                df['Close'] > df['cloud_top'], 'above',
                np.where(df['Close'] < df['cloud_bottom'], 'below', 'inside')
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating Ichimoku: {str(e)}")
            return None
    
    def _get_current_ichimoku_state(self, data: pd.DataFrame) -> Optional[IchimokuComponents]:
        """Get current Ichimoku component values"""
        try:
            current_idx = len(data) - 1
            
            return IchimokuComponents(
                tenkan_sen=data['tenkan_sen'].iloc[current_idx],
                kijun_sen=data['kijun_sen'].iloc[current_idx],
                senkou_span_a=data['senkou_span_a'].iloc[current_idx],
                senkou_span_b=data['senkou_span_b'].iloc[current_idx],
                chikou_span=data['chikou_span'].iloc[current_idx - self.displacement] if current_idx >= self.displacement else 0,
                cloud_top=data['cloud_top'].iloc[current_idx],
                cloud_bottom=data['cloud_bottom'].iloc[current_idx],
                price_vs_cloud=data['price_vs_cloud'].iloc[current_idx],
                cloud_color=data['cloud_color'].iloc[current_idx],
                cloud_thickness=data['cloud_thickness'].iloc[current_idx]
            )
        except Exception as e:
            self.logger.error(f"Error getting current state: {str(e)}")
            return None
    
    def _check_cloud_breakout(self, data: pd.DataFrame, current: IchimokuComponents, 
                             symbol: str, timeframe: str) -> Optional[Signal]:
        """Check for cloud breakout signals"""
        try:
            current_price = data['Close'].iloc[-1]
            prev_position = data['price_vs_cloud'].iloc[-2]
            
            # Bullish breakout
            if prev_position != 'above' and current.price_vs_cloud == 'above':
                if current.cloud_color == 'bullish' and current.cloud_thickness >= self.min_cloud_thickness:
                    confidence = min(0.85, 0.65 + (current.cloud_thickness / 100))
                    
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name=self.strategy_name,
                        signal_type=SignalType.BUY,
                        confidence=confidence,
                        price=current_price,
                        timeframe=timeframe,
                        strength=0.8,
                        stop_loss=current.cloud_bottom - 5,
                        take_profit=current_price + (current_price - current.cloud_bottom) * 2,
                        metadata={
                            'signal_reason': 'cloud_breakout_bullish',
                            'cloud_thickness': current.cloud_thickness,
                            'tenkan_position': current.tenkan_sen
                        }
                    )
            
            # Bearish breakout
            elif prev_position != 'below' and current.price_vs_cloud == 'below':
                if current.cloud_color == 'bearish' and current.cloud_thickness >= self.min_cloud_thickness:
                    confidence = min(0.85, 0.65 + (current.cloud_thickness / 100))
                    
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name=self.strategy_name,
                        signal_type=SignalType.SELL,
                        confidence=confidence,
                        price=current_price,
                        timeframe=timeframe,
                        strength=0.8,
                        stop_loss=current.cloud_top + 5,
                        take_profit=current_price - (current.cloud_top - current_price) * 2,
                        metadata={
                            'signal_reason': 'cloud_breakout_bearish',
                            'cloud_thickness': current.cloud_thickness,
                            'tenkan_position': current.tenkan_sen
                        }
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking cloud breakout: {str(e)}")
            return None
    
    def _check_tk_cross(self, data: pd.DataFrame, current: IchimokuComponents,
                       symbol: str, timeframe: str) -> Optional[Signal]:
        """Check for Tenkan-Kijun cross signals"""
        try:
            current_price = data['Close'].iloc[-1]
            prev_tenkan = data['tenkan_sen'].iloc[-2]
            prev_kijun = data['kijun_sen'].iloc[-2]
            
            # Bullish TK Cross
            if prev_tenkan <= prev_kijun and current.tenkan_sen > current.kijun_sen:
                if current.price_vs_cloud == 'above':
                    confidence = 0.75
                    
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name=self.strategy_name,
                        signal_type=SignalType.BUY,
                        confidence=confidence,
                        price=current_price,
                        timeframe=timeframe,
                        strength=0.7,
                        stop_loss=current.kijun_sen - 5,
                        take_profit=current_price + (current_price - current.kijun_sen) * 2.5,
                        metadata={
                            'signal_reason': 'tk_cross_bullish',
                            'price_position': current.price_vs_cloud
                        }
                    )
            
            # Bearish TK Cross
            elif prev_tenkan >= prev_kijun and current.tenkan_sen < current.kijun_sen:
                if current.price_vs_cloud == 'below':
                    confidence = 0.75
                    
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name=self.strategy_name,
                        signal_type=SignalType.SELL,
                        confidence=confidence,
                        price=current_price,
                        timeframe=timeframe,
                        strength=0.7,
                        stop_loss=current.kijun_sen + 5,
                        take_profit=current_price - (current.kijun_sen - current_price) * 2.5,
                        metadata={
                            'signal_reason': 'tk_cross_bearish',
                            'price_position': current.price_vs_cloud
                        }
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking TK cross: {str(e)}")
            return None
    
    def _check_chikou_span(self, data: pd.DataFrame, current: IchimokuComponents,
                          symbol: str, timeframe: str) -> Optional[Signal]:
        """Check Chikou Span confirmation"""
        try:
            if len(data) < self.displacement + 5:
                return None
            
            current_price = data['Close'].iloc[-1]
            chikou_idx = len(data) - self.displacement - 1
            
            if chikou_idx >= 0:
                chikou_price = data['Close'].iloc[-1]
                historical_price = data['Close'].iloc[chikou_idx]
                historical_cloud_top = data['cloud_top'].iloc[chikou_idx]
                historical_cloud_bottom = data['cloud_bottom'].iloc[chikou_idx]
                
                # Bullish Chikou confirmation
                if chikou_price > historical_cloud_top and current.price_vs_cloud == 'above':
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name=self.strategy_name,
                        signal_type=SignalType.BUY,
                        confidence=0.70,
                        price=current_price,
                        timeframe=timeframe,
                        strength=0.65,
                        stop_loss=current.cloud_bottom - 3,
                        take_profit=current_price + (current_price - current.cloud_bottom) * 2,
                        metadata={'signal_reason': 'chikou_confirmation_bullish'}
                    )
                
                # Bearish Chikou confirmation
                elif chikou_price < historical_cloud_bottom and current.price_vs_cloud == 'below':
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name=self.strategy_name,
                        signal_type=SignalType.SELL,
                        confidence=0.70,
                        price=current_price,
                        timeframe=timeframe,
                        strength=0.65,
                        stop_loss=current.cloud_top + 3,
                        take_profit=current_price - (current.cloud_top - current_price) * 2,
                        metadata={'signal_reason': 'chikou_confirmation_bearish'}
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking Chikou span: {str(e)}")
            return None
    
    def _get_higher_timeframe_confirmation(self, symbol: str, timeframe: str) -> List[Signal]:
        """Get confirmation from higher timeframe"""
        # Implementation for multi-timeframe analysis
        return []
    
    def _detect_cloud_breakout_potential(self, data: pd.DataFrame) -> str:
        """Detect potential for cloud breakout"""
        try:
            current_position = data['price_vs_cloud'].iloc[-1]
            recent_positions = data['price_vs_cloud'].iloc[-5:].value_counts()
            
            if current_position == 'inside':
                if 'above' in recent_positions:
                    return 'potential_bullish_breakout'
                elif 'below' in recent_positions:
                    return 'potential_bearish_breakout'
            return 'no_breakout_detected'
        except:
            return 'unknown'
    
    def _detect_tk_cross_potential(self, data: pd.DataFrame) -> str:
        """Detect potential for TK cross"""
        try:
            tenkan = data['tenkan_sen'].iloc[-1]
            kijun = data['kijun_sen'].iloc[-1]
            distance = abs(tenkan - kijun)
            
            if distance < 2:
                if tenkan > kijun:
                    return 'potential_bearish_cross'
                else:
                    return 'potential_bullish_cross'
            return 'no_cross_imminent'
        except:
            return 'unknown'
    
    def _detect_kumo_twist(self, data: pd.DataFrame) -> str:
        """Detect Kumo twist (cloud color change)"""
        try:
            current_color = data['cloud_color'].iloc[-1]
            future_colors = data['cloud_color'].iloc[-5:]
            
            if len(future_colors.unique()) > 1:
                return f'kumo_twist_detected_{current_color}'
            return 'no_kumo_twist'
        except:
            return 'unknown'
    
    def _check_chikou_confirmation(self, data: pd.DataFrame) -> bool:
        """Check if Chikou span confirms the trend"""
        try:
            if len(data) < self.displacement + 1:
                return False
            
            chikou_idx = len(data) - self.displacement - 1
            if chikou_idx >= 0:
                current_price = data['Close'].iloc[-1]
                historical_price = data['Close'].iloc[chikou_idx]
                return current_price > historical_price
            return False
        except:
            return False
    
    def _calculate_trend_strength(self, data: pd.DataFrame, current: IchimokuComponents) -> float:
        """Calculate overall trend strength (0-1)"""
        try:
            strength_factors = []
            
            # Price vs Cloud (40% weight)
            if current.price_vs_cloud == 'above':
                strength_factors.append(0.4)
            elif current.price_vs_cloud == 'below':
                strength_factors.append(-0.4)
            else:
                strength_factors.append(0)
            
            # TK Relationship (30% weight)
            if current.tenkan_sen > current.kijun_sen:
                strength_factors.append(0.3)
            else:
                strength_factors.append(-0.3)
            
            # Cloud color (20% weight)
            if current.cloud_color == 'bullish':
                strength_factors.append(0.2)
            else:
                strength_factors.append(-0.2)
            
            # Chikou confirmation (10% weight)
            if self._check_chikou_confirmation(data):
                strength_factors.append(0.1)
            else:
                strength_factors.append(-0.1)
            
            total_strength = sum(strength_factors)
            return abs(total_strength)  # Return absolute strength 0-1
            
        except:
            return 0.5
    
    def _generate_recommendation(self, current: IchimokuComponents) -> str:
        """Generate trading recommendation based on current state"""
        if current.price_vs_cloud == 'above' and current.cloud_color == 'bullish':
            if current.tenkan_sen > current.kijun_sen:
                return 'STRONG_BUY'
            return 'BUY'
        elif current.price_vs_cloud == 'below' and current.cloud_color == 'bearish':
            if current.tenkan_sen < current.kijun_sen:
                return 'STRONG_SELL'
            return 'SELL'
        elif current.price_vs_cloud == 'inside':
            return 'WAIT_FOR_BREAKOUT'
        else:
            return 'NEUTRAL'


# Testing function
if __name__ == "__main__":
    """Test the modified Ichimoku strategy"""
    
    # Test configuration
    test_config = {
        'parameters': {
            'confidence_threshold': 0.65,
            'lookback_period': 200,
            'timeframe_primary': 'M15',
            'timeframe_secondary': 'H1'
        }
    }
    
    # Mock MT5 manager for testing
    class MockMT5Manager:
        def get_historical_data(self, symbol, timeframe, bars):
            # Generate sample data
            dates = pd.date_range(start=datetime.now() - timedelta(days=10), 
                                 end=datetime.now(), freq='15Min')[:bars]
            
            np.random.seed(42)
            close_prices = 1950 + np.cumsum(np.random.randn(len(dates)) * 2)
            
            data = pd.DataFrame({
                'Open': close_prices + np.random.randn(len(dates)) * 0.5,
                'High': close_prices + np.abs(np.random.randn(len(dates)) * 3),
                'Low': close_prices - np.abs(np.random.randn(len(dates)) * 3),
                'Close': close_prices,
                'Volume': np.random.randint(100, 1000, len(dates))
            }, index=dates)
            
            return data
    
    # Create strategy instance
    mock_mt5 = MockMT5Manager()
    strategy = IchimokuStrategy(test_config, mock_mt5)
    
    print("="*60)
    print("TESTING MODIFIED ICHIMOKU STRATEGY")
    print("="*60)
    
    # Test signal generation
    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - {signal.signal_type.value} at {signal.price:.2f}, Confidence: {signal.confidence:.2f}")
    
    # Test analysis
    print("\n2. Testing analysis method:")
    mock_data = mock_mt5.get_historical_data("XAUUSDm", "M15", 200)
    analysis = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis keys: {list(analysis.keys())}")
    if 'current_values' in analysis:
        print(f"   Current Tenkan: {analysis['current_values'].get('tenkan_sen', 'N/A'):.2f}")
        print(f"   Current Kijun: {analysis['current_values'].get('kijun_sen', 'N/A'):.2f}")
    
    # Test performance summary
    print("\n3. Testing performance tracking:")
    summary = strategy.get_performance_summary()
    print(f"   {summary}")
    
    print("\n" + "="*60)
    print("ICHIMOKU STRATEGY TEST COMPLETED!")
    print("="*60)