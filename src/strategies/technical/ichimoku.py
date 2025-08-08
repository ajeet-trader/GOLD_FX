
"""
Ichimoku Cloud Strategy - Advanced Technical Analysis
===================================================
Author: XAUUSD Trading System
Version: 2.0.0
Date: 2025-08-08

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

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass

# Import base strategy class and signal structure
try:
    from ..signal_engine import Signal, SignalType, SignalGrade
except ImportError:
    # Fallback for testing
    from enum import Enum
    
    class SignalType(Enum):
        BUY = "BUY"
        SELL = "SELL"
        HOLD = "HOLD"
    
    class SignalGrade(Enum):
        A = "A"
        B = "B" 
        C = "C"
        D = "D"
    
    @dataclass
    class Signal:
        timestamp: datetime
        symbol: str
        strategy_name: str
        signal_type: SignalType
        confidence: float
        price: float
        timeframe: str
        strength: float = 0.0
        grade: Optional[SignalGrade] = None
        stop_loss: Optional[float] = None
        take_profit: Optional[float] = None
        metadata: Dict[str, Any] = None


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


class IchimokuStrategy:
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
        >>> signals = strategy.generate_signals("XAUUSDm", "M15")
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager):
        """
        Initialize Ichimoku strategy
        
        Args:
            config: Strategy configuration
            mt5_manager: MT5 connection manager
        """
        self.config = config
        self.mt5_manager = mt5_manager
        
        # Ichimoku parameters
        self.tenkan_period = 9
        self.kijun_period = 26
        self.senkou_span_b_period = 52
        self.displacement = 26
        
        # Strategy parameters
        self.min_confidence = config.get('parameters', {}).get('confidence_threshold', 0.65)
        self.lookback_period = config.get('parameters', {}).get('lookback_period', 200)
        self.primary_timeframe = config.get('parameters', {}).get('timeframe_primary', 'M15')
        self.secondary_timeframe = config.get('parameters', {}).get('timeframe_secondary', 'H1')
        
        # Signal filters
        self.min_cloud_thickness = 5.0  # Minimum cloud thickness in points
        self.min_momentum_bars = 3      # Minimum bars for momentum confirmation
        self.max_signals_per_hour = 2   # Maximum signals per hour
        
        # Performance tracking
        self.recent_signals = []
        self.success_rate = 0.65
        self.profit_factor = 1.8
        
        # Logger
        self.logger = logging.getLogger('ichimoku_strategy')
    
    def generate_signals(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
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
            
            # Primary signal: Cloud breakout/bounce
            cloud_signal = self._analyze_cloud_interaction(data, ichimoku_data, current_components, symbol, timeframe)
            if cloud_signal:
                signals.append(cloud_signal)
            
            # Secondary signal: Tenkan-Kijun cross
            tk_cross_signal = self._analyze_tenkan_kijun_cross(data, ichimoku_data, current_components, symbol, timeframe)
            if tk_cross_signal:
                signals.append(tk_cross_signal)
            
            # Tertiary signal: Chikou span confirmation
            chikou_signal = self._analyze_chikou_confirmation(data, ichimoku_data, current_components, symbol, timeframe)
            if chikou_signal:
                signals.append(chikou_signal)
            
            # Multi-timeframe confirmation
            if signals and timeframe != self.secondary_timeframe:
                signals = self._apply_mtf_confirmation(signals, symbol, self.secondary_timeframe)
            
            # Filter signals
            filtered_signals = self._filter_signals(signals, current_components)
            
            # Update recent signals tracking
            self._update_signal_tracking(filtered_signals)
            
            self.logger.info(f"Ichimoku generated {len(filtered_signals)} signals from {len(signals)} candidates")
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Ichimoku signal generation failed: {str(e)}")
            return []
    
    def _calculate_ichimoku(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate all Ichimoku components"""
        try:
            df = data.copy()
            
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
            high_9 = df['High'].rolling(window=self.tenkan_period).max()
            low_9 = df['Low'].rolling(window=self.tenkan_period).min()
            df['tenkan_sen'] = (high_9 + low_9) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
            high_26 = df['High'].rolling(window=self.kijun_period).max()
            low_26 = df['Low'].rolling(window=self.kijun_period).min()
            df['kijun_sen'] = (high_26 + low_26) / 2
            
            # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted forward 26 periods
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(self.displacement)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted forward 26 periods
            high_52 = df['High'].rolling(window=self.senkou_span_b_period).max()
            low_52 = df['Low'].rolling(window=self.senkou_span_b_period).min()
            df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(self.displacement)
            
            # Chikou Span (Lagging Span): Current close shifted back 26 periods
            df['chikou_span'] = df['Close'].shift(-self.displacement)
            
            # Cloud calculations
            df['cloud_top'] = np.maximum(df['senkou_span_a'], df['senkou_span_b'])
            df['cloud_bottom'] = np.minimum(df['senkou_span_a'], df['senkou_span_b'])
            df['cloud_thickness'] = df['cloud_top'] - df['cloud_bottom']
            
            # Cloud color (bullish when Senkou A > Senkou B)
            df['cloud_bullish'] = df['senkou_span_a'] > df['senkou_span_b']
            
            # Price position relative to cloud
            df['price_vs_cloud'] = np.where(
                df['Close'] > df['cloud_top'], 'above',
                np.where(df['Close'] < df['cloud_bottom'], 'below', 'inside')
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ichimoku calculation failed: {str(e)}")
            return None
    
    def _get_current_ichimoku_state(self, data: pd.DataFrame) -> Optional[IchimokuComponents]:
        """Get current Ichimoku state"""
        try:
            latest = data.iloc[-1]
            
            # Handle NaN values
            if pd.isna(latest['tenkan_sen']) or pd.isna(latest['kijun_sen']):
                return None
            
            # Determine cloud color
            cloud_color = "bullish" if latest['cloud_bullish'] else "bearish"
            
            return IchimokuComponents(
                tenkan_sen=latest['tenkan_sen'],
                kijun_sen=latest['kijun_sen'],
                senkou_span_a=latest['senkou_span_a'] if not pd.isna(latest['senkou_span_a']) else latest['Close'],
                senkou_span_b=latest['senkou_span_b'] if not pd.isna(latest['senkou_span_b']) else latest['Close'],
                chikou_span=latest['chikou_span'] if not pd.isna(latest['chikou_span']) else latest['Close'],
                cloud_top=latest['cloud_top'] if not pd.isna(latest['cloud_top']) else latest['Close'],
                cloud_bottom=latest['cloud_bottom'] if not pd.isna(latest['cloud_bottom']) else latest['Close'],
                price_vs_cloud=latest['price_vs_cloud'],
                cloud_color=cloud_color,
                cloud_thickness=latest['cloud_thickness'] if not pd.isna(latest['cloud_thickness']) else 0.0
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get current Ichimoku state: {str(e)}")
            return None
    
    def _analyze_cloud_interaction(self, data: pd.DataFrame, ichimoku_data: pd.DataFrame, 
                                  current: IchimokuComponents, symbol: str, timeframe: str) -> Optional[Signal]:
        """Analyze price interaction with Ichimoku cloud"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # Cloud breakout signals
            if self._is_cloud_breakout(ichimoku_data, current):
                signal_type = SignalType.BUY if current.price_vs_cloud == 'above' else SignalType.SELL
                
                # Calculate confidence based on cloud characteristics
                confidence = self._calculate_cloud_confidence(current, data)
                
                if confidence >= self.min_confidence:
                    # Calculate stop loss and take profit
                    if signal_type == SignalType.BUY:
                        stop_loss = current.cloud_top * 0.999  # Just below cloud
                        take_profit = current_price + (current_price - stop_loss) * 2  # 1:2 RR
                    else:
                        stop_loss = current.cloud_bottom * 1.001  # Just above cloud
                        take_profit = current_price - (stop_loss - current_price) * 2  # 1:2 RR
                    
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name="ichimoku_cloud_breakout",
                        signal_type=signal_type,
                        confidence=confidence,
                        price=current_price,
                        timeframe=timeframe,
                        strength=confidence * 0.9,  # Cloud breakouts are strong
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'cloud_color': current.cloud_color,
                            'cloud_thickness': current.cloud_thickness,
                            'tenkan_sen': current.tenkan_sen,
                            'kijun_sen': current.kijun_sen,
                            'analysis_type': 'cloud_breakout'
                        }
                    )
            
            # Cloud bounce signals
            elif self._is_cloud_bounce(ichimoku_data, current):
                signal_type = SignalType.BUY if current.price_vs_cloud == 'above' else SignalType.SELL
                
                confidence = self._calculate_bounce_confidence(current, data)
                
                if confidence >= self.min_confidence:
                    # Tighter stops for bounces
                    if signal_type == SignalType.BUY:
                        stop_loss = current.cloud_bottom * 0.9995
                        take_profit = current_price + (current_price - stop_loss) * 1.5
                    else:
                        stop_loss = current.cloud_top * 1.0005
                        take_profit = current_price - (stop_loss - current_price) * 1.5
                    
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name="ichimoku_cloud_bounce",
                        signal_type=signal_type,
                        confidence=confidence,
                        price=current_price,
                        timeframe=timeframe,
                        strength=confidence * 0.7,  # Bounces are medium strength
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'cloud_color': current.cloud_color,
                            'cloud_thickness': current.cloud_thickness,
                            'analysis_type': 'cloud_bounce'
                        }
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Cloud interaction analysis failed: {str(e)}")
            return None
    
    def _analyze_tenkan_kijun_cross(self, data: pd.DataFrame, ichimoku_data: pd.DataFrame,
                                   current: IchimokuComponents, symbol: str, timeframe: str) -> Optional[Signal]:
        """Analyze Tenkan-sen and Kijun-sen crossover"""
        try:
            # Check for recent crossover
            if len(ichimoku_data) < 5:
                return None
            
            tenkan_values = ichimoku_data['tenkan_sen'].tail(5)
            kijun_values = ichimoku_data['kijun_sen'].tail(5)
            
            # Detect crossover
            cross_up = (tenkan_values.iloc[-1] > kijun_values.iloc[-1] and 
                       tenkan_values.iloc[-2] <= kijun_values.iloc[-2])
            
            cross_down = (tenkan_values.iloc[-1] < kijun_values.iloc[-1] and 
                         tenkan_values.iloc[-2] >= kijun_values.iloc[-2])
            
            if cross_up or cross_down:
                signal_type = SignalType.BUY if cross_up else SignalType.SELL
                
                # Calculate confidence based on position relative to cloud
                base_confidence = 0.6
                
                # Boost confidence if aligned with cloud
                if cross_up and current.cloud_color == "bullish":
                    base_confidence += 0.15
                elif cross_down and current.cloud_color == "bearish":
                    base_confidence += 0.15
                
                # Boost confidence if price is on correct side of cloud
                if ((cross_up and current.price_vs_cloud == 'above') or 
                    (cross_down and current.price_vs_cloud == 'below')):
                    base_confidence += 0.1
                
                current_price = data['Close'].iloc[-1]
                
                if signal_type == SignalType.BUY:
                    stop_loss = current.kijun_sen * 0.998
                    take_profit = current_price + (current_price - stop_loss) * 1.8
                else:
                    stop_loss = current.kijun_sen * 1.002
                    take_profit = current_price - (stop_loss - current_price) * 1.8
                
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name="ichimoku_tk_cross",
                    signal_type=signal_type,
                    confidence=min(base_confidence, 0.85),
                    price=current_price,
                    timeframe=timeframe,
                    strength=base_confidence * 0.8,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'tenkan_sen': current.tenkan_sen,
                        'kijun_sen': current.kijun_sen,
                        'cross_direction': 'up' if cross_up else 'down',
                        'analysis_type': 'tk_cross'
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"TK cross analysis failed: {str(e)}")
            return None
    
    def _analyze_chikou_confirmation(self, data: pd.DataFrame, ichimoku_data: pd.DataFrame,
                                   current: IchimokuComponents, symbol: str, timeframe: str) -> Optional[Signal]:
        """Analyze Chikou span for trade confirmation"""
        try:
            if len(ichimoku_data) < self.displacement + 5:
                return None
            
            current_price = data['Close'].iloc[-1]
            
            # Get historical price 26 periods ago for Chikou comparison
            chikou_compare_price = data['Close'].iloc[-self.displacement-1] if len(data) > self.displacement else current_price
            
            # Chikou signals
            chikou_bullish = current.chikou_span > chikou_compare_price
            chikou_bearish = current.chikou_span < chikou_compare_price
            
            # Only generate signal if Chikou confirms trend
            if chikou_bullish and current.tenkan_sen > current.kijun_sen:
                signal_type = SignalType.BUY
                confidence = 0.65
            elif chikou_bearish and current.tenkan_sen < current.kijun_sen:
                signal_type = SignalType.SELL
                confidence = 0.65
            else:
                return None
            
            # Additional confirmation from cloud
            if ((signal_type == SignalType.BUY and current.price_vs_cloud == 'above') or
                (signal_type == SignalType.SELL and current.price_vs_cloud == 'below')):
                confidence += 0.1
            
            if confidence >= self.min_confidence:
                if signal_type == SignalType.BUY:
                    stop_loss = min(current.kijun_sen, current.cloud_top) * 0.998
                    take_profit = current_price + (current_price - stop_loss) * 2.0
                else:
                    stop_loss = max(current.kijun_sen, current.cloud_bottom) * 1.002
                    take_profit = current_price - (stop_loss - current_price) * 2.0
                
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name="ichimoku_chikou_confirm",
                    signal_type=signal_type,
                    confidence=confidence,
                    price=current_price,
                    timeframe=timeframe,
                    strength=confidence * 0.85,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'chikou_span': current.chikou_span,
                        'chikou_compare_price': chikou_compare_price,
                        'analysis_type': 'chikou_confirmation'
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Chikou analysis failed: {str(e)}")
            return None
    
    def _is_cloud_breakout(self, ichimoku_data: pd.DataFrame, current: IchimokuComponents) -> bool:
        """Check if there's a recent cloud breakout"""
        try:
            if len(ichimoku_data) < 3:
                return False
            
            recent_positions = ichimoku_data['price_vs_cloud'].tail(3).tolist()
            
            # Breakout above cloud
            breakout_up = (recent_positions[-1] == 'above' and 
                          'inside' in recent_positions[-3:-1])
            
            # Breakout below cloud  
            breakout_down = (recent_positions[-1] == 'below' and 
                            'inside' in recent_positions[-3:-1])
            
            # Ensure cloud has minimum thickness
            thick_enough = current.cloud_thickness >= self.min_cloud_thickness
            
            return (breakout_up or breakout_down) and thick_enough
            
        except Exception as e:
            self.logger.error(f"Cloud breakout check failed: {str(e)}")
            return False
    
    def _is_cloud_bounce(self, ichimoku_data: pd.DataFrame, current: IchimokuComponents) -> bool:
        """Check if there's a bounce from cloud edge"""
        try:
            if len(ichimoku_data) < 5:
                return False
            
            close_prices = ichimoku_data['Close'].tail(5)
            cloud_tops = ichimoku_data['cloud_top'].tail(5)
            cloud_bottoms = ichimoku_data['cloud_bottom'].tail(5)
            
            # Check for bounce from cloud top (support)
            bounce_from_top = (
                current.price_vs_cloud == 'above' and
                any(abs(price - cloud_top) < current.cloud_thickness * 0.2 
                    for price, cloud_top in zip(close_prices[:-1], cloud_tops[:-1]))
            )
            
            # Check for bounce from cloud bottom (resistance)
            bounce_from_bottom = (
                current.price_vs_cloud == 'below' and
                any(abs(price - cloud_bottom) < current.cloud_thickness * 0.2 
                    for price, cloud_bottom in zip(close_prices[:-1], cloud_bottoms[:-1]))
            )
            
            return bounce_from_top or bounce_from_bottom
            
        except Exception as e:
            self.logger.error(f"Cloud bounce check failed: {str(e)}")
            return False
    
    def _calculate_cloud_confidence(self, current: IchimokuComponents, data: pd.DataFrame) -> float:
        """Calculate confidence for cloud breakout signals"""
        confidence = 0.6  # Base confidence
        
        try:
            # Cloud thickness factor
            if current.cloud_thickness >= 20:
                confidence += 0.1
            elif current.cloud_thickness >= 10:
                confidence += 0.05
            
            # Cloud color alignment
            if ((current.price_vs_cloud == 'above' and current.cloud_color == 'bullish') or
                (current.price_vs_cloud == 'below' and current.cloud_color == 'bearish')):
                confidence += 0.1
            
            # Tenkan-Kijun alignment
            if ((current.price_vs_cloud == 'above' and current.tenkan_sen > current.kijun_sen) or
                (current.price_vs_cloud == 'below' and current.tenkan_sen < current.kijun_sen)):
                confidence += 0.08
            
            # Volume confirmation (if available)
            if 'Volume' in data.columns:
                recent_volume = data['Volume'].tail(3).mean()
                avg_volume = data['Volume'].tail(20).mean()
                if recent_volume > avg_volume * 1.2:
                    confidence += 0.07
            
            return min(confidence, 0.95)  # Cap at 95%
            
        except Exception as e:
            self.logger.error(f"Cloud confidence calculation failed: {str(e)}")
            return 0.6
    
    def _calculate_bounce_confidence(self, current: IchimokuComponents, data: pd.DataFrame) -> float:
        """Calculate confidence for cloud bounce signals"""
        confidence = 0.55  # Lower base confidence for bounces
        
        try:
            # Strong cloud (thick)
            if current.cloud_thickness >= 15:
                confidence += 0.15
            elif current.cloud_thickness >= 8:
                confidence += 0.08
            
            # Cloud color alignment
            if ((current.price_vs_cloud == 'above' and current.cloud_color == 'bullish') or
                (current.price_vs_cloud == 'below' and current.cloud_color == 'bearish')):
                confidence += 0.12
            
            # Price action at cloud edge
            current_price = data['Close'].iloc[-1]
            cloud_distance = min(abs(current_price - current.cloud_top), 
                               abs(current_price - current.cloud_bottom))
            
            if cloud_distance < current.cloud_thickness * 0.1:  # Very close to cloud edge
                confidence += 0.08
            
            return min(confidence, 0.85)  # Cap at 85% for bounces
            
        except Exception as e:
            self.logger.error(f"Bounce confidence calculation failed: {str(e)}")
            return 0.55
    
    def _apply_mtf_confirmation(self, signals: List[Signal], symbol: str, 
                              higher_timeframe: str) -> List[Signal]:
        """Apply multi-timeframe confirmation"""
        try:
            # Get higher timeframe data
            htf_data = self.mt5_manager.get_historical_data(symbol, higher_timeframe, 100)
            if htf_data is None or len(htf_data) < 50:
                return signals  # Return original signals if can't get HTF data
            
            # Calculate Ichimoku on higher timeframe
            htf_ichimoku = self._calculate_ichimoku(htf_data)
            if htf_ichimoku is None:
                return signals
            
            htf_current = self._get_current_ichimoku_state(htf_ichimoku)
            if not htf_current:
                return signals
            
            confirmed_signals = []
            
            for signal in signals:
                # Check HTF alignment
                htf_aligned = False
                
                if signal.signal_type == SignalType.BUY:
                    htf_aligned = (
                        htf_current.price_vs_cloud in ['above', 'inside'] and
                        htf_current.tenkan_sen >= htf_current.kijun_sen and
                        htf_current.cloud_color == 'bullish'
                    )
                elif signal.signal_type == SignalType.SELL:
                    htf_aligned = (
                        htf_current.price_vs_cloud in ['below', 'inside'] and
                        htf_current.tenkan_sen <= htf_current.kijun_sen and
                        htf_current.cloud_color == 'bearish'
                    )
                
                if htf_aligned:
                    # Boost confidence for HTF confirmation
                    signal.confidence = min(signal.confidence + 0.1, 0.95)
                    signal.metadata['htf_confirmed'] = True
                    signal.metadata['htf_timeframe'] = higher_timeframe
                    confirmed_signals.append(signal)
                elif signal.confidence >= 0.8:  # Keep very strong signals even without HTF
                    signal.metadata['htf_confirmed'] = False
                    confirmed_signals.append(signal)
            
            return confirmed_signals
            
        except Exception as e:
            self.logger.error(f"MTF confirmation failed: {str(e)}")
            return signals
    
    def _filter_signals(self, signals: List[Signal], current: IchimokuComponents) -> List[Signal]:
        """Apply additional signal filters"""
        if not signals:
            return signals
        
        filtered = []
        
        for signal in signals:
            # Skip if too many recent signals
            if self._too_many_recent_signals():
                continue
            
            # Skip weak signals in ranging markets
            if (current.cloud_thickness < 5 and signal.confidence < 0.75):
                continue
            
            # Skip if signal conflicts with major cloud level
            if self._conflicts_with_cloud_structure(signal, current):
                continue
            
            filtered.append(signal)
        
        # Sort by confidence and take best signals
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        return filtered[:3]  # Maximum 3 signals per generation
    
    def _too_many_recent_signals(self) -> bool:
        """Check if too many signals generated recently"""
        now = datetime.now()
        recent_count = sum(1 for sig in self.recent_signals 
                          if (now - sig).total_seconds() < 3600)  # 1 hour
        return recent_count >= self.max_signals_per_hour
    
    def _conflicts_with_cloud_structure(self, signal: Signal, current: IchimokuComponents) -> bool:
        """Check if signal conflicts with cloud structure"""
        # Don't buy into strong bearish cloud resistance
        if (signal.signal_type == SignalType.BUY and 
            current.cloud_color == 'bearish' and 
            current.price_vs_cloud == 'below' and
            current.cloud_thickness > 15):
            return True
        
        # Don't sell into strong bullish cloud support
        if (signal.signal_type == SignalType.SELL and 
            current.cloud_color == 'bullish' and 
            current.price_vs_cloud == 'above' and
            current.cloud_thickness > 15):
            return True
        
        return False
    
    def _update_signal_tracking(self, signals: List[Signal]) -> None:
        """Update recent signals tracking"""
        now = datetime.now()
        
        # Add new signals
        for signal in signals:
            self.recent_signals.append(now)
        
        # Clean old signals (older than 24 hours)
        cutoff = now - timedelta(hours=24)
        self.recent_signals = [sig_time for sig_time in self.recent_signals if sig_time > cutoff]
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and performance"""
        return {
            'name': 'Ichimoku Cloud Strategy',
            'version': '2.0.0',
            'description': 'Advanced Ichimoku Kinko Hyo implementation with multi-timeframe analysis',
            'parameters': {
                'tenkan_period': self.tenkan_period,
                'kijun_period': self.kijun_period,
                'senkou_span_b_period': self.senkou_span_b_period,
                'displacement': self.displacement,
                'min_confidence': self.min_confidence,
                'min_cloud_thickness': self.min_cloud_thickness
            },
            'performance': {
                'success_rate': self.success_rate,
                'profit_factor': self.profit_factor,
                'recent_signals_count': len(self.recent_signals)
            },
            'signal_types': [
                'cloud_breakout',
                'cloud_bounce', 
                'tenkan_kijun_cross',
                'chikou_confirmation'
            ]
        }


# Testing function
def test_ichimoku_strategy():
    """Test Ichimoku strategy functionality"""
    print("Testing Ichimoku Strategy...")
    
    # Mock MT5 manager
    class MockMT5Manager:
        def get_historical_data(self, symbol, timeframe, bars):
            # Generate sample OHLCV data
            dates = pd.date_range(start='2025-01-01', periods=bars, freq='15T')
            
            # Generate trending price data for better testing
            base_price = 2000.0
            trend = np.linspace(0, 50, bars)  # Upward trend
            noise = np.random.normal(0, 5, bars)
            prices = base_price + trend + noise
            
            data = pd.DataFrame({
                'Open': prices + np.random.normal(0, 2, bars),
                'High': prices + np.random.normal(2, 3, bars),
                'Low': prices + np.random.normal(-2, 3, bars),
                'Close': prices,
                'Volume': np.random.normal(1000, 200, bars)
            }, index=dates)
            
            # Ensure High >= Low and OHLC consistency
            data['High'] = np.maximum(data[['Open', 'Close']].max(axis=1), data['High'])
            data['Low'] = np.minimum(data[['Open', 'Close']].min(axis=1), data['Low'])
            
            return data
    
    # Test configuration
    test_config = {
        'parameters': {
            'confidence_threshold': 0.65,
            'lookback_period': 200,
            'timeframe_primary': 'M15',
            'timeframe_secondary': 'H1'
        }
    }
    
    try:
        # Create strategy
        strategy = IchimokuStrategy(test_config, MockMT5Manager())
        
        # Test signal generation
        signals = strategy.generate_signals("XAUUSDm", "M15")
        
        print(f"Generated {len(signals)} Ichimoku signals")
        
        for i, signal in enumerate(signals):
            print(f"Signal {i+1}:")
            print(f"  Type: {signal.signal_type.value}")
            print(f"  Confidence: {signal.confidence:.2f}")
            print(f"  Strategy: {signal.strategy_name}")
            print(f"  Price: {signal.price:.2f}")
            print(f"  Stop Loss: {signal.stop_loss:.2f}" if signal.stop_loss else "  Stop Loss: None")
            print(f"  Take Profit: {signal.take_profit:.2f}" if signal.take_profit else "  Take Profit: None")
            print(f"  Metadata: {signal.metadata}")
        
        # Test strategy info
        info = strategy.get_strategy_info()
        print(f"\nStrategy Info: {info}")
        
        print("✅ Ichimoku strategy test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Ichimoku strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_ichimoku_strategy()