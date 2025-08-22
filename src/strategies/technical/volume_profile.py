
"""
Volume Profile Strategy - Volume-Based Support/Resistance
=======================================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-08-15

Advanced volume profile analysis for XAUUSD trading:
- Point of Control (POC) detection
- Value Area High/Low (VAH/VAL)
- High/Low Volume Nodes (HVN/LVN)
- Volume-based support/resistance
- Breakout and reversal signals

Features:
- Dynamic bin sizing based on price range
- Value area calculation (default 70%)
- Node detection with threshold
- Multi-signal generation near key levels
- Tolerance for approach/rejection signals

Dependencies:
    - pandas
    - numpy
    - datetime
"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

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
class VolumeProfile:
    """Volume profile data structure"""
    poc: float
    vah: float
    val: float
    high_volume_nodes: List[float]
    low_volume_nodes: List[float]
    volume_distribution: Dict[float, float]  # bin_price: volume


class VolumeProfileStrategy(AbstractStrategy):
    """
    Advanced Volume Profile Strategy
    
    This strategy builds a volume profile to identify key levels:
    - POC: Price with highest volume
    - Value Area: Range containing 70% of volume (VAH/VAL)
    - HVNs: High volume nodes (potential support/resistance)
    - LVNs: Low volume nodes (potential breakouts)
    
    Signal Generation:
    - Buy: Approach VAL from below, bounce off VAL, breakout up through LVN
    - Sell: Approach VAH from above, bounce off VAH, breakout down through LVN
    - Near POC: Reversal signals
    - Tolerance: Within 0.3% of levels for signals
    - Multiple signals if near multiple levels (with min distance filter)
    
    Example:
        >>> strategy = VolumeProfileStrategy(config, mt5_manager)
        >>> signals = strategy.generate_signal("XAUUSDm", "M15")
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """
        Initialize Volume Profile strategy
        
        Args:
            config: Strategy configuration
            mt5_manager: MT5 connection manager (optional)
            database: Database manager (optional)
        """
        super().__init__(config, mt5_manager, database)
        
        # Determine mode (CLI overrides config)
        self.mode = parse_mode() or self.config.get('mode', 'mock')

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

        # Strategy parameters - More lenient for more signals
        self.lookback_period = config.get('parameters', {}).get('lookback_period', 200)
        self.value_area_pct = config.get('parameters', {}).get('value_area_pct', 0.7)
        self.volume_node_threshold = config.get('parameters', {}).get('volume_node_threshold', 1.1)  # Lower threshold
        self.min_confidence = config.get('parameters', {}).get('confidence_threshold', 0.55)  # Lower threshold
        self.min_price_distance = config.get('parameters', {}).get('min_price_distance', 0.1)  # Smaller distance
        
        # Additional parameters - More signals
        self.bin_size_factor = 0.003  # 0.3% of price for bin size (more bins)
        self.tolerance_pct = 0.005  # 0.5% tolerance for level proximity (wider range)
        self.max_signals = 10  # Allow up to 10 signals per generation
        
        # Performance tracking (handled by parent)
        self.success_rate = 0.65
        self.profit_factor = 1.8
    
    def _create_mock_mt5(self):
        """Create mock MT5 manager with mode-specific data"""
        class MockMT5Manager:
            def __init__(self, mode):
                self.mode = mode
                
            def get_historical_data(self, symbol, timeframe, bars):
                dates = pd.date_range(start=datetime.now() - timedelta(days=10), 
                                     end=datetime.now(), freq='15Min')[:bars]
                
                np.random.seed(42 if self.mode == 'mock' else 123)
                base_prices = 1950 + np.cumsum(np.random.randn(len(dates)) * 2)
                
                # Create clustered volumes: high around 1950-1960, low around 1940-1950
                volumes = np.where((base_prices > 1950) & (base_prices < 1960), 
                                   np.random.uniform(800, 1200, len(dates)),
                                   np.random.uniform(200, 500, len(dates)))
                
                data = pd.DataFrame({
                    'Open': base_prices + np.random.randn(len(dates)) * 0.5,
                    'High': base_prices + np.abs(np.random.randn(len(dates)) * 3),
                    'Low': base_prices - np.abs(np.random.randn(len(dates)) * 3),
                    'Close': base_prices,
                    'Volume': volumes
                }, index=dates)
                
                return data
        
        return MockMT5Manager(self.mode)

    def generate_signal(self, symbol: str, timeframe: str) -> List[Signal]:
        """
        Generate volume profile-based trading signals
        
        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            List of trading signals
        """
        signals = []
        try:
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available")
                return []
                
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_period)
            if data is None or len(data) < 50:
                self.logger.warning(f"Insufficient data for volume profile: {len(data) if data is not None else 0} bars")
                return []
            
            # Build volume profile
            profile = self._build_volume_profile(data)
            if not profile:
                return []
            
            current_price = data['Close'].iloc[-1]
            tolerance = current_price * self.tolerance_pct
            
            # Generate signals near key levels
            key_levels = [profile.poc, profile.vah, profile.val] + profile.high_volume_nodes + profile.low_volume_nodes
            unique_levels = sorted(set(key_levels))  # Remove duplicates and sort
            
            last_signal_price = None
            for level in unique_levels:
                dist = abs(current_price - level)
                if dist > tolerance:
                    continue
                
                # Skip if too close to previous signal
                if last_signal_price and abs(level - last_signal_price) < self.min_price_distance:
                    continue
                
                # Determine signal type
                if level in profile.low_volume_nodes:
                    # LVN: Breakout signals
                    signal_type = SignalType.BUY if current_price > level else SignalType.SELL
                    base_conf = 0.75 - (dist / tolerance) * 0.1
                    metadata = {'level_type': 'LVN_breakout', 'level': level}
                elif level in profile.high_volume_nodes or level == profile.poc:
                    # HVN/POC: Reversal/bounce signals
                    signal_type = SignalType.BUY if current_price < level else SignalType.SELL
                    base_conf = 0.80 - (dist / tolerance) * 0.1
                    metadata = {'level_type': 'HVN_reversal' if level != profile.poc else 'POC_reversal', 'level': level}
                elif level == profile.val:
                    # VAL: Support (buy approach or bounce)
                    signal_type = SignalType.BUY
                    base_conf = 0.78 - (dist / tolerance) * 0.1
                    metadata = {'level_type': 'VAL_support', 'level': level}
                elif level == profile.vah:
                    # VAH: Resistance (sell approach or bounce)
                    signal_type = SignalType.SELL
                    base_conf = 0.78 - (dist / tolerance) * 0.1
                    metadata = {'level_type': 'VAH_resistance', 'level': level}
                else:
                    continue
                
                # Adjust confidence based on volume strength
                level_volume = profile.volume_distribution.get(level, 0)
                max_volume = max(profile.volume_distribution.values())
                conf_adjust = (level_volume / max_volume) * 0.15
                confidence = min(1.0, max(self.min_confidence, base_conf + conf_adjust))
                
                # Create signal
                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=signal_type,
                    confidence=confidence,
                    price=current_price,
                    timeframe=timeframe,
                    strength=confidence,  # Use confidence as strength
                    stop_loss=level - tolerance if signal_type == SignalType.BUY else level + tolerance,
                    take_profit=level + (tolerance * 2) if signal_type == SignalType.BUY else level - (tolerance * 2),
                    metadata=metadata
                )
                
                # Validate and add
                if self.validate_signal(signal):
                    signals.append(signal)
                    last_signal_price = level
                
                if len(signals) >= self.max_signals:
                    break
            
            # NEW: Add additional signal types if we have fewer signals
            if len(signals) < 5:
                additional_signals = self._generate_additional_signals(profile, current_price, symbol, timeframe, data)
                signals.extend(additional_signals)
                
                # Limit total signals
                signals = signals[:self.max_signals]
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            return []
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze data using volume profile
        
        Args:
            data: Historical OHLCV data
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            Dictionary of analysis results
        """
        try:
            profile = self._build_volume_profile(data)
            if not profile:
                return {}
            
            current_price = data['Close'].iloc[-1]
            
            # Determine position vs value area
            if current_price > profile.vah:
                position_vs_va = 'above'
            elif current_price < profile.val:
                position_vs_va = 'below'
            else:
                position_vs_va = 'within'
            
            return {
                'poc': profile.poc,
                'vah': profile.vah,
                'val': profile.val,
                'high_volume_nodes': profile.high_volume_nodes,
                'low_volume_nodes': profile.low_volume_nodes,
                'current_price': current_price,
                'position_vs_value_area': position_vs_va,
                'volume_distribution': profile.volume_distribution
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {}
    
    def _build_volume_profile(self, data: pd.DataFrame) -> Optional[VolumeProfile]:
        """Build volume profile from OHLCV data"""
        try:
            if len(data) < 50:
                return None
            
            # Determine bin size dynamically
            price_range = data['High'].max() - data['Low'].min()
            bin_size = max(0.5, price_range * self.bin_size_factor)
            
            # Create price bins
            min_price = data['Low'].min()
            max_price = data['High'].max()
            bins = np.arange(min_price, max_price + bin_size, bin_size)
            
            # Assign each bar to bins (using close price for simplicity)
            data['bin'] = pd.cut(data['Close'], bins=bins, labels=bins[:-1], right=False)
            
            # Calculate volume per bin
            volume_dist = data.groupby('bin')['Volume'].sum().to_dict()
            
            if not volume_dist:
                return None
            
            # POC: Price with max volume
            poc = max(volume_dist, key=volume_dist.get)
            
            # Sort by volume descending for value area
            sorted_vol = sorted(volume_dist.items(), key=lambda x: x[1], reverse=True)
            total_vol = sum(volume_dist.values())
            cum_vol = 0
            value_area_prices = []
            
            for price, vol in sorted_vol:
                value_area_prices.append(price)
                cum_vol += vol
                if cum_vol >= total_vol * self.value_area_pct:
                    break
            
            vah = max(value_area_prices)
            val = min(value_area_prices)
            
            # Nodes
            mean_vol = np.mean(list(volume_dist.values()))
            hvns = [p for p, v in volume_dist.items() if v > mean_vol * self.volume_node_threshold]
            lvns = [p for p, v in volume_dist.items() if v < mean_vol / self.volume_node_threshold]
            
            return VolumeProfile(
                poc=float(poc),
                vah=float(vah),
                val=float(val),
                high_volume_nodes=[float(p) for p in hvns],
                low_volume_nodes=[float(p) for p in lvns],
                volume_distribution={float(k): float(v) for k, v in volume_dist.items() if pd.notna(k)}
            )
            
        except Exception as e:
            self.logger.error(f"Volume profile build failed: {str(e)}")
            return None
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and parameters"""
        return {
            'name': 'Volume Profile',
            'version': '1.0.0',
            'description': 'Identifies high/low volume nodes, POC, and value areas for support/resistance and breakout signals.',
            'type': 'Technical',
            'parameters': {
                'lookback_period': self.lookback_period,
                'value_area_pct': self.value_area_pct,
                'volume_node_threshold': self.volume_node_threshold,
                'confidence_threshold': self.min_confidence,
                'min_price_distance': self.min_price_distance
            },
            'performance': {
                'success_rate': self.performance.win_rate,
                'profit_factor': self.performance.profit_factor
            }
        }

    def _generate_additional_signals(self, profile: VolumeProfile, current_price: float, 
                                   symbol: str, timeframe: str, data: pd.DataFrame) -> List[Signal]:
        """Generate additional signals to reach 5-10 signals per run"""
        additional_signals = []
        
        try:
            # 1. Half-distance signals (between major levels)
            major_levels = [profile.val, profile.poc, profile.vah]
            for i in range(len(major_levels) - 1):
                mid_level = (major_levels[i] + major_levels[i + 1]) / 2
                if abs(current_price - mid_level) / current_price < 0.008:  # Within 0.8%
                    signal_type = SignalType.BUY if current_price < mid_level else SignalType.SELL
                    confidence = 0.60 + abs(current_price - mid_level) / mid_level * 5
                    
                    additional_signals.append(Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name=self.strategy_name,
                        signal_type=signal_type,
                        confidence=min(confidence, 0.75),
                        price=current_price,
                        timeframe=timeframe,
                        strength=0.6,
                        stop_loss=mid_level - (mid_level * 0.003) if signal_type == SignalType.BUY else mid_level + (mid_level * 0.003),
                        take_profit=mid_level + (mid_level * 0.006) if signal_type == SignalType.BUY else mid_level - (mid_level * 0.006),
                        metadata={'level_type': 'mid_level', 'level': mid_level}
                    ))
            
            # 2. Volume spike signals (from recent high-volume periods)
            recent_volume = data['Volume'].iloc[-20:] 
            avg_volume = recent_volume.mean()
            for i in range(len(recent_volume) - 5, len(recent_volume)):
                if recent_volume.iloc[i] > avg_volume * 1.5:  # Volume spike
                    spike_price = data['Close'].iloc[-20 + i]
                    if abs(current_price - spike_price) / current_price < 0.01:  # Within 1%
                        signal_type = SignalType.BUY if current_price < spike_price else SignalType.SELL
                        confidence = 0.58 + (recent_volume.iloc[i] / avg_volume - 1.5) * 0.1
                        
                        additional_signals.append(Signal(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            strategy_name=self.strategy_name,
                            signal_type=signal_type,
                            confidence=min(confidence, 0.72),
                            price=current_price,
                            timeframe=timeframe,
                            strength=0.58,
                            stop_loss=spike_price - (spike_price * 0.004) if signal_type == SignalType.BUY else spike_price + (spike_price * 0.004),
                            take_profit=spike_price + (spike_price * 0.008) if signal_type == SignalType.BUY else spike_price - (spike_price * 0.008),
                            metadata={'level_type': 'volume_spike', 'level': spike_price, 'volume_ratio': recent_volume.iloc[i] / avg_volume}
                        ))
            
            # 3. Multiple HVN proximity signals
            nearby_hvns = [hvn for hvn in profile.high_volume_nodes if abs(current_price - hvn) / current_price < 0.015]
            for hvn in nearby_hvns[:5]:  # Max 5 nearby HVN signals
                signal_type = SignalType.SELL if current_price > hvn else SignalType.BUY
                distance_factor = 1 - (abs(current_price - hvn) / current_price / 0.015)
                confidence = 0.58 + distance_factor * 0.12
                
                additional_signals.append(Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=signal_type,
                    confidence=min(confidence, 0.70),
                    price=current_price,
                    timeframe=timeframe,
                    strength=0.58,
                    stop_loss=hvn - (hvn * 0.004) if signal_type == SignalType.BUY else hvn + (hvn * 0.004),
                    take_profit=hvn + (hvn * 0.008) if signal_type == SignalType.BUY else hvn - (hvn * 0.008),
                    metadata={'level_type': 'nearby_hvn', 'level': hvn}
                ))
            
            return additional_signals[:7]  # Max 7 additional signals
            
        except Exception as e:
            self.logger.error(f"Error generating additional signals: {str(e)}")
            return []


# Testing function
if __name__ == "__main__":
    """Test the Volume Profile strategy"""
    
    test_config = {
        'parameters': {
            'lookback_period': 200,
            'value_area_pct': 0.7,
            'volume_node_threshold': 1.3,
            'confidence_threshold': 0.65,
            'min_price_distance': 0.2,
            'mode': 'mock' # Added mode parameter
        }
    }
    
    # Create strategy instance
    strategy = VolumeProfileStrategy(test_config, mt5_manager=None) # Pass mt5_manager=None to trigger internal mock creation
    
    print("="*60)
    print("TESTING VOLUME PROFILE STRATEGY")
    print("============================================================")
    print(f"Running in {strategy.mode.upper()} mode") # Print mode
    
    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
        print(f"     Metadata: {signal.metadata}")
    
    print("\n2. Testing analysis method:")
    mock_data = strategy.mt5_manager.get_historical_data("XAUUSDm", "M15", 200)
    analysis = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis keys: {list(analysis.keys())}")
    print(f"   POC: {analysis.get('poc', 'N/A'):.2f}")
    print(f"   VAH: {analysis.get('vah', 'N/A'):.2f}")
    print(f"   VAL: {analysis.get('val', 'N/A'):.2f}")
    print(f"   HVNs: {analysis.get('high_volume_nodes', [])}")
    print(f"   LVNs: {analysis.get('low_volume_nodes', [])}")
    print(f"   Position vs VA: {analysis.get('position_vs_value_area', 'N/A')}")
    
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
    
    print("\n" + "="*60)
    print("VOLUME PROFILE STRATEGY TEST COMPLETED!")
    print("============================================================")
