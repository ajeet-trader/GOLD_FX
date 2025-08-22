"""
Harmonic Pattern Strategy - Advanced Pattern Recognition
======================================================
Author: XAUUSD Trading System
Version: 2.0.0
Date: 2025-08-09 (Modified for base.py integration: 2025-08-15)

Advanced harmonic pattern recognition for XAUUSD:
- Gartley patterns (bullish/bearish)
- Butterfly patterns
- Bat patterns
- Crab patterns
- Cypher patterns
- ABCD patterns

Features:
- Fibonacci-based pattern validation
- Multi-timeframe pattern detection
- Pattern completion zones
- Risk/reward optimization
"""

import sys
import os
from pathlib import Path

# Standardized sys.path manipulation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

# Standardized imports from src.core.base
from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade

# Standardized cli_args import with fallback
try:
    from src.utils.cli_args import parse_mode, print_mode_banner
except ImportError:
    def parse_mode(*_args, **_kwargs):  # type: ignore
        return 'mock'
    def print_mode_banner(_mode):  # type: ignore
        pass

class PatternType(Enum):
    """Harmonic pattern types"""
    GARTLEY = "GARTLEY"
    BUTTERFLY = "BUTTERFLY"
    BAT = "BAT"
    CRAB = "CRAB"
    CYPHER = "CYPHER"
    ABCD = "ABCD"
    THREE_DRIVES = "THREE_DRIVES"


@dataclass
class HarmonicPattern:
    """Harmonic pattern data structure"""
    pattern_type: PatternType
    direction: str  # bullish or bearish
    x_point: Tuple[int, float]  # (index, price)
    a_point: Tuple[int, float]
    b_point: Tuple[int, float]
    c_point: Tuple[int, float]
    d_point: Tuple[int, float]
    
    # Fibonacci ratios
    xab_ratio: float
    abc_ratio: float
    bcd_ratio: float
    xad_ratio: float
    
    # Pattern characteristics
    completion_zone: Tuple[float, float]  # PRZ range
    pattern_score: float  # Pattern accuracy score
    confidence: float
    
    def __post_init__(self):
        """Calculate derived properties"""
        self.prz_midpoint = (self.completion_zone[0] + self.completion_zone[1]) / 2


class HarmonicStrategy(AbstractStrategy):
    """
    Advanced Harmonic Pattern Recognition Strategy
    
    This strategy identifies and trades harmonic patterns:
    - Gartley: XA=.618, AB=.382/.886, BC=.382/.886, CD=1.13/1.618, XD=.786
    - Butterfly: XA=.786, AB=.382/.886, BC=.382/.886, CD=1.618/2.618, XD=1.27/1.618
    - Bat: XA=.382/.50, AB=.382/.886, BC=.382/.886, CD=1.618/2.618, XD=.886
    - Crab: XA=.382/.618, AB=.382/.886, BC=.382/.886, CD=2.24/3.618, XD=1.618
    - Cypher: XA=.382/.618, AB=1.13/1.414, BC=.382/.886, CD=1.272/2.0, XD=.786
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """Initialize Harmonic strategy"""
        # Call parent class initialization
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
                    self.mt5_manager = self._create_mock_mt5(mode='mock') # Pass mode for seeding
                    self.mode = 'mock'
            except ImportError:
                print("⚠️  MT5Manager not available, using mock data")
                self.mt5_manager = self._create_mock_mt5(mode='mock') # Pass mode for seeding
                self.mode = 'mock'
        elif self.mode == 'mock' or mt5_manager is None:
            self.mt5_manager = self._create_mock_mt5(mode='mock') # Always mock if explicit or no manager
        else:
            self.mt5_manager = mt5_manager
        
        self.min_confidence = self.config.get('parameters', {}).get('confidence_threshold', 0.55)  # Lower threshold
        self.lookback_period = self.config.get('parameters', {}).get('lookback_period', 200)
        self.fib_tolerance = 0.10  # More tolerance for Fibonacci ratios
        self.min_pattern_score = 0.50  # Lower minimum score
        
        self.pattern_ratios = {
            PatternType.GARTLEY: {
                'xab': (0.618, 0.05),
                'abc': (0.382, 0.886),
                'bcd': (1.13, 1.618),
                'xad': (0.786, 0.05)
            },
            PatternType.BUTTERFLY: {
                'xab': (0.786, 0.05),
                'abc': (0.382, 0.886),
                'bcd': (1.618, 2.618),
                'xad': (1.27, 1.618)
            },
            PatternType.BAT: {
                'xab': (0.382, 0.50),
                'abc': (0.382, 0.886),
                'bcd': (1.618, 2.618),
                'xad': (0.886, 0.05)
            },
            PatternType.CRAB: {
                'xab': (0.382, 0.618),
                'abc': (0.382, 0.886),
                'bcd': (2.24, 3.618),
                'xad': (1.618, 0.05)
            },
            PatternType.CYPHER: {
                'xab': (0.382, 0.618),
                'abc': (1.13, 1.414),
                'bcd': (1.272, 2.0),
                'xad': (0.786, 0.05)
            }
        }
        
        self.detected_patterns_count = 0

    def _create_mock_mt5(self, mode: str):
        """Create mock MT5 manager with mode-specific data"""
        class MockMT5Manager:
            def __init__(self, mode_inner: str):
                self.mode = mode_inner
                
            def get_historical_data(self, symbol, timeframe, bars):
                # Generate sample data with different characteristics for mock vs live
                dates = pd.date_range(start=datetime.now() - timedelta(days=10), 
                                     end=datetime.now(), freq='15Min')
                
                # Ensure enough dates for the requested bars
                if len(dates) < bars:
                    # Extend dates if needed
                    dates = pd.date_range(start=datetime.now() - timedelta(days=10 + (bars - len(dates)) // (24*4)), 
                                         end=datetime.now(), freq='15Min')
                
                dates = dates[:bars] # Trim to exact bars
                
                # Use different seeds and base prices for mock vs live
                np.random.seed(42 if self.mode == 'mock' else 123)
                
                x = 0
                prices = []
                
                for i in range(len(dates)):
                    x += 0.1
                    # Slightly vary the pattern generation for different modes
                    if self.mode == 'mock':
                        if i % 50 < 10:
                            price = 1950 + 10 * np.sin(x) + np.random.normal(0, 0.5)
                        elif i % 50 < 20:
                            price = 1960 - 8 * np.sin(x) + np.random.normal(0, 0.5)
                        elif i % 50 < 30:
                            price = 1955 + 12 * np.sin(x) + np.random.normal(0, 0.5)
                        elif i % 50 < 40:
                            price = 1965 - 10 * np.sin(x) + np.random.normal(0, 0.5)
                        else:
                            price = 1958 + 15 * np.sin(x) + np.random.normal(0, 0.5)
                    else: # Live-like data (more random, less clear patterns)
                        price = 1975 + 5 * np.sin(x*0.5) + np.random.normal(0, 2)
                        
                    prices.append(price)
                
                data = pd.DataFrame({
                    'Open': prices + np.random.normal(0, 0.5, len(dates)),
                    'High': np.array(prices) + np.abs(np.random.normal(2, 1, len(dates))),
                    'Low': np.array(prices) - np.abs(np.random.normal(2, 1, len(dates))),
                    'Close': prices,
                    'Volume': np.random.randint(500, 1500, len(dates))
                }, index=dates)
                
                return data
        
        return MockMT5Manager(mode)
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """Generate harmonic pattern trading signals"""
        signals = []
        try:
            if not self.mt5_manager:
                self.logger.warning("MT5 manager not available for signal generation.")
                return []

            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_period)
            if data is None or len(data) < 50:
                self.logger.warning(f"Insufficient data for harmonic analysis: {len(data) if data is not None else 0}")
                return []
            
            pivots = self._find_pivot_points(data)
            if len(pivots) < 5:
                return []
            
            patterns = self._detect_harmonic_patterns(data, pivots)
            
            # NEW: Add simplified ABCD patterns for more signal generation
            abcd_patterns = self._detect_abcd_patterns(data, pivots)
            patterns.extend(abcd_patterns)
            self.detected_patterns_count = len(patterns)
            
            for pattern in patterns:
                signal = self._pattern_to_signal(pattern, data, symbol, timeframe)
                if signal:
                    signals.append(signal)
            
            filtered_signals = self._filter_overlapping_patterns(signals)
            
            self.logger.info(f"Harmonic generated {len(filtered_signals)} signals from {len(patterns)} patterns")
            
            validated_signals = []
            for signal in filtered_signals:
                if self.validate_signal(signal):
                    validated_signals.append(signal)
            
            return validated_signals
            
        except Exception as e:
            self.logger.error(f"Harmonic signal generation failed: {str(e)}", exc_info=True)
            return []
    
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Perform detailed harmonic pattern analysis without generating signals.
        """
        try:
            if data is None or len(data) < 50:
                return {
                    'error': 'Insufficient data for analysis',
                    'required_bars': 50,
                    'available_bars': len(data) if data is not None else 0
                }
            
            pivots = self._find_pivot_points(data)
            patterns = self._detect_harmonic_patterns(data, pivots)

            analysis_results = {
                'strategy': self.strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_time': datetime.now().isoformat(),
                'data_points': len(data),
                'pivots_detected': len(pivots),
                'harmonic_patterns_detected_count': len(patterns),
                'patterns': []
            }

            for pattern in patterns:
                analysis_results['patterns'].append({
                    'type': pattern.pattern_type.value,
                    'direction': pattern.direction,
                    'x_point': pattern.x_point,
                    'd_point': pattern.d_point,
                    'pattern_score': round(pattern.pattern_score, 3),
                    'confidence': round(pattern.confidence, 3),
                    'completion_zone': pattern.completion_zone
                })
            
            return analysis_results

        except Exception as e:
            self.logger.error(f"Error during Harmonic analysis: {str(e)}", exc_info=True)
            return {'error': str(e)}

    def _find_pivot_points(self, data: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """Find significant pivot points (swing highs and lows)"""
        pivots = []
        window = 3  # Reduced window for more sensitivity
        
        for i in range(window, len(data) - window):
            is_high = True
            current_high = data['High'].iloc[i]
            for j in range(i - window, i + window + 1):
                if j != i and data['High'].iloc[j] >= current_high:
                    is_high = False
                    break
            
            if is_high:
                pivots.append((i, current_high, 'high'))
            
            is_low = True
            current_low = data['Low'].iloc[i]
            for j in range(i - window, i + window + 1):
                if j != i and data['Low'].iloc[j] <= current_low:
                    is_low = False
                    break
            
            if is_low:
                pivots.append((i, current_low, 'low'))
        
        pivots.sort(key=lambda x: x[0])
        
        return pivots
    
    def _detect_harmonic_patterns(self, data: pd.DataFrame, pivots: List[Tuple[int, float, str]]) -> List[HarmonicPattern]:
        """Detect harmonic patterns from pivot points"""
        patterns = []
        
        for i in range(len(pivots) - 4):
            points = pivots[i:i+5]
            
            if not self._is_valid_structure(points):
                continue
            
            x_point = (points[0][0], points[0][1])
            a_point = (points[1][0], points[1][1])
            b_point = (points[2][0], points[2][1])
            c_point = (points[3][0], points[3][1])
            d_point = (points[4][0], points[4][1])
            
            direction = 'bullish' if points[0][2] == 'high' else 'bearish'
            
            xab_denom = (a_point[1] - x_point[1])
            xab_ratio = abs((b_point[1] - a_point[1]) / xab_denom) if xab_denom != 0 else 0.0

            abc_denom = (b_point[1] - a_point[1])
            abc_ratio = abs((c_point[1] - b_point[1]) / abc_denom) if abc_denom != 0 else 0.0

            bcd_denom = (c_point[1] - b_point[1])
            bcd_ratio = abs((d_point[1] - c_point[1]) / bcd_denom) if bcd_denom != 0 else 0.0
            
            xad_denom = (a_point[1] - x_point[1])
            xad_ratio = abs((d_point[1] - x_point[1]) / xad_denom) if xad_denom != 0 else 0.0 # Fix: Should be (D-X)/(A-X)

            for pattern_type, ratios in self.pattern_ratios.items():
                if self._check_pattern_ratios(xab_ratio, abc_ratio, bcd_ratio, xad_ratio, ratios):
                    pattern_score = self._calculate_pattern_score(
                        xab_ratio, abc_ratio, bcd_ratio, xad_ratio, ratios
                    )
                    
                    if pattern_score >= self.min_pattern_score:
                        prz = self._calculate_prz(x_point, a_point, b_point, c_point, pattern_type)
                        
                        pattern = HarmonicPattern(
                            pattern_type=pattern_type,
                            direction=direction,
                            x_point=x_point,
                            a_point=a_point,
                            b_point=b_point,
                            c_point=c_point,
                            d_point=d_point,
                            xab_ratio=xab_ratio,
                            abc_ratio=abc_ratio,
                            bcd_ratio=bcd_ratio,
                            xad_ratio=xad_ratio,
                            completion_zone=prz,
                            pattern_score=pattern_score,
                            confidence=self._calculate_pattern_confidence(pattern_score, data, d_point[0])
                        )
                        
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_abcd_patterns(self, data: pd.DataFrame, pivots: List[Tuple[int, float, str]]) -> List[HarmonicPattern]:
        """Detect simple ABCD patterns for more signal generation"""
        patterns = []
        
        if len(pivots) < 4:
            return patterns
        
        for i in range(len(pivots) - 3):
            points = pivots[i:i+4]
            
            if not self._is_valid_abcd_structure(points):
                continue
            
            a_point = (points[0][0], points[0][1])
            b_point = (points[1][0], points[1][1])
            c_point = (points[2][0], points[2][1])
            d_point = (points[3][0], points[3][1])
            
            direction = 'bullish' if points[0][2] == 'low' else 'bearish'
            
            # Calculate AB and CD ratios
            ab_distance = abs(b_point[1] - a_point[1])
            cd_distance = abs(d_point[1] - c_point[1])
            
            if ab_distance == 0:
                continue
            
            cd_ab_ratio = cd_distance / ab_distance
            
            # ABCD pattern: CD should be 1.27 or 1.618 of AB (with tolerance)
            if 0.8 <= cd_ab_ratio <= 2.0:  # Very lenient for more patterns
                pattern_score = 0.6 + (0.3 * min(1.0, 1.0 / abs(cd_ab_ratio - 1.27)))
                
                if pattern_score >= 0.5:
                    # Create pattern with dummy X point for ABCD
                    x_point = a_point  # ABCD doesn't have X, use A
                    
                    pattern = HarmonicPattern(
                        pattern_type=PatternType.ABCD,
                        direction=direction,
                        x_point=x_point,
                        a_point=a_point,
                        b_point=b_point,
                        c_point=c_point,
                        d_point=d_point,
                        xab_ratio=0,  # N/A for ABCD
                        abc_ratio=0,  # N/A for ABCD
                        bcd_ratio=cd_ab_ratio,
                        xad_ratio=0,  # N/A for ABCD
                        completion_zone=(d_point[1] - 5, d_point[1] + 5),
                        pattern_score=pattern_score,
                        confidence=min(0.8, 0.55 + pattern_score * 0.2)
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _is_valid_abcd_structure(self, points: List[Tuple[int, float, str]]) -> bool:
        """Check if points form valid ABCD structure"""
        expected_bull = ['low', 'high', 'low', 'high']
        expected_bear = ['high', 'low', 'high', 'low']
        
        actual_sequence = [p[2] for p in points]
        
        return actual_sequence == expected_bull or actual_sequence == expected_bear
    
    def _is_valid_structure(self, points: List[Tuple[int, float, str]]) -> bool:
        """Check if points form valid pattern structure"""
        expected_sequence = ['high', 'low', 'high', 'low', 'high']
        alt_sequence = ['low', 'high', 'low', 'high', 'low']
        
        actual_sequence = [p[2] for p in points]
        
        return actual_sequence == expected_sequence or actual_sequence == alt_sequence
    
    def _check_pattern_ratios(self, xab: float, abc: float, bcd: float, xad: float, 
                             expected_ratios: Dict) -> bool:
        """Check if ratios match expected pattern ratios - More lenient version"""
        tolerance_multiplier = 2.0  # Double the tolerance for more patterns
        
        xab_range = expected_ratios['xab']
        if isinstance(xab_range[1], float) and xab_range[1] < 1:
            tolerance = xab_range[1] * tolerance_multiplier
            if not (xab_range[0] * (1 - tolerance) <= xab <= xab_range[0] * (1 + tolerance)):
                return False
        else:
            range_width = xab_range[1] - xab_range[0]
            extended_min = xab_range[0] - range_width * 0.5
            extended_max = xab_range[1] + range_width * 0.5
            if not (extended_min <= xab <= extended_max):
                return False
        
        # More lenient ABC range
        abc_range_width = expected_ratios['abc'][1] - expected_ratios['abc'][0]
        abc_min = expected_ratios['abc'][0] - abc_range_width * 0.3
        abc_max = expected_ratios['abc'][1] + abc_range_width * 0.3
        if not (abc_min <= abc <= abc_max):
            return False
        
        # More lenient BCD range
        bcd_range_width = expected_ratios['bcd'][1] - expected_ratios['bcd'][0]
        bcd_min = expected_ratios['bcd'][0] - bcd_range_width * 0.3
        bcd_max = expected_ratios['bcd'][1] + bcd_range_width * 0.3
        if not (bcd_min <= bcd <= bcd_max):
            return False
        
        # More lenient XAD range
        xad_range = expected_ratios['xad']
        if isinstance(xad_range[1], float) and xad_range[1] < 1:
            tolerance = xad_range[1] * tolerance_multiplier
            if not (xad_range[0] * (1 - tolerance) <= xad <= xad_range[0] * (1 + tolerance)):
                return False
        else:
            range_width = xad_range[1] - xad_range[0]
            extended_min = xad_range[0] - range_width * 0.5
            extended_max = xad_range[1] + range_width * 0.5
            if not (extended_min <= xad <= extended_max):
                return False
        
        return True
    
    def _calculate_pattern_score(self, xab: float, abc: float, bcd: float, xad: float,
                               expected_ratios: Dict) -> float:
        """Calculate pattern accuracy score"""
        score = 0.0
        
        xab_target = expected_ratios['xab'][0]
        xab_score = 1.0 - min(abs(xab - xab_target) / xab_target, 1.0)
        score += xab_score * 0.25
        
        abc_mid = (expected_ratios['abc'][0] + expected_ratios['abc'][1]) / 2
        abc_score = 1.0 - min(abs(abc - abc_mid) / abc_mid, 1.0)
        score += abc_score * 0.25
        
        bcd_mid = (expected_ratios['bcd'][0] + expected_ratios['bcd'][1]) / 2
        bcd_score = 1.0 - min(abs(bcd - bcd_mid) / bcd_mid, 1.0)
        score += bcd_score * 0.25
        
        xad_target = expected_ratios['xad'][0]
        xad_score = 1.0 - min(abs(xad - xad_target) / xad_target, 1.0)
        score += xad_score * 0.25
        
        return score
    
    def _calculate_prz(self, x_point: Tuple, a_point: Tuple, b_point: Tuple, 
                      c_point: Tuple, pattern_type: PatternType) -> Tuple[float, float]:
        """Calculate Potential Reversal Zone"""
        xa_distance = abs(a_point[1] - x_point[1])
        
        if pattern_type == PatternType.GARTLEY:
            prz_center = x_point[1] + (xa_distance * 0.786) if x_point[1] < a_point[1] else x_point[1] - (xa_distance * 0.786)
            prz_range = xa_distance * 0.03
        elif pattern_type == PatternType.BUTTERFLY:
            prz_center = x_point[1] + (xa_distance * 1.27) if x_point[1] < a_point[1] else x_point[1] - (xa_distance * 1.27)
            prz_range = xa_distance * 0.05
        elif pattern_type == PatternType.BAT:
            prz_center = x_point[1] + (xa_distance * 0.886) if x_point[1] < a_point[1] else x_point[1] - (xa_distance * 0.886)
            prz_range = xa_distance * 0.03
        elif pattern_type == PatternType.CRAB:
            prz_center = x_point[1] + (xa_distance * 1.618) if x_point[1] < a_point[1] else x_point[1] - (xa_distance * 1.618)
            prz_range = xa_distance * 0.05
        else: # Cypher and default
            prz_center = x_point[1] + (xa_distance * 0.786) if x_point[1] < a_point[1] else x_point[1] - (xa_distance * 0.786)
            prz_range = xa_distance * 0.03
        
        return (prz_center - prz_range, prz_center + prz_range)
    
    def _calculate_pattern_confidence(self, pattern_score: float, data: pd.DataFrame, 
                                    d_index: int) -> float:
        """Calculate confidence based on pattern quality and market context"""
        confidence = pattern_score * 0.5
        
        # Consider how recent the pattern D point is
        # If the D point is the most recent bar (or very recent), it's more actionable
        if len(data) - 1 - d_index < 5: # D point is within last 5 bars
            confidence += 0.2
        
        if 'Volume' in data.columns and d_index > 20:
            recent_volume = data['Volume'].iloc[d_index-5:d_index+5].mean()
            avg_volume = data['Volume'].iloc[d_index-20:d_index].mean()
            if recent_volume > avg_volume * 1.2:
                confidence += 0.1
        
        # Ensure we have enough data for SMAs before calculating
        if len(data) > d_index and d_index >= 50:
            sma_20 = data['Close'].rolling(20).mean().iloc[d_index]
            sma_50 = data['Close'].rolling(50).mean().iloc[d_index]
            
            # Add confidence if price aligns with trend from SMAs
            current_price_at_d = data['Close'].iloc[d_index]
            if current_price_at_d > sma_50 and sma_20 > sma_50: # Bullish alignment
                confidence += 0.1
            elif current_price_at_d < sma_50 and sma_20 < sma_50: # Bearish alignment
                confidence += 0.1
        
        return min(confidence, 0.95)
    
    def _pattern_to_signal(self, pattern: HarmonicPattern, data: pd.DataFrame, 
                          symbol: str, timeframe: str) -> Optional[Signal]:
        """Convert harmonic pattern to trading signal"""
        try:
            current_price = data['Close'].iloc[-1]
            current_time = data.index[-1]
            
            # More lenient PRZ checking - allow signals even outside PRZ
            distance_from_prz = 0
            if not (pattern.completion_zone[0] <= current_price <= pattern.completion_zone[1]):
                distance_from_prz = min(
                    abs(current_price - pattern.completion_zone[0]),
                    abs(current_price - pattern.completion_zone[1])
                )
                
                prz_width = abs(pattern.completion_zone[1] - pattern.completion_zone[0])
                if prz_width == 0:
                    prz_width = 10  # Default width if PRZ is a single point
                
                # Allow signals up to 2x PRZ width away, with penalty
                if distance_from_prz > prz_width * 2: # Reduced leniency for distance from PRZ
                    return None
                
                # Reduce confidence based on distance from PRZ
                distance_penalty = min(0.3, (distance_from_prz / prz_width) * 0.15) # Increased penalty
                pattern.confidence = max(self.min_confidence, pattern.confidence - distance_penalty) # Ensure it stays above min_confidence
            
            signal_type = None
            stop_loss = 0.0
            take_profit = 0.0

            if pattern.direction == 'bullish':
                signal_type = SignalType.BUY
                # Stop loss below X point for safety
                stop_loss = pattern.x_point[1] * 0.995 
                # Take profit at C point or A point retracement
                take_profit = max(pattern.c_point[1], pattern.a_point[1] - abs(pattern.a_point[1] - pattern.b_point[1]) * 0.5)

            elif pattern.direction == 'bearish':
                signal_type = SignalType.SELL
                # Stop loss above X point for safety
                stop_loss = pattern.x_point[1] * 1.005 
                # Take profit at C point or A point retracement
                take_profit = min(pattern.c_point[1], pattern.a_point[1] + abs(pattern.a_point[1] - pattern.b_point[1]) * 0.5)
            
            if pattern.confidence < self.min_confidence:
                return None
            
            return Signal(
                timestamp=current_time,
                symbol=symbol,
                strategy_name=self.strategy_name,
                signal_type=signal_type,
                confidence=pattern.confidence,
                price=current_price,
                timeframe=timeframe,
                strength=pattern.pattern_score,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'pattern_type': pattern.pattern_type.value,
                    'pattern_direction': pattern.direction,
                    'xab_ratio': round(pattern.xab_ratio, 3),
                    'abc_ratio': round(pattern.abc_ratio, 3),
                    'bcd_ratio': round(pattern.bcd_ratio, 3),
                    'xad_ratio': round(pattern.xad_ratio, 3),
                    'prz_range': pattern.completion_zone,
                    'pattern_score': round(pattern.pattern_score, 3)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Pattern to signal conversion failed: {str(e)}", exc_info=True)
            return None
    
    def _filter_overlapping_patterns(self, signals: List[Signal]) -> List[Signal]:
        """Filter overlapping patterns, keeping the best ones"""
        if len(signals) <= 1:
            return signals
        
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        filtered = []
        
        if buy_signals:
            # Sort by combined strength and confidence
            best_buy = max(buy_signals, key=lambda s: s.confidence * s.strength)
            filtered.append(best_buy)
        
        if sell_signals:
            # Sort by combined strength and confidence
            best_sell = max(sell_signals, key=lambda s: s.confidence * s.strength)
            filtered.append(best_sell)
        
        return filtered
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and performance.
        """
        # Ensure self.performance is initialized; it's part of AbstractStrategy
        # For test cases where __init__ might not run fully (e.g., direct script execution),
        # ensure a default is available.
        if not hasattr(self, 'performance'):
            self.performance = self._create_empty_performance_metrics()

        return {
            'name': 'Harmonic Patterns Strategy',
            'type': 'Technical',
            'version': '2.0.0',
            'description': 'Advanced harmonic pattern recognition with Fibonacci validation',
            'patterns_supported': [p.value for p in PatternType],
            'min_confidence': self.min_confidence,
            'fib_tolerance': self.fib_tolerance,
            'min_pattern_score': self.min_pattern_score,
            'detected_patterns_count': self.detected_patterns_count,
            'performance': { # This structure matches the original request
                'success_rate': self.performance.win_rate,
                'profit_factor': self.performance.profit_factor
            }
        }

    # Helper for get_strategy_info to handle uninitialized performance in direct script runs
    def _create_empty_performance_metrics(self):
        from src.core.base import StrategyPerformance
        return StrategyPerformance(
            strategy_name=self.strategy_name,
            win_rate=0.0,
            profit_factor=0.0
        )


if __name__ == "__main__":
    """Test the Harmonic strategy"""
    
    # Test configuration
    test_config = {
        'parameters': {
            'confidence_threshold': 0.72,
            'lookback_period': 200,
            'mode': 'mock' # Default to mock for direct script execution
        }
    }
    
    # Create strategy instance with mode-aware MT5 manager
    strategy = HarmonicStrategy(test_config, mt5_manager=None) # Pass None to trigger internal MT5 creation
    
    print("="*60)
    print("TESTING MODIFIED HARMONIC STRATEGY")
    print("="*60)
    print(f"Running in {strategy.mode.upper()} mode")

    print("\n1. Testing signal generation:")
    signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"   - {signal.signal_type.value} at {signal.price:.2f}, "
              f"Confidence: {signal.confidence:.3f}, Grade: {signal.grade.value}")
        print(f"     Pattern: {signal.metadata.get('pattern_type', 'Unknown')}")
        print(f"     Pattern Score: {signal.metadata.get('pattern_score', 0)}")
    
    print("\n2. Testing analysis method:")
    # Get mock data from the strategy's own MT5 manager instance
    mock_data = strategy.mt5_manager.get_historical_data("XAUUSDm", "M15", 200)
    analysis_results = strategy.analyze(mock_data, "XAUUSDm", "M15")
    print(f"   Analysis results keys: {analysis_results.keys()}")
    if 'patterns' in analysis_results:
        print(f"   Detected patterns in analysis: {len(analysis_results['patterns'])}")

    print("\n3. Testing performance tracking:")
    summary = strategy.get_performance_summary()
    print(f"   {summary}")

    # --- New section to display strategy info and important details ---
    print("\n4. Strategy Information:")
    strategy_info = strategy.get_strategy_info()
    print(f"   Name: {strategy_info['name']}")
    print(f"   Version: {strategy_info['version']}")
    print(f"   Description: {strategy_info['description']}")
    print(f"   Type: {strategy_info['type']}")
    print(f"   Patterns Supported: {', '.join(strategy_info['patterns_supported'])}")
    print(f"   Minimum Confidence: {strategy_info['min_confidence']:.2f}")
    print(f"   Fibonacci Tolerance: {strategy_info['fib_tolerance']:.2f}")
    print(f"   Minimum Pattern Score: {strategy_info['min_pattern_score']:.2f}")
    print(f"   Detected Patterns Count (Last Run): {strategy_info['detected_patterns_count']}")
    print(f"   Performance Summary:")
    print(f"     Success Rate: {strategy_info['performance']['success_rate']:.2%}")
    print(f"     Profit Factor: {strategy_info['performance']['profit_factor']:.2f}")
    # --- End of new section ---
    
    print("\n" + "="*60)
    print(f"HARMONIC STRATEGY TEST COMPLETED IN {strategy.mode.upper()} MODE!")
    print("="*60)