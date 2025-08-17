"""
Elliott Wave Analysis Strategy - Advanced Wave Pattern Recognition
================================================================
Author: XAUUSD Trading System
Version: 2.0.0
Date: 2025-01-08 (Modified for base.py integration: 2025-08-15)

Complete Elliott Wave implementation for XAUUSD trading:
- 5-wave impulse pattern identification
- 3-wave corrective pattern detection (ABC)
- Multiple corrective types (Zigzag, Flat, Triangle, Complex)
- Fibonacci relationship validation
- Wave degree classification
- Multi-timeframe wave analysis
- Real-time wave counting

Elliott Wave Rules:
1. Wave 2 never retraces more than 100% of Wave 1
2. Wave 3 is never the shortest impulse wave
3. Wave 4 never enters Wave 1 territory
4. Wave alternation principle
5. Fibonacci relationships between waves

Dependencies:
    - pandas
    - numpy
    - datetime
    - typing
    - logging
"""

import sys
import os
from pathlib import Path

# Add src to path as in other strategy files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

# Import base classes from src.core.base
from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade, StrategyPerformance


# Configure logging
logging.basicConfig(level=logging.INFO)
# Logger is now handled by AbstractStrategy
# logger = logging.getLogger(__name__)

class WaveType(Enum):
    """Elliott Wave types"""
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"
    DIAGONAL = "diagonal"
    TRIANGLE = "triangle"
    FLAT = "flat"
    ZIGZAG = "zigzag"
    COMPLEX = "complex"

class WaveDegree(Enum):
    """Wave degree levels from largest to smallest"""
    GRAND_SUPERCYCLE = 9
    SUPERCYCLE = 8
    CYCLE = 7
    PRIMARY = 6
    INTERMEDIATE = 5
    MINOR = 4
    MINUTE = 3
    MINUETTE = 2
    SUBMINUETTE = 1

@dataclass
class ElliottWave:
    """Elliott Wave data structure"""
    wave_number: int
    wave_label: str  # 1,2,3,4,5 or A,B,C
    wave_type: WaveType
    degree: WaveDegree
    start_price: float
    end_price: float
    start_time: datetime
    end_time: datetime
    start_index: int
    end_index: int
    retracement: float
    extension: float
    is_valid: bool
    confidence: float

@dataclass
class WavePattern:
    """Complete wave pattern structure"""
    pattern_type: WaveType
    waves: List[ElliottWave]
    degree: WaveDegree
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    is_complete: bool
    confidence: float
    next_wave_projection: Dict
    validation_score: float

class ElliottWaveStrategy(AbstractStrategy): # Inherit from AbstractStrategy
    """
    Elliott Wave Analysis Strategy
    
    Complete implementation with pattern recognition,
    validation, and signal generation for XAUUSDm trading.
    """
    
    # Modified __init__ signature to match AbstractStrategy
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database=None):
        """
        Initialize Elliott Wave Strategy
        
        Args:
            config: Strategy configuration
            mt5_manager: MT5 manager instance
        """
        super().__init__(config, mt5_manager, database) # Call parent __init__
        
        # Wave parameters - use self.config from AbstractStrategy
        # Access parameters from the 'parameters' key in the config dict
        self.min_wave_size = self.config.get('parameters', {}).get('min_wave_size', 10)  # Reduced minimum wave size
        self.lookback_periods = self.config.get('parameters', {}).get('lookback_periods', 200)
        self.min_confidence = self.config.get('parameters', {}).get('min_confidence', 0.55)  # Lower confidence threshold
        self.use_volume = self.config.get('parameters', {}).get('use_volume', True)
        self.strict_rules = self.config.get('parameters', {}).get('strict_rules', False)  # Less strict validation
        
        # Fibonacci ratios for wave relationships
        self.fib_ratios = {
            'retracement': self.config.get('parameters', {}).get('fib_retracement_ratios', [0.236, 0.382, 0.5, 0.618, 0.786]),
            'extension': self.config.get('parameters', {}).get('fib_extension_ratios', [1.0, 1.272, 1.618, 2.0, 2.618, 3.618, 4.236]),
            'tolerance': self.config.get('parameters', {}).get('fibonacci_tolerance', 0.05)
        }
        
        # Pattern storage
        self.current_patterns = {}
        self.completed_patterns = []
        self.wave_counts = {}
        
        # Performance tracking is now handled by AbstractStrategy base class
        # self.signals_generated = 0
        # self.successful_signals = 0
        
        self.logger.info(f"Elliott Wave Strategy initialized with min_wave_size={self.min_wave_size}, lookback={self.lookback_periods}")
    
    # Renamed from generate_signals to generate_signal to match AbstractStrategy
    def generate_signal(self, symbol: str = "XAUUSDm", timeframe: str = "M15") -> List[Signal]:
        """
        Generate trading signals from Elliott Wave patterns
        
        Args:
            symbol: Trading symbol (default: XAUUSDm)
            timeframe: Timeframe for analysis
            
        Returns:
            List of Signal objects
        """
        signals = []
        try:
            if not self.mt5_manager:
                self.logger.warning("No MT5 manager available, returning empty signals")
                return []
            
            # Get historical data
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_periods)
            
            if data is None or len(data) < self.lookback_periods:
                self.logger.warning(f"Insufficient data for {symbol} on {timeframe}")
                return []
            
            # Perform wave analysis, which returns a signal-like dictionary
            analysis_result_dict = self._perform_wave_analysis(data)
            
            # Convert analysis result to Signal object if a tradable pattern is found
            if analysis_result_dict['direction'] != 'NEUTRAL':
                signal = self._create_signal_from_analysis(analysis_result_dict, symbol, timeframe, data)
                if signal:
                    if self.validate_signal(signal): # Validate signal using base class method
                        signals.append(signal)
            
            # NEW: Generate additional signals from all valid patterns
            additional_signals = self._generate_additional_signals(analysis_result_dict, symbol, timeframe, data)
            for signal in additional_signals:
                if self.validate_signal(signal):
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating Elliott Wave signals: {str(e)}", exc_info=True)
            return []
    
    # Renamed from analyze to _perform_wave_analysis (private helper for signal generation)
    def _perform_wave_analysis(self, data: pd.DataFrame) -> Dict:
        """
        Private helper method for Elliott Wave pattern detection and analysis.
        This method's output is used to generate trading signals.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary with analysis results, including a potential signal direction.
        """
        try:
            if len(data) < self.lookback_periods:
                return self._empty_signal_dict()
            
            # Step 1: Identify swing points
            swings = self._identify_swings(data)
            
            if len(swings) < 6:  # Need minimum points for pattern
                return self._empty_signal_dict()
            
            # Step 2: Find wave patterns
            impulse_waves = self._find_impulse_waves(swings, data)
            corrective_waves = self._find_corrective_waves(swings, data)
            
            # Step 3: Validate patterns
            all_patterns = impulse_waves + corrective_waves
            valid_patterns = self._validate_patterns(all_patterns)
            
            # Step 4: Generate signal from best pattern (returns a dict for further processing)
            if valid_patterns:
                signal_dict = self._generate_signal_dict(valid_patterns, data)
                return signal_dict
            
            return self._empty_signal_dict()
            
        except Exception as e:
            self.logger.error(f"Error in Elliott Wave analysis: {str(e)}", exc_info=True)
            return self._empty_signal_dict()

    # New method: analyze, required by AbstractStrategy
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Performs a detailed Elliott Wave analysis without generating executable signals.
        
        Args:
            data: Historical price data.
            symbol: Trading symbol.
            timeframe: Analysis timeframe.
            
        Returns:
            Dictionary containing detailed analysis results of detected wave patterns.
        """
        try:
            if data is None or len(data) < self.lookback_periods:
                return {
                    'status': 'Insufficient data for analysis',
                    'required_bars': self.lookback_periods,
                    'available_bars': len(data) if data is not None else 0
                }
            
            swings = self._identify_swings(data)
            if len(swings) < 6:
                return {'status': 'Not enough swing points detected for full analysis'}
            
            impulse_waves = self._find_impulse_waves(swings, data)
            corrective_waves = self._find_corrective_waves(swings, data)
            all_patterns = impulse_waves + corrective_waves
            valid_patterns = self._validate_patterns(all_patterns)

            analysis_output = {
                'strategy': self.strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_time': datetime.now().isoformat(),
                'swings_detected': len(swings),
                'total_patterns_found': len(all_patterns),
                'valid_patterns_count': len(valid_patterns),
                'detailed_patterns': []
            }

            for pattern in valid_patterns:
                analysis_output['detailed_patterns'].append({
                    'pattern_type': pattern.pattern_type.value,
                    'degree': pattern.degree.name,
                    'start_price': pattern.start_price,
                    'end_price': pattern.end_price,
                    'confidence': round(pattern.confidence, 3),
                    'validation_score': round(pattern.validation_score, 3),
                    'waves_details': [
                        {
                            'label': w.wave_label,
                            'type': w.wave_type.value,
                            'start_price': w.start_price,
                            'end_price': w.end_price,
                            'retracement': round(w.retracement, 3),
                            'extension': round(w.extension, 3)
                        } for w in pattern.waves
                    ],
                    'next_wave_projection': self._project_next_wave([pattern], data) # Re-use projection logic
                })
            
            if not valid_patterns:
                analysis_output['status'] = 'No valid Elliott Wave patterns detected'

            return analysis_output

        except Exception as e:
            self.logger.error(f"Error in Elliott Wave analysis method: {str(e)}", exc_info=True)
            return {'error': str(e)}

    def _identify_swings(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify swing highs and lows using advanced pivot detection
        """
        try:
            swings = []
            window = 3  # Reduced window for more sensitive detection
            
            for i in range(window, len(data) - window):
                high_price = data.iloc[i]['High']
                is_swing_high = True
                
                for j in range(1, window + 1):
                    if data.iloc[i-j]['High'] >= high_price or data.iloc[i+j]['High'] >= high_price:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    swings.append({
                        'index': i,
                        'time': data.index[i] if hasattr(data.index[i], 'to_pydatetime') else data.index[i],
                        'price': high_price,
                        'type': 'high'
                    })
                
                low_price = data.iloc[i]['Low']
                is_swing_low = True
                
                for j in range(1, window + 1):
                    if data.iloc[i-j]['Low'] <= low_price or data.iloc[i+j]['Low'] <= low_price:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    swings.append({
                        'index': i,
                        'time': data.index[i] if hasattr(data.index[i], 'to_pydatetime') else data.index[i],
                        'price': low_price,
                        'type': 'low'
                    })
            
            swings.sort(key=lambda x: x['index'])
            
            if swings:
                return pd.DataFrame(swings)
            else:
                return pd.DataFrame(columns=['index', 'time', 'price', 'type'])
                
        except Exception as e:
            self.logger.error(f"Error identifying swings: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def _find_impulse_waves(self, swings: pd.DataFrame, data: pd.DataFrame) -> List[WavePattern]:
        """
        Find 5-wave impulse patterns
        """
        patterns = []
        
        if len(swings) < 6:
            return patterns
        
        for i in range(len(swings) - 5):
            expected_pattern_bull = ['low', 'high', 'low', 'high', 'low', 'high']
            expected_pattern_bear = ['high', 'low', 'high', 'low', 'high', 'low']
            
            actual_pattern = swings.iloc[i:i+6]['type'].tolist()
            
            is_bullish = actual_pattern == expected_pattern_bull
            is_bearish = actual_pattern == expected_pattern_bear
            
            if not (is_bullish or is_bearish):
                continue
            
            waves = []
            for j in range(5):
                wave_number = j + 1
                wave = ElliottWave(
                    wave_number=wave_number,
                    wave_label=str(wave_number),
                    wave_type=WaveType.IMPULSE,
                    degree=WaveDegree.MINOR,
                    start_price=swings.iloc[i + j]['price'],
                    end_price=swings.iloc[i + j + 1]['price'],
                    start_time=swings.iloc[i + j]['time'],
                    end_time=swings.iloc[i + j + 1]['time'],
                    start_index=swings.iloc[i + j]['index'],
                    end_index=swings.iloc[i + j + 1]['index'],
                    retracement=0,
                    extension=0,
                    is_valid=True,
                    confidence=0
                )
                waves.append(wave)
            
            self._calculate_wave_relationships(waves)
            
            if self._validate_impulse_rules(waves, is_bullish):
                pattern = WavePattern(
                    pattern_type=WaveType.IMPULSE,
                    waves=waves,
                    degree=WaveDegree.MINOR,
                    start_time=waves[0].start_time,
                    end_time=waves[-1].end_time,
                    start_price=waves[0].start_price,
                    end_price=waves[-1].end_price,
                    is_complete=True,
                    confidence=0,
                    next_wave_projection={},
                    validation_score=0
                )
                
                pattern.confidence = self._calculate_pattern_confidence(pattern, data)
                pattern.validation_score = self._calculate_validation_score(pattern)
                
                if pattern.confidence >= self.min_confidence:
                    patterns.append(pattern)
                    self.logger.info(f"Found {'bullish' if is_bullish else 'bearish'} impulse pattern with confidence {pattern.confidence:.2f}")
        
        return patterns
    
    def _find_corrective_waves(self, swings: pd.DataFrame, data: pd.DataFrame) -> List[WavePattern]:
        """
        Find 3-wave corrective patterns (ABC)
        """
        patterns = []
        
        if len(swings) < 4:
            return patterns
        
        for i in range(len(swings) - 3):
            expected_pattern_bear = ['high', 'low', 'high', 'low']
            expected_pattern_bull = ['low', 'high', 'low', 'high']
            
            actual_pattern = swings.iloc[i:i+4]['type'].tolist()
            
            is_bearish_correction = actual_pattern == expected_pattern_bear
            is_bullish_correction = actual_pattern == expected_pattern_bull
            
            if not (is_bearish_correction or is_bullish_correction):
                continue
            
            waves = []
            wave_labels = ['A', 'B', 'C']
            
            for j in range(3):
                wave = ElliottWave(
                    wave_number=j + 1,
                    wave_label=wave_labels[j],
                    wave_type=WaveType.CORRECTIVE,
                    degree=WaveDegree.MINOR,
                    start_price=swings.iloc[i + j]['price'],
                    end_price=swings.iloc[i + j + 1]['price'],
                    start_time=swings.iloc[i + j]['time'],
                    end_time=swings.iloc[i + j + 1]['time'],
                    start_index=swings.iloc[i + j]['index'],
                    end_index=swings.iloc[i + j + 1]['index'],
                    retracement=0,
                    extension=0,
                    is_valid=True,
                    confidence=0
                )
                waves.append(wave)
            
            self._calculate_wave_relationships(waves)
            
            pattern_type = self._identify_corrective_type(waves)
            
            if pattern_type:
                pattern = WavePattern(
                    pattern_type=pattern_type,
                    waves=waves,
                    degree=WaveDegree.MINOR,
                    start_time=waves[0].start_time,
                    end_time=waves[-1].end_time,
                    start_price=waves[0].start_price,
                    end_price=waves[-1].end_price,
                    is_complete=True,
                    confidence=0,
                    next_wave_projection={},
                    validation_score=0
                )
                
                pattern.confidence = self._calculate_pattern_confidence(pattern, data)
                pattern.validation_score = self._calculate_validation_score(pattern)
                
                if pattern.confidence >= self.min_confidence:
                    patterns.append(pattern)
                    self.logger.info(f"Found {pattern_type.value} corrective pattern with confidence {pattern.confidence:.2f}")
        
        return patterns
    
    def _calculate_wave_relationships(self, waves: List[ElliottWave]):
        """
        Calculate Fibonacci relationships between waves
        """
        for i, wave in enumerate(waves):
            wave_size = abs(wave.end_price - wave.start_price)
            
            if i > 0:
                prev_wave_size = abs(waves[i-1].end_price - waves[i-1].start_price)
                if prev_wave_size > 0:
                    wave.retracement = wave_size / prev_wave_size
            
            if i > 0 and len(waves) > 0:
                first_wave_size = abs(waves[0].end_price - waves[0].start_price)
                if first_wave_size > 0:
                    wave.extension = wave_size / first_wave_size
    
    def _validate_impulse_rules(self, waves: List[ElliottWave], is_bullish: bool) -> bool:
        """
        Validate Elliott Wave impulse rules
        """
        if len(waves) != 5:
            return False
        
        wave_sizes = [abs(w.end_price - w.start_price) for w in waves]
        
        if is_bullish:
            if waves[1].end_price <= waves[0].start_price:
                return False
            
            impulse_sizes = [wave_sizes[0], wave_sizes[2], wave_sizes[4]]
            if wave_sizes[2] == min(impulse_sizes):
                return False
            
            if waves[3].end_price <= waves[0].end_price:
                return False
        else:
            if waves[1].end_price >= waves[0].start_price:
                return False
            
            impulse_sizes = [wave_sizes[0], wave_sizes[2], wave_sizes[4]]
            if wave_sizes[2] == min(impulse_sizes):
                return False
            
            if waves[3].end_price >= waves[0].end_price:
                return False
        
        return True
    
    def _identify_corrective_type(self, waves: List[ElliottWave]) -> Optional[WaveType]:
        """
        Identify the type of corrective pattern
        """
        if len(waves) != 3:
            return None
        
        a_size = abs(waves[0].end_price - waves[0].start_price)
        b_size = abs(waves[1].end_price - waves[1].start_price)
        c_size = abs(waves[2].end_price - waves[2].start_price)
        
        if a_size == 0:
            return None
            
        b_retracement = b_size / a_size
        
        if 0.5 <= b_retracement <= 0.618:
            return WaveType.ZIGZAG
        elif 0.786 <= b_retracement <= 1.0:
            return WaveType.FLAT
        elif b_retracement < 0.5 and c_size < a_size:
            return WaveType.TRIANGLE
        else:
            return WaveType.CORRECTIVE
    
    def _validate_patterns(self, patterns: List[WavePattern]) -> List[WavePattern]:
        """
        Validate wave patterns against Elliott Wave guidelines
        """
        valid_patterns = []
        
        for pattern in patterns:
            is_valid = True
            
            for wave in pattern.waves:
                wave_size_pips = abs(wave.end_price - wave.start_price)
                if wave_size_pips < self.min_wave_size:
                    is_valid = False
                    break
            
            if not is_valid:
                continue
            
            if self._check_fibonacci_relationships(pattern):
                # Access parameters from the 'parameters' key in the config dict
                if self.config.get('parameters', {}).get('time_ratios', True): 
                    if self._check_time_relationships(pattern):
                        valid_patterns.append(pattern)
                else:
                    valid_patterns.append(pattern)
        
        return valid_patterns
    
    def _check_fibonacci_relationships(self, pattern: WavePattern) -> bool:
        """
        Check if waves follow Fibonacci relationships
        """
        tolerance = self.fib_ratios['tolerance']
        
        if pattern.pattern_type == WaveType.IMPULSE and len(pattern.waves) >= 3:
            if pattern.waves[1].retracement > 0:
                valid_retracement = False
                for ratio in self.fib_ratios['retracement']:
                    if abs(pattern.waves[1].retracement - ratio) <= tolerance:
                        valid_retracement = True
                        break
                
                if not valid_retracement and self.strict_rules:
                    return False
            
            if pattern.waves[2].extension > 0:
                valid_extension = False
                for ratio in self.fib_ratios['extension']:
                    if abs(pattern.waves[2].extension - ratio) <= tolerance:
                        valid_extension = True
                        break
                
                if pattern.waves[2].extension < 1.0 and self.strict_rules:
                    return False
        
        return True
    
    def _check_time_relationships(self, pattern: WavePattern) -> bool:
        """
        Check if waves follow reasonable time relationships
        """
        for wave in pattern.waves:
            if hasattr(wave.end_time, 'timestamp'):
                duration = (wave.end_time - wave.start_time).total_seconds()
            else:
                duration = wave.end_time - wave.start_time
                
            if duration <= 0:
                return False
        
        return True
    
    def _calculate_pattern_confidence(self, pattern: WavePattern, data: pd.DataFrame) -> float:
        """
        Calculate confidence score for a wave pattern
        """
        confidence = 0.5
        
        if pattern.pattern_type == WaveType.IMPULSE and len(pattern.waves) >= 5:
            wave3_size = abs(pattern.waves[2].end_price - pattern.waves[2].start_price)
            wave1_size = abs(pattern.waves[0].end_price - pattern.waves[0].start_price)
            
            if wave1_size > 0:
                ratio = wave3_size / wave1_size
                if ratio > 1.618:
                    confidence += 0.15
                elif ratio > 1.0:
                    confidence += 0.10
        
        fib_accuracy = self._calculate_fibonacci_accuracy(pattern)
        confidence += fib_accuracy * 0.2
        
        if self.use_volume and 'Volume' in data.columns:
            volume_confirmation = self._check_volume_confirmation(pattern, data)
            confidence += volume_confirmation * 0.1
        
        if pattern.is_complete:
            confidence += 0.1
        
        if pattern.pattern_type == WaveType.IMPULSE:
            if self._check_wave_alternation(pattern):
                confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _calculate_fibonacci_accuracy(self, pattern: WavePattern) -> float:
        """
        Calculate how well the pattern matches Fibonacci ratios
        """
        if not pattern.waves:
            return 0.0
        
        accuracy_scores = []
        
        for wave in pattern.waves:
            if wave.retracement > 0:
                min_distance = min(abs(wave.retracement - ratio) 
                                 for ratio in self.fib_ratios['retracement'])
                accuracy = max(0, 1 - (min_distance / 0.1))
                accuracy_scores.append(accuracy)
        
        return np.mean(accuracy_scores) if accuracy_scores else 0.5
    
    def _check_volume_confirmation(self, pattern: WavePattern, data: pd.DataFrame) -> float:
        """
        Check if volume confirms the wave pattern
        """
        try:
            confirmation_score = 0.5
            
            if pattern.pattern_type == WaveType.IMPULSE and len(pattern.waves) >= 3:
                for i, wave in enumerate(pattern.waves):
                    if i in [0, 2, 4]:
                        wave_volume = data.loc[wave.start_index:wave.end_index, 'Volume'].mean()
                        
                        avg_volume = data['Volume'].mean()
                        
                        if wave_volume > avg_volume * 1.2:
                            confirmation_score += 0.1
            
            return min(confirmation_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error checking volume confirmation: {e}")
            return 0.5
    
    def _check_wave_alternation(self, pattern: WavePattern) -> bool:
        """
        Check if corrective waves (2 and 4) alternate in form
        """
        if pattern.pattern_type != WaveType.IMPULSE or len(pattern.waves) < 4:
            return False
        
        wave2_retracement = pattern.waves[1].retracement
        wave4_retracement = pattern.waves[3].retracement
        
        return abs(wave2_retracement - wave4_retracement) > 0.1
    
    def _calculate_validation_score(self, pattern: WavePattern) -> float:
        """
        Calculate overall validation score for pattern
        """
        score = 0.0
        checks = 0
        
        if all(w.is_valid for w in pattern.waves):
            score += 1.0
        checks += 1
        
        if all(abs(w.end_price - w.start_price) >= self.min_wave_size for w in pattern.waves):
            score += 1.0
        checks += 1
        
        if self._check_fibonacci_relationships(pattern):
            score += 1.0
        checks += 1
        
        if self._check_time_relationships(pattern):
            score += 1.0
        checks += 1
        
        return score / checks if checks > 0 else 0.0
    
    # Renamed from _generate_signal to _generate_signal_dict (private helper for signal generation)
    def _generate_signal_dict(self, patterns: List[WavePattern], data: pd.DataFrame) -> Dict:
        """
        Generate trading signal dictionary from wave patterns.
        This dict is then converted to a Signal object by _create_signal_from_analysis.
        """
        if not patterns:
            return self._empty_signal_dict()
        
        best_pattern = max(patterns, key=lambda p: (p.confidence * 0.7 + p.validation_score * 0.3))
        current_price = float(data['Close'].iloc[-1])
        
        signal_dict = {
            'timestamp': datetime.now(),
            'symbol': 'XAUUSDm',
            'strategy': 'Elliott Wave Analysis',
            'pattern': best_pattern.pattern_type.value,
            'degree': best_pattern.degree.name,
            'confidence': best_pattern.confidence,
            'validation_score': best_pattern.validation_score,
            'current_price': current_price
        }
        
        if best_pattern.pattern_type == WaveType.IMPULSE:
            last_wave = best_pattern.waves[-1]
            is_bullish_impulse = last_wave.end_price > best_pattern.waves[0].start_price
            
            if is_bullish_impulse:
                signal_dict['direction'] = 'SELL'
                signal_dict['entry'] = current_price
                signal_dict['stop_loss'] = last_wave.end_price + 30
                
                impulse_size = last_wave.end_price - best_pattern.waves[0].start_price
                signal_dict['take_profit'] = current_price - (impulse_size * 0.382)
                signal_dict['take_profit_2'] = current_price - (impulse_size * 0.618)
                
                signal_dict['reason'] = f"Bullish impulse wave complete at {last_wave.end_price:.2f}, expecting ABC correction"
            else:
                signal_dict['direction'] = 'BUY'
                signal_dict['entry'] = current_price
                signal_dict['stop_loss'] = last_wave.end_price - 30
                
                impulse_size = best_pattern.waves[0].start_price - last_wave.end_price
                signal_dict['take_profit'] = current_price + (impulse_size * 0.382)
                signal_dict['take_profit_2'] = current_price + (impulse_size * 0.618)
                
                signal_dict['reason'] = f"Bearish impulse wave complete at {last_wave.end_price:.2f}, expecting ABC correction"
        
        elif best_pattern.pattern_type in [WaveType.ZIGZAG, WaveType.FLAT, WaveType.CORRECTIVE]:
            last_wave = best_pattern.waves[-1]
            is_bearish_correction = last_wave.end_price < best_pattern.waves[0].start_price
            
            if is_bearish_correction:
                signal_dict['direction'] = 'BUY'
                signal_dict['entry'] = current_price
                signal_dict['stop_loss'] = last_wave.end_price - 30
                
                correction_size = best_pattern.waves[0].start_price - last_wave.end_price
                signal_dict['take_profit'] = current_price + (correction_size * 1.618)
                signal_dict['take_profit_2'] = current_price + (correction_size * 2.618)
                
                signal_dict['reason'] = f"ABC correction complete at {last_wave.end_price:.2f}, expecting new impulse wave"
            else:
                signal_dict['direction'] = 'SELL'
                signal_dict['entry'] = current_price
                signal_dict['stop_loss'] = last_wave.end_price + 30
                
                correction_size = last_wave.end_price - best_pattern.waves[0].start_price
                signal_dict['take_profit'] = current_price - (correction_size * 1.618)
                signal_dict['take_profit_2'] = current_price - (correction_size * 2.618)
                
                signal_dict['reason'] = f"ABC correction complete at {last_wave.end_price:.2f}, expecting new impulse wave"
        else:
            signal_dict['direction'] = 'NEUTRAL'
            signal_dict['reason'] = "Pattern not clear for trading"
        
        signal_dict['wave_count'] = {
            'pattern_type': best_pattern.pattern_type.value,
            'completed_waves': len(best_pattern.waves),
            'wave_labels': [w.wave_label for w in best_pattern.waves],
            'current_wave': self._identify_current_wave(best_pattern),
            'next_wave_projection': self._project_next_wave([best_pattern], data)
        }
        
        signal_dict['pattern_details'] = {
            'start_price': best_pattern.start_price,
            'end_price': best_pattern.end_price,
            'pattern_size': abs(best_pattern.end_price - best_pattern.start_price),
            'duration': str(best_pattern.end_time - best_pattern.start_time) if hasattr(best_pattern.end_time, '__sub__') else 'N/A',
            'wave_relationships': self._get_wave_relationships(best_pattern)
        }
        
        return signal_dict
    
    def _generate_additional_signals(self, analysis_result: Dict, symbol: str, timeframe: str, data: pd.DataFrame) -> List[Signal]:
        """Generate additional signals from patterns for better signal generation"""
        signals = []
        
        try:
            if not self.mt5_manager:
                return []
            
            # Get data for analysis
            if data is None or len(data) < self.lookback_periods:
                return []
            
            swings = self._identify_swings(data)
            if len(swings) < 6:
                return []
            
            impulse_waves = self._find_impulse_waves(swings, data)
            corrective_waves = self._find_corrective_waves(swings, data)
            all_patterns = impulse_waves + corrective_waves
            valid_patterns = self._validate_patterns(all_patterns)
            
            # Generate signals from multiple valid patterns (not just the best one)
            for i, pattern in enumerate(valid_patterns[:5]):  # Limit to top 5 patterns
                if pattern.confidence >= max(0.50, self.min_confidence - 0.1):  # More lenient threshold
                    additional_signal = self._create_pattern_specific_signal(pattern, data, symbol, timeframe, i)
                    if additional_signal:
                        signals.append(additional_signal)
                        
            # Generate signals from intermediate wave completions
            intermediate_signals = self._generate_intermediate_wave_signals(valid_patterns, data, symbol, timeframe)
            signals.extend(intermediate_signals)
            
            return signals[:10]  # Limit total additional signals to 10
            
        except Exception as e:
            self.logger.error(f"Error generating additional signals: {str(e)}")
            return []
    
    def _create_pattern_specific_signal(self, pattern: WavePattern, data: pd.DataFrame, symbol: str, timeframe: str, index: int) -> Optional[Signal]:
        """Create a signal from a specific pattern"""
        try:
            current_price = float(data['Close'].iloc[-1])
            
            if pattern.pattern_type == WaveType.IMPULSE:
                last_wave = pattern.waves[-1]
                is_bullish_impulse = last_wave.end_price > pattern.waves[0].start_price
                
                confidence_adjustment = 1.0 - (index * 0.1)  # Reduce confidence for lower-ranked patterns
                adjusted_confidence = pattern.confidence * confidence_adjustment
                
                if is_bullish_impulse:
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name=self.strategy_name,
                        signal_type=SignalType.SELL,  # Expect correction after impulse up
                        confidence=max(0.5, adjusted_confidence),
                        price=current_price,
                        timeframe=timeframe,
                        strength=pattern.validation_score,
                        stop_loss=last_wave.end_price + 25,
                        take_profit=current_price - (abs(pattern.end_price - pattern.start_price) * 0.382),
                        metadata={
                            'signal_reason': f'impulse_completion_{index+1}',
                            'pattern_type': pattern.pattern_type.value,
                            'wave_count': len(pattern.waves),
                            'pattern_rank': index + 1
                        }
                    )
                else:
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name=self.strategy_name,
                        signal_type=SignalType.BUY,  # Expect correction after impulse down
                        confidence=max(0.5, adjusted_confidence),
                        price=current_price,
                        timeframe=timeframe,
                        strength=pattern.validation_score,
                        stop_loss=last_wave.end_price - 25,
                        take_profit=current_price + (abs(pattern.end_price - pattern.start_price) * 0.382),
                        metadata={
                            'signal_reason': f'impulse_completion_{index+1}',
                            'pattern_type': pattern.pattern_type.value,
                            'wave_count': len(pattern.waves),
                            'pattern_rank': index + 1
                        }
                    )
            
            elif pattern.pattern_type in [WaveType.CORRECTIVE, WaveType.ZIGZAG, WaveType.FLAT]:
                last_wave = pattern.waves[-1]
                is_bearish_correction = last_wave.end_price < pattern.waves[0].start_price
                
                confidence_adjustment = 1.0 - (index * 0.1)
                adjusted_confidence = pattern.confidence * confidence_adjustment
                
                if is_bearish_correction:
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name=self.strategy_name,
                        signal_type=SignalType.BUY,  # Buy after correction down
                        confidence=max(0.5, adjusted_confidence),
                        price=current_price,
                        timeframe=timeframe,
                        strength=pattern.validation_score,
                        stop_loss=last_wave.end_price - 25,
                        take_profit=current_price + (abs(pattern.end_price - pattern.start_price) * 1.27),
                        metadata={
                            'signal_reason': f'correction_completion_{index+1}',
                            'pattern_type': pattern.pattern_type.value,
                            'wave_count': len(pattern.waves),
                            'pattern_rank': index + 1
                        }
                    )
                else:
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy_name=self.strategy_name,
                        signal_type=SignalType.SELL,  # Sell after correction up
                        confidence=max(0.5, adjusted_confidence),
                        price=current_price,
                        timeframe=timeframe,
                        strength=pattern.validation_score,
                        stop_loss=last_wave.end_price + 25,
                        take_profit=current_price - (abs(pattern.end_price - pattern.start_price) * 1.27),
                        metadata={
                            'signal_reason': f'correction_completion_{index+1}',
                            'pattern_type': pattern.pattern_type.value,
                            'wave_count': len(pattern.waves),
                            'pattern_rank': index + 1
                        }
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating pattern-specific signal: {str(e)}")
            return None
    
    def _generate_intermediate_wave_signals(self, patterns: List[WavePattern], data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """Generate signals from intermediate wave formations"""
        signals = []
        
        try:
            if not patterns:
                return signals
                
            current_price = float(data['Close'].iloc[-1])
            
            # Look for wave 3 completions (strongest waves)
            for pattern in patterns[:3]:  # Top 3 patterns
                if pattern.pattern_type == WaveType.IMPULSE and len(pattern.waves) >= 3:
                    wave3 = pattern.waves[2]  # Third wave (index 2)
                    
                    # Check if wave 3 might be completing (strong momentum)
                    wave3_size = abs(wave3.end_price - wave3.start_price)
                    if wave3_size > self.min_wave_size:
                        
                        if wave3.end_price > wave3.start_price:  # Bullish wave 3
                            signals.append(Signal(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                strategy_name=self.strategy_name,
                                signal_type=SignalType.BUY,  # Continue with wave 5
                                confidence=0.62,
                                price=current_price,
                                timeframe=timeframe,
                                strength=0.65,
                                stop_loss=wave3.start_price - 15,
                                take_profit=current_price + wave3_size * 0.618,
                                metadata={
                                    'signal_reason': 'wave3_momentum_continuation',
                                    'wave_size': wave3_size
                                }
                            ))
                        else:  # Bearish wave 3
                            signals.append(Signal(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                strategy_name=self.strategy_name,
                                signal_type=SignalType.SELL,  # Continue with wave 5
                                confidence=0.62,
                                price=current_price,
                                timeframe=timeframe,
                                strength=0.65,
                                stop_loss=wave3.start_price + 15,
                                take_profit=current_price - wave3_size * 0.618,
                                metadata={
                                    'signal_reason': 'wave3_momentum_continuation',
                                    'wave_size': wave3_size
                                }
                            ))
            
            # Look for ABC correction signals
            for pattern in patterns:
                if pattern.pattern_type in [WaveType.CORRECTIVE, WaveType.ZIGZAG] and len(pattern.waves) >= 2:
                    wave_b = pattern.waves[1]  # B wave
                    
                    # Generate signal on B wave completion (anticipating C wave)
                    if abs(wave_b.end_price - wave_b.start_price) > self.min_wave_size:
                        
                        if wave_b.end_price > wave_b.start_price:  # B wave up, expect C down
                            signals.append(Signal(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                strategy_name=self.strategy_name,
                                signal_type=SignalType.SELL,
                                confidence=0.58,
                                price=current_price,
                                timeframe=timeframe,
                                strength=0.60,
                                stop_loss=wave_b.end_price + 20,
                                take_profit=current_price - abs(wave_b.end_price - wave_b.start_price) * 1.27,
                                metadata={
                                    'signal_reason': 'wave_b_completion',
                                    'expected_wave': 'C'
                                }
                            ))
                        else:  # B wave down, expect C up
                            signals.append(Signal(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                strategy_name=self.strategy_name,
                                signal_type=SignalType.BUY,
                                confidence=0.58,
                                price=current_price,
                                timeframe=timeframe,
                                strength=0.60,
                                stop_loss=wave_b.end_price - 20,
                                take_profit=current_price + abs(wave_b.end_price - wave_b.start_price) * 1.27,
                                metadata={
                                    'signal_reason': 'wave_b_completion',
                                    'expected_wave': 'C'
                                }
                            ))
            
            return signals[:5]  # Limit intermediate signals
            
        except Exception as e:
            self.logger.error(f"Error generating intermediate wave signals: {str(e)}")
            return []
    
    def _create_signal_from_analysis(self, analysis: Dict, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[Signal]:
        """
        Create a Signal object from analysis results dictionary.
        """
        try:
            if analysis['direction'] == 'BUY':
                signal_type = SignalType.BUY
            elif analysis['direction'] == 'SELL':
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            confidence = analysis['confidence']
            
            metadata = {
                'pattern': analysis.get('pattern', 'unknown'),
                'degree': analysis.get('degree', 'MINOR'),
                'wave_count': analysis.get('wave_count', {}),
                'pattern_details': analysis.get('pattern_details', {}),
                'validation_score': analysis.get('validation_score', 0)
            }
            
            signal = Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                strategy_name=self.strategy_name, # Use self.strategy_name from base class
                signal_type=signal_type,
                confidence=confidence,
                price=analysis.get('entry', data['Close'].iloc[-1]),
                timeframe=timeframe,
                strength=confidence * analysis.get('validation_score', 1.0),
                # Grade is now automatically calculated by Signal's __post_init__
                # grade=grade,
                stop_loss=analysis.get('stop_loss'),
                take_profit=analysis.get('take_profit'),
                metadata=metadata
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating signal from analysis: {str(e)}", exc_info=True)
            return None
    
    def _identify_current_wave(self, pattern: WavePattern) -> str:
        """
        Identify which wave is currently forming
        """
        if pattern.pattern_type == WaveType.IMPULSE:
            wave_labels = ['1', '2', '3', '4', '5']
            if pattern.is_complete:
                return "Wave 5 complete - Awaiting correction"
            else:
                current_idx = len(pattern.waves) - 1
                if current_idx < len(wave_labels):
                    return f"Wave {wave_labels[current_idx]} in progress"
                else:
                    return "Extended wave pattern"
        else:
            wave_labels = ['A', 'B', 'C']
            if pattern.is_complete:
                return "Wave C complete - Awaiting new impulse"
            else:
                current_idx = len(pattern.waves) - 1
                if current_idx < len(wave_labels):
                    return f"Wave {wave_labels[current_idx]} in progress"
                else:
                    return "Complex correction"
    
    def _project_next_wave(self, patterns: List[WavePattern], data: pd.DataFrame) -> Dict:
        """
        Project the next wave targets
        """
        if not patterns:
            return {}
        
        pattern = patterns[0]
        current_price = float(data['Close'].iloc[-1])
        
        projections = {
            'current_price': current_price,
            'pattern_type': pattern.pattern_type.value,
            'fibonacci_targets': [],
            'time_projections': []
        }
        
        if pattern.pattern_type == WaveType.IMPULSE and pattern.is_complete:
            impulse_range = abs(pattern.end_price - pattern.start_price)
            
            for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]:
                if pattern.end_price > pattern.start_price:
                    target = pattern.end_price - (impulse_range * ratio)
                else:
                    target = pattern.end_price + (impulse_range * ratio)
                
                projections['fibonacci_targets'].append({
                    'level': f"{ratio*100:.1f}%",
                    'price': round(target, 2),
                    'type': 'retracement'
                })
        
        elif pattern.pattern_type in [WaveType.CORRECTIVE, WaveType.ZIGZAG, WaveType.FLAT]:
            correction_range = abs(pattern.end_price - pattern.start_price)
            
            for ratio in [1.0, 1.272, 1.618, 2.0, 2.618]:
                if pattern.end_price < pattern.start_price:
                    target = pattern.end_price + (correction_range * ratio)
                else:
                    target = pattern.end_price - (correction_range * ratio)
                
                projections['fibonacci_targets'].append({
                    'level': f"{ratio*100:.0f}%",
                    'price': round(target, 2),
                    'type': 'extension'
                })
        
        return projections
    
    def _get_wave_relationships(self, pattern: WavePattern) -> Dict:
        """
        Get Fibonacci relationships between waves
        """
        relationships = {}
        
        if len(pattern.waves) < 2:
            return relationships
        
        for i, wave in enumerate(pattern.waves):
            if i == 0:
                continue
                
            wave_size = abs(wave.end_price - wave.start_price)
            prev_wave_size = abs(pattern.waves[i-1].end_price - pattern.waves[i-1].start_price)
            
            if prev_wave_size > 0:
                ratio = wave_size / prev_wave_size
                relationships[f"Wave_{wave.wave_label}_to_{pattern.waves[i-1].wave_label}"] = round(ratio, 3)
        
        if pattern.pattern_type == WaveType.IMPULSE and len(pattern.waves) >= 3:
            wave1_size = abs(pattern.waves[0].end_price - pattern.waves[0].start_price)
            wave3_size = abs(pattern.waves[2].end_price - pattern.waves[2].start_price)
            
            if wave1_size > 0:
                relationships["Wave_3_to_1"] = round(wave3_size / wave1_size, 3)
            
            if len(pattern.waves) >= 5:
                wave5_size = abs(pattern.waves[4].end_price - pattern.waves[4].start_price)
                if wave1_size > 0:
                    relationships["Wave_5_to_1"] = round(wave5_size / wave1_size, 3)
                if wave3_size > 0:
                    relationships["Wave_5_to_3"] = round(wave5_size / wave3_size, 3)
        
        return relationships
    
    def _empty_signal_dict(self) -> Dict:
        """Generate empty/neutral signal dictionary for internal use"""
        return {
            'timestamp': datetime.now(),
            'symbol': 'XAUUSDm',
            'strategy': self.strategy_name,
            'direction': 'NEUTRAL',
            'confidence': 0.0,
            'pattern': None,
            'reason': 'No valid Elliott Wave pattern detected',
            'wave_count': {},
            'projections': {}
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and performance"""
        # Access performance attributes directly from the self.performance object
        # which is an instance of StrategyPerformance (from base.py)
        
        return {
            'name': 'Elliott Wave Analysis Strategy',
            'type': 'Technical',
            'version': '2.0.0',
            'description': 'Advanced Elliott Wave pattern recognition with Fibonacci validation',
            'parameters': {
                'min_wave_size': self.min_wave_size,
                'lookback_periods': self.lookback_periods,
                'min_confidence': self.min_confidence,
                'fibonacci_tolerance': self.fib_ratios['tolerance']
            },
            'performance': {
                'total_signals_generated': self.performance.total_signals, # Direct access
                'successful_signals': self.performance.successful_signals, # Direct access
                'win_rate': self.performance.win_rate,
                'profit_factor': self.performance.profit_factor
            },
            'wave_types_supported': [wt.value for wt in WaveType],
            'wave_degrees': [wd.name for wd in WaveDegree]
        }


# Testing function
if __name__ == "__main__":
    """Test Elliott Wave strategy functionality"""
    
    # Mock MT5 manager for testing
    class MockMT5Manager:
        def get_historical_data(self, symbol, timeframe, bars):
            """Generate synthetic wave-like price data"""
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            dates = pd.date_range(start=datetime.now() - timedelta(days=10), 
                                 periods=bars, freq='15min')
            
            x = np.linspace(0, 4*np.pi, bars)
            
            prices = []
            base_price = 2000.0
            
            wave1 = base_price + np.linspace(0, 50, bars//8)
            prices.extend(wave1)
            
            wave2 = wave1[-1] - np.linspace(0, 19, bars//8)
            prices.extend(wave2)
            
            wave3 = wave2[-1] + np.linspace(0, 81, bars//8)
            prices.extend(wave3)
            
            wave4 = wave3[-1] - np.linspace(0, 31, bars//8)
            prices.extend(wave4)
            
            wave5 = wave4[-1] + np.linspace(0, 50, bars//8)
            prices.extend(wave5)
            
            waveA = wave5[-1] - np.linspace(0, 40, bars//8)
            prices.extend(waveA)
            
            waveB = waveA[-1] + np.linspace(0, 25, bars//8)
            prices.extend(waveB)
            
            waveC = waveB[-1] - np.linspace(0, 40, bars//8)
            prices.extend(waveC)
            
            remaining = bars - len(prices)
            if remaining > 0:
                prices.extend([prices[-1] + np.random.normal(0, 2) for _ in range(remaining)])
            
            prices = prices[:bars]
            
            prices = np.array(prices) + np.random.normal(0, 1, len(prices))
            
            data = pd.DataFrame({
                'Open': prices - np.random.uniform(0, 2, len(prices)),
                'High': prices + np.random.uniform(2, 5, len(prices)),
                'Low': prices - np.random.uniform(2, 5, len(prices)),
                'Close': prices,
                'Volume': np.random.uniform(1000, 5000, len(prices))
            }, index=dates)
            
            data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
            data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
            
            return data
    
    # Test configuration
    test_config = {
        'parameters': {
            'min_wave_size': 10,
            'lookback_periods': 200,
            'min_confidence': 0.60,
            'use_volume': True,
            'strict_rules': False,
            'fibonacci_tolerance': 0.10
        }
    }
    
    try:
        # Initialize strategy
        mock_mt5 = MockMT5Manager()
        # Pass test_config directly, AbstractStrategy will assign it to self.config
        elliott_strategy = ElliottWaveStrategy(test_config, mock_mt5, database=None)
        
        # Output header matching other strategy files
        print("============================================================")
        print("TESTING MODIFIED ELLIOTT WAVE STRATEGY")
        print("============================================================")

        # 1. Testing signal generation
        print("\n1. Testing signal generation:")
        signals = elliott_strategy.generate_signal("XAUUSDm", "M15")
        print(f"   Generated {len(signals)} signals")
        for i, signal in enumerate(signals, 1):
            print(f"   - Signal {i}:")
            print(f"     Type: {signal.signal_type.value}")
            print(f"     Confidence: {signal.confidence:.2%}")
            print(f"     Grade: {signal.grade.value if signal.grade else 'N/A'}")
            print(f"     Price: {signal.price:.2f}")
            print(f"     Stop Loss: {signal.stop_loss:.2f}" if signal.stop_loss else "     Stop Loss: N/A")
            print(f"     Take Profit: {signal.take_profit:.2f}" if signal.take_profit else "     Take Profit: N/A")
            if signal.metadata:
                print(f"     Pattern: {signal.metadata.get('pattern', 'N/A')}")
                print(f"     Degree: {signal.metadata.get('degree', 'N/A')}")
                if 'wave_count' in signal.metadata:
                    wave_count = signal.metadata['wave_count']
                    if isinstance(wave_count, dict):
                        print(f"     Current Wave: {wave_count.get('current_wave', 'N/A')}")
                    else:
                        print(f"     Wave Count: {wave_count}")
        
        # 2. Testing analysis method
        print("\n2. Testing analysis method:")
        mock_data = mock_mt5.get_historical_data("XAUUSDm", "M15", 200)
        analysis = elliott_strategy.analyze(mock_data, "XAUUSDm", "M15")
        print(f"   Analysis results keys: {analysis.keys()}")
        if 'detailed_patterns' in analysis:
            print(f"   Detected patterns in analysis: {len(analysis['detailed_patterns'])}")
            if analysis['detailed_patterns']:
                print(f"   First pattern type: {analysis['detailed_patterns'][0]['pattern_type']}")
        
        # 3. Testing performance tracking
        print("\n3. Testing performance tracking:")
        summary = elliott_strategy.get_performance_summary()
        print(f"   {summary}")
        
        # 4. Strategy Information
        print("\n4. Strategy Information:")
        strategy_info = elliott_strategy.get_strategy_info()
        print(f"   Name: {strategy_info['name']}")
        print(f"   Version: {strategy_info['version']}")
        print(f"   Description: {strategy_info['description']}")
        print(f"   Type: {strategy_info['type']}")
        print(f"   Wave Types Supported: {', '.join(strategy_info['wave_types_supported'])}")
        print(f"   Wave Degrees: {', '.join(strategy_info['wave_degrees'])}")
        print(f"   Parameters:")
        for param, value in strategy_info['parameters'].items():
            print(f"     - {param}: {value}")
        print(f"   Performance Summary:")
        # Corrected: Accessing total_signals and successful_signals directly from self.performance
        print(f"     Total Signals Generated: {strategy_info['performance']['total_signals_generated']}")
        print(f"     Successful Signals: {strategy_info['performance']['successful_signals']}")
        print(f"     Win Rate: {strategy_info['performance']['win_rate']:.2%}")
        print(f"     Profit Factor: {strategy_info['performance']['profit_factor']:.2f}")

        # Footer matching other strategy files
        print("\n============================================================")
        print("ELLIOTT WAVE STRATEGY TEST COMPLETED!")
        print("============================================================")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()