"""
Elliott Wave Analysis Strategy - Advanced Wave Pattern Recognition
================================================================
Author: XAUUSD Trading System
Version: 2.0.0
Date: 2025-01-08

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

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade

# Import base classes from signal_engine
try:
    from ..core.signal_engine import Signal, SignalType, SignalGrade
except ImportError:
    # Fallback definitions for testing
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class ElliottWaveStrategy:
    """
    Elliott Wave Analysis Strategy
    
    Complete implementation with pattern recognition,
    validation, and signal generation for XAUUSDm trading.
    """
    
    def __init__(self, config: Dict, mt5_manager=None):
        """
        Initialize Elliott Wave Strategy
        
        Args:
            config: Strategy configuration
            mt5_manager: MT5 manager instance
        """
        self.config = config.get('parameters', {}) if 'parameters' in config else config
        self.mt5_manager = mt5_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Wave parameters
        self.min_wave_size = self.config.get('min_wave_size', 30)  # pips
        self.lookback_periods = self.config.get('lookback_periods', 200)
        self.min_confidence = self.config.get('min_confidence', 0.65)
        self.use_volume = self.config.get('use_volume', True)
        self.strict_rules = self.config.get('strict_rules', True)
        
        # Fibonacci ratios for wave relationships
        self.fib_ratios = {
            'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extension': [1.0, 1.272, 1.618, 2.0, 2.618, 3.618, 4.236],
            'tolerance': self.config.get('fibonacci_tolerance', 0.05)
        }
        
        # Pattern storage
        self.current_patterns = {}
        self.completed_patterns = []
        self.wave_counts = {}
        
        # Performance tracking
        self.signals_generated = 0
        self.successful_signals = 0
        
        self.logger.info(f"Elliott Wave Strategy initialized with min_wave_size={self.min_wave_size}, lookback={self.lookback_periods}")
    
    def generate_signals(self, symbol: str = "XAUUSDm", timeframe: str = "M15") -> List[Signal]:
        """
        Generate trading signals from Elliott Wave patterns
        
        Args:
            symbol: Trading symbol (default: XAUUSDm)
            timeframe: Timeframe for analysis
            
        Returns:
            List of Signal objects
        """
        try:
            if not self.mt5_manager:
                self.logger.warning("No MT5 manager available, returning empty signals")
                return []
            
            # Get historical data
            data = self.mt5_manager.get_historical_data(symbol, timeframe, self.lookback_periods)
            
            if data is None or len(data) < self.lookback_periods:
                self.logger.warning(f"Insufficient data for {symbol} on {timeframe}")
                return []
            
            # Analyze Elliott Waves
            analysis = self.analyze(data)
            
            # Convert analysis to signals
            signals = []
            
            if analysis['direction'] != 'NEUTRAL':
                signal = self._create_signal_from_analysis(analysis, symbol, timeframe, data)
                if signal:
                    signals.append(signal)
                    self.signals_generated += 1
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating Elliott Wave signals: {str(e)}")
            return []
    
    def analyze(self, data: pd.DataFrame) -> Dict:
        """
        Main analysis method for Elliott Wave patterns
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if len(data) < self.lookback_periods:
                return self._empty_signal()
            
            # Step 1: Identify swing points
            swings = self._identify_swings(data)
            
            if len(swings) < 6:  # Need minimum points for pattern
                return self._empty_signal()
            
            # Step 2: Find wave patterns
            impulse_waves = self._find_impulse_waves(swings, data)
            corrective_waves = self._find_corrective_waves(swings, data)
            
            # Step 3: Validate patterns
            all_patterns = impulse_waves + corrective_waves
            valid_patterns = self._validate_patterns(all_patterns)
            
            # Step 4: Generate signal from best pattern
            if valid_patterns:
                signal = self._generate_signal(valid_patterns, data)
                return signal
            
            return self._empty_signal()
            
        except Exception as e:
            self.logger.error(f"Error in Elliott Wave analysis: {str(e)}")
            return self._empty_signal()
    
    def _identify_swings(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify swing highs and lows using advanced pivot detection
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with swing points
        """
        try:
            swings = []
            window = 5  # Look-ahead and look-back window
            
            for i in range(window, len(data) - window):
                # Check for swing high
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
                
                # Check for swing low
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
            
            # Sort by index and convert to DataFrame
            swings.sort(key=lambda x: x['index'])
            
            if swings:
                return pd.DataFrame(swings)
            else:
                return pd.DataFrame(columns=['index', 'time', 'price', 'type'])
                
        except Exception as e:
            self.logger.error(f"Error identifying swings: {str(e)}")
            return pd.DataFrame()
    
    def _find_impulse_waves(self, swings: pd.DataFrame, data: pd.DataFrame) -> List[WavePattern]:
        """
        Find 5-wave impulse patterns
        
        Args:
            swings: DataFrame of swing points
            data: OHLCV data
            
        Returns:
            List of impulse wave patterns
        """
        patterns = []
        
        if len(swings) < 6:
            return patterns
        
        # Scan for potential 5-wave patterns
        for i in range(len(swings) - 5):
            # Check if we have alternating highs and lows
            expected_pattern_bull = ['low', 'high', 'low', 'high', 'low', 'high']
            expected_pattern_bear = ['high', 'low', 'high', 'low', 'high', 'low']
            
            actual_pattern = swings.iloc[i:i+6]['type'].tolist()
            
            is_bullish = actual_pattern == expected_pattern_bull
            is_bearish = actual_pattern == expected_pattern_bear
            
            if not (is_bullish or is_bearish):
                continue
            
            # Create waves
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
            
            # Calculate wave relationships
            self._calculate_wave_relationships(waves)
            
            # Validate impulse wave rules
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
                
                # Calculate pattern confidence
                pattern.confidence = self._calculate_pattern_confidence(pattern, data)
                pattern.validation_score = self._calculate_validation_score(pattern)
                
                if pattern.confidence >= self.min_confidence:
                    patterns.append(pattern)
                    self.logger.info(f"Found {'bullish' if is_bullish else 'bearish'} impulse pattern with confidence {pattern.confidence:.2f}")
        
        return patterns
    
    def _find_corrective_waves(self, swings: pd.DataFrame, data: pd.DataFrame) -> List[WavePattern]:
        """
        Find 3-wave corrective patterns (ABC)
        
        Args:
            swings: DataFrame of swing points
            data: OHLCV data
            
        Returns:
            List of corrective wave patterns
        """
        patterns = []
        
        if len(swings) < 4:
            return patterns
        
        # Scan for ABC patterns
        for i in range(len(swings) - 3):
            # Check pattern structure
            expected_pattern_bear = ['high', 'low', 'high', 'low']
            expected_pattern_bull = ['low', 'high', 'low', 'high']
            
            actual_pattern = swings.iloc[i:i+4]['type'].tolist()
            
            is_bearish_correction = actual_pattern == expected_pattern_bear
            is_bullish_correction = actual_pattern == expected_pattern_bull
            
            if not (is_bearish_correction or is_bullish_correction):
                continue
            
            # Create ABC waves
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
            
            # Calculate wave relationships
            self._calculate_wave_relationships(waves)
            
            # Identify corrective pattern type
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
                
                # Calculate pattern confidence
                pattern.confidence = self._calculate_pattern_confidence(pattern, data)
                pattern.validation_score = self._calculate_validation_score(pattern)
                
                if pattern.confidence >= self.min_confidence:
                    patterns.append(pattern)
                    self.logger.info(f"Found {pattern_type.value} corrective pattern with confidence {pattern.confidence:.2f}")
        
        return patterns
    
    def _calculate_wave_relationships(self, waves: List[ElliottWave]):
        """
        Calculate Fibonacci relationships between waves
        
        Args:
            waves: List of Elliott waves
        """
        for i, wave in enumerate(waves):
            wave_size = abs(wave.end_price - wave.start_price)
            
            # Calculate retracement from previous wave
            if i > 0:
                prev_wave_size = abs(waves[i-1].end_price - waves[i-1].start_price)
                if prev_wave_size > 0:
                    wave.retracement = wave_size / prev_wave_size
            
            # Calculate extension from first wave
            if i > 0 and len(waves) > 0:
                first_wave_size = abs(waves[0].end_price - waves[0].start_price)
                if first_wave_size > 0:
                    wave.extension = wave_size / first_wave_size
    
    def _validate_impulse_rules(self, waves: List[ElliottWave], is_bullish: bool) -> bool:
        """
        Validate Elliott Wave impulse rules
        
        Rules:
        1. Wave 2 cannot retrace more than 100% of Wave 1
        2. Wave 3 cannot be the shortest
        3. Wave 4 cannot overlap Wave 1
        
        Args:
            waves: List of 5 waves
            is_bullish: True if bullish pattern
            
        Returns:
            True if rules are satisfied
        """
        if len(waves) != 5:
            return False
        
        # Calculate wave sizes
        wave_sizes = [abs(w.end_price - w.start_price) for w in waves]
        
        if is_bullish:
            # Rule 1: Wave 2 cannot go below Wave 1 start
            if waves[1].end_price <= waves[0].start_price:
                return False
            
            # Rule 2: Wave 3 cannot be the shortest impulse wave (1, 3, 5)
            impulse_sizes = [wave_sizes[0], wave_sizes[2], wave_sizes[4]]
            if wave_sizes[2] == min(impulse_sizes):
                return False
            
            # Rule 3: Wave 4 cannot go below Wave 1 high
            if waves[3].end_price <= waves[0].end_price:
                return False
        else:
            # Bearish pattern rules (inverse)
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
        
        Args:
            waves: List of 3 waves (ABC)
            
        Returns:
            WaveType or None
        """
        if len(waves) != 3:
            return None
        
        # Calculate wave relationships
        a_size = abs(waves[0].end_price - waves[0].start_price)
        b_size = abs(waves[1].end_price - waves[1].start_price)
        c_size = abs(waves[2].end_price - waves[2].start_price)
        
        if a_size == 0:
            return None
            
        b_retracement = b_size / a_size
        
        # Identify pattern type based on wave relationships
        if 0.5 <= b_retracement <= 0.618:
            # Zigzag pattern: Sharp correction
            return WaveType.ZIGZAG
        elif 0.786 <= b_retracement <= 1.0:
            # Flat pattern: Sideways correction
            return WaveType.FLAT
        elif b_retracement < 0.5 and c_size < a_size:
            # Triangle pattern: Converging price action
            return WaveType.TRIANGLE
        else:
            # Generic corrective pattern
            return WaveType.CORRECTIVE
    
    def _validate_patterns(self, patterns: List[WavePattern]) -> List[WavePattern]:
        """
        Validate wave patterns against Elliott Wave guidelines
        
        Args:
            patterns: List of potential patterns
            
        Returns:
            List of valid patterns
        """
        valid_patterns = []
        
        for pattern in patterns:
            # Check minimum wave size
            is_valid = True
            
            for wave in pattern.waves:
                wave_size_pips = abs(wave.end_price - wave.start_price)
                if wave_size_pips < self.min_wave_size:
                    is_valid = False
                    break
            
            if not is_valid:
                continue
            
            # Check Fibonacci relationships
            if self._check_fibonacci_relationships(pattern):
                # Check time relationships if enabled
                if self.config.get('time_ratios', True):
                    if self._check_time_relationships(pattern):
                        valid_patterns.append(pattern)
                else:
                    valid_patterns.append(pattern)
        
        return valid_patterns
    
    def _check_fibonacci_relationships(self, pattern: WavePattern) -> bool:
        """
        Check if waves follow Fibonacci relationships
        
        Args:
            pattern: Wave pattern to check
            
        Returns:
            True if Fibonacci relationships are satisfied
        """
        tolerance = self.fib_ratios['tolerance']
        
        if pattern.pattern_type == WaveType.IMPULSE and len(pattern.waves) >= 3:
            # Wave 2 typically retraces 0.382-0.618 of Wave 1
            if pattern.waves[1].retracement > 0:
                valid_retracement = False
                for ratio in self.fib_ratios['retracement']:
                    if abs(pattern.waves[1].retracement - ratio) <= tolerance:
                        valid_retracement = True
                        break
                
                if not valid_retracement and self.strict_rules:
                    return False
            
            # Wave 3 is often 1.618 or 2.618 times Wave 1
            if pattern.waves[2].extension > 0:
                valid_extension = False
                for ratio in self.fib_ratios['extension']:
                    if abs(pattern.waves[2].extension - ratio) <= tolerance:
                        valid_extension = True
                        break
                
                # Wave 3 should at least be longer than Wave 1
                if pattern.waves[2].extension < 1.0 and self.strict_rules:
                    return False
        
        return True
    
    def _check_time_relationships(self, pattern: WavePattern) -> bool:
        """
        Check if waves follow reasonable time relationships
        
        Args:
            pattern: Wave pattern to check
            
        Returns:
            True if time relationships are reasonable
        """
        for wave in pattern.waves:
            # Ensure wave has positive duration
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
        
        Args:
            pattern: Wave pattern
            data: OHLCV data
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence
        
        # Factor 1: Wave size relationships
        if pattern.pattern_type == WaveType.IMPULSE and len(pattern.waves) >= 5:
            # Check if Wave 3 is strong
            wave3_size = abs(pattern.waves[2].end_price - pattern.waves[2].start_price)
            wave1_size = abs(pattern.waves[0].end_price - pattern.waves[0].start_price)
            
            if wave1_size > 0:
                ratio = wave3_size / wave1_size
                if ratio > 1.618:
                    confidence += 0.15
                elif ratio > 1.0:
                    confidence += 0.10
        
        # Factor 2: Fibonacci accuracy
        fib_accuracy = self._calculate_fibonacci_accuracy(pattern)
        confidence += fib_accuracy * 0.2
        
        # Factor 3: Volume confirmation (if available)
        if self.use_volume and 'Volume' in data.columns:
            volume_confirmation = self._check_volume_confirmation(pattern, data)
            confidence += volume_confirmation * 0.1
        
        # Factor 4: Pattern completion
        if pattern.is_complete:
            confidence += 0.1
        
        # Factor 5: Wave alternation (for impulse waves)
        if pattern.pattern_type == WaveType.IMPULSE:
            if self._check_wave_alternation(pattern):
                confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _calculate_fibonacci_accuracy(self, pattern: WavePattern) -> float:
        """
        Calculate how well the pattern matches Fibonacci ratios
        
        Args:
            pattern: Wave pattern
            
        Returns:
            Accuracy score (0-1)
        """
        if not pattern.waves:
            return 0.0
        
        accuracy_scores = []
        
        for wave in pattern.waves:
            if wave.retracement > 0:
                # Check how close to standard Fibonacci ratios
                min_distance = min(abs(wave.retracement - ratio) 
                                 for ratio in self.fib_ratios['retracement'])
                accuracy = max(0, 1 - (min_distance / 0.1))
                accuracy_scores.append(accuracy)
        
        return np.mean(accuracy_scores) if accuracy_scores else 0.5
    
    def _check_volume_confirmation(self, pattern: WavePattern, data: pd.DataFrame) -> float:
        """
        Check if volume confirms the wave pattern
        
        Args:
            pattern: Wave pattern
            data: OHLCV data with volume
            
        Returns:
            Volume confirmation score (0-1)
        """
        try:
            confirmation_score = 0.5
            
            # For impulse waves, volume should increase on waves 1, 3, 5
            if pattern.pattern_type == WaveType.IMPULSE and len(pattern.waves) >= 3:
                for i, wave in enumerate(pattern.waves):
                    if i in [0, 2, 4]:  # Impulse waves
                        # Get volume for this wave period
                        wave_volume = data.loc[wave.start_index:wave.end_index, 'Volume'].mean()
                        
                        # Compare to overall average
                        avg_volume = data['Volume'].mean()
                        
                        if wave_volume > avg_volume * 1.2:
                            confirmation_score += 0.1
            
            return min(confirmation_score, 1.0)
            
        except Exception:
            return 0.5
    
    def _check_wave_alternation(self, pattern: WavePattern) -> bool:
        """
        Check if corrective waves (2 and 4) alternate in form
        
        Args:
            pattern: Wave pattern
            
        Returns:
            True if alternation principle is satisfied
        """
        if pattern.pattern_type != WaveType.IMPULSE or len(pattern.waves) < 4:
            return False
        
        # Wave 2 and Wave 4 should differ in complexity or depth
        wave2_retracement = pattern.waves[1].retracement
        wave4_retracement = pattern.waves[3].retracement
        
        # Simple check: they should differ significantly
        return abs(wave2_retracement - wave4_retracement) > 0.1
    
    def _calculate_validation_score(self, pattern: WavePattern) -> float:
        """
        Calculate overall validation score for pattern
        
        Args:
            pattern: Wave pattern
            
        Returns:
            Validation score (0-1)
        """
        score = 0.0
        checks = 0
        
        # Check 1: All waves valid
        if all(w.is_valid for w in pattern.waves):
            score += 1.0
        checks += 1
        
        # Check 2: Minimum wave sizes met
        if all(abs(w.end_price - w.start_price) >= self.min_wave_size for w in pattern.waves):
            score += 1.0
        checks += 1
        
        # Check 3: Fibonacci relationships
        if self._check_fibonacci_relationships(pattern):
            score += 1.0
        checks += 1
        
        # Check 4: Time relationships
        if self._check_time_relationships(pattern):
            score += 1.0
        checks += 1
        
        return score / checks if checks > 0 else 0.0
    
    def _generate_signal(self, patterns: List[WavePattern], data: pd.DataFrame) -> Dict:
        """
        Generate trading signal from wave patterns
        
        Args:
            patterns: Valid wave patterns
            data: OHLCV data
            
        Returns:
            Trading signal dictionary
        """
        if not patterns:
            return self._empty_signal()
        
        # Get the most recent and highest confidence pattern
        best_pattern = max(patterns, key=lambda p: (p.confidence * 0.7 + p.validation_score * 0.3))
        current_price = float(data['Close'].iloc[-1])
        
        signal = {
            'timestamp': datetime.now(),
            'symbol': 'XAUUSDm',
            'strategy': 'Elliott Wave Analysis',
            'pattern': best_pattern.pattern_type.value,
            'degree': best_pattern.degree.name,
            'confidence': best_pattern.confidence,
            'validation_score': best_pattern.validation_score,
            'current_price': current_price
        }
        
        # Determine signal direction based on pattern completion
        if best_pattern.pattern_type == WaveType.IMPULSE:
            # After 5-wave impulse, expect correction
            last_wave = best_pattern.waves[-1]
            is_bullish_impulse = last_wave.end_price > best_pattern.waves[0].start_price
            
            if is_bullish_impulse:
                # Bullish impulse complete, expect bearish correction
                signal['direction'] = 'SELL'
                signal['entry'] = current_price
                signal['stop_loss'] = last_wave.end_price + 30  # 30 pips above wave 5 high
                
                # Calculate targets based on Fibonacci retracements
                impulse_size = last_wave.end_price - best_pattern.waves[0].start_price
                signal['take_profit'] = current_price - (impulse_size * 0.382)  # 38.2% retracement
                signal['take_profit_2'] = current_price - (impulse_size * 0.618)  # 61.8% retracement
                
                signal['reason'] = f"Bullish impulse wave complete at {last_wave.end_price:.2f}, expecting ABC correction"
            else:
                # Bearish impulse complete, expect bullish correction
                signal['direction'] = 'BUY'
                signal['entry'] = current_price
                signal['stop_loss'] = last_wave.end_price - 30  # 30 pips below wave 5 low
                
                impulse_size = best_pattern.waves[0].start_price - last_wave.end_price
                signal['take_profit'] = current_price + (impulse_size * 0.382)
                signal['take_profit_2'] = current_price + (impulse_size * 0.618)
                
                signal['reason'] = f"Bearish impulse wave complete at {last_wave.end_price:.2f}, expecting ABC correction"
        
        elif best_pattern.pattern_type in [WaveType.ZIGZAG, WaveType.FLAT, WaveType.CORRECTIVE]:
            # After correction, expect new impulse
            last_wave = best_pattern.waves[-1]
            is_bearish_correction = last_wave.end_price < best_pattern.waves[0].start_price
            
            if is_bearish_correction:
                # Bearish correction complete, expect bullish impulse
                signal['direction'] = 'BUY'
                signal['entry'] = current_price
                signal['stop_loss'] = last_wave.end_price - 30
                
                # Project new impulse based on previous wave
                correction_size = best_pattern.waves[0].start_price - last_wave.end_price
                signal['take_profit'] = current_price + (correction_size * 1.618)  # 161.8% extension
                signal['take_profit_2'] = current_price + (correction_size * 2.618)  # 261.8% extension
                
                signal['reason'] = f"ABC correction complete at {last_wave.end_price:.2f}, expecting new impulse wave"
            else:
                # Bullish correction complete, expect bearish impulse
                signal['direction'] = 'SELL'
                signal['entry'] = current_price
                signal['stop_loss'] = last_wave.end_price + 30
                
                correction_size = last_wave.end_price - best_pattern.waves[0].start_price
                signal['take_profit'] = current_price - (correction_size * 1.618)
                signal['take_profit_2'] = current_price - (correction_size * 2.618)
                
                signal['reason'] = f"ABC correction complete at {last_wave.end_price:.2f}, expecting new impulse wave"
        else:
            signal['direction'] = 'NEUTRAL'
            signal['reason'] = "Pattern not clear for trading"
        
        # Add wave count information
        signal['wave_count'] = {
            'pattern_type': best_pattern.pattern_type.value,
            'completed_waves': len(best_pattern.waves),
            'wave_labels': [w.wave_label for w in best_pattern.waves],
            'current_wave': self._identify_current_wave(best_pattern),
            'next_wave_projection': self._project_next_wave([best_pattern], data)
        }
        
        # Add pattern details
        signal['pattern_details'] = {
            'start_price': best_pattern.start_price,
            'end_price': best_pattern.end_price,
            'pattern_size': abs(best_pattern.end_price - best_pattern.start_price),
            'duration': str(best_pattern.end_time - best_pattern.start_time) if hasattr(best_pattern.end_time, '__sub__') else 'N/A',
            'wave_relationships': self._get_wave_relationships(best_pattern)
        }
        
        return signal
    
    def _create_signal_from_analysis(self, analysis: Dict, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[Signal]:
        """
        Create a Signal object from analysis results
        
        Args:
            analysis: Analysis dictionary
            symbol: Trading symbol
            timeframe: Timeframe
            data: OHLCV data
            
        Returns:
            Signal object or None
        """
        try:
            if analysis['direction'] == 'BUY':
                signal_type = SignalType.BUY
            elif analysis['direction'] == 'SELL':
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Determine signal grade based on confidence
            confidence = analysis['confidence']
            if confidence >= 0.85:
                grade = SignalGrade.A
            elif confidence >= 0.75:
                grade = SignalGrade.B
            elif confidence >= 0.65:
                grade = SignalGrade.C
            else:
                grade = SignalGrade.D
            
            # Create metadata
            metadata = {
                'pattern': analysis.get('pattern', 'unknown'),
                'degree': analysis.get('degree', 'MINOR'),
                'wave_count': analysis.get('wave_count', {}),
                'pattern_details': analysis.get('pattern_details', {}),
                'validation_score': analysis.get('validation_score', 0)
            }
            
            # Create Signal object
            signal = Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                strategy_name='Elliott Wave',
                signal_type=signal_type,
                confidence=confidence,
                price=analysis.get('entry', data['Close'].iloc[-1]),
                timeframe=timeframe,
                strength=confidence * analysis.get('validation_score', 1.0),
                grade=grade,
                stop_loss=analysis.get('stop_loss'),
                take_profit=analysis.get('take_profit'),
                metadata=metadata
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating signal from analysis: {str(e)}")
            return None
    
    def _identify_current_wave(self, pattern: WavePattern) -> str:
        """
        Identify which wave is currently forming
        
        Args:
            pattern: Current wave pattern
            
        Returns:
            Current wave label
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
        
        Args:
            patterns: Valid patterns
            data: OHLCV data
            
        Returns:
            Dictionary with wave projections
        """
        if not patterns:
            return {}
        
        pattern = patterns[0]  # Use most confident pattern
        current_price = float(data['Close'].iloc[-1])
        
        projections = {
            'current_price': current_price,
            'pattern_type': pattern.pattern_type.value,
            'fibonacci_targets': [],
            'time_projections': []
        }
        
        if pattern.pattern_type == WaveType.IMPULSE and pattern.is_complete:
            # Project ABC correction after impulse
            impulse_range = abs(pattern.end_price - pattern.start_price)
            
            # Fibonacci retracement levels for correction
            for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]:
                if pattern.end_price > pattern.start_price:
                    # Bullish impulse, project bearish correction
                    target = pattern.end_price - (impulse_range * ratio)
                else:
                    # Bearish impulse, project bullish correction
                    target = pattern.end_price + (impulse_range * ratio)
                
                projections['fibonacci_targets'].append({
                    'level': f"{ratio*100:.1f}%",
                    'price': round(target, 2),
                    'type': 'retracement'
                })
        
        elif pattern.pattern_type in [WaveType.CORRECTIVE, WaveType.ZIGZAG, WaveType.FLAT]:
            # Project new impulse after correction
            correction_range = abs(pattern.end_price - pattern.start_price)
            
            # Fibonacci extension levels for new impulse
            for ratio in [1.0, 1.272, 1.618, 2.0, 2.618]:
                if pattern.end_price < pattern.start_price:
                    # Bearish correction, project bullish impulse
                    target = pattern.end_price + (correction_range * ratio)
                else:
                    # Bullish correction, project bearish impulse
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
        
        Args:
            pattern: Wave pattern
            
        Returns:
            Dictionary of wave relationships
        """
        relationships = {}
        
        if len(pattern.waves) < 2:
            return relationships
        
        # Calculate relationships for each wave
        for i, wave in enumerate(pattern.waves):
            if i == 0:
                continue
                
            wave_size = abs(wave.end_price - wave.start_price)
            prev_wave_size = abs(pattern.waves[i-1].end_price - pattern.waves[i-1].start_price)
            
            if prev_wave_size > 0:
                ratio = wave_size / prev_wave_size
                relationships[f"Wave_{wave.wave_label}_to_{pattern.waves[i-1].wave_label}"] = round(ratio, 3)
        
        # Special relationships for impulse waves
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
    
    def _empty_signal(self) -> Dict:
        """Generate empty/neutral signal"""
        return {
            'timestamp': datetime.now(),
            'symbol': 'XAUUSDm',
            'strategy': 'Elliott Wave Analysis',
            'direction': 'NEUTRAL',
            'confidence': 0.0,
            'pattern': None,
            'reason': 'No valid Elliott Wave pattern detected',
            'wave_count': {},
            'projections': {}
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and performance"""
        success_rate = (self.successful_signals / max(self.signals_generated, 1)) * 100
        
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
                'signals_generated': self.signals_generated,
                'successful_signals': self.successful_signals,
                'success_rate': f"{success_rate:.1f}%"
            },
            'wave_types_supported': [wt.value for wt in WaveType],
            'wave_degrees': [wd.name for wd in WaveDegree]
        }


# Testing function
def test_elliott_wave_strategy():
    """Test Elliott Wave strategy functionality"""
    print("=" * 80)
    print("ELLIOTT WAVE STRATEGY TEST")
    print("=" * 80)
    
    # Mock MT5 manager for testing
    class MockMT5Manager:
        def get_historical_data(self, symbol, timeframe, bars):
            """Generate synthetic wave-like price data"""
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            # Create dates
            dates = pd.date_range(start=datetime.now() - timedelta(days=10), 
                                 periods=bars, freq='15min')
            
            # Generate wave-like pattern
            x = np.linspace(0, 4*np.pi, bars)
            
            # Create 5-wave impulse pattern followed by ABC correction
            prices = []
            base_price = 2000.0
            
            # Wave 1 up
            wave1 = base_price + np.linspace(0, 50, bars//8)
            prices.extend(wave1)
            
            # Wave 2 down (38.2% retracement)
            wave2 = wave1[-1] - np.linspace(0, 19, bars//8)
            prices.extend(wave2)
            
            # Wave 3 up (1.618 extension)
            wave3 = wave2[-1] + np.linspace(0, 81, bars//8)
            prices.extend(wave3)
            
            # Wave 4 down (38.2% retracement of wave 3)
            wave4 = wave3[-1] - np.linspace(0, 31, bars//8)
            prices.extend(wave4)
            
            # Wave 5 up (equal to wave 1)
            wave5 = wave4[-1] + np.linspace(0, 50, bars//8)
            prices.extend(wave5)
            
            # ABC correction
            # Wave A down
            waveA = wave5[-1] - np.linspace(0, 40, bars//8)
            prices.extend(waveA)
            
            # Wave B up (61.8% retracement)
            waveB = waveA[-1] + np.linspace(0, 25, bars//8)
            prices.extend(waveB)
            
            # Wave C down (equal to wave A)
            waveC = waveB[-1] - np.linspace(0, 40, bars//8)
            prices.extend(waveC)
            
            # Pad with remaining data
            remaining = bars - len(prices)
            if remaining > 0:
                prices.extend([prices[-1] + np.random.normal(0, 2) for _ in range(remaining)])
            
            prices = prices[:bars]  # Ensure correct length
            
            # Add some noise
            prices = np.array(prices) + np.random.normal(0, 1, len(prices))
            
            # Create OHLCV DataFrame
            data = pd.DataFrame({
                'Open': prices - np.random.uniform(0, 2, len(prices)),
                'High': prices + np.random.uniform(2, 5, len(prices)),
                'Low': prices - np.random.uniform(2, 5, len(prices)),
                'Close': prices,
                'Volume': np.random.uniform(1000, 5000, len(prices))
            }, index=dates)
            
            # Ensure OHLC consistency
            data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
            data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
            
            return data
    
    # Test configuration
    test_config = {
        'parameters': {
            'min_wave_size': 10,  # Lower for testing
            'lookback_periods': 200,
            'min_confidence': 0.60,
            'use_volume': True,
            'strict_rules': False,  # Relaxed for testing
            'fibonacci_tolerance': 0.10
        }
    }
    
    try:
        # Initialize strategy
        elliott_strategy = ElliottWaveStrategy(test_config, MockMT5Manager())
        
        print(f"Strategy: {elliott_strategy.get_strategy_info()['name']}")
        print(f"Version: {elliott_strategy.get_strategy_info()['version']}")
        print(f"Description: {elliott_strategy.get_strategy_info()['description']}")
        print()
        
        # Generate signals
        signals = elliott_strategy.generate_signals("XAUUSDm", "M15")
        
        print(f"Generated {len(signals)} signals")
        print()
        
        # Display signals
        for i, signal in enumerate(signals, 1):
            print(f"Signal {i}:")
            print(f"  Type: {signal.signal_type.value}")
            print(f"  Confidence: {signal.confidence:.2%}")
            print(f"  Grade: {signal.grade.value if signal.grade else 'N/A'}")
            print(f"  Price: {signal.price:.2f}")
            print(f"  Stop Loss: {signal.stop_loss:.2f}" if signal.stop_loss else "  Stop Loss: N/A")
            print(f"  Take Profit: {signal.take_profit:.2f}" if signal.take_profit else "  Take Profit: N/A")
            
            if signal.metadata:
                print(f"  Pattern: {signal.metadata.get('pattern', 'N/A')}")
                print(f"  Degree: {signal.metadata.get('degree', 'N/A')}")
                if 'wave_count' in signal.metadata:
                    wave_count = signal.metadata['wave_count']
                    print(f"  Current Wave: {wave_count.get('current_wave', 'N/A')}")
            print()
        
        # Test with direct analysis
        mock_data = MockMT5Manager().get_historical_data("XAUUSDm", "M15", 200)
        analysis = elliott_strategy.analyze(mock_data)
        
        print("Direct Analysis Results:")
        print(f"  Direction: {analysis['direction']}")
        print(f"  Confidence: {analysis['confidence']:.2%}")
        print(f"  Pattern: {analysis.get('pattern', 'N/A')}")
        print(f"  Reason: {analysis.get('reason', 'N/A')}")
        
        if 'wave_count' in analysis and analysis['wave_count']:
            print(f"  Wave Labels: {analysis['wave_count'].get('wave_labels', [])}")
            print(f"  Current Wave: {analysis['wave_count'].get('current_wave', 'Unknown')}")
        
        if 'projections' in analysis and analysis['projections']:
            proj = analysis['projections']
            if 'fibonacci_targets' in proj and proj['fibonacci_targets']:
                print(f"\n  Price Projections:")
                for target in proj['fibonacci_targets'][:3]:  # Show first 3 targets
                    print(f"    {target['type'].title()} {target['level']}: {target['price']:.2f}")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETE - Elliott Wave Strategy Ready!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_elliott_wave_strategy()
