"""
Base Module - Abstract Base Classes and Core Data Structures
=============================================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-01-15

This module provides the foundation for all trading strategies with:
- Abstract base class for strategies
- Standardized Signal, SignalType, and SignalGrade dataclasses
- Market regime and condition enumerations
- Common interfaces for all strategy implementations
- Backward compatibility with existing strategies

Usage:
    from src.core.base import AbstractStrategy, Signal, SignalType, SignalGrade
    
    class MyStrategy(AbstractStrategy):
        def generate_signal(self, symbol: str, timeframe: str) -> List[Signal]:
            # Implementation here
            pass

Dependencies:
    - pandas
    - numpy
    - datetime
    - abc
    - dataclasses
    - enum
    - typing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging

# Import LoggerManager
try:
    from src.utils.logger import get_logger_manager
except ImportError:
    # Fallback for when running as standalone script
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.utils.logger import get_logger_manager


# ============================================================================
# ENUMERATIONS
# ============================================================================

class SignalType(Enum):
    """Signal type enumeration for trading directions"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def from_string(cls, value: str) -> 'SignalType':
        """Create SignalType from string value"""
        value_upper = value.upper()
        if value_upper in cls._value2member_map_:
            return cls(value_upper)
        raise ValueError(f"Invalid SignalType: {value}")


class SignalGrade(Enum):
    """Signal quality grade enumeration"""
    A = "A"  # Highest quality (85%+ confidence)
    B = "B"  # Good quality (70%+ confidence)
    C = "C"  # Acceptable (60%+ confidence)
    D = "D"  # Poor quality (below 60%)
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def from_confidence(cls, confidence: float) -> 'SignalGrade':
        """Determine grade based on confidence level"""
        if confidence >= 0.85:
            return cls.A
        elif confidence >= 0.70:
            return cls.B
        elif confidence >= 0.60:
            return cls.C
        else:
            return cls.D
    
    def get_min_confidence(self) -> float:
        """Get minimum confidence for this grade"""
        grade_thresholds = {
            self.A: 0.85,
            self.B: 0.70,
            self.C: 0.60,
            self.D: 0.0
        }
        return grade_thresholds[self]


class MarketRegime(Enum):
    """Market regime type enumeration"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNCERTAIN = "UNCERTAIN"
    
    def __str__(self) -> str:
        return self.value
    
    def is_trending(self) -> bool:
        """Check if regime is trending"""
        return self in [self.TRENDING_UP, self.TRENDING_DOWN]
    
    def is_volatile(self) -> bool:
        """Check if regime is volatile"""
        return self == self.HIGH_VOLATILITY


class TradingSession(Enum):
    """Trading session enumeration"""
    ASIAN = "ASIAN"
    EUROPEAN = "EUROPEAN"
    AMERICAN = "AMERICAN"
    OVERLAP_EU_US = "OVERLAP_EU_US"
    OVERLAP_AS_EU = "OVERLAP_AS_EU"
    WEEKEND = "WEEKEND"
    
    @classmethod
    def get_current_session(cls, timestamp: datetime = None) -> 'TradingSession':
        """Get current trading session based on time"""
        if timestamp is None:
            timestamp = datetime.now()
        
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # Weekend check
        if weekday >= 5:  # Saturday or Sunday
            return cls.WEEKEND
        
        # Session times (UTC)
        if 23 <= hour or hour < 8:  # Asian: 23:00 - 08:00 UTC
            return cls.ASIAN
        elif 7 <= hour < 9:  # Asian-European overlap
            return cls.OVERLAP_AS_EU
        elif 8 <= hour < 16:  # European: 08:00 - 16:00 UTC
            return cls.EUROPEAN
        elif 14 <= hour < 17:  # European-American overlap
            return cls.OVERLAP_EU_US
        elif 13 <= hour < 22:  # American: 13:00 - 22:00 UTC
            return cls.AMERICAN
        else:
            return cls.ASIAN


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Signal:
    """
    Trading signal data structure
    
    This is the core signal class used by all strategies.
    It contains all necessary information for trade execution.
    """
    # Required fields
    timestamp: datetime
    symbol: str
    strategy_name: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    price: float
    timeframe: str
    
    # Optional technical indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    atr: Optional[float] = None
    volume: Optional[float] = None
    
    # Signal quality and strength
    strength: float = 0.0  # Signal strength (0.0 to 1.0)
    grade: Optional[SignalGrade] = None
    risk_reward_ratio: Optional[float] = None
    
    # Trade levels
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_price: Optional[float] = None  # Entry price for the signal
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    
    # Additional trade parameters
    position_size: Optional[float] = None
    max_risk_amount: Optional[float] = None
    trailing_stop: Optional[float] = None
    
    # Metadata for additional information
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Auto-calculate grade if not provided
        if self.grade is None:
            self.grade = SignalGrade.from_confidence(self.confidence)
        
        # Ensure metadata is a dict
        if self.metadata is None:
            self.metadata = {}
        
        # Validate confidence range
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Validate strength range
        self.strength = max(0.0, min(1.0, self.strength))
        
        # Store original timestamp if not present
        if 'original_timestamp' not in self.metadata:
            self.metadata['original_timestamp'] = self.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'strategy_name': self.strategy_name,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'price': self.price,
            'timeframe': self.timeframe,
            'strength': self.strength,
            'grade': self.grade.value if self.grade else None,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_price': self.entry_price,
            'entry_zone_low': self.entry_zone_low,
            'entry_zone_high': self.entry_zone_high,
            'risk_reward_ratio': self.risk_reward_ratio,
            'rsi': self.rsi,
            'macd': self.macd,
            'atr': self.atr,
            'volume': self.volume,
            'position_size': self.position_size,
            'max_risk_amount': self.max_risk_amount,
            'trailing_stop': self.trailing_stop,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create Signal from dictionary"""
        # Handle timestamp
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Handle signal_type
        if isinstance(data.get('signal_type'), str):
            data['signal_type'] = SignalType.from_string(data['signal_type'])
        
        # Handle grade
        if isinstance(data.get('grade'), str):
            data['grade'] = SignalGrade(data['grade'])
        
        return cls(**data)
    
    def is_valid(self) -> bool:
        """Check if signal is valid for execution"""
        return (
            self.signal_type != SignalType.HOLD and
            self.confidence > 0.5 and
            self.stop_loss is not None and
            self.take_profit is not None
        )
    
    def calculate_risk_reward(self) -> float:
        """Calculate risk-reward ratio"""
        if self.stop_loss and self.take_profit and self.price:
            risk = abs(self.price - self.stop_loss)
            reward = abs(self.take_profit - self.price)
            if risk > 0:
                self.risk_reward_ratio = reward / risk
                return self.risk_reward_ratio
        return 0.0


@dataclass
class MarketCondition:
    """Current market condition assessment"""
    timestamp: datetime
    regime: MarketRegime
    volatility: float  # ATR or volatility measure
    trend_strength: float  # 0.0 to 1.0
    volume_profile: str  # 'low', 'normal', 'high'
    session: TradingSession
    confidence: float  # Confidence in assessment
    
    # Technical conditions
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    key_levels: List[float] = field(default_factory=list)
    pivot_points: Dict[str, float] = field(default_factory=dict)
    
    # Market metrics
    spread: Optional[float] = None
    liquidity: Optional[str] = None  # 'low', 'medium', 'high'
    momentum: Optional[float] = None  # -1.0 to 1.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.key_levels is None:
            self.key_levels = []
        if self.pivot_points is None:
            self.pivot_points = {}
        if self.metadata is None:
            self.metadata = {}
        
        # Auto-detect session if not provided
        if self.session is None:
            self.session = TradingSession.get_current_session(self.timestamp)
    
    def is_favorable_for_trading(self) -> bool:
        """Check if conditions are favorable for trading"""
        return (
            self.regime != MarketRegime.UNCERTAIN and
            self.confidence >= 0.6 and
            self.session != TradingSession.WEEKEND and
            self.volume_profile != 'low'
        )


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    win_rate: float = 0.0
    average_profit: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0
    
    # Grade distribution
    a_grade_signals: int = 0
    b_grade_signals: int = 0
    c_grade_signals: int = 0
    d_grade_signals: int = 0
    
    # Time metrics
    last_signal_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    
    def update_metrics(self):
        """Update calculated metrics"""
        if self.total_signals > 0:
            self.win_rate = self.successful_signals / self.total_signals
        
        if self.average_profit > 0 and self.average_loss > 0:
            self.profit_factor = abs(self.average_profit / self.average_loss)
        
        self.last_update_time = datetime.now()


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class AbstractStrategy(ABC):
    """
    Abstract base class for all trading strategies
    
    This class defines the interface that all strategies must implement.
    It provides common functionality and ensures consistency across strategies.
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager: Any = None, 
                 database: Any = None):
        """
        Initialize the strategy
        
        Args:
            config: Strategy configuration dictionary
            mt5_manager: MT5 connection manager (optional)
            database: Database manager (optional)
        """
        self.config = config
        self.mt5_manager = mt5_manager
        self.database = database
        
        # Extract common configuration
        self.strategy_name = self.__class__.__name__
        self.enabled = config.get('enabled', True)
        self.min_confidence = config.get('min_confidence', 0.6)
        self.max_signals_per_hour = config.get('max_signals_per_hour', 3)
        
        # Performance tracking
        self.performance = StrategyPerformance(strategy_name=self.strategy_name)
        self.signal_history: List[Signal] = []
        self.last_signal_time: Optional[datetime] = None
        
        # Setup logging with LoggerManager
        try:
            logger_manager = get_logger_manager()
            self.logger = logger_manager.get_logger('strategy')
        except Exception:
            # Fallback to standard logging if LoggerManager fails
            self.logger = logging.getLogger(self.strategy_name)
        
        # Initialize strategy-specific components
        self._initialize()
    
    def _initialize(self):
        """
        Initialize strategy-specific components
        Override in child classes for custom initialization
        """
        pass
    
    @abstractmethod
    def generate_signal(self, symbol: str, timeframe: str) -> List[Signal]:
        """
        Generate trading signals
        
        This is the main method that must be implemented by all strategies.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSDm')
            timeframe: Timeframe for analysis (e.g., 'M15', 'H1')
            
        Returns:
            List of Signal objects
        """
        pass
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Perform detailed analysis without generating signals
        
        This method should return analysis results that can be used
        for visualization or further processing.
        
        Args:
            data: Historical price data
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate a signal before execution
        
        Override in child classes for custom validation logic.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid, False otherwise
        """
        # Basic validation
        if not signal.is_valid():
            self.logger.warning(f"Invalid signal: {signal}")
            return False
        
        # Check confidence threshold
        if signal.confidence < self.min_confidence:
            self.logger.info(f"Signal confidence {signal.confidence} below threshold {self.min_confidence}")
            return False
        
        # Check rate limiting
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for signal generation")
            return False
        
        # Check market conditions if available
        if hasattr(self, 'market_condition') and self.market_condition:
            if not self.market_condition.is_favorable_for_trading():
                self.logger.info("Market conditions not favorable for trading")
                return False
        
        return True
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit for signals has been exceeded"""
        if self.last_signal_time:
            time_since_last = (datetime.now() - self.last_signal_time).total_seconds() / 3600
            if time_since_last < 1.0:  # Within the last hour
                recent_signals = sum(1 for s in self.signal_history 
                                   if (datetime.now() - s.timestamp).total_seconds() < 3600)
                if recent_signals >= self.max_signals_per_hour:
                    return False
        return True
    
    def update_performance(self, signal: Signal, result: str, pnl: float = 0.0):
        """
        Update strategy performance metrics
        
        Args:
            signal: The executed signal
            result: 'success' or 'failure'
            pnl: Profit/loss from the signal
        """
        self.performance.total_signals += 1
        
        if result == 'success':
            self.performance.successful_signals += 1
            if pnl > 0:
                self.performance.average_profit = (
                    (self.performance.average_profit * (self.performance.successful_signals - 1) + pnl) /
                    self.performance.successful_signals
                )
        else:
            self.performance.failed_signals += 1
            if pnl < 0:
                self.performance.average_loss = (
                    (self.performance.average_loss * (self.performance.failed_signals - 1) + abs(pnl)) /
                    self.performance.failed_signals
                )
        
        # Update grade distribution
        if signal.grade == SignalGrade.A:
            self.performance.a_grade_signals += 1
        elif signal.grade == SignalGrade.B:
            self.performance.b_grade_signals += 1
        elif signal.grade == SignalGrade.C:
            self.performance.c_grade_signals += 1
        else:
            self.performance.d_grade_signals += 1
        
        self.performance.total_pnl += pnl
        self.performance.last_signal_time = signal.timestamp
        self.performance.update_metrics()
        
        # Store in history
        self.signal_history.append(signal)
        self.last_signal_time = datetime.now()
        
        # Log performance update
        self.logger.info(f"Performance updated: Win rate: {self.performance.win_rate:.2%}, "
                        f"Total PnL: {self.performance.total_pnl:.2f}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary"""
        return {
            'strategy_name': self.strategy_name,
            'total_signals': self.performance.total_signals,
            'win_rate': self.performance.win_rate,
            'profit_factor': self.performance.profit_factor,
            'total_pnl': self.performance.total_pnl,
            'grade_distribution': {
                'A': self.performance.a_grade_signals,
                'B': self.performance.b_grade_signals,
                'C': self.performance.c_grade_signals,
                'D': self.performance.d_grade_signals
            },
            'last_signal': self.performance.last_signal_time.isoformat() if self.performance.last_signal_time else None
        }
    
    def reset_performance(self):
        """Reset performance metrics"""
        self.performance = StrategyPerformance(strategy_name=self.strategy_name)
        self.signal_history.clear()
        self.last_signal_time = None
        self.logger.info("Performance metrics reset")
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save strategy state for persistence
        
        Override in child classes to save additional state.
        
        Returns:
            Dictionary containing strategy state
        """
        return {
            'strategy_name': self.strategy_name,
            'enabled': self.enabled,
            'performance': self.performance.__dict__,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'signal_history_count': len(self.signal_history)
        }
    
    def load_state(self, state: Dict[str, Any]):
        """
        Load strategy state from persistence
        
        Override in child classes to load additional state.
        
        Args:
            state: Dictionary containing strategy state
        """
        self.enabled = state.get('enabled', True)
        
        # Load performance metrics
        if 'performance' in state:
            for key, value in state['performance'].items():
                if hasattr(self.performance, key):
                    setattr(self.performance, key, value)
        
        # Load last signal time
        if state.get('last_signal_time'):
            self.last_signal_time = datetime.fromisoformat(state['last_signal_time'])
        
        self.logger.info(f"State loaded for {self.strategy_name}")
    
    def __str__(self) -> str:
        """String representation of the strategy"""
        return f"{self.strategy_name}(enabled={self.enabled}, signals={self.performance.total_signals})"
    
    def __repr__(self) -> str:
        """Detailed representation of the strategy"""
        return (f"{self.__class__.__name__}("
                f"config={self.config}, "
                f"performance={self.performance}, "
                f"has_mt5={self.mt5_manager is not None}, "
                f"has_db={self.database is not None})")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_position_size(signal: Signal, account_balance: float, 
                           risk_percentage: float = 0.02) -> float:
    """
    Calculate position size based on signal and risk parameters
    
    Args:
        signal: Trading signal
        account_balance: Current account balance
        risk_percentage: Risk per trade (default 2%)
        
    Returns:
        Calculated position size
    """
    if not signal.stop_loss or not signal.price:
        return 0.0
    
    risk_amount = account_balance * risk_percentage
    stop_distance = abs(signal.price - signal.stop_loss)
    
    if stop_distance > 0:
        position_size = risk_amount / stop_distance
        return round(position_size, 2)
    
    return 0.0


def merge_signals(signals: List[Signal], weights: Dict[str, float] = None) -> Optional[Signal]:
    """
    Merge multiple signals into a consensus signal
    
    Args:
        signals: List of signals to merge
        weights: Optional dictionary of strategy weights
        
    Returns:
        Merged consensus signal or None if no consensus
    """
    if not signals:
        return None
    
    # Default equal weights if not provided
    if weights is None:
        weights = {signal.strategy_name: 1.0 for signal in signals}
    
    # Calculate weighted consensus
    buy_score = 0.0
    sell_score = 0.0
    total_weight = 0.0
    
    for signal in signals:
        weight = weights.get(signal.strategy_name, 1.0)
        confidence_weighted = signal.confidence * weight
        
        if signal.signal_type == SignalType.BUY:
            buy_score += confidence_weighted
        elif signal.signal_type == SignalType.SELL:
            sell_score += confidence_weighted
        
        total_weight += weight
    
    if total_weight == 0:
        return None
    
    # Normalize scores
    buy_score /= total_weight
    sell_score /= total_weight
    
    # Determine consensus
    if buy_score > sell_score and buy_score > 0.5:
        signal_type = SignalType.BUY
        confidence = buy_score
    elif sell_score > buy_score and sell_score > 0.5:
        signal_type = SignalType.SELL
        confidence = sell_score
    else:
        signal_type = SignalType.HOLD
        confidence = max(buy_score, sell_score)
    
    # Create consensus signal
    consensus = Signal(
        timestamp=datetime.now(),
        symbol=signals[0].symbol,
        strategy_name="Consensus",
        signal_type=signal_type,
        confidence=confidence,
        price=signals[0].price,  # Use first signal's price
        timeframe=signals[0].timeframe,
        strength=confidence,
        metadata={
            'source_signals': len(signals),
            'buy_score': buy_score,
            'sell_score': sell_score,
            'strategies': [s.strategy_name for s in signals]
        }
    )
    
    # Average the stop loss and take profit levels
    valid_stops = [s.stop_loss for s in signals if s.stop_loss]
    valid_targets = [s.take_profit for s in signals if s.take_profit]
    
    if valid_stops:
        consensus.stop_loss = sum(valid_stops) / len(valid_stops)
    if valid_targets:
        consensus.take_profit = sum(valid_targets) / len(valid_targets)
    
    return consensus


# ============================================================================
# EXCEPTIONS
# ============================================================================

class StrategyError(Exception):
    """Base exception for strategy-related errors"""
    pass


class SignalGenerationError(StrategyError):
    """Exception raised when signal generation fails"""
    pass


class DataValidationError(StrategyError):
    """Exception raised when data validation fails"""
    pass


class ConfigurationError(StrategyError):
    """Exception raised for configuration issues"""
    pass


# ============================================================================
# MODULE TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the base module components"""
    
    print("="*60)
    print("BASE MODULE TEST")
    print("="*60)
    
    # Test SignalType
    print("\n1. Testing SignalType:")
    st = SignalType.BUY
    print(f"   SignalType.BUY: {st}")
    print(f"   From string 'sell': {SignalType.from_string('sell')}")
    
    # Test SignalGrade
    print("\n2. Testing SignalGrade:")
    print(f"   Grade for 0.9 confidence: {SignalGrade.from_confidence(0.9)}")
    print(f"   Grade for 0.75 confidence: {SignalGrade.from_confidence(0.75)}")
    print(f"   Grade for 0.65 confidence: {SignalGrade.from_confidence(0.65)}")
    print(f"   Grade for 0.5 confidence: {SignalGrade.from_confidence(0.5)}")
    
    # Test Signal
    print("\n3. Testing Signal:")
    signal = Signal(
        timestamp=datetime.now(),
        symbol="XAUUSDm",
        strategy_name="TestStrategy",
        signal_type=SignalType.BUY,
        confidence=0.85,
        price=1950.50,
        timeframe="M15",
        stop_loss=1945.00,
        take_profit=1960.00
    )
    print(f"   Created signal: {signal.signal_type} at {signal.price}")
    print(f"   Signal grade: {signal.grade}")
    print(f"   Risk/Reward: {signal.calculate_risk_reward():.2f}")
    print(f"   Is valid: {signal.is_valid()}")
    
    # Test MarketCondition
    print("\n4. Testing MarketCondition:")
    condition = MarketCondition(
        timestamp=datetime.now(),
        regime=MarketRegime.TRENDING_UP,
        volatility=15.5,
        trend_strength=0.75,
        volume_profile="normal",
        session=TradingSession.EUROPEAN,
        confidence=0.8
    )
    print(f"   Market regime: {condition.regime}")
    print(f"   Trading session: {condition.session}")
    print(f"   Favorable for trading: {condition.is_favorable_for_trading()}")
    
    # Test signal merging
    print("\n5. Testing Signal Merging:")
    signals = [
        Signal(datetime.now(), "XAUUSDm", "Strategy1", SignalType.BUY, 0.8, 1950, "M15"),
        Signal(datetime.now(), "XAUUSDm", "Strategy2", SignalType.BUY, 0.7, 1950, "M15"),
        Signal(datetime.now(), "XAUUSDm", "Strategy3", SignalType.SELL, 0.6, 1950, "M15"),
    ]
    consensus = merge_signals(signals)
    if consensus:
        print(f"   Consensus: {consensus.signal_type} with confidence {consensus.confidence:.2f}")
        print(f"   Metadata: {consensus.metadata}")
    
    # Test position sizing
    print("\n6. Testing Position Sizing:")
    test_signal = Signal(
        datetime.now(), "XAUUSDm", "Test", SignalType.BUY, 
        0.8, 1950, "M15", stop_loss=1945, take_profit=1960
    )
    size = calculate_position_size(test_signal, account_balance=10000, risk_percentage=0.02)
    print(f"   Position size for $10,000 balance with 2% risk: {size} lots")
    
    # Example of implementing a strategy using AbstractStrategy
    print("\n7. Example Strategy Implementation:")
    
    class ExampleStrategy(AbstractStrategy):
        """Example strategy implementation for testing"""
        
        def generate_signal(self, symbol: str, timeframe: str) -> List[Signal]:
            """Generate example signals"""
            return [
                Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    signal_type=SignalType.BUY,
                    confidence=0.75,
                    price=1950.00,
                    timeframe=timeframe,
                    stop_loss=1945.00,
                    take_profit=1960.00,
                    metadata={'test': True}
                )
            ]
        
        def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
            """Perform example analysis"""
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'data_points': len(data) if data is not None else 0,
                'analysis_time': datetime.now().isoformat(),
                'indicators': {
                    'trend': 'bullish',
                    'momentum': 'positive',
                    'volume': 'average'
                }
            }
    
    # Test the example strategy
    config = {'enabled': True, 'min_confidence': 0.6}
    strategy = ExampleStrategy(config)
    test_signals = strategy.generate_signal("XAUUSDm", "M15")
    print(f"   Generated {len(test_signals)} signals")
    print(f"   Strategy summary: {strategy}")
    
    print("\n" + "="*60)
    print("BASE MODULE TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
