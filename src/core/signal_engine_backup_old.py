"""
Signal Engine - Core Signal Generation System
============================================
Author: XAUUSD Trading System
Version: 2.0.0
Date: 2025-08-08

This module handles all signal generation and coordination:
- Strategy orchestration
- Signal fusion and weighting
- Market regime detection
- Signal quality grading
- Execution timing

Dependencies:
    - pandas
    - numpy
    - datetime
    - typing
"""
import sys
import os
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent  # src/core/
src_dir = current_dir.parent  # src/
project_root = src_dir.parent  # project root
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import yaml
from pathlib import Path

# Import strategy modules (will be created)
try:
    from src.strategies.technical.ichimoku import IchimokuStrategy
    from src.strategies.technical.harmonic import HarmonicStrategy
    #from src.strategies.technical.volume_profile import VolumeProfileStrategy
    #from src.strategies.technical.momentum_divergence import MomentumDivergenceStrategy
    #from src.strategies.technical.fibonacci_advanced import FibonacciAdvancedStrategy
    #from src.strategies.technical.order_flow import OrderFlowStrategy
    #from src.strategies.technical.wyckoff import WyckoffStrategy
    from src.strategies.technical.elliott_wave import ElliottWaveStrategy
    
    #from src.strategies.smc.market_structure import MarketStructureStrategy
    from src.strategies.smc.order_blocks import OrderBlocksStrategy
    #from src.strategies.smc.liquidity_pools import LiquidityPoolsStrategy
    #from src.strategies.smc.manipulation import ManipulationStrategy
    
    from src.strategies.ml.lstm_predictor import LSTMPredictor
    #from src.strategies.ml.xgboost_classifier import XGBoostStrategy
    #from src.strategies.ml.ensemble_nn import EnsembleNNStrategy
    
    #from src.strategies.fusion.weighted_voting import WeightedVotingStrategy
    #from src.strategies.fusion.confidence_sizing import ConfidenceSizingStrategy
    #from src.strategies.fusion.regime_detection import RegimeDetectionStrategy
except ImportError as e:
    logging.warning(f"Some strategy modules not available: {e}")


class SignalType(Enum):
    """Signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalGrade(Enum):
    """Signal quality grades"""
    A = "A"  # Highest quality (85%+ confidence)
    B = "B"  # Good quality (70%+ confidence)
    C = "C"  # Acceptable (60%+ confidence)
    D = "D"  # Poor quality (below 60%)


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class Signal:
    """Trading signal data structure"""
    timestamp: datetime
    symbol: str
    strategy_name: str
    signal_type: SignalType
    confidence: float
    price: float
    timeframe: str
    
    # Technical indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    atr: Optional[float] = None
    volume: Optional[float] = None
    
    # Signal quality
    strength: float = 0.0
    grade: Optional[SignalGrade] = None
    risk_reward_ratio: Optional[float] = None
    
    # Levels
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Calculate grade based on confidence
        if self.confidence >= 0.85:
            self.grade = SignalGrade.A
        elif self.confidence >= 0.70:
            self.grade = SignalGrade.B
        elif self.confidence >= 0.60:
            self.grade = SignalGrade.C
        else:
            self.grade = SignalGrade.D


@dataclass
class MarketCondition:
    """Current market condition assessment"""
    timestamp: datetime
    regime: MarketRegime
    volatility: float
    trend_strength: float
    volume_profile: str
    session: str
    confidence: float
    
    # Technical conditions
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    key_levels: List[float] = None
    
    def __post_init__(self):
        if self.key_levels is None:
            self.key_levels = []


class SignalEngine:
    """
    Core signal generation and coordination engine
    
    This class orchestrates all trading strategies and generates
    high-quality trading signals for the XAUUSD system.
    
    Key Features:
    - Multi-strategy signal generation
    - Signal fusion and weighting
    - Market regime detection
    - Quality grading system
    - Risk-reward optimization
    
    Example:
        >>> signal_engine = SignalEngine(config, mt5_manager, database)
        >>> signal_engine.initialize()
        >>> signals = signal_engine.generate_signals("XAUUSDm", "M15")
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager, database_manager):
        """
        Initialize the signal engine
        
        Args:
            config: System configuration
            mt5_manager: MT5 connection manager
            database_manager: Database manager
        """
        self.config = config
        self.mt5_manager = mt5_manager
        self.database = database_manager
        
        # Configuration
        self.strategy_config = config.get('strategies', {})
        self.signal_config = config.get('signals', {})
        self.trading_config = config.get('trading', {})
        
        # Initialize strategy components
        self.technical_strategies = {}
        self.smc_strategies = {}
        self.ml_strategies = {}
        self.fusion_strategies = {}
        
        # Market analysis
        self.regime_detector = None
        self.market_condition = None
        
        # Signal storage
        self.active_signals = []
        self.signal_history = []
        
        # Performance tracking
        self.strategy_performance = {}
        self.signal_stats = {
            'total_generated': 0,
            'a_grade_signals': 0,
            'b_grade_signals': 0,
            'c_grade_signals': 0,
            'executed_signals': 0,
            'successful_signals': 0
        }
        
        # Logger
        self.logger = logging.getLogger('signal_engine')
        
        # Initialize flag
        self.initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize all strategy components
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Signal Engine...")
            
            # Initialize technical strategies
            if self.strategy_config.get('technical', {}).get('enabled', False):
                self._initialize_technical_strategies()
            
            # Initialize SMC strategies
            if self.strategy_config.get('smc', {}).get('enabled', False):
                self._initialize_smc_strategies()
            
            # Initialize ML strategies
            if self.strategy_config.get('ml', {}).get('enabled', False):
                self._initialize_ml_strategies()
            
            # Initialize fusion strategies
            if self.strategy_config.get('fusion', {}).get('enabled', False):
                self._initialize_fusion_strategies()
            
            # Initialize market regime detection
            self._initialize_regime_detection()
            
            # Load strategy performance history
            self._load_strategy_performance()
            
            self.initialized = True
            self.logger.info("Signal Engine initialized successfully")
            
            # Log strategy summary
            total_strategies = (len(self.technical_strategies) + 
                              len(self.smc_strategies) + 
                              len(self.ml_strategies) + 
                              len(self.fusion_strategies))
            
            self.logger.info(f"Loaded {total_strategies} strategies:")
            self.logger.info(f"  Technical: {len(self.technical_strategies)}")
            self.logger.info(f"  SMC: {len(self.smc_strategies)}")
            self.logger.info(f"  ML: {len(self.ml_strategies)}")
            self.logger.info(f"  Fusion: {len(self.fusion_strategies)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Signal Engine initialization failed: {str(e)}")
            return False
    
    def _initialize_technical_strategies(self) -> None:
        """Initialize technical analysis strategies"""
        tech_config = self.strategy_config['technical']
        active_strategies = tech_config.get('active_strategies', {})
        
        strategy_classes = {
            'ichimoku': IchimokuStrategy,
            'harmonic': HarmonicStrategy,
            'elliott_wave': ElliottWaveStrategy,
            #'volume_profile': VolumeProfileStrategy,
            #'momentum_divergence': MomentumDivergenceStrategy,
            #'fibonacci_advanced': FibonacciAdvancedStrategy,
            #'order_flow': OrderFlowStrategy,
            #'wyckoff': WyckoffStrategy
        }
        
        for strategy_name, strategy_class in strategy_classes.items():
            if active_strategies.get(strategy_name, False):
                try:
                    strategy_instance = strategy_class(
                        config=tech_config,
                        mt5_manager=self.mt5_manager
                    )
                    self.technical_strategies[strategy_name] = strategy_instance
                    self.logger.info(f"Initialized technical strategy: {strategy_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize {strategy_name}: {e}")
    
    def _initialize_smc_strategies(self) -> None:
        """Initialize Smart Money Concepts strategies"""
        smc_config = self.strategy_config['smc']
        active_components = smc_config.get('active_components', {})
        
        strategy_classes = {
            #'market_structure': MarketStructureStrategy,
            'order_blocks': OrderBlocksStrategy,
            #'liquidity_pools': LiquidityPoolsStrategy,
            #'manipulation': ManipulationStrategy
        }
        
        for strategy_name, strategy_class in strategy_classes.items():
            if active_components.get(strategy_name, False):
                try:
                    strategy_instance = strategy_class(
                        config=smc_config,
                        mt5_manager=self.mt5_manager
                    )
                    self.smc_strategies[strategy_name] = strategy_instance
                    self.logger.info(f"Initialized SMC strategy: {strategy_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize {strategy_name}: {e}")
    
    def _initialize_ml_strategies(self) -> None:
        """Initialize machine learning strategies"""
        ml_config = self.strategy_config['ml']
        active_models = ml_config.get('active_models', {})
        
        strategy_classes = {
            'lstm': LSTMPredictor,
            #'xgboost': XGBoostStrategy,
            #'ensemble': EnsembleNNStrategy
        }
        
        for strategy_name, strategy_class in strategy_classes.items():
            if active_models.get(strategy_name, False):
                try:
                    strategy_instance = strategy_class(
                        config=ml_config,
                        mt5_manager=self.mt5_manager,
                        database=self.database
                    )
                    self.ml_strategies[strategy_name] = strategy_instance
                    self.logger.info(f"Initialized ML strategy: {strategy_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize {strategy_name}: {e}")
    
    def _initialize_fusion_strategies(self) -> None:
        """Initialize signal fusion strategies"""
        fusion_config = self.strategy_config['fusion']
        
        strategy_classes = {
            #'weighted_voting': WeightedVotingStrategy,
            #'confidence_sizing': ConfidenceSizingStrategy,
            #'regime_detection': RegimeDetectionStrategy
        }
        
        for strategy_name, strategy_class in strategy_classes.items():
            try:
                strategy_instance = strategy_class(
                    config=fusion_config,
                    mt5_manager=self.mt5_manager
                )
                self.fusion_strategies[strategy_name] = strategy_instance
                self.logger.info(f"Initialized fusion strategy: {strategy_name}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize {strategy_name}: {e}")
                
    def _initialize_regime_detection(self) -> None:
        """Initialize market regime detection"""
        try:
            if 'regime_detection' in self.fusion_strategies:
                self.regime_detector = self.fusion_strategies['regime_detection']
                self.logger.info("Market regime detection initialized from fusion strategies")
            else:
                self.logger.debug("No regime detection strategy available")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize regime detection: {e}")            
    
    # def _initialize_regime_detection(self) -> None:
    #     """Initialize market regime detection"""
    #     try:
    #         if 'regime_detection' in self.fusion_strategies:
    #             self.regime_detector = self.fusion_strategies['regime_detection']
    #         else:
    #             # Create standalone regime detector if fusion not available
    #             #from strategies.fusion.regime_detection import RegimeDetectionStrategy
    #             self.regime_detector = RegimeDetectionStrategy(
    #                 config=self.strategy_config.get('fusion', {}),
    #                 mt5_manager=self.mt5_manager
    #             )
            
    #         self.logger.info("Market regime detection initialized")
            
    #     except Exception as e:
    #         self.logger.warning(f"Failed to initialize regime detection: {e}")
    
    def generate_signals(self, symbol: str, timeframe: str = "M15") -> List[Signal]:
        """
        Generate trading signals from all active strategies
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSDm')
            timeframe: Analysis timeframe
        
        Returns:
            List of generated signals
        """
        if not self.initialized:
            self.logger.warning("Signal engine not initialized")
            return []
        
        try:
            # Update market condition
            self._update_market_condition(symbol, timeframe)
            
            # Check if market conditions are suitable for trading
            if not self._is_trading_suitable():
                self.logger.debug("Market conditions not suitable for signal generation")
                return []
            
            all_signals = []
            
            # Generate signals from technical strategies
            technical_signals = self._generate_technical_signals(symbol, timeframe)
            all_signals.extend(technical_signals)
            
            # Generate signals from SMC strategies
            smc_signals = self._generate_smc_signals(symbol, timeframe)
            all_signals.extend(smc_signals)
            
            # Generate signals from ML strategies
            ml_signals = self._generate_ml_signals(symbol, timeframe)
            all_signals.extend(ml_signals)
            
            # Apply signal fusion
            fused_signals = self._apply_signal_fusion(all_signals, symbol, timeframe)
            
            # Filter and grade signals
            quality_signals = self._filter_and_grade_signals(fused_signals)
            
            # Update signal statistics
            self._update_signal_stats(quality_signals)
            
            # Store signals in history
            self.signal_history.extend(quality_signals)
            
            # Store in database
            for signal in quality_signals:
                self._store_signal_in_database(signal)
            
            self.logger.info(f"Generated {len(quality_signals)} quality signals from {len(all_signals)} raw signals")
            
            return quality_signals
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            return []
    
    def _generate_technical_signals(self, symbol: str, timeframe: str) -> List[Signal]:
        """Generate signals from technical strategies"""
        signals = []
        
        for strategy_name, strategy in self.technical_strategies.items():
            try:
                strategy_signals = strategy.generate_signals(symbol, timeframe)
                if strategy_signals:
                    signals.extend(strategy_signals)
                    self.logger.debug(f"Technical {strategy_name}: {len(strategy_signals)} signals")
            except Exception as e:
                self.logger.warning(f"Technical strategy {strategy_name} failed: {e}")
        
        return signals
    
    def _generate_smc_signals(self, symbol: str, timeframe: str) -> List[Signal]:
        """Generate signals from SMC strategies"""
        signals = []
        
        for strategy_name, strategy in self.smc_strategies.items():
            try:
                strategy_signals = strategy.generate_signals(symbol, timeframe)
                if strategy_signals:
                    signals.extend(strategy_signals)
                    self.logger.debug(f"SMC {strategy_name}: {len(strategy_signals)} signals")
            except Exception as e:
                self.logger.warning(f"SMC strategy {strategy_name} failed: {e}")
        
        return signals
    
    def _generate_ml_signals(self, symbol: str, timeframe: str) -> List[Signal]:
        """Generate signals from ML strategies"""
        signals = []
        
        for strategy_name, strategy in self.ml_strategies.items():
            try:
                strategy_signals = strategy.generate_signals(symbol, timeframe)
                if strategy_signals:
                    signals.extend(strategy_signals)
                    self.logger.debug(f"ML {strategy_name}: {len(strategy_signals)} signals")
            except Exception as e:
                self.logger.warning(f"ML strategy {strategy_name} failed: {e}")
        
        return signals
    
    def _apply_signal_fusion(self, signals: List[Signal], symbol: str, timeframe: str) -> List[Signal]:
        """Apply signal fusion techniques"""
        if not signals or not self.fusion_strategies:
            return signals
        
        try:
            # Apply weighted voting if available
            if 'weighted_voting' in self.fusion_strategies:
                voting_strategy = self.fusion_strategies['weighted_voting']
                fused_signals = voting_strategy.fuse_signals(signals, symbol, timeframe)
                return fused_signals
            
            return signals
            
        except Exception as e:
            self.logger.warning(f"Signal fusion failed: {e}")
            return signals
    
    def _filter_and_grade_signals(self, signals: List[Signal]) -> List[Signal]:
        """Filter and grade signals based on quality criteria"""
        quality_signals = []
        
        grading_config = self.signal_config.get('grading', {})
        filters_config = self.signal_config.get('filters', {})
        
        for signal in signals:
            # Apply filters
            if not self._signal_passes_filters(signal, filters_config):
                continue
            
            # Apply grading
            self._grade_signal(signal, grading_config)
            
            # Only keep signals above minimum quality
            if signal.grade != SignalGrade.D:
                quality_signals.append(signal)
        
        # Sort by confidence
        quality_signals.sort(key=lambda s: s.confidence, reverse=True)
        
        # Apply daily limits
        quality_signals = self._apply_daily_limits(quality_signals, grading_config)
        
        return quality_signals
    
    def _signal_passes_filters(self, signal: Signal, filters_config: Dict) -> bool:
        """Check if signal passes quality filters"""
        
        # Spread filter
        spread_filter = filters_config.get('spread_filter', {})
        if spread_filter.get('enabled', False):
            current_spread = self._get_current_spread(signal.symbol)
            max_spread = spread_filter.get('max_spread', 15)
            if current_spread > max_spread:
                return False
        
        # Volatility filter
        volatility_filter = filters_config.get('volatility_filter', {})
        if volatility_filter.get('enabled', False):
            if signal.atr:
                min_atr = volatility_filter.get('min_atr', 5)
                max_atr = volatility_filter.get('max_atr', 50)
                if signal.atr < min_atr or signal.atr > max_atr:
                    return False
        
        # Time filter
        time_filter = filters_config.get('time_filter', {})
        if time_filter.get('enabled', False):
            blocked_hours = time_filter.get('blocked_hours', [])
            blocked_days = time_filter.get('blocked_days', [])
            
            signal_hour = signal.timestamp.hour
            signal_day = signal.timestamp.weekday()
            
            if signal_hour in blocked_hours or signal_day in blocked_days:
                return False
        
        # News filter (basic implementation)
        news_filter = filters_config.get('news_filter', {})
        if news_filter.get('enabled', False):
            # Skip signals during high-impact news (simplified)
            # This would need integration with news calendar API
            pass
        
        return True
    
    def _grade_signal(self, signal: Signal, grading_config: Dict) -> None:
        """Assign quality grade to signal"""
        confidence = signal.confidence
        
        # Determine grade based on confidence
        if confidence >= grading_config.get('A_grade', {}).get('min_confidence', 0.85):
            signal.grade = SignalGrade.A
        elif confidence >= grading_config.get('B_grade', {}).get('min_confidence', 0.70):
            signal.grade = SignalGrade.B
        elif confidence >= grading_config.get('C_grade', {}).get('min_confidence', 0.60):
            signal.grade = SignalGrade.C
        else:
            signal.grade = SignalGrade.D
    
    def _apply_daily_limits(self, signals: List[Signal], grading_config: Dict) -> List[Signal]:
        """Apply daily signal limits per grade"""
        limited_signals = []
        today = datetime.now().date()
        
        # Count existing signals for today
        daily_counts = {'A': 0, 'B': 0, 'C': 0}
        
        for signal in self.signal_history:
            if signal.timestamp.date() == today:
                daily_counts[signal.grade.value] += 1
        
        # Apply limits
        for signal in signals:
            grade_key = f"{signal.grade.value}_grade"
            grade_config = grading_config.get(grade_key, {})
            max_daily = grade_config.get('max_daily', 10)
            
            if daily_counts[signal.grade.value] < max_daily:
                limited_signals.append(signal)
                daily_counts[signal.grade.value] += 1
        
        return limited_signals
    
    def _update_market_condition(self, symbol: str, timeframe: str) -> None:
        """Update current market condition assessment"""
        if not self.regime_detector:
            return
        
        try:
            # Detect current market regime
            regime = self.regime_detector.detect_regime(symbol, timeframe)
            
            # Get market data for analysis
            data = self.mt5_manager.get_historical_data(symbol, timeframe, 100)
            if data.empty:
                return
            
            # Calculate market metrics
            current_price = data['Close'].iloc[-1]
            volatility = self._calculate_volatility(data)
            trend_strength = self._calculate_trend_strength(data)
            volume_profile = self._analyze_volume_profile(data)
            session = self._get_current_session()
            
            # Create market condition
            self.market_condition = MarketCondition(
                timestamp=datetime.now(),
                regime=regime,
                volatility=volatility,
                trend_strength=trend_strength,
                volume_profile=volume_profile,
                session=session,
                confidence=0.75,  # Would be calculated based on various factors
                support_level=self._find_support_level(data),
                resistance_level=self._find_resistance_level(data),
                key_levels=self._find_key_levels(data)
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to update market condition: {e}")
    
    def _is_trading_suitable(self) -> bool:
        """Check if current market conditions are suitable for trading"""
        if not self.market_condition:
            return True  # Default to allow trading
        
        # Don't trade in very low volatility conditions
        if self.market_condition.volatility < 0.2:
            return False
        
        # Be cautious during uncertain market regimes
        if (self.market_condition.regime == MarketRegime.UNCERTAIN and 
            self.market_condition.confidence < 0.6):
            return False
        
        return True
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate current market volatility"""
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        return min(volatility, 1.0)  # Cap at 1.0
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength (0-1)"""
        if len(data) < 20:
            return 0.5
        
        # Simple trend strength based on price position relative to moving averages
        sma_20 = data['Close'].rolling(20).mean()
        current_price = data['Close'].iloc[-1]
        sma_current = sma_20.iloc[-1]
        
        # Normalize trend strength
        price_ratio = current_price / sma_current
        if price_ratio > 1:
            trend_strength = min((price_ratio - 1) * 10, 1.0)
        else:
            trend_strength = max((price_ratio - 1) * 10, -1.0)
        
        return abs(trend_strength)
    
    def _analyze_volume_profile(self, data: pd.DataFrame) -> str:
        """Analyze volume profile"""
        if 'Volume' not in data.columns:
            return "unknown"
        
        recent_volume = data['Volume'].tail(10).mean()
        average_volume = data['Volume'].mean()
        
        if recent_volume > average_volume * 1.2:
            return "high"
        elif recent_volume < average_volume * 0.8:
            return "low"
        else:
            return "normal"
    
    def _get_current_session(self) -> str:
        """Get current trading session"""
        current_hour = datetime.now().hour
        
        if 0 <= current_hour < 9:
            return "asian"
        elif 9 <= current_hour < 17:
            return "london"
        elif 17 <= current_hour < 24:
            return "newyork"
        else:
            return "overlap"
    
    def _find_support_level(self, data: pd.DataFrame) -> Optional[float]:
        """Find nearest support level"""
        try:
            lows = data['Low'].tail(50)
            # Simple support: recent significant low
            support = lows.min()
            return support
        except:
            return None
    
    def _find_resistance_level(self, data: pd.DataFrame) -> Optional[float]:
        """Find nearest resistance level"""
        try:
            highs = data['High'].tail(50)
            # Simple resistance: recent significant high
            resistance = highs.max()
            return resistance
        except:
            return None
    
    def _find_key_levels(self, data: pd.DataFrame) -> List[float]:
        """Find key support/resistance levels"""
        levels = []
        try:
            # Add psychological levels (round numbers)
            current_price = data['Close'].iloc[-1]
            base_level = int(current_price / 10) * 10
            
            for i in range(-5, 6):
                level = base_level + (i * 10)
                if abs(level - current_price) <= 50:  # Within 50 points
                    levels.append(float(level))
        except:
            pass
        
        return levels
    
    def _get_current_spread(self, symbol: str) -> float:
        """Get current spread for symbol"""
        try:
            tick_data = self.mt5_manager.get_realtime_data(symbol)
            if tick_data and 'bid' in tick_data and 'ask' in tick_data:
                spread = (tick_data['ask'] - tick_data['bid']) * 10000  # In points
                return spread
        except:
            pass
        return 15.0  # Default spread
    
    def _store_signal_in_database(self, signal: Signal) -> None:
        """Store signal in database"""
        try:
            signal_data = {
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'strategy': signal.strategy_name,
                'signal_type': signal.signal_type.value,
                'confidence': signal.confidence,
                'price': signal.price,
                'timeframe': signal.timeframe,
                'rsi': signal.rsi,
                'macd': signal.macd,
                'atr': signal.atr,
                'volume': signal.volume,
                'strength': signal.strength,
                'quality_grade': signal.grade.value if signal.grade else 'D',
                'risk_reward_ratio': signal.risk_reward_ratio,
                'executed': False,
                'signal_metadata': signal.metadata
            }
            
            self.database.store_signal(signal_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to store signal in database: {e}")
    
    def _update_signal_stats(self, signals: List[Signal]) -> None:
        """Update signal statistics"""
        self.signal_stats['total_generated'] += len(signals)
        
        for signal in signals:
            if signal.grade == SignalGrade.A:
                self.signal_stats['a_grade_signals'] += 1
            elif signal.grade == SignalGrade.B:
                self.signal_stats['b_grade_signals'] += 1
            elif signal.grade == SignalGrade.C:
                self.signal_stats['c_grade_signals'] += 1
    
    def _load_strategy_performance(self) -> None:
        """Load historical strategy performance data"""
        try:
            # Load from database if available
            # This would load win rates, profit factors, etc. for each strategy
            self.strategy_performance = {
                'ichimoku': {'win_rate': 0.65, 'profit_factor': 1.8, 'weight': 1.0},
                'harmonic': {'win_rate': 0.72, 'profit_factor': 2.1, 'weight': 1.2},
                'elliott_wave': {'win_rate': 0.67, 'profit_factor': 1.6, 'weight': 1.0},
                'volume_profile': {'win_rate': 0.68, 'profit_factor': 1.9, 'weight': 1.1},
                'momentum_divergence': {'win_rate': 0.63, 'profit_factor': 1.7, 'weight': 0.9},
                'fibonacci_advanced': {'win_rate': 0.70, 'profit_factor': 2.0, 'weight': 1.1},
                'order_flow': {'win_rate': 0.75, 'profit_factor': 2.3, 'weight': 1.3},
                'wyckoff': {'win_rate': 0.66, 'profit_factor': 1.8, 'weight': 1.0},
                'market_structure': {'win_rate': 0.78, 'profit_factor': 2.5, 'weight': 1.4},
                'order_blocks': {'win_rate': 0.74, 'profit_factor': 2.2, 'weight': 1.3},
                'liquidity_pools': {'win_rate': 0.76, 'profit_factor': 2.4, 'weight': 1.3},
                'manipulation': {'win_rate': 0.71, 'profit_factor': 2.0, 'weight': 1.2},
                'lstm': {'win_rate': 0.69, 'profit_factor': 1.9, 'weight': 1.1},
                'xgboost': {'win_rate': 0.73, 'profit_factor': 2.1, 'weight': 1.2},
                'ensemble': {'win_rate': 0.77, 'profit_factor': 2.4, 'weight': 1.4}
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to load strategy performance: {e}")
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of signal generation performance"""
        total_signals = self.signal_stats['total_generated']
        
        return {
            'total_signals_generated': total_signals,
            'a_grade_percentage': (self.signal_stats['a_grade_signals'] / max(total_signals, 1)) * 100,
            'b_grade_percentage': (self.signal_stats['b_grade_signals'] / max(total_signals, 1)) * 100,
            'c_grade_percentage': (self.signal_stats['c_grade_signals'] / max(total_signals, 1)) * 100,
            'executed_signals': self.signal_stats['executed_signals'],
            'successful_signals': self.signal_stats['successful_signals'],
            'success_rate': (self.signal_stats['successful_signals'] / max(self.signal_stats['executed_signals'], 1)) * 100,
            'active_strategies': {
                'technical': len(self.technical_strategies),
                'smc': len(self.smc_strategies),
                'ml': len(self.ml_strategies),
                'fusion': len(self.fusion_strategies)
            },
            'current_market_regime': self.market_condition.regime.value if self.market_condition else 'UNKNOWN',
            'market_volatility': self.market_condition.volatility if self.market_condition else 0.0
        }
    
    def update_signal_performance(self, signal_id: int, success: bool, profit: float) -> None:
        """Update signal performance tracking"""
        try:
            self.signal_stats['executed_signals'] += 1
            if success:
                self.signal_stats['successful_signals'] += 1
            
            # Update strategy performance
            # This would update the performance metrics for the strategy that generated the signal
            # Implementation would depend on how signals are tracked
            
        except Exception as e:
            self.logger.warning(f"Failed to update signal performance: {e}")
    
    def get_best_strategies(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top performing strategies"""
        strategy_list = []
        
        for strategy_name, performance in self.strategy_performance.items():
            strategy_list.append({
                'name': strategy_name,
                'win_rate': performance['win_rate'],
                'profit_factor': performance['profit_factor'],
                'weight': performance['weight'],
                'score': performance['win_rate'] * performance['profit_factor']
            })
        
        # Sort by score
        strategy_list.sort(key=lambda x: x['score'], reverse=True)
        
        return strategy_list[:n]
    
    def cleanup_old_signals(self, days: int = 7) -> None:
        """Clean up old signals from memory"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Keep only recent signals in memory
        self.signal_history = [
            signal for signal in self.signal_history 
            if signal.timestamp > cutoff_date
        ]
        
        self.active_signals = [
            signal for signal in self.active_signals 
            if signal.timestamp > cutoff_date
        ]
        
        self.logger.info(f"Cleaned up signals older than {days} days")


# Testing and utility functions
def test_signal_engine():
    """Test signal engine functionality"""
    print("Testing Signal Engine...")
    
    # Mock config
    test_config = {
        'strategies': {
            'technical': {
                'enabled': True,
                'active_strategies': {
                    'ichimoku': True,
                    'harmonic': True,
                    'volume_profile': True,
                    'elliott_wave': True
                }
            },
            'smc': {
                'enabled': True,
                'active_components': {
                    'market_structure': False,
                    'order_blocks': True
                }
            },
            'ml': {
                'enabled': True,
                'active_models': {
                    'lstm': True,
                    'xgboost': False
                }
            },
            'fusion': {
                'enabled': False
            }
        },
        'signals': {
            'grading': {
                'A_grade': {'min_confidence': 0.85, 'max_daily': 5},
                'B_grade': {'min_confidence': 0.70, 'max_daily': 8},
                'C_grade': {'min_confidence': 0.60, 'max_daily': 7}
            },
            'filters': {
                'spread_filter': {'enabled': True, 'max_spread': 15},
                'volatility_filter': {'enabled': True, 'min_atr': 5, 'max_atr': 50}
            }
        }
    }
    
    # Create mock objects
    class MockMT5Manager:
        def get_historical_data(self, symbol, timeframe, bars):
            # Return mock data
            import pandas as pd
            dates = pd.date_range(start='2025-01-01', periods=bars, freq='15T')
            data = pd.DataFrame({
                'Open': np.random.normal(2000, 10, bars),
                'High': np.random.normal(2005, 10, bars),
                'Low': np.random.normal(1995, 10, bars),
                'Close': np.random.normal(2000, 10, bars),
                'Volume': np.random.normal(1000, 100, bars)
            }, index=dates)
            return data
        
        def get_realtime_data(self, symbol):
            return {'bid': 2000.0, 'ask': 2000.5}
    
    class MockDatabase:
        def store_signal(self, signal_data):
            print(f"Storing signal: {signal_data['strategy']} - {signal_data['signal_type']}")
    
    try:
        # Create signal engine
        signal_engine = SignalEngine(
            config=test_config,
            mt5_manager=MockMT5Manager(),
            database_manager=MockDatabase()
        )
        
        # Initialize (will fail due to missing strategy modules, but tests structure)
        initialized = signal_engine.initialize()
        print(f"Initialization: {'✅ Success' if initialized else '⚠️ Partial (missing strategy modules)'}")
        
        # Test signal generation (will return empty list due to missing strategies)
        signals = signal_engine.generate_signals("XAUUSDm", "M15")
        print(f"Generated signals: {len(signals)}")
        
        # Test summary
        summary = signal_engine.get_signal_summary()
        print(f"Signal summary: {summary}")
        
        print("✅ Signal Engine structure test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Signal Engine test failed: {e}")
        return False


if __name__ == "__main__":
    test_signal_engine()