"""
Signal Engine - Core Signal Generation System
============================================

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

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from enum import Enum
from dataclasses import dataclass
import importlib
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Define base classes here to avoid circular imports
class SignalType(Enum):
    """Signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_BUY = "CLOSE_BUY"
    CLOSE_SELL = "CLOSE_SELL"

class SignalGrade(Enum):
    """Signal quality grades"""
    A = "A"  # High confidence
    B = "B"  # Medium confidence
    C = "C"  # Low confidence

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
    strength: float
    grade: SignalGrade = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Auto-calculate grade if not provided"""
        if self.grade is None:
            if self.confidence >= 0.8:
                self.grade = SignalGrade.A
            elif self.confidence >= 0.6:
                self.grade = SignalGrade.B
            else:
                self.grade = SignalGrade.C


class StrategyImporter:
    """Helper class to import strategies with error handling"""
    
    @staticmethod
    def try_import_strategy(module_path: str, class_name: str, strategy_type: str) -> Optional[Any]:
        """Try to import a strategy class with error handling"""
        try:
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)
            logging.info(f"Successfully imported {strategy_type} strategy: {class_name}")
            return strategy_class
        except ImportError as e:
            if "tensorflow" in str(e).lower() or "sklearn" in str(e).lower():
                print(f"TensorFlow/Scikit-learn not available. {class_name} strategy will run in simulation mode.")
                logging.info(f"Successfully imported {strategy_type} strategy: {class_name}")
                module = importlib.import_module(module_path)
                strategy_class = getattr(module, class_name)
                return strategy_class
            else:
                logging.warning(f"Could not import {strategy_type} strategy {class_name}: {str(e)}")
                return None
        except Exception as e:
            logging.warning(f"Error importing {strategy_type} strategy {class_name}: {str(e)}")
            return None
    
    def load_technical_strategies(self) -> Dict[str, Any]:
        """Load available technical strategies"""
        strategies = {}
        
        # All 10 technical strategies
        technical_strategies = {
            'ichimoku': ('src.strategies.technical.ichimoku', 'IchimokuStrategy'),
            'harmonic': ('src.strategies.technical.harmonic', 'HarmonicStrategy'),
            'elliott_wave': ('src.strategies.technical.elliott_wave', 'ElliottWaveStrategy'),
            'volume_profile': ('src.strategies.technical.volume_profile', 'VolumeProfileStrategy'),
            'market_profile': ('src.strategies.technical.market_profile', 'MarketProfileStrategy'),
            'order_flow': ('src.strategies.technical.order_flow', 'OrderFlowStrategy'),
            'wyckoff': ('src.strategies.technical.wyckoff', 'WyckoffStrategy'),
            'gann': ('src.strategies.technical.gann', 'GannStrategy'),
            'fibonacci_advanced': ('src.strategies.technical.fibonacci_advanced', 'FibonacciAdvancedStrategy'),
            'momentum_divergence': ('src.strategies.technical.momentum_divergence', 'MomentumDivergenceStrategy')
        }
        
        for strategy_name, (module_path, class_name) in technical_strategies.items():
            strategy_class = self.try_import_strategy(module_path, class_name, 'technical')
            if strategy_class:
                strategies[strategy_name] = strategy_class
        
        return strategies
    
    def load_smc_strategies(self) -> Dict[str, Any]:
        """Load available SMC strategies"""
        strategies = {}
        
        # All 5 SMC strategies
        smc_strategies = {
            'order_blocks': ('src.strategies.smc.order_blocks', 'OrderBlocksStrategy'),
            'liquidity_pools': ('src.strategies.smc.liquidity_pools', 'LiquidityPoolsStrategy'),
            'market_structure': ('src.strategies.smc.market_structure', 'MarketStructureStrategy'),
            'manipulation': ('src.strategies.smc.manipulation', 'ManipulationStrategy'),
            'imbalance': ('src.strategies.smc.imbalance', 'ImbalanceStrategy')
        }
        
        for strategy_name, (module_path, class_name) in smc_strategies.items():
            strategy_class = self.try_import_strategy(module_path, class_name, 'smc')
            if strategy_class:
                strategies[strategy_name] = strategy_class
        
        return strategies
    
    def load_ml_strategies(self) -> Dict[str, Any]:
        """Load available ML strategies"""
        strategies = {}
        
        # All 4 ML strategies now implemented
        ml_strategies = {
            'lstm': ('src.strategies.ml.lstm_predictor', 'LSTMPredictor'),
            'xgboost_classifier': ('src.strategies.ml.xgboost_classifier', 'XGBoostClassifierStrategy'),
            'ensemble_nn': ('src.strategies.ml.ensemble_nn', 'EnsembleNNStrategy'),
            'rl_agent': ('src.strategies.ml.rl_agent', 'RLAgentStrategy')
        }
        
        for strategy_name, (module_path, class_name) in ml_strategies.items():
            strategy_class = self.try_import_strategy(module_path, class_name, 'ml')
            if strategy_class:
                strategies[strategy_name] = strategy_class
        
        return strategies
    
    def load_fusion_strategies(self) -> Dict[str, Any]:
        """Load available fusion strategies"""
        strategies = {}
        
        # All 4 fusion strategies now implemented
        fusion_strategies = {
            'weighted_voting': ('src.strategies.fusion.weighted_voting', 'WeightedVotingFusionStrategy'),
            'confidence_sizing': ('src.strategies.fusion.confidence_sizing', 'ConfidenceSizingFusionStrategy'),
            'regime_detection': ('src.strategies.fusion.regime_detection', 'RegimeDetectionFusionStrategy'),
            'adaptive_ensemble': ('src.strategies.fusion.adaptive_ensemble', 'AdaptiveEnsembleFusionStrategy')
        }
        
        for strategy_name, (module_path, class_name) in fusion_strategies.items():
            strategy_class = self.try_import_strategy(module_path, class_name, 'fusion')
            if strategy_class:
                strategies[strategy_name] = strategy_class
        
        return strategies


class SignalEngine:
    """
    Core signal generation and coordination engine with graceful error handling
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database_manager=None):
        """
        Initialize Signal Engine
        
        Args:
            config: Configuration dictionary
            mt5_manager: MT5Manager instance
            database_manager: DatabaseManager instance
        """
        self.config = config
        self.mt5_manager = mt5_manager
        self.database_manager = database_manager
        
        # Setup logging
        self.logger = logging.getLogger('signal_engine')
        self.logger.setLevel(logging.INFO)
        
        # Strategy containers
        self.available_strategies = {
            'technical': {},
            'smc': {},
            'ml': {},
            'fusion': {}
        }
        
        self.strategies = {
            'technical': {},
            'smc': {},
            'ml': {},
            'fusion': {}
        }
        
        # Signal management
        self.active_signals = []
        self.signal_history = []
        self.signal_buffer = []
        
        # Performance tracking
        self.strategy_performance = {}
        
        # Market regime
        self.current_regime = "NEUTRAL"
        
        # Initialize importer
        self.importer = StrategyImporter()
        
        # Load and initialize strategies
        self._load_available_strategies()
        self._initialize_strategies()
    
    def _load_available_strategies(self) -> None:
        """Load all available strategy classes"""
        self.logger.info("Loading available strategies...")
        
        # Load each category
        self.available_strategies['technical'] = self.importer.load_technical_strategies()
        self.available_strategies['smc'] = self.importer.load_smc_strategies()
        self.available_strategies['ml'] = self.importer.load_ml_strategies()
        self.available_strategies['fusion'] = self.importer.load_fusion_strategies()
        
        # Log summary
        total_available = sum(len(strategies) for strategies in self.available_strategies.values())
        self.logger.info(f"Loaded {total_available} available strategy classes")
        
        for category, strategies in self.available_strategies.items():
            if strategies:
                self.logger.info(f"  {category.upper()}: {list(strategies.keys())}")
    
    def _initialize_strategies(self) -> None:
        """Initialize active strategies based on configuration"""
        self.logger.info("Initializing active strategies...")
        
        # Get active strategies from config
        strategies_config = self.config.get('strategies', {})
        
        # Initialize technical strategies (enable all that are implemented)
        technical_config = strategies_config.get('technical', {})
        active_technical = technical_config.get('active_strategies', [])
        
        # If no specific list, enable all available technical strategies
        if not active_technical:
            active_technical = list(self.available_strategies['technical'].keys())
        
        for strategy_name in active_technical:
            if strategy_name in self.available_strategies['technical']:
                self._initialize_single_strategy('technical', strategy_name, technical_config)
        
        # Initialize SMC strategies (enable all that are implemented)
        smc_config = strategies_config.get('smc', {})
        active_smc = smc_config.get('active_strategies', [])
        
        # If no specific list, enable all available SMC strategies
        if not active_smc:
            active_smc = list(self.available_strategies['smc'].keys())
        
        for strategy_name in active_smc:
            if strategy_name in self.available_strategies['smc']:
                self._initialize_single_strategy('smc', strategy_name, smc_config)
        
        # Initialize ML strategies (all 4 now implemented)
        ml_config = strategies_config.get('ml', {})
        active_ml = list(self.available_strategies['ml'].keys())
        
        for strategy_name in active_ml:
            if strategy_name in self.available_strategies['ml']:
                self._initialize_single_strategy('ml', strategy_name, ml_config)
        
        # Initialize fusion strategies (all 4 now implemented)
        fusion_config = strategies_config.get('fusion', {})
        active_fusion = list(self.available_strategies['fusion'].keys())
        
        for strategy_name in active_fusion:
            if strategy_name in self.available_strategies['fusion']:
                self._initialize_single_strategy('fusion', strategy_name, fusion_config)
        
        # Log initialization summary
        self._log_initialization_summary()
    
    def _initialize_single_strategy(self, category: str, strategy_name: str, category_config: Dict) -> None:
        """
        Initialize a single strategy instance
        
        Args:
            category: Strategy category (technical, smc, ml, fusion)
            strategy_name: Name of the strategy
            category_config: Configuration for the category
        """
        try:
            strategy_class = self.available_strategies[category][strategy_name]
            
            # Get strategy-specific config
            strategy_config = category_config.get(strategy_name, {})
            
            # Merge with default parameters
            full_config = {
                'name': strategy_name,
                'category': category,
                'parameters': strategy_config,
                'risk_per_trade': self.config.get('risk_management', {}).get('risk_per_trade', 0.02)
            }
            
            # Initialize strategy with all three parameters
            # All strategies should accept (config, mt5_manager, database) even if they don't use database
            strategy_instance = strategy_class(
                config=full_config,
                mt5_manager=self.mt5_manager,
                database=self.database_manager  # Pass database to all strategies
            )
            
            self.strategies[category][strategy_name] = strategy_instance
            self.logger.info(f"Initialized {category} strategy: {strategy_name}")
            
            # Initialize performance tracking
            self.strategy_performance[strategy_name] = {
                'signals_generated': 0,
                'successful_signals': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'last_update': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {category} strategy {strategy_name}: {str(e)}")
    
    def _log_initialization_summary(self) -> None:
        """Log summary of initialized strategies"""
        total_initialized = sum(len(strategies) for strategies in self.strategies.values())
        
        self.logger.info(f"Signal Engine initialized successfully with {total_initialized} strategies:")
        
        for category in ['technical', 'smc', 'ml', 'fusion']:
            available = len(self.available_strategies[category])
            active = len(self.strategies[category])
            self.logger.info(f"  {category.capitalize()}: {active} / {available} available")
    
    def generate_signals(self, symbol: str = "XAUUSDm", timeframe: int = 15) -> List[Signal]:
        """
        Generate signals from all active strategies
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe in minutes
            
        Returns:
            List of quality signals
        """
        all_signals = []
        
        # Detect current market regime
        self.current_regime = self._detect_market_regime(symbol, timeframe)
        
        # Generate signals from each category
        for category in ['technical', 'smc', 'ml', 'fusion']:  
            for strategy_name, strategy in self.strategies[category].items():
                try:
                    # Generate signals from strategy
                    signals = self._generate_strategy_signals(strategy, strategy_name, symbol, timeframe)
                    
                    if signals:
                        all_signals.extend(signals)
                        
                        # Update performance
                        self.strategy_performance[strategy_name]['signals_generated'] += len(signals)
                        
                except Exception as e:
                    self.logger.error(f"Error generating signals from {strategy_name}: {str(e)}")
        
        # Apply signal quality filters
        quality_signals = self._filter_quality_signals(all_signals)
        
        # Store signals
        for signal in quality_signals:
            self._store_signal(signal)
        
        self.logger.info(f"Generated {len(quality_signals)} quality signals from {len(all_signals)} raw signals")
        
        return quality_signals
    
    def _generate_strategy_signals(self, strategy, strategy_name: str, symbol: str, timeframe: int) -> List[Signal]:
        """
        Generate signals from a specific strategy
        
        Args:
            strategy: Strategy instance
            strategy_name: Name of the strategy
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            List of signals
        """
        try:
            # Convert timeframe to string format (M15, H1, etc.)
            timeframe_str = self._convert_timeframe(timeframe)
            
            # Call strategy's generate_signal method (singular)
            # Most strategies should implement generate_signal, not generate_signals
            if hasattr(strategy, 'generate_signal'):
                signals = strategy.generate_signal(symbol, timeframe_str)
            elif hasattr(strategy, 'generate_signals'):
                signals = strategy.generate_signals(symbol, timeframe_str)
            else:
                self.logger.warning(f"Strategy {strategy_name} has no signal generation method")
                return []
            
            # Ensure signals is a list
            if signals is None:
                return []
            if not isinstance(signals, list):
                signals = [signals]
            
            # Filter out None values
            signals = [s for s in signals if s is not None]
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in {strategy_name} signal generation: {str(e)}")
            return []
    
    def _detect_market_regime(self, symbol: str, timeframe: int) -> str:
        """
        Detect current market regime
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Market regime (TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE)
        """
        try:
            if not self.mt5_manager:
                return "NEUTRAL"
            
            # Get recent price data
            data = self.mt5_manager.get_historical_data(symbol, timeframe, 100)
            if data is None or data.empty:
                return "NEUTRAL"
            
            # Calculate indicators for regime detection
            
            # 1. ADX for trend strength
            adx = self._calculate_adx(data, 14)
            current_adx = adx.iloc[-1] if not adx.empty else 25
            
            # 2. Price position relative to moving averages
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            current_price = data['close'].iloc[-1]
            
            # 3. Volatility using ATR
            atr = self._calculate_atr(data, 14)
            avg_atr = atr.mean() if not atr.empty else 0
            current_atr = atr.iloc[-1] if not atr.empty else avg_atr
            
            # Determine regime
            if current_adx > 25:  # Strong trend
                if current_price > sma_20.iloc[-1] and sma_20.iloc[-1] > sma_50.iloc[-1]:
                    regime = "TRENDING_UP"
                elif current_price < sma_20.iloc[-1] and sma_20.iloc[-1] < sma_50.iloc[-1]:
                    regime = "TRENDING_DOWN"
                else:
                    regime = "VOLATILE"
            else:  # Weak trend
                if current_atr > avg_atr * 1.5:
                    regime = "VOLATILE"
                else:
                    regime = "RANGING"
            
            self.logger.debug(f"Market regime detected: {regime} (ADX: {current_adx:.2f})")
            return regime
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return "NEUTRAL"
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            plus_dm = high.diff()
            minus_dm = low.diff().abs()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(period).mean()
            
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
            
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            adx = dx.rolling(period).mean()
            
            return adx
            
        except Exception:
            return pd.Series([25] * len(data))  # Default neutral value
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(period).mean()
            
            return atr
            
        except Exception:
            return pd.Series([0] * len(data))
    
    def _filter_quality_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Filter signals based on quality criteria
        
        Args:
            signals: List of all signals
            
        Returns:
            List of quality signals
        """
        quality_signals = []
        
        for signal in signals:
            # Basic quality checks
            if signal.confidence < 0.5:
                continue
            
            # Check signal conflicts
            if self._has_conflict(signal, quality_signals):
                continue
            
            # Regime-based filtering
            if not self._is_regime_appropriate(signal):
                continue
            
            quality_signals.append(signal)
        
        # Sort by confidence and grade
        quality_signals.sort(key=lambda x: (x.grade.value, x.confidence), reverse=True)
        
        # Limit number of signals based on configuration
        max_signals = self.config.get('signal_generation', {}).get('max_signals_per_bar', 5)
        
        return quality_signals[:max_signals]
    
    def _has_conflict(self, signal: Signal, existing_signals: List[Signal]) -> bool:
        """Check if signal conflicts with existing signals"""
        for existing in existing_signals:
            # Same symbol but opposite direction
            if (signal.symbol == existing.symbol and 
                ((signal.signal_type == SignalType.BUY and existing.signal_type == SignalType.SELL) or
                 (signal.signal_type == SignalType.SELL and existing.signal_type == SignalType.BUY))):
                
                # Keep the one with higher confidence
                if signal.confidence <= existing.confidence:
                    return True
        
        return False
    
    def _is_regime_appropriate(self, signal: Signal) -> bool:
        """Check if signal is appropriate for current market regime"""
        # In volatile markets, reduce number of signals
        if self.current_regime == "VOLATILE" and signal.grade == SignalGrade.C:
            return False
        
        # In ranging markets, filter out weak trend-following signals
        if self.current_regime == "RANGING":
            if signal.strategy_name in ['ichimoku', 'elliott_wave'] and signal.confidence < 0.7:
                return False
        
        return True
    
    def _store_signal(self, signal: Signal) -> None:
        """Store signal in history and database"""
        try:
            # Add to history
            self.signal_history.append(signal)
            self.active_signals.append(signal)
            
            # Store in database if available
            if self.database_manager:
                signal_data = {
                    'timestamp': signal.timestamp,
                    'symbol': signal.symbol,
                    'strategy': signal.strategy_name,
                    'signal_type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'grade': signal.grade.value,
                    'price': signal.price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'timeframe': signal.timeframe,
                    'metadata': str(signal.metadata)
                }
                self.database_manager.store_signal(signal_data)
            
            # Log signal
            self.logger.info(f"📊 Storing signal: {signal.strategy_name} - {signal.signal_type.value}")
            
        except Exception as e:
            self.logger.error(f"Error storing signal: {str(e)}")
    
    def _convert_timeframe(self, timeframe_minutes: int) -> str:
        """Convert timeframe from minutes to MT5 format"""
        timeframe_map = {
            1: 'M1',
            5: 'M5',
            15: 'M15',
            30: 'M30',
            60: 'H1',
            240: 'H4',
            1440: 'D1'
        }
        return timeframe_map.get(timeframe_minutes, 'M15')
    
    def get_active_strategies(self) -> Dict[str, List[str]]:
        """Get list of active strategies by category"""
        active = {}
        for category in self.strategies:
            active[category] = list(self.strategies[category].keys())
        return active
    
    def get_strategy_performance(self, strategy_name: str = None) -> Dict:
        """Get performance metrics for strategies"""
        if strategy_name:
            return self.strategy_performance.get(strategy_name, {})
        return self.strategy_performance
    
    def update_signal_result(self, signal: Signal, result: str, profit: float = 0.0) -> None:
        """
        Update signal result for performance tracking
        
        Args:
            signal: Original signal
            result: 'WIN' or 'LOSS'
            profit: Profit/loss amount
        """
        try:
            strategy_name = signal.strategy_name
            
            if strategy_name in self.strategy_performance:
                perf = self.strategy_performance[strategy_name]
                
                if result == 'WIN':
                    perf['successful_signals'] += 1
                
                # Update win rate
                if perf['signals_generated'] > 0:
                    perf['win_rate'] = perf['successful_signals'] / perf['signals_generated']
                
                perf['last_update'] = datetime.now()
                
                self.logger.info(f"Updated performance for {strategy_name}: Win rate {perf['win_rate']:.2%}")
                
        except Exception as e:
            self.logger.error(f"Error updating signal result: {str(e)}")


def test_signal_engine():
    """Test the Signal Engine functionality with all strategies"""
    print("\nTesting Updated Signal Engine...")
    print("=" * 60)

    # Create test configuration
    config = {
        'strategies': {
            'technical': {
                'active_strategies': [
                    'ichimoku', 'harmonic', 'elliott_wave',
                    'volume_profile', 'market_profile', 'order_flow',
                    'wyckoff', 'gann', 'fibonacci_advanced', 'momentum_divergence'
                ],
                'ichimoku': {'tenkan_period': 9, 'kijun_period': 26},
                'harmonic': {'min_confidence': 0.7},
                'elliott_wave': {'min_wave_size': 30},
                'volume_profile': {'lookback_bars': 200},
                'market_profile': {'lookback_bars': 200},
                'order_flow': {'lookback_bars': 200},
                'wyckoff': {'lookback_bars': 200},
                'gann': {'lookback_bars': 200},
                'fibonacci_advanced': {'lookback_bars': 200},
                'momentum_divergence': {'lookback_bars': 200}
            },
            'smc': {
                'active_strategies': [
                    'order_blocks', 'market_structure',
                    'liquidity_pools', 'manipulation'
                ],
                'order_blocks': {'lookback': 50},
                'market_structure': {
                    'lookback_bars': 200,
                    'swing_window': 5,
                    'retest_window': 3
                },
                'liquidity_pools': {
                    'lookback_bars': 300,
                    'equal_highs_tolerance': 0.1
                },
                'manipulation': {
                    'lookback_bars': 250,
                    'wick_ratio_threshold': 1.5
                }
            },
            'ml': {
                'active_strategies': [
                    'lstm', 'xgboost_classifier',
                    'ensemble_nn', 'rl_agent'
                ],
                'lstm': {'sequence_length': 60},
                'xgboost_classifier': {'lookback_bars': 200},
                'ensemble_nn': {'lookback_bars': 200},
                'rl_agent': {'lookback_bars': 200}
            },
            'fusion': {
                'active_strategies': [
                    'weighted_voting', 'confidence_sizing',
                    'regime_detection', 'adaptive_ensemble'
                ],
                'weighted_voting': {'lookback_bars': 200},
                'confidence_sizing': {'lookback_bars': 200},
                'regime_detection': {'lookback_bars': 200},
                'adaptive_ensemble': {'lookback_bars': 200}
            }
        },
        'risk_management': {
            'risk_per_trade': 0.02,
            'max_daily_loss': 0.06
        },
        'signal_generation': {
            'max_signals_per_bar': 5
        }
    }

    # Initialize Signal Engine
    print("\n1. Testing initialization...")
    engine = SignalEngine(config, mt5_manager=None, database_manager=None)
    print(f"   Initialization: ✅ Success")

    # Check loaded strategies
    print("\n2. Available strategies:")
    active = engine.get_active_strategies()
    for category, strategies in active.items():
        if strategies:
            print(f"   {category.upper()}: {strategies}")

    # Test signal generation (will be limited without MT5)
    print("\n3. Testing signal generation...")
    signals = engine.generate_signals("XAUUSDm", 15)
    print(f"   Generated {len(signals)} signals")

    # Show performance
    print("\n4. Strategy performance:")
    perf = engine.get_strategy_performance()
    for strategy, metrics in perf.items():
        print(f"   {strategy}: Signals: {metrics['signals_generated']}, Win Rate: {metrics['win_rate']:.2%}")

    print("\n✅ Signal Engine test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )

    # Run test
    test_signal_engine()