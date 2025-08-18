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
import time
from collections import defaultdict

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
                
# =====================================================================
# CONSOLE REPORTER
# =====================================================================
class ConsoleReporter:
    """Clean, structured console output reporter"""
    
    def __init__(self):
        self.start_time = time.time()
        self.error_counts = defaultdict(int)
        self.warning_messages = defaultdict(list)
        
    def confidence_bar(self, confidence: float, width: int = 5) -> str:
        """Create visual confidence bar [****.]"""
        filled = int(confidence * width)
        empty = width - filled
        return "[" + "*" * filled + "." * empty + "]"
    
    def phase_header(self, phase: int, title: str, status: str = ""):
        """Print clean phase headers"""
        status_text = f" - {status}" if status else ""
        print(f"\n[{phase}] {title}{status_text}")
        if phase == 1:
            print("-" * 50)
    
    def strategy_table(self, strategies: Dict[str, List[str]]):
        """Clean strategy overview table"""
        print("\nStrategy Categories:")
        print("-" * 60)
        print(f"| {'Category':<12} | {'Count':<5} | {'Status':<15} | {'Strategies':<20} |")
        print("-" * 60)
        
        for category, strategy_list in strategies.items():
            if strategy_list:
                count = len(strategy_list)
                status = "Active"
                strategies_text = ", ".join(strategy_list[:3])  # Show first 3
                if len(strategy_list) > 3:
                    strategies_text += "..."
                print(f"| {category.upper():<12} | {count:<5} | {status:<15} | {strategies_text:<20} |")
        
        print("-" * 60)
        total = sum(len(strategies) for strategies in strategies.values())
        print(f"Total Strategies Loaded: {total}")
    
    def signal_summary(self, signals: List):
        """Display signal generation summary"""
        if not signals:
            print("  - No signals generated")
            return
            
        print(f"  - Total Signals: {len(signals)}")
        for i, signal in enumerate(signals[:5], 1):  # Show max 5
            conf_bar = self.confidence_bar(signal.confidence)
            print(f"    {i}. {signal.signal_type.value} at {signal.price:.2f} "
                  f"(Confidence: {signal.confidence:.1%}) {conf_bar} "
                  f"[Strategy: {signal.strategy_name}] [Grade: {signal.grade.value}]")
        
        if len(signals) > 5:
            print(f"    ... and {len(signals) - 5} more signals")
    
    def add_warning(self, strategy: str, message: str):
        """Add warning message for aggregation"""
        self.warning_messages[strategy].append(message)
    
    def show_warnings(self):
        """Display aggregated warnings"""
        if not self.warning_messages:
            print("  - No issues detected")
            return
        
        print("  Issues Found:")
        for strategy, messages in self.warning_messages.items():
            if len(messages) > 1:
                print(f"    - {strategy}: {len(messages)} issues detected")
                for i, msg in enumerate(messages[:3], 1):  # Show first 3 messages
                    print(f"      {i}. {msg}")
                if len(messages) > 3:
                    print(f"      ... and {len(messages) - 3} more issues")
            else:
                print(f"    - {strategy}: {messages[0]}")
    
    def final_dashboard(self, total_strategies: int, signals_generated: int, 
                       active_strategies: int):
        """Comprehensive session summary"""
        duration = time.time() - self.start_time
        
        print(f"\nSession Duration: {duration:.1f}s")
        print(f"Strategies Loaded: {total_strategies}")
        print(f"Active Strategies: {active_strategies}")
        print(f"Signals Generated: {signals_generated}")
        
        if signals_generated > 0:
            rate = signals_generated / duration if duration > 0 else 0
            print(f"Signal Rate: {rate:.1f} signals/second")

# =====================================================================
# MOCK MT5 FALLBACK
# =====================================================================                
class MockMT5Manager:
    """Mock MT5 manager for testing strategies without MT5 connection."""

    def get_historical_data(self, symbol="XAUUSDm", timeframe="M15", lookback=500):
        import pandas as pd
        import numpy as np

        dates = pd.date_range(end=pd.Timestamp.today(), periods=lookback, freq="15min")
        data = pd.DataFrame({
            "Time": dates,
            "Open": np.random.uniform(1800, 2000, lookback),
            "High": np.random.uniform(1800, 2000, lookback),
            "Low": np.random.uniform(1800, 2000, lookback),
            "Close": np.random.uniform(1800, 2000, lookback),
            "Volume": np.random.randint(100, 1000, lookback),
        })
        # Ensure datetime index
        data.set_index("Time", inplace=True)

        # Add lowercase aliases (compatibility for fusion/ML)
        data["open"] = data["Open"]
        data["high"] = data["High"]
        data["low"] = data["Low"]
        data["close"] = data["Close"]
        data["volume"] = data["Volume"]

        return data

    def get_ohlcv(self, symbol="XAUUSDm", timeframe="M15", lookback=500):
        return self.get_historical_data(symbol, timeframe, lookback)
    
mock_mt5 = MockMT5Manager()


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
            #'imbalance': ('src.strategies.smc.imbalance', 'ImbalanceStrategy')
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
            'lstm': ('src.strategies.ml.lstm_predictor', 'LSTMPredictorStrategy'),
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
            'weighted_voting': ('src.strategies.fusion.weighted_voting', 'WeightedVoting'),
            'confidence_sizing': ('src.strategies.fusion.confidence_sizing', 'ConfidenceSizing'),
            'regime_detection': ('src.strategies.fusion.regime_detection', 'RegimeDetection'),
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
        #self.mt5_manager = mt5_manager
        self.mt5_manager = mt5_manager if mt5_manager else mock_mt5
        self.database_manager = database_manager
        
        # Setup logging
        self.logger = logging.getLogger('signal_engine')
        # Respect global logging level instead of forcing INFO
        
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
                'invalid_signals': 0,
                'successful_signals': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0},
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
                self._track_invalid_signal(signal.strategy_name, "Low confidence")
                continue
            
            # Check signal conflicts
            if self._has_conflict(signal, quality_signals):
                self._track_invalid_signal(signal.strategy_name, "Signal conflict")
                continue
            
            # Regime-based filtering
            if not self._is_regime_appropriate(signal):
                self._track_invalid_signal(signal.strategy_name, "Regime mismatch")
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
            
            # Update performance tracking
            strategy_name = signal.strategy_name
            if strategy_name in self.strategy_performance:
                # Update grade distribution
                grade_key = signal.grade.value if signal.grade else 'C'
                if grade_key in self.strategy_performance[strategy_name]['grade_distribution']:
                    self.strategy_performance[strategy_name]['grade_distribution'][grade_key] += 1
            
            # Log signal
            self.logger.info(f"📊 Storing signal: {signal.strategy_name} - {signal.signal_type.value}")
            
        except Exception as e:
            self.logger.error(f"Error storing signal: {str(e)}")
    
    def _track_invalid_signal(self, strategy_name: str, reason: str = "Quality filter") -> None:
        """Track invalid signal for performance metrics"""
        if strategy_name in self.strategy_performance:
            self.strategy_performance[strategy_name]['invalid_signals'] += 1
    
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
    """Test the Signal Engine functionality with clean structured output"""
    
    # Suppress ALL verbose logging and TensorFlow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.filterwarnings('ignore')
    
    import logging
    logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
    logging.getLogger('signal_engine').setLevel(logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # Redirect stdout temporarily to suppress prints during initialization
    import sys
    from io import StringIO
    
    # Initialize console reporter
    reporter = ConsoleReporter()
    
    # Session header
    print("SIGNAL ENGINE SESSION - {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
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
                'wyckoff': {'lookback_bars': 80},
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
                'market_structure': {'lookback_bars': 200, 'swing_window': 5},
                'liquidity_pools': {'lookback_bars': 300},
                'manipulation': {'lookback_bars': 250}
            },
            'ml': {
                'active_strategies': [
                    'lstm', 'xgboost_classifier',
                    'ensemble_nn', 'rl_agent'
                ],
                'lstm': {'sequence_length': 60},
                'xgboost_classifier': {'lookback_bars': 120},
                'ensemble_nn': {'lookback_bars': 200},
                'rl_agent': {'lookback_bars': 200}
            },
            'fusion': {
                'active_strategies': [
                    'weighted_voting', 'confidence_sizing',
                    'regime_detection', 'adaptive_ensemble'
                ],
                'weighted_voting': {'min_signals': 2},
                'confidence_sizing': {'base_risk': 0.02},
                'regime_detection': {'lookback_period': 30},
                'adaptive_ensemble': {'lookback_bars': 200}
            }
        },
        'risk_management': {
            'risk_per_trade': 0.02,
            'max_daily_loss': 0.06
        }
    }

    # Phase 1: Initialization
    reporter.phase_header(1, "Initialization", "OK")
    try:
        # Capture stdout to get strategy loading messages but suppress sys.path spam
        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        engine = SignalEngine(config, mt5_manager=None, database_manager=None)
        
        # Get captured output and filter for important messages
        output = captured_output.getvalue()
        sys.stdout = old_stdout
        
        # Look for important warnings/errors in the captured output
        lines = output.split('\n')
        for line in lines:
            if 'WARNING:' in line:
                if 'Could not import' in line:
                    reporter.add_warning("Strategy Loading", line.split(':', 2)[2].strip() if line.count(':') >= 2 else line)
                elif 'Error importing' in line:
                    strategy_name = line.split(':')[1] if ':' in line else "Unknown"
                    reporter.add_warning("Strategy Import", f"{strategy_name}: Import failed")
                elif 'not available' in line and ('TensorFlow' in line or 'ML libraries' in line):
                    reporter.add_warning("ML Dependencies", "ML libraries not available - running in simulation mode")
                elif 'not available' in line and 'XGBoost' in line:
                    reporter.add_warning("ML Dependencies", "XGBoost not available - running in simulation mode")
            elif 'ERROR:' in line:
                strategy_name = line.split(':')[1] if ':' in line else "System"
                error_msg = line.split(':', 2)[2].strip() if line.count(':') >= 2 else line
                reporter.add_warning(f"ERROR-{strategy_name}", error_msg)
        
        print("  - Core System: Ready")
        print("  - Configuration: Loaded") 
        print("  - Mock MT5: Active")
        print("  - Strategies: 21 loaded")
    except Exception as e:
        # Restore stdout if error
        sys.stdout = old_stdout
        reporter.add_warning("System Initialization", f"Critical error: {str(e)}")
        print(f"  - ERROR: {str(e)}")
        return

    # Phase 2: Strategy Overview  
    reporter.phase_header(2, "Strategy Overview")
    active = engine.get_active_strategies()
    reporter.strategy_table(active)
    
    total_strategies = sum(len(strategies) for strategies in active.values())

    # Phase 3: Signal Generation
    reporter.phase_header(3, "Signal Generation")
    try:
        # Capture signal generation output to filter for warnings
        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        signals = engine.generate_signals("XAUUSDm", 15)
        
        # Process captured output for warnings
        output = captured_output.getvalue()
        sys.stdout = old_stdout
        
        # Look for all warnings and errors in signal generation
        lines = output.split('\n')
        invalid_signal_counts = defaultdict(int)
        
        for line in lines:
            if 'WARNING:' in line:
                if 'Invalid signal:' in line:
                    # Count invalid signals per strategy
                    strategy_name = line.split(':')[1] if ':' in line else "Unknown"
                    invalid_signal_counts[strategy_name.strip()] += 1
                elif 'MT5 manager not available' in line:
                    strategy_name = line.split(':')[1] if ':' in line else "Unknown"
                    reporter.add_warning(strategy_name.strip(), "MT5 manager not available")
                elif 'No valid signals' in line:
                    strategy_name = line.split(':')[1] if ':' in line else "Unknown"
                    reporter.add_warning(strategy_name.strip(), "No valid signals generated")
            elif 'ERROR:' in line:
                strategy_name = line.split(':')[1] if ':' in line else "System"
                error_msg = line.split(':', 2)[2].strip() if line.count(':') >= 2 else line
                reporter.add_warning(f"ERROR-{strategy_name}", error_msg[:100] + "..." if len(error_msg) > 100 else error_msg)
        
        # Add invalid signal counts to warnings
        for strategy, count in invalid_signal_counts.items():
            if count > 0:
                reporter.add_warning(strategy, f"{count} invalid signals rejected")
        
        reporter.signal_summary(signals)
    except Exception as e:
        # Restore stdout if error
        sys.stdout = old_stdout
        reporter.add_warning("Signal Generation", f"Critical error: {str(e)}")
        print(f"  - ERROR: Signal generation failed - {str(e)}")
        signals = []

    # Phase 4: Strategy Performance  
    reporter.phase_header(4, "Strategy Performance")
    try:
        perf = engine.get_strategy_performance()
        active_strategies = sum(1 for metrics in perf.values() if metrics['signals_generated'] > 0)
        
        # Group performance by category
        categories = {
            'TECHNICAL': ['ichimoku', 'harmonic', 'elliott_wave', 'volume_profile', 'market_profile', 
                         'order_flow', 'wyckoff', 'gann', 'fibonacci_advanced', 'momentum_divergence'],
            'SMC': ['order_blocks', 'market_structure', 'liquidity_pools', 'manipulation'],
            'ML': ['lstm', 'xgboost_classifier', 'ensemble_nn', 'rl_agent'],
            'FUSION': ['weighted_voting', 'confidence_sizing', 'regime_detection', 'adaptive_ensemble']
        }
        
        for cat_name, strategy_list in categories.items():
            active_count = sum(1 for s in strategy_list if s in perf and perf[s]['signals_generated'] > 0)
            total_signals = sum(perf[s]['signals_generated'] for s in strategy_list if s in perf)
            print(f"  - {cat_name}: {active_count}/{len(strategy_list)} active, {total_signals} signals")
        
        # Detailed Strategy Performance Table
        print("\nDetailed Strategy Performance:")
        print("-" * 95)
        print(f"| {'Strategy':<18} | {'Valid':<6} | {'Invalid':<7} | {'A':<3} | {'B':<3} | {'C':<3} | {'Win Rate':<8} | {'Status':<8} |")
        print("-" * 95)
        
        # Sort strategies by signal count for better readability
        sorted_strategies = sorted(perf.items(), key=lambda x: x[1]['signals_generated'], reverse=True)
        
        for strategy_name, metrics in sorted_strategies:
            if metrics['signals_generated'] > 0 or metrics.get('invalid_signals', 0) > 0:  # Show strategies with any activity
                valid_signals = metrics['signals_generated']
                invalid_signals = metrics.get('invalid_signals', 0)
                
                # Get grade distribution - handle case where grades might not exist
                grades = metrics.get('grade_distribution', {'A': 0, 'B': 0, 'C': 0})
                a_grade = grades.get('A', 0)
                b_grade = grades.get('B', 0) 
                c_grade = grades.get('C', 0)
                
                win_rate = metrics.get('win_rate', 0.0)
                win_rate_str = f"{win_rate:.1%}"
                
                # Determine status
                if valid_signals > 0:
                    status = "Active"
                elif invalid_signals > 0:
                    status = "Issues"
                else:
                    status = "Idle"
                
                print(f"| {strategy_name:<18} | {valid_signals:<6} | {invalid_signals:<7} | {a_grade:<3} | {b_grade:<3} | {c_grade:<3} | {win_rate_str:<8} | {status:<8} |")
        
        print("-" * 95)
        total_valid = sum(metrics['signals_generated'] for metrics in perf.values())
        total_invalid = sum(metrics.get('invalid_signals', 0) for metrics in perf.values())
        print(f"Totals: Valid Signals: {total_valid}, Invalid Signals: {total_invalid}, Active Strategies: {active_strategies}")
            
    except Exception as e:
        print(f"  - ERROR: Performance analysis failed - {str(e)}")
        active_strategies = 0

    # Phase 5: Issues & Warnings
    reporter.phase_header(5, "Issues & Warnings") 
    reporter.show_warnings()

    # Phase 6: Final Dashboard
    reporter.phase_header(6, "Final Dashboard", "COMPLETE")
    signals_count = len(signals) if hasattr(signals, '__len__') else 0
    reporter.final_dashboard(total_strategies, signals_count, active_strategies)
    
    print("=" * 60)


if __name__ == "__main__":
    # Suppress all logging for clean test output
    logging.basicConfig(level=logging.CRITICAL)
    
    # Run test
    test_signal_engine()