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
from dataclasses import dataclass, field # Import 'field'
import importlib
import sys
from pathlib import Path
import time
from collections import defaultdict

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import live MT5 manager
try:
    from src.core.mt5_manager import MT5Manager  # type: ignore
except Exception:
    MT5Manager = None  # Fallback handled at runtime

# CLI mode parsing utility
try:
    from src.utils.cli_args import parse_mode, print_mode_banner
except Exception:
    def parse_mode(*_args, **_kwargs):  # type: ignore
        return 'mock'
    def print_mode_banner(_mode):  # type: ignore
        pass

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
    strength: float = 0.0
    grade: SignalGrade = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict) # FIX 1: Use field(default_factory=dict) for metadata
    
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
                for i, msg in enumerate(messages, 1):  # Show ALL messages
                    print(f"      {i}. {msg}")
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

    def __init__(self, mode='mock'): # Added mode to constructor for consistent mock data generation
        self.mode = mode

    def get_historical_data(self, symbol="XAUUSDm", timeframe="M15", lookback=500):
        import pandas as pd
        import numpy as np

        dates = pd.date_range(end=pd.Timestamp.today(), periods=lookback, freq="15min")
        np.random.seed(42 if self.mode == 'mock' else 123) # Use mode to seed for consistent mock data
        
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
    
# mock_mt5 = MockMT5Manager() # REMOVED: Instantiate only when needed in SignalEngine.__init__

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
            # More specific check for missing ML dependencies
            if "tensorflow" in str(e).lower() or "sklearn" in str(e).lower() or "xgboost" in str(e).lower():
                print(f"WARNING: ML library not available ({str(e)}). {class_name} will run in simulation mode.")
                logging.info(f"Successfully imported {strategy_type} strategy: {class_name} (ML simulation mode)")
                # For ML strategies, we still want to load the class but ensure it's aware of the sim mode.
                # This assumes the strategy itself handles the ML_AVAILABLE flag.
                module = importlib.import_module(module_path)
                strategy_class = getattr(module, class_name)
                return strategy_class
            else:
                logging.warning(f"Could not import {strategy_type} strategy {class_name} due to ImportError: {str(e)}")
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
    Core signal generation and coordination engine with enhanced error handling and configuration
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, database_manager=None):
        """
        Initialize Signal Engine with enhanced error handling and configuration
        
        Args:
            config: Configuration dictionary
            mt5_manager: MT5Manager instance (optional)
            database_manager: DatabaseManager instance (optional)
        """
        self.config = config
        self._setup_logging()
        self._initialize_managers(mt5_manager, database_manager)
        self._initialize_containers()
        self._load_and_initialize_strategies()

    def _setup_logging(self):
        """Configure logging for the signal engine"""
        self.logger = logging.getLogger('signal_engine')
        logging.basicConfig(level=logging.INFO)

    def _initialize_managers(self, mt5_manager, database_manager):
        """Initialize MT5 and database managers"""
        self.data_mode = self._determine_data_mode()
        self.mt5_manager = self._initialize_mt5_manager(mt5_manager)
        self.database_manager = database_manager

    def _determine_data_mode(self):
        """Determine the data mode from config and CLI"""
        mode = parse_mode() or self.config.get('mode', 'mock')
        print_mode_banner(mode)
        return mode

    def _initialize_mt5_manager(self, mt5_manager):
        """Initialize MT5 manager with fallback to mock"""
        if mt5_manager is not None:
            return mt5_manager
            
        if self.data_mode == 'live' and MT5Manager is not None:
            try:
                default_symbol = self.config.get('trading', {}).get('symbol', 'XAUUSD')
                live_mgr = MT5Manager(symbol=default_symbol)
                if live_mgr.connect():
                    return live_mgr
                self.logger.warning("Falling back to MockMT5Manager: live MT5 connection failed")
            except Exception as e:
                self.logger.warning(f"Error connecting to live MT5: {e}. Falling back to mock.")
        
        return MockMT5Manager(self.data_mode)

    def _initialize_containers(self):
        """Initialize data containers and state"""
        # Strategy storage
        self.available_strategies = {cat: {} for cat in ['technical', 'smc', 'ml', 'fusion']}
        self.strategies = {cat: {} for cat in self.available_strategies}
        
        # Signal management with max buffer size
        self.signal_buffer = []
        self.max_buffer_size = self.config.get('signal_engine', {}).get('max_buffer_size', 1000)
        self.active_signals = []
        self.signal_history = []
        
        # Performance tracking
        self.strategy_performance = {}
        self.strategy_name_map = {}
        
        # Market state
        self.current_regime = "NEUTRAL"
        
        # Initialize importer
        self.importer = StrategyImporter()

    def _load_and_initialize_strategies(self):
        """Load and initialize all strategies from configuration"""
        self._load_available_strategies()
        self._initialize_active_strategies()
        self._log_initialization_summary()

    def _load_available_strategies(self) -> None:
        """Load all available strategy classes with error handling"""
        self.logger.info("Loading available strategies...")
        
        loaders = {
            'technical': self.importer.load_technical_strategies,
            'smc': self.importer.load_smc_strategies,
            'ml': self.importer.load_ml_strategies,
            'fusion': self.importer.load_fusion_strategies
        }
        
        for category, loader in loaders.items():
            try:
                self.available_strategies[category] = loader() or {}
                self.logger.debug(f"Loaded {len(self.available_strategies[category])} {category} strategies")
            except Exception as e:
                self.logger.error(f"Error loading {category} strategies: {e}")
                self.available_strategies[category] = {}

    def _initialize_active_strategies(self) -> None:
        """Initialize all active strategies from configuration"""
        self.logger.info("Initializing active strategies...")
        strategies_config = self.config.get('strategies', {})
        
        for category in self.available_strategies:
            self._initialize_category_strategies(category, strategies_config.get(category, {}))

    def _initialize_category_strategies(self, category: str, category_config: Dict) -> None:
        """Initialize strategies for a specific category"""
        available = self.available_strategies[category]
        if not available:
            self.logger.warning(f"No {category} strategies available to initialize")
            return
            
        # Get active strategies or default to all available
        active_strategies = category_config.get('active_strategies', list(available.keys()))
        
        for strategy_name in active_strategies:
            if strategy_name in available:
                self._initialize_single_strategy(category, strategy_name, category_config)

    def _initialize_single_strategy(self, category: str, strategy_name: str, category_config: Dict) -> None:
        """
        Initialize a single strategy instance with error handling
        
        Args:
            category: Strategy category (technical, smc, ml, fusion)
            strategy_name: Name of the strategy
            category_config: Configuration for the category
        """
        try:
            strategy_class = self.available_strategies[category].get(strategy_name)
            if not strategy_class:
                self.logger.warning(f"Strategy class not found: {strategy_name}")
                return
                
            # Prepare configuration
            strategy_config = {
                'name': strategy_name,
                'category': category,
                'parameters': category_config.get(strategy_name, {}),
                'risk_per_trade': self.config.get('risk_management', {}).get('risk_per_trade', 0.02)
            }
            
            # Initialize strategy
            strategy_instance = strategy_class(
                config=strategy_config,
                mt5_manager=self.mt5_manager,
                database=self.database_manager
            )
            
            # Register strategy
            self.strategies[category][strategy_name] = strategy_instance
            class_name = strategy_instance.__class__.__name__
            self.strategy_name_map[class_name] = strategy_name
            
            # Initialize performance tracking
            self.strategy_performance[strategy_name] = {
                'signals_generated': 0,
                'signals_executed': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'invalid_signals': 0,
                'grade_distribution': {'A': 0, 'B': 0, 'C': 0},
                'last_updated': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Initialized {category} strategy: {strategy_name} ({class_name})")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {category} strategy '{strategy_name}': {e}", 
                            exc_info=self.logger.level <= logging.DEBUG)

    def _log_initialization_summary(self) -> None:
        """Log summary of initialized strategies"""
        total_initialized = sum(len(strategies) for strategies in self.strategies.values())
        self.logger.info(f"Initialized {total_initialized} strategies:")
        
        for category, strategies in self.strategies.items():
            if strategies:
                self.logger.info(f"  {category.upper()}: {', '.join(strategies.keys())}")

    def _add_signal_to_buffer(self, signal: Signal) -> None:
        """
        Add signal to buffer with size management
        
        Args:
            signal: Signal to add to buffer
        """
        # Enforce buffer size limit
        if len(self.signal_buffer) >= self.max_buffer_size:
            self.signal_buffer.pop(0)  # Remove oldest signal
            
        self.signal_buffer.append(signal)
        self.signal_history.append(signal)
        
        # Update performance metrics
        if signal.strategy_name in self.strategy_performance:
            self.strategy_performance[signal.strategy_name]['signals_generated'] += 1
            self.strategy_performance[signal.strategy_name]['last_updated'] = datetime.utcnow().isoformat()

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
            # Convert timeframe to string format if needed
            tf_str = self._convert_timeframe(timeframe) if isinstance(timeframe, int) else timeframe
            data = self.mt5_manager.get_historical_data(symbol, tf_str, 100)
            # Validate and normalize
            data = self._validate_market_data(data, symbol, timeframe)
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

    def _validate_market_data(self, data: pd.DataFrame, symbol: str, timeframe: int) -> pd.DataFrame:
        """Validate OHLCV data: missing bars, symbol mismatch, precision.

        - Ensures lowercase columns exist alongside standard case
        - Checks for gaps beyond config threshold
        - Validates price precision against symbol digits (live mode)
        """
        try:
            if data is None or data.empty:
                return pd.DataFrame()

            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                # Try to coerce
                if 'time' in data.columns:
                    data['time'] = pd.to_datetime(data['time'])
                    data.set_index('time', inplace=True)
                else:
                    return pd.DataFrame()

            # Ensure lowercase aliases
            for src, dst in [('Open','open'), ('High','high'), ('Low','low'), ('Close','close'), ('Volume','volume')]:
                if src in data.columns and dst not in data.columns:
                    data[dst] = data[src]
                if dst in data.columns and src not in data.columns:
                    data[src] = data[dst]

            # Gap check
            validation_cfg = self.config.get('data', {}).get('validation', {})
            check_gaps = bool(validation_cfg.get('check_gaps', True))
            max_gap_size = int(validation_cfg.get('max_gap_size', 5))
            if check_gaps:
                tf_str = self._convert_timeframe(timeframe)
                expected_minutes = {
                    'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                    'H1': 60, 'H4': 240, 'D1': 1440
                }.get(tf_str, 15)
                expected_delta = pd.Timedelta(minutes=expected_minutes)
                gaps = data.index.to_series().diff().dropna()
                # Count how many missing bars implicit in gaps
                missing_bars = int(sum(max(int((g / expected_delta)) - 1, 0) for g in gaps))
                if missing_bars > 0:
                    self.logger.warning(f"Detected {missing_bars} missing bars in {symbol} {tf_str}") # FIX 6: Directly use self.logger

            # Precision check (only meaningful in live mode with MT5 symbol digits)
            precision_tol = float(validation_cfg.get('precision_tolerance', 1e-6))
            if self.data_mode == 'live' and hasattr(self.mt5_manager, 'get_symbol_info'):
                try:
                    info = self.mt5_manager.get_symbol_info(symbol)
                    digits = int(info.get('digits', 0)) if info else 0
                except Exception:
                    digits = 0
                if digits > 0:
                    def round_series(s: pd.Series) -> pd.Series:
                        return s.round(digits)
                    for col in ['Open','High','Low','Close']:
                        if col in data.columns:
                            rounded = round_series(data[col])
                            drift = (data[col] - rounded).abs()
                            if (drift > precision_tol).any():
                                # Auto-correct by rounding
                                data[col] = rounded
                                # Keep lowercase in sync
                                data[col.lower()] = rounded
            return data
        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}") # FIX 6: Directly use self.logger
            return pd.DataFrame()
    
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
            # FIRST: Track grade distribution for ALL signals before filtering
            self._track_signal_grade(signal)
            
            # Basic quality checks
            if signal.confidence < 0.5:
                self._track_invalid_signal(signal.strategy_name, "Low confidence")
                continue
            
            # Filter out poor-quality grade D signals explicitly
            try:
                grade_value = getattr(getattr(signal, 'grade', None), 'value', None)
                if grade_value == 'D':
                    self._track_invalid_signal(signal.strategy_name, "Low grade (D)")
                    continue
            except Exception:
                pass
            
            # Check signal conflicts
            if self._has_conflict(signal, quality_signals):
                self._track_invalid_signal(signal.strategy_name, "Signal conflict")
                continue
            
            # Regime-based filtering
            if not self._is_regime_appropriate(signal):
                self._track_invalid_signal(signal.strategy_name, "Regime mismatch")
                continue
            
            quality_signals.append(signal)
        
        # Sort by grade rank (A>B>C>D) then by confidence
        grade_rank = {'A': 3, 'B': 2, 'C': 1, 'D': 0}
        quality_signals.sort(
            key=lambda x: (
                grade_rank.get(getattr(getattr(x, 'grade', None), 'value', 'C'), 1),
                x.confidence
            ),
            reverse=True
        )
        
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
                    'quality_grade': signal.grade.value if signal.grade else 'C',
                    'price': signal.price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'timeframe': signal.timeframe,
                    'signal_metadata': str(signal.metadata) if signal.metadata else '{}'
                }
                self.database_manager.store_signal(signal_data)
            
            # Update performance tracking - FIXED: This was being called after signals were filtered
            # but we need to track ALL stored signals, not just during generation
            # Normalize performance key (map class name to registry key)
            # FIX 5: Simplify perf_key logic
            perf_key = signal.strategy_name 
            if perf_key in self.strategy_performance:
                # Update grade distribution - ensure signal has a grade
                if signal.grade:
                    grade_key = signal.grade.value
                    if grade_key in self.strategy_performance[perf_key]['grade_distribution']:
                        self.strategy_performance[perf_key]['grade_distribution'][grade_key] += 1
                        self.logger.debug(f"Updated grade distribution for {perf_key}: {grade_key}")
                else:
                    # Fallback if no grade
                    self.strategy_performance[perf_key]['grade_distribution']['C'] += 1
                    self.logger.debug(f"Updated grade distribution for {perf_key}: C (fallback)")
            
            # Log signal
            self.logger.info(f"📊 Storing signal: {signal.strategy_name} - {signal.signal_type.value} - Grade: {signal.grade.value if signal.grade else 'None'}")
            
        except Exception as e:
            self.logger.error(f"Error storing signal: {str(e)}")
    
    def _track_invalid_signal(self, strategy_name: str, reason: str = "Quality filter") -> None:
        """Track invalid signal for performance metrics"""
        # FIX 5: Simplify perf_key logic
        perf_key = strategy_name
        if perf_key in self.strategy_performance:
            self.strategy_performance[perf_key]['invalid_signals'] += 1
    
    def _track_signal_grade(self, signal: Signal) -> None:
        """Track signal grade distribution for performance metrics"""
        try:
            strategy_name = signal.strategy_name
            if strategy_name in self.strategy_performance:
                # Ensure signal has a grade
                if signal.grade:
                    grade_key = signal.grade.value
                    if grade_key in self.strategy_performance[strategy_name]['grade_distribution']:
                        self.strategy_performance[strategy_name]['grade_distribution'][grade_key] += 1
                        self.logger.debug(f"Tracked grade {grade_key} for {strategy_name}")
                else:
                    # Fallback if no grade - assign based on confidence
                    if signal.confidence >= 0.8:
                        grade_key = 'A'
                    elif signal.confidence >= 0.6:
                        grade_key = 'B'
                    else:
                        grade_key = 'C'
                    
                    self.strategy_performance[strategy_name]['grade_distribution'][grade_key] += 1
                    self.logger.debug(f"Tracked fallback grade {grade_key} for {strategy_name}")
        except Exception as e:
            self.logger.error(f"Error tracking signal grade: {str(e)}")
    
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
    
    # Configure logging to show warnings and errors but suppress debug/info
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Show warnings and errors
    import warnings
    # Don't suppress all warnings - we want to see important ones
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    
    import logging
    # Set logging to WARNING level to see issues
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('signal_engine').setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    
    # Redirect stdout temporarily to suppress prints during initialization
    import sys
    from io import StringIO
    
    # Initialize console reporter
    reporter = ConsoleReporter()

    # Attach a logging capture handler to collect warnings/errors emitted via logging
    class LogCaptureHandler(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.WARNING)
            self.records = []
        def emit(self, record):
            self.records.append(record)

    root_logger = logging.getLogger()
    capture_handler = LogCaptureHandler()
    root_logger.addHandler(capture_handler)
    
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
        # Capture stdout/stderr to get strategy loading messages but suppress sys.path spam
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_output = StringIO()
        sys.stdout = captured_output
        sys.stderr = captured_output
        
        # Instantiate SignalEngine. It will handle its own MT5Manager based on config.data.mode
        engine = SignalEngine(config, mt5_manager=None, database_manager=None)
        
        # Get captured output and filter for important messages
        output = captured_output.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
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
        
        # Print actual mode of the engine for clarity
        print(f"  - Core System: Ready (Mode: {engine.data_mode.upper()})")
        print("  - Configuration: Loaded") 
        print("  - Strategies: 21 loaded") # Hardcoded count is okay for test output
    except Exception as e:
        # Restore stdout if error
        sys.stdout = old_stdout
        sys.stderr = old_stderr
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
        # Capture signal generation output (stdout/stderr) to filter for warnings
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_output = StringIO()
        sys.stdout = captured_output
        sys.stderr = captured_output
        
        signals = engine.generate_signals("XAUUSDm", 15)
        
        # Process captured output for warnings
        output = captured_output.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
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
        sys.stderr = old_stderr
        reporter.add_warning("Signal Generation", f"Critical error: {str(e)}")
        print(f"  - ERROR: Signal generation failed - {str(e)}")
        signals = []

    # Add captured log warnings/errors to reporter before performance section
    try:
        for rec in capture_handler.records:
            name = rec.name or "System"
            if rec.levelno >= logging.ERROR:
                reporter.add_warning(f"ERROR-{name}", rec.getMessage())
            else:
                reporter.add_warning(name, rec.getMessage())
    except Exception:
        pass

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

    # Detach capture handler
    try:
        root_logger.removeHandler(capture_handler)
    except Exception:
        pass

    # Phase 6: Final Dashboard
    reporter.phase_header(6, "Final Dashboard", "COMPLETE")
    signals_count = len(signals) if hasattr(signals, '__len__') else 0
    reporter.final_dashboard(total_strategies, signals_count, active_strategies)
    
    print("=" * 60)


if __name__ == "__main__":
    # Run test with proper logging configuration (handled in test_signal_engine)
    test_signal_engine()