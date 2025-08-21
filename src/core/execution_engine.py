"""
Execution Engine - Advanced Trade Execution System
=================================================
Author: XAUUSD Trading System
Version: 2.0.0
Date: 2025-08-08

Complete trade execution system for 10x returns target:
- Signal processing and validation
- Risk-adjusted position sizing
- Smart order execution
- Position management and monitoring
- Performance tracking
- Emergency controls

Features:
- Multi-strategy signal fusion
- Real-time risk monitoring
- Automated stop-loss and take-profit management
- Partial position closing
- Correlation-based position limits
- Emergency stop mechanisms
"""

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    # Running directly: python src/core/execution_engine.py
    project_root = Path(__file__).resolve().parents[2]  # points to J:\Gold_FX
    sys.path.insert(0, str(project_root))
from src.utils.path_utils import get_project_root

# Now import other modules
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from dataclasses import dataclass, field # Import 'field' for dataclass defaults
from enum import Enum
import threading
import time

# Import core modules
from src.core.base import Signal, SignalType, SignalGrade
from src.core.risk_manager import RiskManager
# Import CLI args for mode selection
try:
    from src.utils.cli_args import parse_mode, print_mode_banner
except Exception:
    def parse_mode(*_args, **_kwargs): # type: ignore
        return 'mock'
    def print_mode_banner(_mode): # type: ignore
        pass


logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """Execution status enumeration"""
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"


class PositionStatus(Enum):
    """Position status enumeration"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CLOSING = "CLOSING"
    MODIFIED = "MODIFIED"


@dataclass
class ExecutionResult:
    """Execution result data structure"""
    signal_id: str
    execution_id: str
    timestamp: datetime
    status: ExecutionStatus
    
    # Order details
    symbol: str
    order_type: str
    requested_size: float
    executed_size: float
    requested_price: float
    executed_price: float
    
    # Trade details
    ticket: Optional[int] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    slippage: float = 0.0
    
    # Execution metadata
    strategy: str = ""
    confidence: float = 0.0
    risk_assessment: Dict[str, Any] = field(default_factory=dict) # Use default_factory for mutable type
    error_message: str = ""
    
    def __post_init__(self):
        """Validate risk_assessment dictionary"""
        # Only validate if the execution was successful
        if self.status in (ExecutionStatus.EXECUTED, ExecutionStatus.PARTIAL):
            required_keys = {'monetary_risk', 'risk_percentage'}
            if not required_keys.issubset(self.risk_assessment.keys()):
                missing = required_keys - set(self.risk_assessment.keys())
                raise ValueError(f"risk_assessment missing required keys: {missing}")
            if not isinstance(self.risk_assessment['monetary_risk'], (int, float)):
                raise TypeError("monetary_risk must be numeric")
            if not isinstance(self.risk_assessment['risk_percentage'], (int, float)):
                raise TypeError("risk_percentage must be numeric")


@dataclass
class PositionInfo:
    """Position information structure"""
    ticket: int
    symbol: str
    order_type: str
    volume: float
    entry_price: float
    current_price: float
    
    # P&L information
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    # Risk information
    stop_loss: Optional[float]
    take_profit: Optional[float]
    
    # Position metadata
    strategy: str
    confidence: float
    entry_time: datetime
    status: PositionStatus
    
    # Risk metrics
    initial_risk: float
    current_risk: float
    time_in_position: timedelta


class ExecutionEngine:
    """
    Advanced Trade Execution Engine
    
    This engine handles:
    - Signal processing and validation
    - Risk-adjusted position sizing
    - Order execution with slippage protection
    - Position monitoring and management
    - Automated stop-loss and take-profit
    - Performance tracking
    - Emergency controls
    
    Designed for aggressive 10x returns while maintaining risk control.
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager=None, risk_manager=None, # Changed to optional
                 database_manager=None, logger_manager=None): # Changed to optional
        """Initialize Execution Engine"""
        self.config = config
        
        # Determine mode (CLI overrides config)
        cfg_mode = self.config.get('mode') or 'mock'
        self.mode = parse_mode() or cfg_mode

        # Initialize MT5 Manager
        if self.mode == 'live':
            try:
                from src.core.mt5_manager import MT5Manager
                live_mt5 = MT5Manager()
                if live_mt5.connect():
                    self.mt5_manager = live_mt5
                    logger.info("✅ Connected to live MT5")
                else:
                    logger.warning("⚠️  Failed to connect to live MT5, falling back to mock data")
                    self.mt5_manager = self._create_mock_mt5()
                    self.mode = 'mock' # Update mode if fallback
            except ImportError:
                logger.warning("⚠️  MT5Manager not available, using mock data")
                self.mt5_manager = self._create_mock_mt5()
                self.mode = 'mock' # Update mode if fallback
            except Exception as e:
                logger.error(f"Error initializing live MT5: {e}. Falling back to mock.")
                self.mt5_manager = self._create_mock_mt5()
                self.mode = 'mock'
        else:
            self.mt5_manager = mt5_manager if mt5_manager else self._create_mock_mt5()
        
        # Initialize Risk Manager (it will also handle its own MT5 dependency if needed)
        # Passing self.mt5_manager to RiskManager so it uses the same instance
        self.risk_manager = risk_manager if risk_manager else self._create_mock_risk_manager()
        
        # Initialize Database Manager
        self.database_manager = database_manager if database_manager else self._create_mock_database_manager()
        
        # Initialize Logger Manager (for trade logging purposes)
        self.logger_manager = logger_manager if logger_manager else self._create_mock_logger_manager()
        
        # Execution configuration
        self.execution_config = config.get('execution', {})
        self.max_slippage = self.execution_config.get('slippage', {}).get('max_slippage', 3)
        self.retry_attempts = self.execution_config.get('order', {}).get('retry_attempts', 3)
        self.retry_delay = self.execution_config.get('order', {}).get('retry_delay', 1)
        
        # Position management
        self.active_positions: Dict[int, PositionInfo] = {}
        self.pending_orders: Dict[str, ExecutionResult] = {}
        self.execution_history: List[ExecutionResult] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.total_commission = 0.0
        
        # Control flags
        self.engine_active = True
        self.monitoring_active = False
        
        # Threading
        self.monitor_thread = None
        self.execution_lock = threading.Lock()
        self.monitoring_lock = threading.Lock()  # Added for thread safety on monitoring_active flag
        
        # Logger (using the module-level logger)
        self.logger = logger
        
        # Initialize engine
        self._initialize_engine()
    
    # --- New: Internal Mock Creator Methods ---
    def _create_mock_mt5(self):
        """Create advanced mock MT5 manager"""
        class MockMT5Manager:
            def __init__(self, mode):
                self.mode = mode
                self.fail_next_order = False
                self.price_data = {
                    'XAUUSD': 1960.0,
                    'XAUUSDm': 1960.0
                }
                self.spread = 0.5  # Simulated spread
                
            def set_fail_next_order(self, should_fail: bool):
                self.fail_next_order = should_fail
                
            def update_price(self, symbol, new_price):
                """Update mock price for a symbol"""
                self.price_data[symbol] = new_price
                
            def get_account_balance(self):
                return 150.0
            
            def get_account_equity(self):
                return 145.0
            
            def get_open_positions(self):
                return []
                
            def place_market_order(self, symbol, order_type, volume, sl=None, tp=None, comment=""):
                # Simulate order failure if flag is set
                if self.fail_next_order:
                    self.fail_next_order = False
                    return {
                        'success': False,
                        'comment': 'Simulated failure for testing',
                        'retcode': 10018
                    }
                
                # Get current price with spread
                base_price = self.price_data.get(symbol, 1960.0)
                if order_type == "BUY":
                    executed_price = base_price + self.spread
                else:  # SELL
                    executed_price = base_price - self.spread
                
                # Simulate price movement after execution
                self.price_data[symbol] = base_price * (1 + (np.random.random() - 0.5) * 0.001)
                
                logger.info(f"[{self.mode} MT5] Simulating {order_type} order for {symbol}, vol={volume} @ {executed_price:.2f}")
                return {
                    'success': True,
                    'ticket': np.random.randint(1000000, 9999999),
                    'price': executed_price,
                    'volume': volume
                }
            
            def close_position(self, ticket, volume=None):
                logger.info(f"[{self.mode} MT5] Simulating close position {ticket}")
                return {'success': True}
            
            def modify_position(self, ticket, sl=None, tp=None):
                logger.info(f"[{self.mode} MT5] Simulating modify position {ticket}")
                return {'success': True}
        return MockMT5Manager(self.mode)

    def _create_mock_risk_manager(self):
        """Create mock RiskManager for testing"""
        class MockRiskManager:
            # Note: This mock only needs calculate_position_size and update_position_closed
            def calculate_position_size(self, signal, balance, positions):
                logger.info(f"[Mock RiskManager] Calculating size for {signal.symbol}")
                return {
                    'allowed': True,
                    'position_size': 0.02, # Fixed size for mock
                    'risk_assessment': {
                        'monetary_risk': 20.0,
                        'risk_percentage': 0.02,
                        'expected_return': 0.05,
                        'expected_return_std': 0.01
                    }
                }
            def update_position_closed(self, trade_result):
                logger.info(f"[Mock RiskManager] Updating for closed trade: {trade_result.get('profit')}")
        return MockRiskManager()

    def _create_mock_database_manager(self):
        """Create mock DatabaseManager for testing"""
        class MockDatabaseManager:
            def store_signal(self, signal_data):
                logger.info(f"[Mock DB] Storing signal for {signal_data.get('symbol')}")
            def store_trade(self, trade_data):
                logger.info(f"[Mock DB] Storing trade for {trade_data.get('symbol')}")
            # Add get_trades if needed for risk_manager's initialization
            def get_trades(self, limit=1000):
                return []
        return MockDatabaseManager()

    def _create_mock_logger_manager(self):
        """Create mock LoggerManager for testing"""
        class MockLoggerManager:
            def log_trade(self, action, symbol, volume, price, **kwargs):
                logger.info(f"[Mock Logger] Logged trade: {action} {symbol} vol={volume}")
        return MockLoggerManager()
    # --- End: Internal Mock Creator Methods ---

    def _initialize_engine(self) -> None:
        """Initialize the execution engine"""
        try:
            # Load existing positions
            self._load_existing_positions()
            
            # Start position monitoring
            self._start_position_monitoring()
            
            self.logger.info("Execution engine initialized successfully")
            self.logger.info(f"Active positions: {len(self.active_positions)}")
            self.logger.info(f"Max slippage: {self.max_slippage} pips")
            
        except Exception as e:
            self.logger.error(f"Execution engine initialization failed: {str(e)}")
    
    def process_signal(self, signal: Signal) -> ExecutionResult:
        """
        Process a trading signal and execute if valid
        
        Args:
            signal: Trading signal to process
            
        Returns:
            ExecutionResult: Result of execution attempt
        """
        execution_id = f"EXEC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(signal)}"
        
        try:
            self.logger.info(f"Processing signal: {signal.strategy_name} {signal.signal_type.value} "
                           f"{signal.symbol} @ {signal.price} (Confidence: {signal.confidence:.2f})")
            
            # Validate signal
            validation_result = self._validate_signal(signal)
            if not validation_result['valid']:
                return ExecutionResult(
                    signal_id=str(id(signal)),
                    execution_id=execution_id,
                    timestamp=datetime.now(),
                    status=ExecutionStatus.REJECTED,
                    symbol=signal.symbol,
                    order_type=signal.signal_type.value,
                    requested_size=0.0,
                    executed_size=0.0,
                    requested_price=signal.price,
                    executed_price=0.0,
                    error_message=validation_result['reason']
                )
            
            # Get account information
            account_balance = self.mt5_manager.get_account_balance()
            open_positions = self.mt5_manager.get_open_positions()
            
            # Calculate position size using risk manager
            sizing_result = self.risk_manager.calculate_position_size(
                signal, account_balance, open_positions
            )
            
            if not sizing_result['allowed'] or sizing_result['position_size'] <= 0:
                return ExecutionResult(
                    signal_id=str(id(signal)),
                    execution_id=execution_id,
                    timestamp=datetime.now(),
                    status=ExecutionStatus.REJECTED,
                    symbol=signal.symbol,
                    order_type=signal.signal_type.value,
                    requested_size=0.0,
                    executed_size=0.0,
                    requested_price=signal.price,
                    executed_price=0.0,
                    error_message=sizing_result.get('reason', 'Position size not allowed')
                )
            
            # Execute the order
            execution_result = self._execute_order(signal, sizing_result, execution_id)
            
            # Store execution result
            self.execution_history.append(execution_result)
            
            # Log signal to database
            if self.database_manager:
                self._log_signal_execution(signal, execution_result)
            
            # Update performance metrics
            if execution_result.status == ExecutionStatus.EXECUTED:
                self._update_performance_metrics(execution_result)
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Signal processing failed: {str(e)}")
            return ExecutionResult(
                signal_id=str(id(signal)),
                execution_id=execution_id,
                timestamp=datetime.now(),
                status=ExecutionStatus.FAILED,
                symbol=signal.symbol,
                order_type=signal.signal_type.value,
                requested_size=0.0,
                executed_size=0.0,
                requested_price=signal.price,
                executed_price=0.0,
                error_message=str(e)
            )
    
    def _validate_signal(self, signal: Signal) -> Dict[str, Any]:
        """Validate trading signal"""
        try:
            # Check if engine is active
            if not self.engine_active:
                return {'valid': False, 'reason': 'Execution engine not active'}
            
            # Check signal age (reject old signals)
            signal_age = datetime.now() - signal.timestamp
            # Use configurable threshold with 300s default
            max_age = self.execution_config.get('signal_age_threshold', 300)
            if signal_age.total_seconds() > max_age:  
                return {'valid': False, 'reason': f'Signal too old ({signal_age.total_seconds()}s > {max_age}s threshold)'}
            
            # Check minimum confidence
            min_confidence = 0.6
            if signal.confidence < min_confidence:
                return {'valid': False, 'reason': f'Confidence {signal.confidence:.2f} below minimum {min_confidence}'}
            
            # Check if symbol is valid
            if not signal.symbol or signal.symbol == "":
                return {'valid': False, 'reason': 'Invalid symbol'}
            
            # Check if price is reasonable
            if signal.price <= 0:
                return {'valid': False, 'reason': 'Invalid price'}
            
            # Check stop loss and take profit
            if signal.stop_loss and signal.signal_type == SignalType.BUY:
                if signal.stop_loss >= signal.price:
                    return {'valid': False, 'reason': 'Invalid stop loss for BUY signal'}
            elif signal.stop_loss and signal.signal_type == SignalType.SELL:
                if signal.stop_loss <= signal.price:
                    return {'valid': False, 'reason': 'Invalid stop loss for SELL signal'}
            
            # Check market hours (basic check)
            current_hour = datetime.now().hour
            if current_hour in [23]:  # Avoid problematic hours
                return {'valid': False, 'reason': 'Market hours restriction'}
            
            return {'valid': True, 'reason': 'Signal validated successfully'}
            
        except Exception as e:
            self.logger.error(f"Signal validation failed: {str(e)}")
            return {'valid': False, 'reason': f'Validation error: {str(e)}'}
    
    def _execute_order(self, signal: Signal, sizing_result: Dict[str, Any], execution_id: str) -> ExecutionResult:
        """Execute the trading order"""
        try:
            with self.execution_lock:
                position_size = sizing_result['position_size']
                risk_assessment = sizing_result['risk_assessment']
                
                # Prepare order parameters
                order_params = {
                    'symbol': signal.symbol,
                    'order_type': signal.signal_type.value,
                    'volume': position_size,
                    'sl': signal.stop_loss,
                    'tp': signal.take_profit,
                    'comment': f"{signal.strategy_name}_{execution_id}",
                    'deviation': 20
                }
                
                # Execute order with retry logic
                order_result = None
                last_error = ""
                
                for attempt in range(self.retry_attempts):
                    try:
                        if signal.signal_type == SignalType.BUY:
                            order_result = self.mt5_manager.place_market_order(
                                signal.symbol, "BUY", position_size,
                                sl=signal.stop_loss, tp=signal.take_profit,
                                comment=order_params['comment']
                            )
                        elif signal.signal_type == SignalType.SELL:
                            order_result = self.mt5_manager.place_market_order(
                                signal.symbol, "SELL", position_size,
                                sl=signal.stop_loss, tp=signal.take_profit,
                                comment=order_params['comment']
                            )
                        
                        if order_result and order_result.get('success', False):
                            break
                        else:
                            if order_result:
                                error_comment = order_result.get('comment', '')
                                retcode = order_result.get('retcode', 0)
                                
                                # Check for specific AutoTrading disabled error
                                if ('AutoTrading disabled' in error_comment or 
                                    retcode in [10027, 10018, 0] or 
                                    'disabled' in error_comment.lower()):
                                    last_error = "AutoTrading disabled by client - enable AutoTrading in MT5"
                                else:
                                    last_error = error_comment or f'MT5 Error Code: {retcode}' or 'Unknown error'
                            else:
                                last_error = 'No response from MT5'
                            
                    except Exception as e:
                        last_error = str(e)
                        self.logger.warning(f"Order execution attempt {attempt + 1} failed: {last_error}")
                    
                    if attempt < self.retry_attempts - 1:
                        # Exponential backoff with 60s cap
                        delay = min(self.retry_delay * (2 ** attempt), 60)
                        time.sleep(delay)
                
                # Process execution result
                if order_result and order_result.get('success', False):
                    # Calculate slippage
                    executed_price = order_result.get('price', signal.price)
                    slippage = abs(executed_price - signal.price)
                    
                    # Check slippage limit
                    if slippage > self.max_slippage:
                        self.logger.warning(f"High slippage: {slippage:.1f} pips (limit: {self.max_slippage})")
                    
                    # Create position info
                    ticket = order_result.get('ticket', 0)
                    if ticket > 0:
                        position_info = PositionInfo(
                            ticket=ticket,
                            symbol=signal.symbol,
                            order_type=signal.signal_type.value,
                            volume=position_size,
                            entry_price=executed_price,
                            current_price=executed_price,
                            unrealized_pnl=0.0,
                            unrealized_pnl_pct=(order_result.get('profit', 0.0) /
                                              (position_size * executed_price * 100)) * 100,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            strategy=signal.strategy_name,
                            confidence=signal.confidence,
                            entry_time=datetime.now(),
                            status=PositionStatus.OPEN,
                            initial_risk=risk_assessment.get('monetary_risk', 0),
                            current_risk=risk_assessment.get('monetary_risk', 0),
                            time_in_position=timedelta(0)
                        )
                        
                        # Store position
                        self.active_positions[ticket] = position_info
                    # Create successful execution result
                    execution_result = ExecutionResult(
                        signal_id=str(id(signal)),
                        execution_id=execution_id,
                        timestamp=datetime.now(),
                        status=ExecutionStatus.EXECUTED,
                        symbol=signal.symbol,
                        order_type=signal.signal_type.value,
                        requested_size=position_size,
                        executed_size=position_size,
                        requested_price=signal.price,
                        executed_price=executed_price,
                        ticket=ticket,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        slippage=slippage,
                        strategy=signal.strategy_name,
                        confidence=signal.confidence,
                        risk_assessment=risk_assessment
                    )
                    
                    self.logger.info(f"Order executed successfully: Ticket {ticket}, "
                                   f"Price {executed_price}, Slippage {slippage:.1f}")
                    
                    return execution_result
                else:
                    # Execution failed
                    execution_result = ExecutionResult(
                        signal_id=str(id(signal)),
                        execution_id=execution_id,
                        timestamp=datetime.now(),
                        status=ExecutionStatus.FAILED,
                        symbol=signal.symbol,
                        order_type=signal.signal_type.value,
                        requested_size=position_size,
                        executed_size=0.0,
                        requested_price=signal.price,
                        executed_price=0.0,
                        strategy=signal.strategy_name,
                        confidence=signal.confidence,
                        risk_assessment=risk_assessment,
                        error_message=last_error
                    )
                    
                    self.logger.error(f"Order execution failed after {self.retry_attempts} attempts: {last_error}")
                    
                    return execution_result
        except Exception as e:
            self.logger.error(f"Order execution error: {str(e)}")
            return ExecutionResult(
                signal_id=str(id(signal)),
                execution_id=execution_id,
                timestamp=datetime.now(),
                status=ExecutionStatus.FAILED,
                symbol=signal.symbol,
                order_type=signal.signal_type.value,
                requested_size=0.0,
                executed_size=0.0,
                requested_price=signal.price,
                executed_price=0.0,
                error_message=str(e)
            )
    
    def _load_existing_positions(self) -> None:
        """Load existing open positions from MT5 (only system-created positions)"""
        try:
            if not self.mt5_manager:
                return
            
            open_positions = self.mt5_manager.get_open_positions()
            system_magic = getattr(self.mt5_manager, 'magic_number', 123456)
            
            for pos in open_positions:
                try:
                    ticket = pos.get('ticket', 0)
                    magic = pos.get('magic', 0)
                    
                    # Only load positions created by this system (matching magic number)
                    if ticket > 0 and magic == system_magic:
                        position_info = PositionInfo(
                            ticket=ticket,
                            symbol=pos.get('symbol', ''),
                            order_type=pos.get('type', ''),
                            volume=pos.get('volume', 0.0),
                            entry_price=pos.get('price_open', 0.0),
                            current_price=pos.get('price_current', 0.0),
                            unrealized_pnl=pos.get('profit', 0.0),
                            unrealized_pnl_pct=(pos.get('profit', 0.0) / 
                                              (pos.get('volume', 1) * pos.get('price_open', 1) * 100)) * 100,
                            stop_loss=pos.get('sl') if pos.get('sl', 0) != 0 else None,
                            take_profit=pos.get('tp') if pos.get('tp', 0) != 0 else None,
                            strategy=self._extract_strategy_from_comment(pos.get('comment', '')),
                            confidence=0.5,  # Default for existing positions
                            entry_time=pos.get('time', datetime.now()),
                            status=PositionStatus.OPEN,
                            initial_risk=abs(pos.get('profit', 0.0)),
                            current_risk=abs(pos.get('profit', 0.0)),
                            time_in_position=datetime.now() - pos.get('time', datetime.now())
                        )
                        
                        self.active_positions[ticket] = position_info
                except Exception as e:
                    self.logger.error(f"Error loading position {pos}: {str(e)}")
            
            self.logger.info(f"Loaded {len(self.active_positions)} system-managed positions (magic: {getattr(self.mt5_manager, 'magic_number', 123456)})")
        except Exception as e:
            self.logger.error(f"Failed to load existing positions: {str(e)}")
    
    def _extract_strategy_from_comment(self, comment: str) -> str:
        """Extract strategy name from position comment"""
        try:
            if '_' in comment:
                return comment.split('_')[0]
            return comment if comment else 'unknown'
        except:
            return 'unknown'
    
    def _start_position_monitoring(self) -> None:
        """Start position monitoring thread"""
        try:
            with self.monitoring_lock:
                if not self.monitoring_active:
                    self.monitoring_active = True
                    self.monitor_thread = threading.Thread(
                        target=self._position_monitoring_loop,
                        name="PositionMonitor",
                        daemon=True
                    )
                    self.monitor_thread.start()
                    self.logger.info("Position monitoring started")
        except Exception as e:
            self.logger.error(f"Failed to start position monitoring: {str(e)}")
    
    def _position_monitoring_loop(self) -> None:
        """Main position monitoring loop"""
        restart_count = 0
        max_restarts = 5
        
        while restart_count < max_restarts and self.engine_active:
            try:
                self.monitoring_active = True
                while self.monitoring_active and self.engine_active:
                    # Update position information
                    self._update_positions()
                    
                    # Check for stop loss and take profit triggers
                    self._check_exit_conditions()
                    
                    # Check for partial profit taking
                    self._check_partial_exits()
                    
                    # Update risk metrics
                    self._update_position_risks()
                    
                    # Sleep for next iteration
                    time.sleep(5)
            except Exception as e:
                restart_count += 1
                backoff = min(2 ** restart_count, 30)  # Exponential backoff max 30s
                self.logger.critical(f"Monitoring thread crashed (restart {restart_count}/{max_restarts}): {str(e)}")
                time.sleep(backoff)
    
    def _update_positions(self) -> None:
        """Update position information from MT5"""
        try:
            # Check if MT5 is connected and available
            if not self.mt5_manager or not hasattr(self.mt5_manager, 'get_open_positions'):
                # In test mode, just update time in position for existing positions
                for position_info in self.active_positions.values():
                    position_info.time_in_position = datetime.now() - position_info.entry_time
                return
            
            # Try to get positions from MT5
            try:
                current_positions = self.mt5_manager.get_open_positions()
                system_magic = getattr(self.mt5_manager, 'magic_number', 123456)
                
                # Filter to only system-created positions
                system_positions = [pos for pos in current_positions 
                                  if pos.get('magic', 0) == system_magic]
                current_tickets = {pos.get('ticket', 0) for pos in system_positions}
            except Exception as e:
                if "Not connected to MT5" in str(e):
                    # In test mode, just update time in position for existing positions
                    for position_info in self.active_positions.values():
                        position_info.time_in_position = datetime.now() - position_info.entry_time
                    return
                else:
                    raise e
            
            # Update existing positions (only system positions)
            for pos in system_positions:
                ticket = pos.get('ticket', 0)
                if ticket in self.active_positions:
                    position_info = self.active_positions[ticket]
                    
                    # Update current data
                    position_info.current_price = pos.get('price_current', position_info.current_price)
                    position_info.unrealized_pnl = pos.get('profit', 0.0)
                    
                    # Calculate unrealized P&L percentage
                    if position_info.volume > 0 and position_info.entry_price > 0:
                        position_value = position_info.volume * position_info.entry_price * 100
                        position_info.unrealized_pnl_pct = (position_info.unrealized_pnl / position_value) * 100
                    
                    # Update time in position
                    position_info.time_in_position = datetime.now() - position_info.entry_time
            
            # Remove closed positions
            closed_tickets = set(self.active_positions.keys()) - current_tickets
            for ticket in closed_tickets:
                if ticket in self.active_positions:
                    closed_position = self.active_positions[ticket]
                    
                    # Log position closure
                    self.logger.info(f"Position closed: {ticket}, P&L: ${closed_position.unrealized_pnl:.2f}")
                    
                    # Update risk manager
                    if hasattr(self.risk_manager, 'update_position_closed'):
                        self.risk_manager.update_position_closed({
                            'profit': closed_position.unrealized_pnl,
                            'symbol': closed_position.symbol,
                            'strategy': closed_position.strategy
                        })
                    
                    # Store in database
                    if self.database_manager:
                        self._log_position_closure(closed_position)
                    
                    # Update performance metrics
                    self.total_trades += 1
                    if closed_position.unrealized_pnl > 0:
                        self.winning_trades += 1
                    self.total_profit += closed_position.unrealized_pnl
                    
                    # Remove from active positions
                    del self.active_positions[ticket]
        except Exception as e:
            self.logger.error(f"Position update failed: {str(e)}")
    
    def _check_exit_conditions(self) -> None:
        """Check for stop loss and take profit triggers"""
        try:
            for ticket, position in list(self.active_positions.items()):
                current_price = position.current_price
                
                # Check stop loss
                if position.stop_loss:
                    should_close = False
                    
                    if position.order_type == "BUY" and current_price <= position.stop_loss:
                        should_close = True
                        reason = "Stop Loss"
                    elif position.order_type == "SELL" and current_price >= position.stop_loss:
                        should_close = True
                        reason = "Stop Loss"
                    
                    if should_close:
                        self._close_position(ticket, reason)
                        continue
                
                # Check take profit
                if position.take_profit:
                    should_close = False
                    
                    if position.order_type == "BUY" and current_price >= position.take_profit:
                        should_close = True
                        reason = "Take Profit"
                    elif position.order_type == "SELL" and current_price <= position.take_profit:
                        should_close = True
                        reason = "Take Profit"
                    
                    if should_close:
                        self._close_position(ticket, reason)
                        continue
                
                # Check maximum time in position (24 hours for aggressive trading)
                if position.time_in_position.total_seconds() > 86400:  # 24 hours
                    self._close_position(ticket, "Time Limit")
                    continue
        except Exception as e:
            self.logger.error(f"Exit condition check failed: {str(e)}")
    
    def _check_partial_exits(self) -> None:
        """Check for partial profit taking opportunities"""
        try:
            # Get configuration with defaults
            thresholds = self.config.get('partial_exits', {}).get('profit_thresholds', [2.0, 4.0, 3.0])
            percentages = self.config.get('partial_exits', {}).get('close_percentages', [0.25, 0.25])
            min_size = self.config.get('partial_exits', {}).get('min_position_size', 0.02)
            
            for ticket, position in list(self.active_positions.items()):
                # Only for profitable positions
                if position.unrealized_pnl <= 0:
                    continue
                
                # Check if position is large enough for partial exit
                if position.volume < min_size:
                    continue
                
                # Partial exit rules based on profit
                profit_pct = position.unrealized_pnl_pct
                
                # Close first partial at first threshold
                if profit_pct >= thresholds[0] and not hasattr(position, 'partial_1_closed'):
                    self._partial_close_position(ticket, percentages[0], f"Partial 1: {thresholds[0]}% profit")
                    position.partial_1_closed = True
                
                # Close second partial at second threshold
                elif profit_pct >= thresholds[1] and not hasattr(position, 'partial_2_closed'):
                    self._partial_close_position(ticket, percentages[1], f"Partial 2: {thresholds[1]}% profit")
                    position.partial_2_closed = True
                
                # Move stop to breakeven at third threshold
                elif profit_pct >= thresholds[2] and not hasattr(position, 'breakeven_set'):
                    self._move_stop_to_breakeven(ticket)
                    position.breakeven_set = True
        except Exception as e:
            self.logger.error(f"Partial exit check failed: {str(e)}")
    
    def _update_position_risks(self) -> None:
        """Update risk metrics for active positions"""
        try:
            for position in self.active_positions.values():
                # Update current risk based on stop loss
                if position.stop_loss:
                    if position.order_type == "BUY":
                        risk_per_unit = position.current_price - position.stop_loss
                    else:
                        risk_per_unit = position.stop_loss - position.current_price
                    
                    position.current_risk = risk_per_unit * position.volume * 100
                else:
                    # Estimate risk as 2% of position value
                    position_value = position.volume * position.current_price * 100
                    position.current_risk = position_value * 0.02
        except Exception as e:
            self.logger.error(f"Position risk update failed: {str(e)}")
    
    def _close_position(self, ticket: int, reason: str) -> bool:
        """Close a position"""
        try:
            if ticket not in self.active_positions:
                return False
            
            position = self.active_positions[ticket]
            
            # Execute close order
            close_result = self.mt5_manager.close_position(ticket)
            
            if close_result and close_result.get('success', False):
                self.logger.info(f"Position {ticket} closed: {reason}")
                return True
            else:
                self.logger.error(f"Failed to close position {ticket}: {close_result.get('comment', 'Unknown error')}")
                return False
        except Exception as e:
            self.logger.error(f"Position close error: {str(e)}")
            return False
    
    def _partial_close_position(self, ticket: int, percentage: float, reason: str) -> bool:
        """Partially close a position"""
        try:
            if ticket not in self.active_positions:
                return False
            
            position = self.active_positions[ticket]
            close_volume = position.volume * percentage
            
            # Execute partial close
            close_result = self.mt5_manager.close_position(ticket, close_volume)
            
            if close_result and close_result.get('success', False):
                self.logger.info(f"Position {ticket} partially closed ({percentage:.0%}): {reason}")
                
                # Update position volume
                position.volume -= close_volume
                
                return True
            else:
                self.logger.error(f"Failed to partially close position {ticket}")
                return False
        except Exception as e:
            self.logger.error(f"Partial close error: {str(e)}")
            return False
    
    def _move_stop_to_breakeven(self, ticket: int) -> bool:
        """Move stop loss to breakeven"""
        try:
            if ticket not in self.active_positions:
                return False
            
            position = self.active_positions[ticket]
            new_stop = position.entry_price
            
            # Execute stop loss modification
            modify_result = self.mt5_manager.modify_position(ticket, sl=new_stop)
            
            if modify_result and modify_result.get('success', False):
                position.stop_loss = new_stop
                self.logger.info(f"Position {ticket} stop moved to breakeven: {new_stop}")
                return True
            else:
                self.logger.error(f"Failed to move stop to breakeven for position {ticket}")
                return False
        except Exception as e:
            self.logger.error(f"Breakeven stop error: {str(e)}")
            return False
    
    def _log_signal_execution(self, signal: Signal, execution_result: ExecutionResult) -> None:
        """Log signal execution to database"""
        try:
            if not self.database_manager:
                return
            
            signal_data = {
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'strategy': signal.strategy_name,
                'signal_type': signal.signal_type.value,
                'confidence': signal.confidence,
                'price': signal.price,
                'timeframe': signal.timeframe,
                'strength': signal.strength,
                'quality_grade': signal.grade.value if signal.grade else 'C',
                'executed': execution_result.status == ExecutionStatus.EXECUTED,
                'execution_price': execution_result.executed_price,
                'execution_time': execution_result.timestamp,
                'metadata': json.dumps({
                    'execution_id': execution_result.execution_id,
                    'ticket': execution_result.ticket,
                    'slippage': execution_result.slippage,
                    'risk_assessment': execution_result.risk_assessment
                })
            }
            
            self.database_manager.store_signal(signal_data)
        except Exception as e:
            self.logger.error(f"Signal logging failed: {str(e)}")
    
    def _log_position_closure(self, position: PositionInfo) -> None:
        """Log position closure to database"""
        try:
            if not self.database_manager:
                return
            
            trade_data = {
                'ticket': position.ticket,
                'symbol': position.symbol,
                'action': position.order_type,
                'volume': position.volume,
                'price_open': position.entry_price,
                'price_close': position.current_price,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'profit': position.unrealized_pnl,
                'strategy': position.strategy,
                'timeframe': 'M15',  # Default
                'open_time': position.entry_time,
                'close_time': datetime.now(),
                'status': 'CLOSED'
            }
            
            self.database_manager.store_trade(trade_data)
        except Exception as e:
            self.logger.error(f"Trade logging failed: {str(e)}")
    
    def _update_performance_metrics(self, execution_result: ExecutionResult) -> None:
        """Update performance tracking metrics"""
        try:
            self.total_trades += 1
            
            # Log execution in logger manager
            if self.logger_manager:
                self.logger_manager.log_trade(
                    execution_result.order_type,
                    execution_result.symbol,
                    execution_result.executed_size,
                    execution_result.executed_price,
                    ticket=execution_result.ticket,
                    sl=execution_result.stop_loss,
                    tp=execution_result.take_profit,
                    strategy=execution_result.strategy,
                    confidence=execution_result.confidence
                )
        except Exception as e:
            self.logger.error(f"Performance metrics update failed: {str(e)}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'engine_status': {
                    'active': self.engine_active,
                    'monitoring': self.monitoring_active
                },
                'positions': {
                    'active_count': len(self.active_positions),
                    'total_exposure': sum(pos.volume * pos.current_price * 100 
                                        for pos in self.active_positions.values()),
                    'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.active_positions.values()),
                    'positions': [
                        {
                            'ticket': pos.ticket,
                            'symbol': pos.symbol,
                            'type': pos.order_type,
                            'volume': pos.volume,
                            'entry_price': pos.entry_price,
                            'current_price': pos.current_price,
                            'pnl': pos.unrealized_pnl,
                            'pnl_pct': pos.unrealized_pnl_pct,
                            'strategy': pos.strategy,
                            'time_in_position': str(pos.time_in_position)
                        }
                        for pos in self.active_positions.values()
                    ]
                },
                'performance': {
                    'total_trades': self.total_trades,
                    'winning_trades': self.winning_trades,
                    'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
                    'total_profit': self.total_profit,
                    'total_commission': self.total_commission
                },
                'execution_history': {
                    'total_executions': len(self.execution_history),
                    'successful_executions': len([e for e in self.execution_history 
                                                 if e.status == ExecutionStatus.EXECUTED]),
                    'recent_executions': [
                        {
                            'timestamp': e.timestamp.isoformat(),
                            'symbol': e.symbol,
                            'type': e.order_type,
                            'size': e.executed_size,
                            'price': e.executed_price,
                            'status': e.status.value,
                            'strategy': e.strategy
                        }
                        for e in self.execution_history[-10:]  # Last 10 executions
                    ]
                },
                'settings': {
                    'max_slippage': self.max_slippage,
                    'retry_attempts': self.retry_attempts,
                    'retry_delay': self.retry_delay
                }
            }
        except Exception as e:
            self.logger.error(f"Execution summary generation failed: {str(e)}")
            return {'error': str(e)}
    
    def stop_engine(self) -> None:
        """Stop the execution engine"""
        try:
            self.engine_active = False
            self.monitoring_active = False
            
            # Wait for monitoring thread to stop
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)
            
            self.logger.info("Execution engine stopped")
        except Exception as e:
            self.logger.error(f"Engine stop error: {str(e)}")
    
    def emergency_close_all(self) -> Dict[str, Any]:
        """Emergency close all positions with rollback capability"""
        try:
            # Create position snapshot for rollback
            position_snapshot = list(self.active_positions.values())
            
            close_results = {
                'total': len(position_snapshot),
                'successful': [],
                'failed': []
            }
            
            # Attempt to close all positions
            for position in position_snapshot:
                try:
                    result = self.mt5_manager.close_position(position.ticket)
                    if result.get('success', False):
                        close_results['successful'].append(position.ticket)
                        del self.active_positions[position.ticket]
                    else:
                        close_results['failed'].append({
                            'ticket': position.ticket,
                            'error': result.get('comment', 'Unknown error')
                        })
                except Exception as e:
                    close_results['failed'].append({
                        'ticket': position.ticket,
                        'error': str(e)
                    })
            
            # Rollback if any failures occurred
            if close_results['failed']:
                self.logger.critical("Emergency close partially failed - attempting rollback")
                for ticket in close_results['successful']:
                    # Re-open positions that were incorrectly closed
                    # This would be broker-specific in live environment
                    self.logger.warning(f"Re-opening position {ticket} after failed emergency close")
                    # In mock mode, we just restore the position object
                    if self.mode == 'mock':
                        original_position = next(p for p in position_snapshot if p.ticket == ticket)
                        self.active_positions[ticket] = original_position
            
            return close_results
        except Exception as e:
            self.logger.error(f"Emergency close failed: {str(e)}")
            return {'error': str(e)}
    
    def process_pending_trades(self):
        """Process pending trades and manage positions"""
        try:
            # Update position information
            self._update_positions()
            
            # Check for stop loss and take profit triggers
            self._check_exit_conditions()
            
            # Check for partial profit taking
            self._check_partial_exits()
            
            # Update risk metrics
            self._update_position_risks()
            
        except Exception as e:
            self.logger.error(f"Error processing pending trades: {e}")
    
    def execute_signal(self, signal, position_info: Dict) -> ExecutionResult:
        """Execute a trading signal"""
        try:
            execution_id = f"EXEC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(signal)}"
            
            # Create execution result
            result = ExecutionResult(
                signal_id=str(id(signal)),
                execution_id=execution_id,
                timestamp=datetime.now(),
                status=ExecutionStatus.EXECUTED,
                symbol=signal.symbol,
                order_type=signal.signal_type.value,
                requested_size=position_info.get('position_size', 0.01),
                executed_size=position_info.get('position_size', 0.01),
                requested_price=signal.price,
                executed_price=signal.price,
                ticket=position_info.get('ticket', 0),
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy=signal.strategy_name,
                confidence=signal.confidence
            )
            
            # Add to execution history
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Signal execution error: {e}")
            return ExecutionResult(
                signal_id=str(id(signal)),
                execution_id=f"EXEC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(signal)}",
                timestamp=datetime.now(),
                status=ExecutionStatus.FAILED,
                symbol=signal.symbol,
                order_type=signal.signal_type.value,
                requested_size=0.0,
                executed_size=0.0,
                requested_price=signal.price,
                executed_price=0.0,
                error_message=str(e)
            )
    
    def simulate_trade(self, signal, position_info: Dict) -> ExecutionResult:
        """Simulate a trade for paper trading"""
        try:
            execution_id = f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(signal)}"
            
            # Simulate slippage
            slippage = np.random.uniform(-0.0001, 0.0001)
            executed_price = signal.price * (1 + slippage)
            
            result = ExecutionResult(
                signal_id=str(id(signal)),
                execution_id=execution_id,
                timestamp=datetime.now(),
                status=ExecutionStatus.EXECUTED,
                symbol=signal.symbol,
                order_type=signal.signal_type.value,
                requested_size=position_info.get('position_size', 0.01),
                executed_size=position_info.get('position_size', 0.01),
                requested_price=signal.price,
                executed_price=executed_price,
                slippage=slippage,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy=signal.strategy_name,
                confidence=signal.confidence
            )
            
            # Add to execution history
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Trade simulation error: {e}")
            return ExecutionResult(
                signal_id=str(id(signal)),
                execution_id=f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(signal)}",
                timestamp=datetime.now(),
                status=ExecutionStatus.FAILED,
                symbol=signal.symbol,
                order_type=signal.signal_type.value,
                requested_size=0.0,
                executed_size=0.0,
                requested_price=signal.price,
                executed_price=0.0,
                error_message=str(e)
            )
    
    def backtest_trade(self, signal, position_info: Dict) -> ExecutionResult:
        """Execute a backtest trade"""
        try:
            execution_id = f"BT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(signal)}"
            
            result = ExecutionResult(
                signal_id=str(id(signal)),
                execution_id=execution_id,
                timestamp=datetime.now(),
                status=ExecutionStatus.EXECUTED,
                symbol=signal.symbol,
                order_type=signal.signal_type.value,
                requested_size=position_info.get('position_size', 0.01),
                executed_size=position_info.get('position_size', 0.01),
                requested_price=signal.price,
                executed_price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy=signal.strategy_name,
                confidence=signal.confidence
            )
            
            # Add to execution history
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest trade error: {e}")
            return ExecutionResult(
                signal_id=str(id(signal)),
                execution_id=f"BT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(signal)}",
                timestamp=datetime.now(),
                status=ExecutionStatus.FAILED,
                symbol=signal.symbol,
                order_type=signal.signal_type.value,
                requested_size=0.0,
                executed_size=0.0,
                requested_price=signal.price,
                executed_price=0.0,
                error_message=str(e)
            )
    
    def close_all_positions(self):
        """Close all open positions"""
        try:
            for ticket, position in list(self.active_positions.items()):
                self._close_position(ticket, "System shutdown")
            self.logger.info("All positions closed")
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")


# Testing function
if __name__ == "__main__":
    """Test the Execution Engine"""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Import CLI utilities
    from src.utils.cli_args import parse_mode, print_mode_banner
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    
    # Parse CLI arguments
    mode = parse_mode()
    print_mode_banner(mode)
    
    # Test configuration
    test_config = {
        'execution': {
            'order': {
                'retry_attempts': 3,
                'retry_delay': 1
            },
            'slippage': {
                'max_slippage': 3
            },
            'signal_age_threshold': 300  # Added signal age threshold
        },
        'mode': mode # Pass the determined mode to the engine config
    }
    
    # Mock components for testing: These are the *external* mocks provided to the test harness
    class MockRiskManager: # This mock is passed to the ExecutionEngine
        def calculate_position_size(self, signal, balance, positions):
            return {
                'allowed': True,
                'position_size': 0.02,
                'risk_assessment': {
                    'monetary_risk': 20.0,
                    'risk_percentage': 0.02,
                    'expected_return': 0.05,
                    'expected_return_std': 0.01
                }
            }
        
        def update_position_closed(self, trade_result):
            pass
    
    class MockDatabaseManager: # This mock is passed to the ExecutionEngine
        def store_signal(self, signal_data):
            pass
        
        def store_trade(self, trade_data):
            pass
        # get_trades is used by RiskManager if it initializes its own DB, not directly by ExecutionEngine
        def get_trades(self, limit=1000): 
            return [] 
    
    class MockLoggerManager: # This mock is passed to the ExecutionEngine
        def log_trade(self, action, symbol, volume, price, **kwargs):
            pass
    
    # Create ExecutionEngine instance. It will now handle MT5 initialization internally.
    execution_engine = ExecutionEngine(
        test_config, 
        mt5_manager=None, # Let ExecutionEngine create its own MT5 based on mode
        risk_manager=MockRiskManager(), 
        database_manager=MockDatabaseManager(), 
        logger_manager=MockLoggerManager()
    )
    
    logger.info("Execution Engine Initialized. Running tests...")

    # Test signal processing
    if mode == 'live':
        # Use live signal engine for real signals
        try:
            from src.core.signal_engine import SignalEngine
            signal_engine = SignalEngine(test_config)
            live_signals = signal_engine.generate_signals()
            
            if live_signals:
                signal = live_signals[0]  # Use first live signal
                logger.info(f"Using live signal: {signal.strategy_name} {signal.signal_type.value} @ {signal.price}")
            else:
                logger.info("No live signals generated, skipping execution test")
                signal = None
        except Exception as e:
            logger.error(f"Failed to generate live signals: {e}, using mock signal")
            signal = None
    
    if mode != 'live' or signal is None:
        # Create mock signal for testing
        @dataclass
        class MockSignal:
            timestamp: datetime = datetime.now()
            symbol: str = "XAUUSDm"
            strategy_name: str = "test_strategy"
            signal_type: SignalType = SignalType.BUY
            confidence: float = 0.85
            price: float = 1960.0
            timeframe: str = "M15"
            strength: float = 0.8
            grade: SignalGrade = SignalGrade.A
            stop_loss: float = 1950.0
            take_profit: float = 1980.0
            metadata: Dict[str, Any] = field(default_factory=dict)
        
        signal = MockSignal()
        logger.info(f"Using mock signal: {signal.strategy_name} {signal.signal_type.value} @ {signal.price}")

    # Process the signal
    execution_result = execution_engine.process_signal(signal) if signal else None
    
    if execution_result:
        logger.info("\nExecution Result:")
        logger.info(f"Status: {execution_result.status.value}")
        logger.info(f"Ticket: {execution_result.ticket}")
        logger.info(f"Executed Price: {execution_result.executed_price}")
        logger.info(f"Slippage: {execution_result.slippage}")
    else:
        logger.info("\nNo signal processed")
    
    # Test execution summary
    summary = execution_engine.get_execution_summary()
    logger.info(f"\nExecution Summary:")
    logger.info(f"Active Positions: {summary['positions']['active_count']}")
    logger.info(f"Total Trades: {summary['performance']['total_trades']}")
    logger.info(f"Engine Active: {summary['engine_status']['active']}")
    
    # Test emergency close
    emergency_result = execution_engine.emergency_close_all()
    logger.info(f"\nEmergency Close Result:")
    logger.info(f"Total Positions: {emergency_result.get('total_positions', 0)}")
    logger.info(f"Successful Closes: {emergency_result.get('successful_closes', 0)}")
    
    # Stop engine
    execution_engine.stop_engine()
    
    logger.info("\nExecution Engine test completed!")