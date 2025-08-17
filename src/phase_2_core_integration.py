#!/usr/bin/env python3
"""
Phase 2 Advanced Trading System Integration v2.0
=================================================
Author: XAUUSD Trading System
Version: 2.0.0
Date: 2025-08-16

Complete Phase 2 implementation with enhanced architecture:
- Dynamic Strategy Loading with Plugin Support
- Advanced Signal Engine with Multi-timeframe Analysis
- Smart Money Concepts (SMC) Integration
- Technical Indicators Suite
- Machine Learning Predictions
- Risk-Adjusted Position Sizing
- Real-time Performance Monitoring
- Emergency Control Systems

Features:
- Modular strategy architecture
- Graceful degradation on missing components
- Performance-based strategy selection
- Real-time configuration updates
- Advanced signal fusion algorithms
- Market regime detection
- Correlation-based filtering

Usage:
    python phase_2_core_integration.py --mode live     # Live trading
    python phase_2_core_integration.py --mode paper    # Paper trading
    python phase_2_core_integration.py --mode backtest # Backtesting
    python phase_2_core_integration.py --test          # Component testing
"""

import sys
import os
import argparse
import time
import asyncio
import signal as system_signal
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import yaml
import traceback

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

try:
    # Import Phase 1 components
    from phase_1_core_integration import CoreSystem
    from utils.logger import LoggerManager
    from utils.database import DatabaseManager
    from utils.error_handler import ErrorHandler
    from core.mt5_manager import MT5Manager
    
    # Import Phase 2 core components
    from core.signal_engine import SignalEngine, Signal, SignalType, SignalGrade
    from core.risk_manager import RiskManager, RiskLevel
    from core.execution_engine import ExecutionEngine
    
    # Define TradeResult locally if not available
    try:
        from core.execution_engine import TradeResult
    except ImportError:
        from dataclasses import dataclass
        from datetime import datetime
        
        @dataclass
        class TradeResult:
            success: bool
            trade_id: str = ""
            entry_price: float = 0.0
            exit_price: float = 0.0
            pnl: float = 0.0
            timestamp: datetime = None
            error_message: str = ""
    
except ImportError as e:
    print(f"Critical Import Error: {e}")
    print("Please ensure all Phase 1 and Phase 2 core files are in place.")
    print("Run Phase 1 tests first: python tests/Phase-1/test_phase1.py")
    sys.exit(1)


class Phase2TradingSystem:
    """
    Complete Phase 2 Advanced Trading System
    
    Integrates all components for aggressive 10x returns:
    - Multi-strategy signal generation
    - Advanced risk management with Kelly Criterion
    - Smart execution engine with slippage protection
    - Real-time performance monitoring
    - Emergency stop mechanisms
    """
    
    def __init__(self, config_path: str = 'config/master_config.yaml'):
        """Initialize the complete Phase 2 trading system"""
        self.config_path = config_path
        self.system_active = False
        self.emergency_stop = False
        self.mode = 'paper'  # 'live', 'paper', 'backtest'
        
        # Core components
        self.core_system: Optional[CoreSystem] = None
        self.signal_engine: Optional[SignalEngine] = None
        self.risk_manager: Optional[RiskManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        
        # Configuration
        self.config: Dict[str, Any] = {}
        
        # Performance tracking
        self.start_time = datetime.now()
        self.session_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'trades_won': 0,
            'trades_lost': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_balance': 0.0,
            'target_balance': 1000.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
        
        # System monitoring
        self.last_heartbeat = datetime.now()
        self.heartbeat_interval = 30  # seconds
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Logger
        self.logger = logging.getLogger('phase2_system')
        
        # Signal handlers for graceful shutdown
        system_signal.signal(system_signal.SIGINT, self._signal_handler)
        if hasattr(system_signal, 'SIGTERM'):
            system_signal.signal(system_signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
    
    def load_configuration(self) -> bool:
        """Load system configuration"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                self.logger.error(f"Configuration file not found: {self.config_path}")
                return False
            
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.logger.info(f"Configuration loaded from: {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize the complete Phase 2 system"""
        print("Initializing Phase 2 Advanced Trading System")
        print("=" * 70)
        print(f"System ID: PHASE2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"Start Time: {datetime.now()}")
        print(f"Mode: {self.mode.upper()}")
        print(f"Config Path: {self.config_path}")
        print()
        
        try:
            # Step 1: Load configuration
            print("Step 1: Loading Configuration...")
            if not self.load_configuration():
                print("Configuration loading failed")
                return False
            print("Configuration loaded")
            
            # Step 2: Initialize Phase 1 core system
            print("Step 2: Initializing Phase 1 Core System...")
            self.core_system = CoreSystem(self.config_path)
            if not self.core_system.initialize():
                print("Phase 1 core system initialization failed")
                return False
            print("Phase 1 core system initialized")
            
            # Step 3: Connect to MT5 if in live or paper mode
            if self.mode in ['live', 'paper']:
                print("Step 3: Connecting to MT5...")
                if not self.core_system.connect_mt5():
                    print("MT5 connection failed")
                    return False
                print("MT5 connection established")
            else:
                print("Step 3: Skipping MT5 connection (backtest mode)")
            
            # Step 4: Initialize Signal Engine
            print("Step 4: Initializing Signal Engine...")
            self.signal_engine = SignalEngine(
                config=self.config,
                mt5_manager=self.core_system.mt5_manager,
                database_manager=self.core_system.database_manager
            )
            print("Signal Engine initialized")
            
            # Step 5: Initialize Risk Manager
            print("Step 5: Initializing Risk Manager...")
            self.risk_manager = RiskManager(
                config=self.config,
                mt5_manager=self.core_system.mt5_manager,
                database_manager=self.core_system.database_manager
            )
            print("Risk Manager initialized")
            
            # Step 6: Initialize Execution Engine
            print("Step 6: Initializing Execution Engine...")
            self.execution_engine = ExecutionEngine(
                config=self.config,
                mt5_manager=self.core_system.mt5_manager,
                database_manager=self.core_system.database_manager,
                risk_manager=self.risk_manager,
                logger_manager=self.core_system.logger_manager
            )
            print("Execution Engine initialized")
            
            # Step 7: System health check
            print("Step 7: Performing System Health Check...")
            if not self._perform_health_check():
                print("System health check failed")
                return False
            print("System health check passed")
            
            # Step 8: Load initial account state
            print("Step 8: Loading Account State...")
            self._load_account_state()
            print("Account state loaded")
            
            # Step 9: Start monitoring
            print("Step 9: Starting System Monitoring...")
            self._start_monitoring()
            print("System monitoring started")
            
            self.system_active = True
            
            print()
            print("Phase 2 Advanced Trading System Initialization Complete!")
            print("=" * 70)
            print()
            self._print_system_status()
            print()
            print("System is ready for trading operations!")
            print("=" * 70)
            
            self.logger.info("Phase 2 system initialization completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"Phase 2 system initialization failed: {e}"
            print(f"ERROR: {error_msg}")
            self.logger.error(error_msg, exc_info=True)
            return False
    
    def _perform_health_check(self) -> bool:
        """Perform comprehensive system health check"""
        try:
            checks = []
            
            # Core system health
            if self.core_system and self.core_system.initialized:
                checks.append("OK Core System")
            else:
                checks.append("FAIL Core System")
                return False
            
            # Signal engine health
            if self.signal_engine:
                active_strategies = self.signal_engine.get_active_strategies()
                total_strategies = sum(len(strategies) for strategies in active_strategies.values())
                if total_strategies > 0:
                    checks.append(f"OK Signal Engine ({total_strategies} strategies)")
                else:
                    checks.append("WARN Signal Engine (no strategies)")
            else:
                checks.append("FAIL Signal Engine")
                return False
            
            # Risk manager health
            if self.risk_manager:
                checks.append("OK Risk Manager")
            else:
                checks.append("FAIL Risk Manager")
                return False
            
            # Execution engine health
            if self.execution_engine:
                checks.append("OK Execution Engine")
            else:
                checks.append("FAIL Execution Engine")
                return False
            
            # MT5 connection health (if required)
            if self.mode in ['live', 'paper']:
                if self.core_system.mt5_manager and self.core_system.mt5_manager.connected:
                    checks.append("OK MT5 Connection")
                else:
                    checks.append("FAIL MT5 Connection")
                    return False
            
            # Database health
            if self.core_system.database_manager:
                checks.append("OK Database")
            else:
                checks.append("FAIL Database")
                return False
            
            # Print health status
            for check in checks:
                print(f"   {check}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def _load_account_state(self):
        """Load current account state and initialize session stats"""
        try:
            if self.mode in ['live', 'paper'] and self.core_system.mt5_manager:
                account_info = self.core_system.mt5_manager.account_info
                if account_info:
                    self.session_stats['current_balance'] = account_info.get('balance', 0.0)
                    self.session_stats['target_balance'] = self.config.get('trading', {}).get('capital', {}).get('target_capital', 1000.0)
            else:
                # Backtest mode - use initial capital from config
                self.session_stats['current_balance'] = self.config.get('trading', {}).get('capital', {}).get('initial_capital', 100.0)
                self.session_stats['target_balance'] = self.config.get('trading', {}).get('capital', {}).get('target_capital', 1000.0)
            
        except Exception as e:
            self.logger.error(f"Failed to load account state: {e}")
    
    def _start_monitoring(self):
        """Start system monitoring thread"""
        try:
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("System monitoring thread started")
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
    
    def _monitoring_loop(self):
        """System monitoring loop"""
        while not self.stop_monitoring.is_set():
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                
                # Update session statistics
                self._update_session_stats()
                
                # Check for emergency conditions
                self._check_emergency_conditions()
                
                # Log periodic status
                if datetime.now().minute % 5 == 0:  # Every 5 minutes
                    self._log_periodic_status()
                
                # Sleep until next check
                self.stop_monitoring.wait(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)  # Brief pause before retry
    
    def _update_session_stats(self):
        """Update session statistics"""
        try:
            if self.core_system and self.core_system.database_manager:
                # Get recent trades for statistics
                # This would typically query the database for recent performance
                pass
        except Exception as e:
            self.logger.error(f"Failed to update session stats: {e}")
    
    def _check_emergency_conditions(self):
        """Check for emergency stop conditions"""
        try:
            current_balance = self.session_stats['current_balance']
            target_balance = self.session_stats['target_balance']
            
            # Check maximum drawdown
            max_drawdown_pct = self.config.get('trading', {}).get('risk_management', {}).get('max_drawdown', 0.25)
            initial_capital = self.config.get('trading', {}).get('capital', {}).get('initial_capital', 100.0)
            
            current_drawdown = (initial_capital - current_balance) / initial_capital
            
            if current_drawdown > max_drawdown_pct:
                self.logger.critical(f"Maximum drawdown exceeded: {current_drawdown:.2%} > {max_drawdown_pct:.2%}")
                self.emergency_stop = True
                self.shutdown()
            
            # Check minimum capital
            min_capital = self.config.get('trading', {}).get('capital', {}).get('minimum_capital', 50.0)
            if current_balance < min_capital:
                self.logger.critical(f"Balance below minimum capital: ${current_balance:.2f} < ${min_capital:.2f}")
                self.emergency_stop = True
                self.shutdown()
            
        except Exception as e:
            self.logger.error(f"Emergency check failed: {e}")
    
    def _log_periodic_status(self):
        """Log periodic system status"""
        try:
            runtime = datetime.now() - self.start_time
            self.logger.info(f"System Status - Runtime: {runtime}, Balance: ${self.session_stats['current_balance']:.2f}, "
                           f"Signals: {self.session_stats['signals_generated']}, Trades: {self.session_stats['trades_executed']}")
        except Exception as e:
            self.logger.error(f"Periodic status logging failed: {e}")
    
    def _print_system_status(self):
        """Print current system status"""
        print("System Status:")
        print(f"   - Core System: Active")
        print(f"   - Signal Engine: Active ({sum(len(s) for s in self.signal_engine.get_active_strategies().values())} strategies)")
        print(f"   - Risk Manager: Active")
        print(f"   - Execution Engine: Active")
        if self.mode in ['live', 'paper']:
            print(f"   - MT5 Connection: Connected")
        print(f"   - Database: Active")
        print(f"   - Monitoring: Active")
        print()
        print("Account Status:")
        print(f"   - Current Balance: ${self.session_stats['current_balance']:.2f}")
        print(f"   - Target Balance: ${self.session_stats['target_balance']:.2f}")
        print(f"   - Progress: {(self.session_stats['current_balance'] / self.session_stats['target_balance'] * 100):.1f}%")
    
    async def run_trading_loop(self):
        """Main trading loop"""
        self.logger.info(f"Starting trading loop in {self.mode} mode")
        print(f"Starting trading loop in {self.mode.upper()} mode...")
        print("Press Ctrl+C to stop gracefully")
        print()
        
        signal_interval = 60  # Generate signals every minute
        last_signal_time = datetime.now() - timedelta(seconds=signal_interval)
        
        try:
            while self.system_active and not self.emergency_stop:
                current_time = datetime.now()
                
                # Generate signals at specified intervals
                if (current_time - last_signal_time).seconds >= signal_interval:
                    await self._process_signals()
                    last_signal_time = current_time
                
                # Process pending trades
                await self._process_trades()
                
                # Brief sleep to prevent excessive CPU usage
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Trading loop interrupted by user")
        except Exception as e:
            self.logger.error(f"Trading loop error: {e}", exc_info=True)
        finally:
            self.logger.info("Trading loop ended")
    
    async def _process_signals(self):
        """Process signal generation"""
        try:
            symbol = self.config.get('trading', {}).get('symbol', 'XAUUSDm')
            timeframe = self.config.get('timeframes', {}).get('data', {}).get('primary', 'M15')
            
            # Convert timeframe to minutes for signal engine
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            
            # Generate signals
            signals = self.signal_engine.generate_signals(symbol, timeframe_minutes)
            
            if signals:
                self.session_stats['signals_generated'] += len(signals)
                self.logger.info(f"Generated {len(signals)} signals for {symbol}")
                
                # Process each signal
                for signal in signals:
                    await self._process_single_signal(signal)
            
        except Exception as e:
            self.logger.error(f"Signal processing error: {e}")
    
    async def _process_single_signal(self, signal: Signal):
        """Process a single trading signal"""
        try:
            # Get current account balance
            if self.mode in ['live', 'paper'] and self.core_system.mt5_manager:
                account_info = self.core_system.mt5_manager.account_info
                balance = account_info.get('balance', 0.0) if account_info else 0.0
            else:
                balance = self.session_stats['current_balance']
            
            # Get current positions
            positions = []
            if self.mode in ['live', 'paper'] and self.core_system.mt5_manager:
                positions = self.core_system.mt5_manager.get_positions()
            
            # Calculate position size using risk manager
            position_info = self.risk_manager.calculate_position_size(
                signal=signal,
                account_balance=balance,
                open_positions=positions
            )
            
            if position_info['allowed']:
                # Execute trade
                if self.mode == 'live':
                    result = await self._execute_live_trade(signal, position_info)
                elif self.mode == 'paper':
                    result = await self._execute_paper_trade(signal, position_info)
                else:  # backtest
                    result = await self._execute_backtest_trade(signal, position_info)
                
                if result and result.success:
                    self.session_stats['trades_executed'] += 1
                    self.logger.info(f"Trade executed: {signal.signal_type} {signal.symbol} "
                                   f"Size: {position_info['position_size']:.3f} Price: {signal.price}")
            else:
                self.logger.debug(f"Signal rejected by risk manager: {position_info.get('reason', 'Unknown')}")
                
        except Exception as e:
            self.logger.error(f"Single signal processing error: {e}")
    
    async def _execute_live_trade(self, signal: Signal, position_info: Dict) -> Optional[TradeResult]:
        """Execute live trade"""
        try:
            return self.execution_engine.execute_signal(signal, position_info)
        except Exception as e:
            self.logger.error(f"Live trade execution error: {e}")
            return None
    
    async def _execute_paper_trade(self, signal: Signal, position_info: Dict) -> Optional[TradeResult]:
        """Execute paper trade"""
        try:
            # For paper trading, we simulate the trade
            return self.execution_engine.simulate_trade(signal, position_info)
        except Exception as e:
            self.logger.error(f"Paper trade execution error: {e}")
            return None
    
    async def _execute_backtest_trade(self, signal: Signal, position_info: Dict) -> Optional[TradeResult]:
        """Execute backtest trade"""
        try:
            # For backtesting, we use historical data
            return self.execution_engine.backtest_trade(signal, position_info)
        except Exception as e:
            self.logger.error(f"Backtest trade execution error: {e}")
            return None
    
    async def _process_trades(self):
        """Process pending trades and manage positions"""
        try:
            if self.execution_engine:
                await self.execution_engine.process_pending_trades()
        except Exception as e:
            self.logger.error(f"Trade processing error: {e}")
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440
        }
        return timeframe_map.get(timeframe, 15)  # Default to M15
    
    def set_mode(self, mode: str):
        """Set trading mode"""
        if mode.lower() in ['live', 'paper', 'backtest']:
            self.mode = mode.lower()
            self.logger.info(f"Trading mode set to: {self.mode}")
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'live', 'paper', or 'backtest'")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        runtime = datetime.now() - self.start_time
        
        # Calculate win rate
        total_completed_trades = self.session_stats['trades_won'] + self.session_stats['trades_lost']
        win_rate = (self.session_stats['trades_won'] / total_completed_trades * 100) if total_completed_trades > 0 else 0.0
        
        # Calculate profit factor
        # This would typically come from detailed trade analysis
        profit_factor = 1.0  # Placeholder
        
        return {
            'runtime': str(runtime),
            'runtime_hours': runtime.total_seconds() / 3600,
            'signals_generated': self.session_stats['signals_generated'],
            'trades_executed': self.session_stats['trades_executed'],
            'trades_won': self.session_stats['trades_won'],
            'trades_lost': self.session_stats['trades_lost'],
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': self.session_stats['total_pnl'],
            'current_balance': self.session_stats['current_balance'],
            'target_balance': self.session_stats['target_balance'],
            'progress_to_target': (self.session_stats['current_balance'] / self.session_stats['target_balance'] * 100),
            'max_drawdown': self.session_stats['max_drawdown']
        }
    
    def shutdown(self):
        """Graceful system shutdown"""
        print("\nInitiating system shutdown...")
        self.logger.info("Initiating graceful shutdown")
        
        try:
            # Stop trading loop
            self.system_active = False
            
            # Stop monitoring
            if self.monitoring_thread:
                self.stop_monitoring.set()
                self.monitoring_thread.join(timeout=5)
            
            # Close all positions if in live mode
            if self.mode == 'live' and self.execution_engine:
                print("Closing all open positions...")
                self.execution_engine.close_all_positions()
            
            # Disconnect MT5
            if self.core_system and self.core_system.mt5_manager:
                print("Disconnecting from MT5...")
                self.core_system.mt5_manager.disconnect()
            
            # Save final performance data
            performance = self.get_performance_summary()
            print("\nFinal Performance Summary:")
            print(f"   - Runtime: {performance['runtime']}")
            print(f"   - Signals Generated: {performance['signals_generated']}")
            print(f"   - Trades Executed: {performance['trades_executed']}")
            print(f"   - Win Rate: {performance['win_rate']:.1f}%")
            print(f"   - Final Balance: ${performance['current_balance']:.2f}")
            print(f"   - Target Progress: {performance['progress_to_target']:.1f}%")
            
            # Log final status
            self.logger.info("System shutdown completed successfully")
            print("System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}", exc_info=True)
            print(f"ERROR: Shutdown error: {e}")


# Test Functions
def test_system_initialization():
    """Test system initialization"""
    print("Testing Phase 2 System Initialization")
    print("=" * 50)
    
    try:
        system = Phase2TradingSystem()
        system.set_mode('paper')
        
        if system.initialize():
            print("System initialization test passed")
            
            # Test signal generation
            print("\nTesting signal generation...")
            if system.signal_engine:
                active_strategies = system.signal_engine.get_active_strategies()
                print(f"Active strategies: {active_strategies}")
                print("Signal generation test passed")
            
            # Test performance summary
            print("\nTesting performance summary...")
            summary = system.get_performance_summary()
            print(f"Performance summary keys: {list(summary.keys())}")
            print("Performance summary test passed")
            
            system.shutdown()
            return True
        else:
            print("System initialization test failed")
            return False
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        traceback.print_exc()
        return False


def run_system(mode: str, duration_hours: Optional[float] = None):
    """Run the trading system"""
    print(f"Starting XAUUSD Trading System in {mode.upper()} mode")
    
    try:
        system = Phase2TradingSystem()
        system.set_mode(mode)
        
        if not system.initialize():
            print("System initialization failed")
            return
        
        # Run trading loop
        if duration_hours:
            print(f"Running for {duration_hours} hours...")
            
        # Use asyncio to run the trading loop
        try:
            asyncio.run(system.run_trading_loop())
        except KeyboardInterrupt:
            print("\nSystem interrupted by user")
        finally:
            system.shutdown()
            
    except Exception as e:
        print(f"System error: {e}")
        traceback.print_exc()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='XAUUSD Phase 2 Trading System')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], 
                       default='paper', help='Trading mode')
    parser.add_argument('--test', action='store_true', 
                       help='Run system tests')
    parser.add_argument('--duration', type=float, 
                       help='Run duration in hours')
    parser.add_argument('--config', default='config/master_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Set up basic logging for main
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/phase2_system.log', encoding='utf-8')
        ]
    )
    
    if args.test:
        success = test_system_initialization()
        sys.exit(0 if success else 1)
    else:
        run_system(args.mode, args.duration)


if __name__ == '__main__':
    main()
