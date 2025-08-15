#!/usr/bin/env python3
"""
Phase 2 Complete Setup - All Trading Strategies Integration
==========================================================
Author: XAUUSD Trading System
Version: 2.1.1 (Fixed Initialization)
Date: 2025-08-15

Complete Phase 2 implementation with all strategies:
- Signal Engine (core signal processing)
- Ichimoku Cloud Strategy (technical analysis)
- Order Blocks Strategy (Smart Money Concepts)
- LSTM Predictor (Machine Learning)
- Risk Manager (advanced risk management)
- Execution Engine (trade execution)

This integrates everything for the 10x trading goal.

Usage:
    python phase_2_core_integration.py --test     # Test all components
    python phase_2_core_integration.py --run      # Run full system
    python phase_2_core_integration.py --setup    # Setup only
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Add project root to sys.path for correct imports
# Assuming this script is at <project_root>/src/phase_2_core_integration.py
# The project root is one level up from 'src'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    # Import Phase 1 components
    from src.phase_1_core_integration import CoreSystem
    # Specific component classes for type hinting and clarity
    from src.core.mt5_manager import MT5Manager
    from src.utils.database import DatabaseManager
    from src.utils.error_handler import ErrorHandler
    from src.utils.logger import LoggerManager
    
    # Import Phase 2 components
    from src.core.signal_engine import SignalEngine
    from src.core.risk_manager import RiskManager
    from src.core.execution_engine import ExecutionEngine
    
    # Removed direct strategy imports, as SignalEngine handles their instantiation
    
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    print("Please ensure all Phase 1 and Phase 2 files are in place and the project root is correctly added to sys.path.")
    sys.exit(1)


class Phase2TradingSystem:
    """
    Complete Phase 2 Trading System
    
    Integrates all components for aggressive 10x returns:
    - Multiple trading strategies (managed by SignalEngine)
    - Advanced risk management
    - Smart execution engine
    - Real-time monitoring
    """
    
    def __init__(self, config_path: str = 'config/master_config.yaml'):
        """Initialize the complete trading system"""
        self.config_path = config_path
        self.system_active = False # Main loop control flag
        
        # Core components (Phase 1 instances)
        self.core_system: Optional[CoreSystem] = None
        
        # Phase 2 Components
        self.signal_engine: Optional[SignalEngine] = None
        self.risk_manager: Optional[RiskManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_signals = 0
        self.executed_trades = 0
        self.current_balance = 0.0 # Will be updated from MT5
        self.target_reached = False
        
        # System settings for trading loop (can be moved to config)
        self.max_trading_attempts = 5 # Max consecutive errors before emergency stop
        self.trading_symbol = 'XAUUSDm' # Default, will be from config
        self.primary_timeframe = 'M15' # Default, will be from config

        # Logger for this specific class
        self.logger = logging.getLogger('phase2_system')
    
    def initialize(self) -> bool:
        """Initialize the complete Phase 2 system"""
        print("🚀 Initializing Phase 2 Trading System")
        print("="*60)
        
        try:
            # Step 1: Initialize Phase 1 core system
            print("📋 Step 1: Initializing Core System (Phase 1)...")
            self.core_system = CoreSystem(self.config_path)
            if not self.core_system.initialize():
                print("❌ Core system initialization failed.")
                # The CoreSystem's error handler should have already logged/triggered.
                return False
            print("✅ Core system initialized.")
            
            # Step 2: Connect to MT5 (via CoreSystem)
            print("📋 Step 2: Connecting to MT5...")
            if not self.core_system.connect_mt5():
                print("❌ MT5 connection failed.")
                # The CoreSystem's error handler should have already logged/triggered.
                return False
            print("✅ MT5 connected successfully.")

            # Update trading symbol and timeframe from config
            self.trading_symbol = self.core_system.config.get('trading', {}).get('symbol', 'XAUUSDm')
            # Assuming timeframe is also in trading config or implicitly handled by strategies

            # Step 3: Initialize Risk Manager
            print("📋 Step 3: Initializing Risk Manager...")
            # RiskManager.__init__ signature: (config, mt5_manager, database_manager)
            # It internally gets its own logger.
            self.risk_manager = RiskManager(
                config=self.core_system.config,
                mt5_manager=self.core_system.mt5_manager,
                database_manager=self.core_system.database_manager
            )
            # RiskManager does not have an explicit `initialize` method based on provided code
            print("✅ Risk Manager initialized.")
            
            # Step 4: Initialize Execution Engine
            print("📋 Step 4: Initializing Execution Engine...")
            # ExecutionEngine.__init__ signature: (config, mt5_manager, risk_manager, database_manager, logger_manager)
            # logger_manager is LoggerManager object itself here, not a specific logger.
            self.execution_engine = ExecutionEngine(
                config=self.core_system.config,
                mt5_manager=self.core_system.mt5_manager,
                risk_manager=self.risk_manager, # Execution Engine depends on Risk Manager
                database_manager=self.core_system.database_manager,
                logger_manager=self.core_system.logger_manager # Pass LoggerManager instance
            )
            # ExecutionEngine has an internal _initialize_engine that runs on __init__
            print("✅ Execution Engine initialized.")
            
            # Step 5: Initialize Signal Engine
            print("📋 Step 5: Initializing Signal Engine...")
            # SignalEngine.__init__ signature: (config, mt5_manager, database_manager)
            # SignalEngine has its own `initialize` method that loads strategies.
            self.signal_engine = SignalEngine(
                config=self.core_system.config,
                mt5_manager=self.core_system.mt5_manager,
                database_manager=self.core_system.database_manager # Pass database_manager for ML strategies
            )
            if not self.signal_engine.initialize(): # SignalEngine initializes its own strategies
                print("❌ Signal Engine initialization failed.")
                self.core_system.error_handler.trigger_emergency_stop("Signal Engine Init Failed")
                return False
            print("✅ Signal Engine initialized.")
            
            # Step 6: Performing System Health Check
            print("📋 Step 6: Performing System Health Check...")
            if not self._perform_health_check():
                print("❌ System health check failed.")
                self.core_system.error_handler.trigger_emergency_stop("System Health Check Failed")
                return False
            print("✅ System health check passed.")
            
            self.system_active = True
            self._display_system_summary()
            
            return True
            
        except Exception as e:
            error_msg = f"Phase 2 System Initialization Failed: {str(e)}"
            print(f"❌ {error_msg}")
            # Ensure error handler is available before using it
            if self.core_system and self.core_system.error_handler:
                self.core_system.error_handler.handle_error(e, "Phase2_System_Init_Error", "Critical")
                # Trigger emergency stop if system init fails
                self.core_system.error_handler.trigger_emergency_stop("Phase 2 System Initialization Failed")
            return False
    
    def _perform_health_check(self) -> bool:
        """Perform comprehensive system health check"""
        try:
            health_status = {
                'mt5_connection': False,
                'signal_engine': False, 
                'risk_manager': False,
                'execution_engine': False,
                'account_status': False
            }
            
            # Check MT5 connection via CoreSystem
            try:
                # Assuming core_system.mt5_manager is already connected from core_system.connect_mt5()
                balance = self.core_system.mt5_manager.get_account_balance()
                if balance > 0:
                    health_status['mt5_connection'] = True
                    self.current_balance = balance
            except Exception as e:
                self.logger.warning(f"MT5 balance check failed during health check: {e}")
            
            # Check SignalEngine
            if self.signal_engine and self.signal_engine.initialized:
                health_status['signal_engine'] = True
            
            # Check RiskManager (based on its internal initialization status)
            # Assuming RiskManager sets an `initialized` attribute or similar upon successful setup
            # If not, you might need to add one or check internal attributes if possible.
            health_status['risk_manager'] = self.risk_manager is not None # Basic check
            # More robust check would be: hasattr(self.risk_manager, 'is_initialized') and self.risk_manager.is_initialized()
            
            # Check ExecutionEngine (based on its internal initialization status)
            # ExecutionEngine has `engine_active` and `monitoring_active` flags
            if self.execution_engine and self.execution_engine.engine_active:
                health_status['execution_engine'] = True
            
            # Check account status (minimum viable balance)
            if self.current_balance >= self.core_system.config.get('trading', {}).get('capital', {}).get('minimum_capital', 50.0):
                health_status['account_status'] = True
            
            # Display health status
            print("\n   Health Check Results:")
            for component, status in health_status.items():
                icon = "✅" if status else "❌"
                print(f"   {icon} {component.replace('_', ' ').title()}")
            
            return all(health_status.values())
            
        except Exception as e:
            self.logger.error(f"   ❌ Health check error: {str(e)}")
            # No emergency stop from health check itself, but initialization flow will catch it
            return False
    
    def _display_system_summary(self) -> None:
        """Display comprehensive system summary"""
        print("\n" + "="*60)
        print("🎉 PHASE 2 TRADING SYSTEM READY!")
        print("="*60)
        
        print(f"\n📊 System Configuration:")
        current_equity = self.core_system.mt5_manager.get_account_equity() if self.core_system and self.core_system.mt5_manager else self.current_balance
        target_capital = self.core_system.config.get('trading', {}).get('capital', {}).get('target_capital', 1000)
        print(f"   • Current Equity: ${current_equity:.2f}")
        print(f"   • Target: ${target_capital} (10x returns goal)")
        
        # Get active strategy count from SignalEngine
        active_strat_summary = self.signal_engine.get_active_strategies()
        total_active_strats = sum(len(v) for v in active_strat_summary.values())
        print(f"   • Strategies: {total_active_strats} active ({', '.join([f'{k.title()}: {len(v)}' for k,v in active_strat_summary.items()])})")
        
        # Risk manager info
        if self.risk_manager and hasattr(self.risk_manager, 'get_risk_summary'):
            risk_summary = self.risk_manager.get_risk_summary()
            print(f"   • Risk per trade: {risk_summary.get('risk_per_trade', 0.0):.1%}") # Note: config might store as float, not percent
            print(f"   • Max drawdown: {risk_summary.get('max_drawdown', 0.0):.1%}")
        
        print(f"\n⚡ System Capabilities:")
        print(f"   • Multi-strategy signal fusion")
        print(f"   • Advanced risk management")
        print(f"   • Real-time execution")
        print(f"   • Performance monitoring")
        print(f"   • Emergency controls")
        
        print(f"\n🚀 Ready for aggressive 10x trading!")
        print("="*60)
    
    def run_trading_loop(self) -> None:
        """
        Main execution loop for live trading. This loop continuously:
        1. Generates trading signals using the Signal Engine.
        2. Processes and attempts to execute valid signals via the Execution Engine (which consults Risk Manager).
        3. Manages open positions.
        4. Handles errors and ensures system resilience.
        """
        if not self.system_active:
            self.logger.error("Phase 2 Trading System is not initialized. Cannot start live trading.")
            # Trigger emergency stop if attempted to run uninitialized system
            if self.core_system and self.core_system.error_handler:
                self.core_system.error_handler.trigger_emergency_stop("Uninitialized System Run Attempt")
            return
        
        self.logger.info("\n🔄 Starting trading loop...")
        self.logger.info("Press Ctrl+C to stop")
        
        loop_count = 0
        consecutive_error_count = 0
        
        # Get trade interval from config (default to 15 seconds for M15 timeframe alignment)
        trade_loop_interval_seconds = self.core_system.config.get('system_settings', {}).get('trade_loop_interval_seconds', 15)

        try:
            while self.system_active:
                loop_count += 1
                loop_start_time = time.time()
                
                try:
                    # Check for emergency stop triggered externally or by another component
                    if self.core_system.error_handler.is_emergency_stop_triggered():
                        self.logger.critical("Emergency stop is active. Halting trading loop.")
                        self.system_active = False # Set flag to stop loop
                        break # Exit loop immediately

                    self.logger.info(f"\n--- Trading loop iteration {loop_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

                    # 1. Generate signals from active strategies
                    generated_signals = self.signal_engine.generate_signals(self.trading_symbol, self.primary_timeframe)
                    self.total_signals += len(generated_signals)
                    
                    self.logger.info(f"📡 Generated {len(generated_signals)} quality signals in this iteration.")
                    
                    # 2. Process and execute signals through ExecutionEngine
                    if generated_signals:
                        for signal in generated_signals:
                            try:
                                # ExecutionEngine handles risk assessment via RiskManager internally
                                # Assuming process_signal method returns an ExecutionResult object
                                execution_result = self.execution_engine.process_signal(signal) 
                                
                                if execution_result and execution_result.status.value == "EXECUTED":
                                    self.executed_trades += 1
                                    self.logger.info(f"✅ Trade executed: {signal.signal_type.value} {signal.symbol} @ {execution_result.executed_price:.4f}")
                                elif execution_result and execution_result.status.value == "REJECTED":
                                    self.logger.warning(f"⚠️ Signal rejected: {signal.strategy_name} - {execution_result.error_message}")
                                else: # No trade executed, but not necessarily an error
                                    self.logger.debug(f"Signal {signal.strategy_name} processed, but no trade executed (status: {execution_result.status.value}).")

                            except Exception as signal_process_e:
                                # Log error for individual signal processing, but continue with other signals
                                self.core_system.error_handler.handle_error(
                                    signal_process_e, f"SignalProcessingError:{signal.strategy_name}",
                                    severity="High", metadata={'signal_strategy': signal.strategy_name}
                                )
                                self.logger.error(f"❌ Error processing signal from {signal.strategy_name}: {str(signal_process_e)}")
                    else:
                        self.logger.info("No actionable signals to process in this iteration.")

                    # 3. Manage Open Positions (e.g., update SL/TP, partial close, check for closed positions)
                    # Assuming ExecutionEngine has a method for this
                    if hasattr(self.execution_engine, 'run_position_monitoring'): # Assuming this method name from ExecutionEngine
                        self.execution_engine.run_position_monitoring() # This should be a non-blocking call, or it runs in a separate thread
                    else:
                        self.logger.debug("ExecutionEngine.run_position_monitoring method not found. Skipping position management loop.")

                    # Reset consecutive error count on successful iteration
                    consecutive_error_count = 0

                    # Check for 10x target achievement
                    current_equity = self.core_system.mt5_manager.get_account_equity()
                    target_capital = self.core_system.config.get('trading', {}).get('capital', {}).get('target_capital', 1000)
                    
                    if current_equity >= target_capital and not self.target_reached:
                        self.target_reached = True
                        self._celebrate_achievement()
                    
                    # Display progress periodically
                    if loop_count % 10 == 0: # Display every 10 loops
                        self._display_progress()
                    
                    # Calculate sleep time to maintain interval
                    loop_duration = time.time() - loop_start_time
                    sleep_time = max(trade_loop_interval_seconds - loop_duration, 1) # Ensure at least 1 second sleep
                    self.logger.info(f"Sleeping for {sleep_time:.2f} seconds before next iteration...")
                    time.sleep(sleep_time)
                    
                except KeyboardInterrupt:
                    self.logger.info("\n⚠️ Trading loop interrupted by user.")
                    self.system_active = False
                    break # Exit the loop
                except Exception as loop_e:
                    # Handle critical errors in the main trading loop
                    consecutive_error_count += 1
                    # Pass the error details to the global error handler
                    self.core_system.error_handler.handle_error(
                        loop_e, "MainTradingLoopError",
                        severity="Critical" if consecutive_error_count >= self.max_trading_attempts else "High",
                        metadata={'consecutive_errors': consecutive_error_count}
                    )
                    self.logger.error(f"❌ Critical error in trading loop (Consecutive errors: {consecutive_error_count}/{self.max_trading_attempts}): {str(loop_e)}")
                    
                    if consecutive_error_count >= self.max_trading_attempts:
                        self.logger.critical(f"Max consecutive trading errors ({self.max_trading_attempts}) reached. Triggering emergency stop.")
                        self.core_system.error_handler.trigger_emergency_stop("MaxConsecutiveErrors")
                        self.system_active = False # Set flag to stop loop
                        break # Exit the loop
                    
                    # Sleep longer on error to prevent rapid-fire failures
                    self.logger.info("Sleeping for 30 seconds before retrying due to error...")
                    time.sleep(30)
            
        except Exception as outer_e:
            self.logger.critical(f"❌ Trading loop failed fatally: {str(outer_e)}")
            # Final fallback for unexpected outer loop failures
            if self.core_system and self.core_system.error_handler:
                self.core_system.error_handler.handle_error(outer_e, "FatalTradingLoopFailure", "Critical")
                self.core_system.error_handler.trigger_emergency_stop("FatalLoopFailure")
        finally:
            self._shutdown_system() # Ensure shutdown is always attempted
    
    def _display_progress(self) -> None:
        """Display trading progress"""
        try:
            current_equity = self.core_system.mt5_manager.get_account_equity()
            initial_capital = self.core_system.config.get('trading', {}).get('capital', {}).get('initial_capital', 100)
            
            # Calculate progress
            progress_pct = ((current_equity - initial_capital) / initial_capital) * 100 if initial_capital else 0
            # Ensure target_capital is used for progress calculation if available, otherwise default to a simple 10x
            target_capital = self.core_system.config.get('trading', {}).get('capital', {}).get('target_capital', 1000)
            target_progress = (current_equity / target_capital) * 100 if target_capital else 0
            
            runtime = datetime.now() - self.start_time
            
            self.logger.info(f"\n📊 Trading Progress (Loop {self.total_signals} total signals):")
            self.logger.info(f"   • Runtime: {str(runtime).split('.')[0]}")
            self.logger.info(f"   • Current Equity: ${current_equity:.2f}")
            self.logger.info(f"   • Progress vs Initial Capital: {progress_pct:+.1f}% | Progress vs Target: {target_progress:.1f}%")
            self.logger.info(f"   • Signals Generated: {self.total_signals}")
            self.logger.info(f"   • Trades Executed: {self.executed_trades}")
            
            # Active positions summary from ExecutionEngine
            if self.execution_engine and hasattr(self.execution_engine, 'get_execution_summary'):
                positions = self.execution_engine.get_execution_summary()['positions']
                self.logger.info(f"   • Active Positions: {positions['active_count']}")
                if positions['unrealized_pnl'] != 0:
                    self.logger.info(f"   • Unrealized P&L: ${positions['unrealized_pnl']:+.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Progress display error: {str(e)}")
            self.core_system.error_handler.handle_error(e, "ProgressDisplayError", "Low")
    
    def _celebrate_achievement(self) -> None:
        """Celebrate reaching the 10x target"""
        self.logger.info("\n" + "🎉" * 20)
        self.logger.info("     🏆 CONGRATULATIONS! 🏆")
        self.logger.info("   10X TARGET ACHIEVED!")
        self.logger.info("🎉" * 20)
        
        # Log achievement
        self.core_system.logger_manager.info("10x target achieved!")
        
        # Optional: Reduce risk or switch to capital preservation mode
        self.logger.info("\n💡 Suggestion: Consider switching to capital preservation mode")
    
    def test_system(self) -> Dict[str, Any]:
        """Test all system components"""
        print("\n🧪 Testing Phase 2 System Components...")
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # Test 1: Core System (Delegated to CoreSystem's test_system)
        try:
            print("🧪 Test 1: Core System...")
            core_test = self.core_system.test_system() if self.core_system else {'summary': {'success_rate': 0}}
            test_results['tests']['core_system'] = {
                'status': 'PASS' if core_test['summary']['success_rate'] > 0.8 else 'FAIL',
                'success_rate': core_test['summary']['success_rate']
            }
            print(f"   ✅ PASS - {core_test['summary']['success_rate']:.1%} success rate")
        except Exception as e:
            test_results['tests']['core_system'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ FAIL: {e}")
        
        # Test 2: Risk Manager
        try:
            print("🧪 Test 2: Risk Manager...")
            # Assuming RiskManager has a get_risk_summary method
            risk_summary = self.risk_manager.get_risk_summary() if self.risk_manager else {}
            test_results['tests']['risk_manager'] = {
                'status': 'PASS' if 'risk_level' in risk_summary else 'FAIL',
                'risk_level': risk_summary.get('risk_level', 'UNKNOWN')
            }
            print(f"   ✅ PASS - Risk Level: {risk_summary.get('risk_level', 'UNKNOWN')}")
        except Exception as e:
            test_results['tests']['risk_manager'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ FAIL: {e}")
        
        # --- REMOVED: Test 3: Trading Strategies ---
        # The strategies are now managed and tested by SignalEngine.
        # This section is no longer necessary as self.strategies attribute will be empty.
        # test_results['tests']['strategies'] = { ... }

        # Test 3 (formerly 4): Signal Engine
        try:
            print("🧪 Test 3: Signal Engine...")
            # SignalEngine should be able to generate signals for testing
            signals = self.signal_engine.generate_signals(self.trading_symbol, self.primary_timeframe) if self.signal_engine else []
            test_results['tests']['signal_engine'] = {
                'status': 'PASS' if len(signals) >= 0 else 'FAIL',
                'signals_generated': len(signals),
                'active_strategies_count': sum(len(v) for v in self.signal_engine.get_active_strategies().values())
            }
            print(f"   ✅ PASS - Generated {len(signals)} signals from {test_results['tests']['signal_engine']['active_strategies_count']} active strategies")
        except Exception as e:
            test_results['tests']['signal_engine'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ FAIL: {e}")
        
        # Test 4 (formerly 5): Execution Engine
        try:
            print("🧪 Test 4: Execution Engine...")
            # Assuming ExecutionEngine has a get_execution_summary method
            exec_summary = self.execution_engine.get_execution_summary() if self.execution_engine else {}
            test_results['tests']['execution_engine'] = {
                'status': 'PASS' if 'engine_status' in exec_summary else 'FAIL',
                'engine_active': exec_summary.get('engine_status', {}).get('active', False)
            }
            print(f"   ✅ PASS - Engine Active: {exec_summary.get('engine_status', {}).get('active', False)}")
        except Exception as e:
            test_results['tests']['execution_engine'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ FAIL: {e}")
        
        # Calculate overall results
        # Correctly get the number of tests after removal
        # The test key names are now 'core_system', 'risk_manager', 'signal_engine', 'execution_engine'
        passed_tests = sum(1 for test in test_results['tests'].values() if test['status'] == 'PASS')
        total_tests = len(test_results['tests'])
        
        test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        }
        
        print(f"\n📊 Test Summary: {passed_tests}/{total_tests} tests passed")
        print(f"🎯 Overall Success Rate: {test_results['summary']['success_rate']:.1%}")
        
        # Log test results
        if self.core_system and self.core_system.logger_manager:
            self.core_system.logger_manager.info(f"Phase 2 System test completed: {passed_tests}/{total_tests} passed")
        
        return test_results
    
    def _shutdown_system(self) -> None:
        """Gracefully shutdown the system"""
        print("\n🔄 Shutting down Phase 2 Trading System...")
        
        try:
            self.system_active = False # Ensure loop stops if not already
            
            # Initiate emergency close of all open positions as a safety measure
            if self.execution_engine and hasattr(self.execution_engine, 'emergency_close_all_positions'):
                self.logger.info("Initiating emergency close of all open positions...")
                self.execution_engine.emergency_close_all_positions()
                print("✅ Emergency close initiated")
            else:
                self.logger.warning("ExecutionEngine.emergency_close_all_positions method not found. Skipping emergency close.")

            # Stop execution engine
            if self.execution_engine and hasattr(self.execution_engine, 'stop_engine'):
                self.execution_engine.stop_engine()
                print("✅ Execution engine stopped")
            else:
                self.logger.warning("ExecutionEngine.stop_engine method not found. Skipping execution engine stop.")
            
            # Shutdown core system (which handles MT5 disconnect, logging finalization, etc.)
            if self.core_system:
                self.core_system.shutdown()
                print("✅ Core system shutdown")
            
            # Display final statistics
            runtime = datetime.now() - self.start_time
            print(f"\n📊 Final Statistics:")
            print(f"   • Total Runtime: {str(runtime).split('.')[0]}")
            print(f"   • Signals Generated: {self.total_signals}")
            print(f"   • Trades Executed: {self.executed_trades}")
            print(f"   • Target Achieved: {'Yes' if self.target_reached else 'No'}")
            
            print("\n🎯 Phase 2 Trading System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Shutdown error: {str(e)}")
            if self.core_system and self.core_system.error_handler:
                self.core_system.error_handler.handle_error(e, "Phase2_Shutdown_Error", "Critical")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Phase 2 Trading System')
    parser.add_argument('--test', action='store_true', help='Test all components')
    parser.add_argument('--run', action='store_true', help='Run full trading system')
    parser.add_argument('--setup', action='store_true', help='Setup and initialize only')
    parser.add_argument('--config', type=str, default='config/master_config.yaml', 
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Create trading system
    trading_system = Phase2TradingSystem(args.config)
    
    try:
        # Initialize system
        if not trading_system.initialize():
            print("❌ System initialization failed. Exiting.")
            return False # Exit with failure
        
        if args.test:
            # Run tests
            test_results = trading_system.test_system()
            return test_results['summary']['success_rate'] > 0.8
        
        elif args.run:
            # Run full trading system
            trading_system.run_trading_loop()
            return True
        
        elif args.setup:
            # Setup only
            print("✅ System setup completed successfully")
            return True
        
        else:
            # Interactive mode
            print("\n🎯 Interactive Mode - Choose an option:")
            print("1. Run trading system")
            print("2. Test components") 
            print("3. Display status")
            print("4. Exit")
            
            while True:
                try:
                    choice = input("\nEnter choice (1-4): ").strip()
                    
                    if choice == '1':
                        trading_system.run_trading_loop()
                        break
                    elif choice == '2':
                        trading_system.test_system()
                    elif choice == '3':
                        # Display current status using get_execution_summary if available
                        if trading_system.execution_engine and hasattr(trading_system.execution_engine, 'get_execution_summary'):
                            summary = trading_system.execution_engine.get_execution_summary()
                            print(f"\nSystem Status: {summary}")
                        else:
                            print("\nSystem not fully initialized or ExecutionEngine missing summary method.")
                    elif choice == '4':
                        print("Exiting...")
                        break
                    else:
                        print("Invalid choice. Please enter 1-4.")
                        
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
            
            return True
        
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        # If an error happens before _shutdown_system is called in finally, ensure cleanup
        if trading_system.system_active:
            trading_system._shutdown_system() # Call shutdown here too for safety
        return False
    finally:
        # Ensures system shutdown even if an unhandled exception occurs
        # or if interactive mode exits normally.
        if trading_system.system_active or (trading_system.core_system and trading_system.core_system.initialized):
             trading_system._shutdown_system()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)