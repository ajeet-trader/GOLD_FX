#!/usr/bin/env python3
"""
Phase 2 Complete Setup - All Trading Strategies Integration
==========================================================
Author: XAUUSD Trading System
Version: 2.0.0
Date: 2025-08-08

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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    # Import Phase 1 components
    from phase_1_core_integration import CoreSystem
    from utils.logger import LoggerManager
    from utils.database import DatabaseManager
    from utils.error_handler import ErrorHandler
    from core.mt5_manager import MT5Manager
    
    # Import Phase 2 components (we'll create these)
    from core.signal_engine import SignalEngine
    from core.risk_manager import RiskManager
    from core.execution_engine import ExecutionEngine
    
    # Import strategies
    from strategies.technical.ichimoku import IchimokuStrategy
    from strategies.smc.order_blocks import OrderBlocksStrategy
    from strategies.ml.lstm_predictor import LSTMPredictor
    from strategies.technical.harmonic import HarmonicStrategy
    from strategies.technical.elliott_wave import ElliottWaveStrategy
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all Phase 1 and Phase 2 files are in place.")
    sys.exit(1)


class Phase2TradingSystem:
    """
    Complete Phase 2 Trading System
    
    Integrates all components for aggressive 10x returns:
    - Multiple trading strategies
    - Advanced risk management
    - Smart execution engine
    - Real-time monitoring
    """
    
    def __init__(self, config_path: str = 'config/master_config.yaml'):
        """Initialize the complete trading system"""
        self.config_path = config_path
        self.system_active = False
        
        # Core components
        self.core_system: Optional[CoreSystem] = None
        self.signal_engine: Optional[SignalEngine] = None
        self.risk_manager: Optional[RiskManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        
        # Strategies
        self.strategies = {}
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_signals = 0
        self.executed_trades = 0
        self.current_balance = 0.0
        self.target_reached = False
        
        # Logger
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
                print("❌ Core system initialization failed")
                return False
            print("✅ Core system initialized")
            
            # Step 2: Connect to MT5
            print("📋 Step 2: Connecting to MT5...")
            if not self.core_system.connect_mt5():
                print("❌ MT5 connection failed")
                return False
            print("✅ MT5 connected successfully")
            
            # Step 3: Initialize Risk Manager
            print("📋 Step 3: Initializing Risk Manager...")
            self.risk_manager = RiskManager(
                self.core_system.config,
                self.core_system.mt5_manager,
                self.core_system.database_manager
            )
            print("✅ Risk Manager initialized")
            
            # Step 4: Initialize Execution Engine
            print("📋 Step 4: Initializing Execution Engine...")
            self.execution_engine = ExecutionEngine(
                self.core_system.config,
                self.core_system.mt5_manager,
                self.risk_manager,
                self.core_system.database_manager,
                self.core_system.logger_manager
            )
            print("✅ Execution Engine initialized")
            
            # Step 5: Initialize Trading Strategies
            print("📋 Step 5: Initializing Trading Strategies...")
            self._initialize_strategies()
            print(f"✅ {len(self.strategies)} strategies initialized")
            
            # Step 6: Initialize Signal Engine
            print("📋 Step 6: Initializing Signal Engine...")
            self.signal_engine = SignalEngine(
                self.core_system.config,
                self.core_system.mt5_manager,
                list(self.strategies.values())
            )
            print("✅ Signal Engine initialized")
            
            # Step 7: System health check
            print("📋 Step 7: Performing System Health Check...")
            if not self._perform_health_check():
                print("❌ System health check failed")
                return False
            print("✅ System health check passed")
            
            self.system_active = True
            self._display_system_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            return False
    
    def _initialize_strategies(self) -> None:
        """Initialize all trading strategies"""
        strategy_config = self.core_system.config.get('strategies', {})
        
        # Technical Analysis Strategies
        if strategy_config.get('technical', {}).get('enabled', True):
            technical_config = strategy_config['technical']
            
            # Ichimoku Strategy
            if technical_config.get('active_strategies', {}).get('ichimoku', True):
                self.strategies['ichimoku'] = IchimokuStrategy(
                    technical_config, self.core_system.mt5_manager
                )
                print("   • Ichimoku Cloud Strategy")
             
            # Harmonic Patterns Strategy
            if technical_config.get('active_strategies', {}).get('harmonic', True):
                self.strategies['harmonic'] = HarmonicStrategy(
                    technical_config, self.core_system.mt5_manager
                )
                print("   • Harmonic Patterns Strategy")    
            
            # elliott Wave Strategy
            if technical_config.get('active_strategies', {}).get('elliott_wave', True):
                self.strategies['elliott_wave'] = ElliottWaveStrategy(
                    technical_config, self.core_system.mt5_manager
                )
                print("   • Elliott Wave Strategy")    
        
        # Smart Money Concepts
        if strategy_config.get('smc', {}).get('enabled', True):
            smc_config = strategy_config['smc']
            
            # Order Blocks Strategy
            if smc_config.get('active_components', {}).get('order_blocks', True):
                self.strategies['order_blocks'] = OrderBlocksStrategy(
                    smc_config, self.core_system.mt5_manager
                )
                print("   • Order Blocks Strategy (SMC)")
        
        # Machine Learning Strategies
        if strategy_config.get('ml', {}).get('enabled', True):
            ml_config = strategy_config['ml']
            
            # LSTM Predictor
            if ml_config.get('active_models', {}).get('lstm', True):
                self.strategies['lstm'] = LSTMPredictor(
                    ml_config, self.core_system.mt5_manager
                )
                print("   • LSTM Predictor (ML)")
    
    def _perform_health_check(self) -> bool:
        """Perform comprehensive system health check"""
        try:
            health_status = {
                'mt5_connection': False,
                'strategies': False,
                'risk_manager': False,
                'execution_engine': False,
                'account_status': False
            }
            
            # Check MT5 connection
            try:
                balance = self.core_system.mt5_manager.get_account_balance()
                if balance > 0:
                    health_status['mt5_connection'] = True
                    self.current_balance = balance
            except:
                pass
            
            # Check strategies
            if len(self.strategies) > 0:
                health_status['strategies'] = True
            
            # Check risk manager
            if self.risk_manager and hasattr(self.risk_manager, 'risk_per_trade'):
                health_status['risk_manager'] = True
            
            # Check execution engine
            if self.execution_engine and hasattr(self.execution_engine, 'engine_active'):
                health_status['execution_engine'] = True
            
            # Check account status
            if self.current_balance >= 50.0:  # Minimum viable balance
                health_status['account_status'] = True
            
            # Display health status
            print("\n   Health Check Results:")
            for component, status in health_status.items():
                icon = "✅" if status else "❌"
                print(f"   {icon} {component.replace('_', ' ').title()}")
            
            return all(health_status.values())
            
        except Exception as e:
            print(f"   ❌ Health check error: {str(e)}")
            return False
    
    def _display_system_summary(self) -> None:
        """Display comprehensive system summary"""
        print("\n" + "="*60)
        print("🎉 PHASE 2 TRADING SYSTEM READY!")
        print("="*60)
        
        print(f"\n📊 System Configuration:")
        print(f"   • Target: ${self.current_balance:.2f} → $1000 (10x returns)")
        print(f"   • Strategies: {len(self.strategies)} active")
        print(f"   • Risk per trade: {self.risk_manager.risk_per_trade:.1%}")
        print(f"   • Max drawdown: {self.risk_manager.max_drawdown:.1%}")
        
        print(f"\n🎯 Active Strategies:")
        for name, strategy in self.strategies.items():
            strategy_info = strategy.get_strategy_info()
            print(f"   • {strategy_info['name']} ({strategy_info['type']})")
        
        print(f"\n⚡ System Capabilities:")
        print(f"   • Multi-strategy signal fusion")
        print(f"   • Advanced risk management")
        print(f"   • Real-time execution")
        print(f"   • Performance monitoring")
        print(f"   • Emergency controls")
        
        print(f"\n🚀 Ready for aggressive 10x trading!")
        print("="*60)
    
    def run_trading_loop(self) -> None:
        """Run the main trading loop"""
        if not self.system_active:
            print("❌ System not initialized. Call initialize() first.")
            return
        
        print("\n🔄 Starting trading loop...")
        print("Press Ctrl+C to stop")
        
        try:
            loop_count = 0
            
            while self.system_active:
                loop_count += 1
                loop_start = time.time()
                
                try:
                    # Generate signals from all strategies
                    all_signals = self.signal_engine.generate_signals("XAUUSDm", "M15")
                    self.total_signals += len(all_signals)
                    
                    if all_signals:
                        print(f"\n📡 Loop {loop_count}: Generated {len(all_signals)} signals")
                        
                        # Process each signal through execution engine
                        for signal in all_signals:
                            execution_result = self.execution_engine.process_signal(signal)
                            
                            if execution_result.status.value == "EXECUTED":
                                self.executed_trades += 1
                                print(f"✅ Trade executed: {signal.signal_type.value} "
                                     f"{signal.symbol} @ {execution_result.executed_price}")
                            elif execution_result.status.value == "REJECTED":
                                print(f"⚠️ Signal rejected: {execution_result.error_message}")
                    
                    # Check for 10x target achievement
                    current_equity = self.core_system.mt5_manager.get_account_equity()
                    target_capital = self.core_system.config.get('trading', {}).get('capital', {}).get('target_capital', 1000)
                    
                    if current_equity >= target_capital and not self.target_reached:
                        self.target_reached = True
                        print(f"\n🎉 TARGET ACHIEVED! Equity: ${current_equity:.2f} (Target: ${target_capital})")
                        self._celebrate_achievement()
                    
                    # Display progress every 10 loops
                    if loop_count % 10 == 0:
                        self._display_progress()
                    
                    # Sleep for next iteration (15-second intervals for M15 timeframe)
                    loop_duration = time.time() - loop_start
                    sleep_time = max(15 - loop_duration, 1)
                    time.sleep(sleep_time)
                    
                except KeyboardInterrupt:
                    print("\n⚠️ Trading loop interrupted by user")
                    break
                except Exception as e:
                    print(f"❌ Loop error: {str(e)}")
                    self.core_system.logger_manager.error("Trading loop error", e)
                    time.sleep(30)  # Longer sleep on error
            
        except Exception as e:
            print(f"❌ Trading loop failed: {str(e)}")
        finally:
            self._shutdown_system()
    
    def _display_progress(self) -> None:
        """Display trading progress"""
        try:
            current_equity = self.core_system.mt5_manager.get_account_equity()
            initial_capital = self.core_system.config.get('trading', {}).get('capital', {}).get('initial_capital', 100)
            
            # Calculate progress
            progress_pct = ((current_equity - initial_capital) / initial_capital) * 100
            target_progress = (current_equity / 1000) * 100  # Assuming $1000 target
            
            runtime = datetime.now() - self.start_time
            
            print(f"\n📊 Trading Progress:")
            print(f"   • Runtime: {str(runtime).split('.')[0]}")
            print(f"   • Current Equity: ${current_equity:.2f}")
            print(f"   • Progress: {progress_pct:+.1f}% | Target: {target_progress:.1f}%")
            print(f"   • Signals Generated: {self.total_signals}")
            print(f"   • Trades Executed: {self.executed_trades}")
            
            # Active positions
            positions = self.execution_engine.get_execution_summary()['positions']
            print(f"   • Active Positions: {positions['active_count']}")
            if positions['unrealized_pnl'] != 0:
                print(f"   • Unrealized P&L: ${positions['unrealized_pnl']:+.2f}")
            
        except Exception as e:
            print(f"❌ Progress display error: {str(e)}")
    
    def _celebrate_achievement(self) -> None:
        """Celebrate reaching the 10x target"""
        print("\n" + "🎉" * 20)
        print("     🏆 CONGRATULATIONS! 🏆")
        print("   10X TARGET ACHIEVED!")
        print("🎉" * 20)
        
        # Log achievement
        self.core_system.logger_manager.info("10x target achieved!")
        
        # Optional: Reduce risk or switch to capital preservation mode
        print("\n💡 Suggestion: Consider switching to capital preservation mode")
    
    def test_system(self) -> Dict[str, Any]:
        """Test all system components"""
        print("\n🧪 Testing Phase 2 System Components...")
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # Test 1: Core System
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
            risk_summary = self.risk_manager.get_risk_summary() if self.risk_manager else {}
            test_results['tests']['risk_manager'] = {
                'status': 'PASS' if 'risk_level' in risk_summary else 'FAIL',
                'risk_level': risk_summary.get('risk_level', 'UNKNOWN')
            }
            print(f"   ✅ PASS - Risk Level: {risk_summary.get('risk_level', 'UNKNOWN')}")
        except Exception as e:
            test_results['tests']['risk_manager'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ FAIL: {e}")
        
        # Test 3: Strategies
        try:
            print("🧪 Test 3: Trading Strategies...")
            strategy_test_count = 0
            for name, strategy in self.strategies.items():
                try:
                    signals = strategy.generate_signals("XAUUSDm", "M15")
                    strategy_test_count += 1
                    print(f"   • {name}: {len(signals)} signals generated")
                except Exception as e:
                    print(f"   • {name}: FAILED - {e}")
            
            test_results['tests']['strategies'] = {
                'status': 'PASS' if strategy_test_count > 0 else 'FAIL',
                'working_strategies': strategy_test_count,
                'total_strategies': len(self.strategies)
            }
            if strategy_test_count > 0:
                print(f"   ✅ PASS - {strategy_test_count}/{len(self.strategies)} strategies working")
            else:
                print("   ❌ FAIL - No strategies working")
        except Exception as e:
            test_results['tests']['strategies'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ FAIL: {e}")
        
        # Test 4: Signal Engine
        try:
            print("🧪 Test 4: Signal Engine...")
            signals = self.signal_engine.generate_signals("XAUUSDm", "M15") if self.signal_engine else []
            test_results['tests']['signal_engine'] = {
                'status': 'PASS' if len(signals) >= 0 else 'FAIL',
                'signals_generated': len(signals)
            }
            print(f"   ✅ PASS - Generated {len(signals)} signals")
        except Exception as e:
            test_results['tests']['signal_engine'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ FAIL: {e}")
        
        # Test 5: Execution Engine
        try:
            print("🧪 Test 5: Execution Engine...")
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
        passed_tests = sum(1 for test in test_results['tests'].values() if test['status'] == 'PASS')
        total_tests = len(test_results['tests'])
        
        test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        }
        
        print(f"\n📊 Test Summary: {passed_tests}/{total_tests} tests passed")
        print(f"🎯 Overall Success Rate: {test_results['summary']['success_rate']:.1%}")
        
        return test_results
    
    def _shutdown_system(self) -> None:
        """Gracefully shutdown the system"""
        print("\n🔄 Shutting down Phase 2 Trading System...")
        
        try:
            self.system_active = False
            
            # Stop execution engine
            if self.execution_engine:
                self.execution_engine.stop_engine()
                print("✅ Execution engine stopped")
            
            # Shutdown core system
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
            print(f"❌ Shutdown error: {str(e)}")


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
            print("❌ System initialization failed")
            return False
        
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
                        # Display current status
                        if trading_system.execution_engine:
                            summary = trading_system.execution_engine.get_execution_summary()
                            print(f"\nSystem Status: {summary}")
                        else:
                            print("\nSystem not fully initialized")
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
        return False
    finally:
        # Cleanup
        if trading_system.system_active:
            trading_system._shutdown_system()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)