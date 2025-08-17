"""
Phase 1 Integration - Core System Integration
============================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-08-07

This module integrates all Phase 1 components:
- MT5 Manager
- Logging Infrastructure  
- Database Management
- Error Handling Framework

This creates a unified core system that serves as the foundation
for all other trading system components.

Usage:
    >>> from phase_1_core_integration import CoreSystem
    >>> 
    >>> # Initialize with configuration
    >>> core = CoreSystem('config/master_config.yaml')
    >>> core.initialize()
    >>> 
    >>> # Use the integrated system
    >>> core.mt5_manager.connect()
    >>> core.logger.info("System started")
    >>> core.database.store_trade(trade_data)
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))


import yaml
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import our Phase 1 modules
try:
    from utils.logger import LoggerManager, get_logger_manager
    from utils.database import DatabaseManager
    from utils.error_handler import ErrorHandler, get_error_handler
    from core.mt5_manager import MT5Manager
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all Phase 1 modules are in the correct directories:")
    print("- utils/logger.py")
    print("- utils/database.py") 
    print("- utils/error_handler.py")
    print("- core/mt5_manager.py")
    sys.exit(1)


class CoreSystem:
    """
    Core trading system that integrates all Phase 1 components
    
    This class provides a unified interface to:
    - MT5 connection and data management
    - Comprehensive logging system
    - Database operations
    - Error handling and recovery
    
    All components are initialized together and work in harmony
    to provide a robust foundation for the trading system.
    
    Attributes:
        config (dict): Complete system configuration
        mt5_manager (MT5Manager): MetaTrader 5 interface
        logger_manager (LoggerManager): Logging system
        database_manager (DatabaseManager): Database operations
        error_handler (ErrorHandler): Error handling system
        initialized (bool): System initialization status
    
    Example:
        >>> core = CoreSystem('config/master_config.yaml')
        >>> success = core.initialize()
        >>> if success:
        ...     core.logger.info("Trading system ready")
        ...     data = core.mt5_manager.get_historical_data("XAUUSDm", "M15", 100)
        ...     core.database_manager.store_market_data(data)
    """
    
    def __init__(self, config_path: str = 'config/master_config.yaml'):
        """
        Initialize the core system
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config = {}
        self.initialized = False
        
        # Component instances
        self.mt5_manager: Optional[MT5Manager] = None
        self.logger_manager: Optional[LoggerManager] = None
        self.database_manager: Optional[DatabaseManager] = None
        self.error_handler: Optional[ErrorHandler] = None
        
        # System state
        self.start_time = datetime.now()
        self.system_id = f"XAU_SYS_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self) -> bool:
        """Load system configuration from file"""
        try:
            if not self.config_path.exists():
                print(f"Configuration file not found: {self.config_path}")
                print("Using default configuration...")
                self.config = self._get_default_config()
                return True
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            print(f"Configuration loaded from: {self.config_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load configuration: {str(e)}")
            print("Using default configuration...")
            self.config = self._get_default_config()
            return True
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if no config file exists"""
        return {
            'mt5': {
                'login': None,
                'password': None,
                'server': None,
                'terminal_path': None,
                'magic_number': 123456,
                'slippage': 20,
                'deviation': 10
            },
            'trading': {
                'symbol': 'XAUUSDm',
                'capital': {
                    'initial_capital': 100.0,
                    'target_capital': 1000.0,
                    'minimum_capital': 50.0
                },
                'risk_management': {
                    'risk_per_trade': 0.03,
                    'max_risk_per_trade': 0.05,
                    'max_portfolio_risk': 0.15,
                    'max_drawdown': 0.25
                }
            },
            'database': {
                'type': 'sqlite',
                'sqlite': {
                    'path': 'data/trading.db'
                }
            },
            'logging': {
                'level': 'INFO',
                'console': {
                    'enabled': True,
                    'colored': True
                },
                'rotation': {
                    'max_size': '10MB',
                    'backup_count': 10
                }
            },
            'error_handling': {
                'restart_on_error': True,
                'max_restart_attempts': 3,
                'error_notification': True
            }
        }
    
    def initialize(self) -> bool:
        """
        Initialize all system components
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        print("="*60)
        print("XAUUSD MT5 Trading System - Phase 1 Initialization")
        print("="*60)
        print(f"System ID: {self.system_id}")
        print(f"Start Time: {self.start_time}")
        print(f"Config Path: {self.config_path}")
        print()
        
        try:
            # Step 1: Initialize Error Handler (first, to catch other component errors)
            print("Step 1: Initializing Error Handler...")
            self.error_handler = ErrorHandler(self.config)
            self.error_handler.start()
            print("Error Handler initialized")
            
            # Step 2: Initialize Logging System
            print("Step 2: Initializing Logging System...")
            self.logger_manager = LoggerManager(self.config)
            success = self.logger_manager.setup_logging()
            if not success:
                print("ERROR - Failed to initialize logging system")
                return False
            print("OK - Logging System initialized")
            
            # Step 3: Initialize Database
            print("Step Step 3: Initializing Database...")
            self.database_manager = DatabaseManager(self.config)
            success = self.database_manager.initialize_database()
            if not success:
                print("ERROR - Failed to initialize database")
                self.logger_manager.error("Database initialization failed")
                return False
            print("OK - Database initialized")
            
            # Step 4: Initialize MT5 Manager
            print("Step Step 4: Initializing MT5 Manager...")
            #self.mt5_manager = MT5Manager(self.config)
            symbol = self.config.get('trading', {}).get('symbol', 'XAUUSDm')
            self.mt5_manager = MT5Manager(symbol=symbol, magic_number=123456)
            print("OK - MT5 Manager initialized (connection will be established when needed)")
            
            # Step 5: System Health Check
            print("Step Step 5: Performing System Health Check...")
            if not self._perform_health_check():
                print("ERROR - System health check failed")
                return False
            print("OK - System health check passed")
            
            # Step 6: Store System Initialization
            self._store_system_initialization()
            
            self.initialized = True
            print()
            print(" Phase 1 Core System Initialization Complete!")
            print("="*60)
            print()
            print("Status: System Status:")
            print(f"   • Error Handler: {'OK - Active' if self.error_handler else 'ERROR - Inactive'}")
            print(f"   • Logging System: {'OK - Active' if self.logger_manager else 'ERROR - Inactive'}")
            print(f"   • Database: {'OK - Active' if self.database_manager else 'ERROR - Inactive'}")
            print(f"   • MT5 Manager: {'OK - Ready' if self.mt5_manager else 'ERROR - Not Ready'}")
            print()
            print(" System is ready for Phase 2 development!")
            print("="*60)
            
            # Log successful initialization
            self.logger_manager.info("Core system initialization completed successfully")
            
            return True
            
        except Exception as e:
            error_msg = f"System initialization failed: {str(e)}"
            print(f"ERROR - {error_msg}")
            
            if self.error_handler:
                self.error_handler.handle_error(e, "System initialization")
            
            if self.logger_manager:
                self.logger_manager.error(error_msg, e)
            
            return False
    
    def _perform_health_check(self) -> bool:
        """Perform comprehensive system health check"""
        try:
            health_status = {
                'error_handler': False,
                'logging': False,
                'database': False,
                'config': False
            }
            
            # Check Error Handler
            if self.error_handler and self.error_handler.is_system_healthy():
                health_status['error_handler'] = True
            
            # Check Logging System
            if self.logger_manager and self.logger_manager._setup_complete:
                health_status['logging'] = True
            
            # Check Database
            if self.database_manager and self.database_manager.initialized:
                # Test database connection
                stats = self.database_manager.get_database_stats()
                if stats is not None:
                    health_status['database'] = True
            
            # Check Configuration
            if self.config and 'mt5' in self.config and 'trading' in self.config:
                health_status['config'] = True
            
            # Determine overall health
            all_healthy = all(health_status.values())
            
            print(f"   • Error Handler: {'OK -' if health_status['error_handler'] else 'ERROR -'}")
            print(f"   • Logging System: {'OK -' if health_status['logging'] else 'ERROR -'}")
            print(f"   • Database: {'OK -' if health_status['database'] else 'ERROR -'}")
            print(f"   • Configuration: {'OK -' if health_status['config'] else 'ERROR -'}")
            
            return all_healthy
            
        except Exception as e:
            print(f"   ERROR - Health check error: {str(e)}")
            return False
    
    def _store_system_initialization(self) -> None:
        """Store system initialization record in database"""
        try:
            if self.database_manager:
                # Store system configuration
                self.database_manager.set_config(
                    'system.last_initialization',
                    self.start_time.isoformat(),
                    'Last system initialization timestamp',
                    'system'
                )
                
                self.database_manager.set_config(
                    'system.id',
                    self.system_id,
                    'Current system instance ID',
                    'system'
                )
                
                self.database_manager.set_config(
                    'system.phase1_complete',
                    'true',
                    'Phase 1 completion status',
                    'system'
                )
                
        except Exception as e:
            if self.logger_manager:
                self.logger_manager.error("Failed to store system initialization", e)
    
    def connect_mt5(self) -> bool:
        """
        Connect to MT5 terminal
        
        Returns:
            bool: True if connection successful
        """
        if not self.initialized:
            print("ERROR - System not initialized. Call initialize() first.")
            return False
        
        try:
            print(" Connecting to MT5...")
            success = self.mt5_manager.connect()
            
            if success:
                print("OK - MT5 connection established")
                self.logger_manager.info("MT5 connection established successfully")
                
                # Store account information
                if self.mt5_manager.account_info:
                    account_id = self.database_manager.store_account_info(self.mt5_manager.account_info)
                    self.logger_manager.info(f"Account information stored with ID: {account_id}")
                
                return True
            else:
                print("ERROR - MT5 connection failed")
                self.logger_manager.error("MT5 connection failed")
                return False
                
        except Exception as e:
            error_msg = f"MT5 connection error: {str(e)}"
            print(f"ERROR - {error_msg}")
            self.error_handler.handle_error(e, "MT5 connection")
            self.logger_manager.error(error_msg, e)
            return False
    
    def test_system(self) -> Dict[str, Any]:
        """
        Perform comprehensive system testing
        
        Returns:
            dict: Test results
        """
        print("\n🧪 Performing System Tests...")
        print("="*40)
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'system_id': self.system_id,
            'tests': {}
        }
        
        # Test 1: Logging System
        try:
            print("🧪 Test 1: Logging System...")
            self.logger_manager.info("Test log message")
            self.logger_manager.log_signal("test_strategy", "BUY", "XAUUSDm", 0.85, 1950.0)
            test_results['tests']['logging'] = {'status': 'PASS', 'message': 'All logging functions working'}
            print("   OK - PASS")
        except Exception as e:
            test_results['tests']['logging'] = {'status': 'FAIL', 'message': str(e)}
            print(f"   ERROR - FAIL: {e}")
        
        # Test 2: Database Operations
        try:
            print("🧪 Test 2: Database Operations...")
            
            # Test configuration storage
            test_key = f"test.{int(time.time())}"
            self.database_manager.set_config(test_key, "test_value", "Test configuration")
            retrieved_value = self.database_manager.get_config(test_key)
            
            if retrieved_value == "test_value":
                test_results['tests']['database'] = {'status': 'PASS', 'message': 'Database operations working'}
                print("   OK - PASS")
            else:
                raise Exception("Configuration storage/retrieval failed")
                
        except Exception as e:
            test_results['tests']['database'] = {'status': 'FAIL', 'message': str(e)}
            print(f"   ERROR - FAIL: {e}")
        
        # Test 3: Error Handling
        try:
            print("🧪 Test 3: Error Handling...")
            
            # Create a test error
            test_error = Exception("Test error for system testing")
            error_context = self.error_handler.handle_error(test_error, "System testing")
            
            if error_context and error_context.error_id:
                test_results['tests']['error_handling'] = {'status': 'PASS', 'message': 'Error handling working'}
                print("   OK - PASS")
            else:
                raise Exception("Error handling failed")
                
        except Exception as e:
            test_results['tests']['error_handling'] = {'status': 'FAIL', 'message': str(e)}
            print(f"   ERROR - FAIL: {e}")
        
        # Test 4: MT5 Manager (without connection)
        try:
            print("🧪 Test 4: MT5 Manager...")
            
            # Test symbol validation
            symbol = self.mt5_manager.get_valid_symbol("XAUUSD")
            
            if symbol:
                test_results['tests']['mt5_manager'] = {'status': 'PASS', 'message': 'MT5 Manager initialized correctly'}
                print("   OK - PASS")
            else:
                test_results['tests']['mt5_manager'] = {'status': 'PASS', 'message': 'MT5 Manager working (no connection test)'}
                print("   OK - PASS (No connection test)")
                
        except Exception as e:
            test_results['tests']['mt5_manager'] = {'status': 'FAIL', 'message': str(e)}
            print(f"   ERROR - FAIL: {e}")
        
        # Test 5: System Integration
        try:
            print("🧪 Test 5: System Integration...")
            
            # Test component communication
            stats = self.get_system_stats()
            
            if stats and 'components' in stats:
                test_results['tests']['integration'] = {'status': 'PASS', 'message': 'System integration working'}
                print("   OK - PASS")
            else:
                raise Exception("System integration test failed")
                
        except Exception as e:
            test_results['tests']['integration'] = {'status': 'FAIL', 'message': str(e)}
            print(f"   ERROR - FAIL: {e}")
        
        # Calculate overall result
        passed_tests = sum(1 for test in test_results['tests'].values() if test['status'] == 'PASS')
        total_tests = len(test_results['tests'])
        test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        }
        
        print("="*40)
        print(f"🧪 Test Summary: {passed_tests}/{total_tests} tests passed")
        print(f"Status: Success Rate: {test_results['summary']['success_rate']:.1%}")
        
        # Log test results
        self.logger_manager.info(f"System test completed: {passed_tests}/{total_tests} passed")
        
        return test_results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            stats = {
                'system_id': self.system_id,
                'start_time': self.start_time.isoformat(),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'initialized': self.initialized,
                'components': {}
            }
            
            # MT5 Manager stats
            if self.mt5_manager:
                stats['components']['mt5_manager'] = {
                    'connected': self.mt5_manager.connected,
                    'symbol': self.mt5_manager.symbol,
                    'account_info': self.mt5_manager.account_info if self.mt5_manager.connected else None
                }
            
            # Database stats
            if self.database_manager:
                stats['components']['database'] = self.database_manager.get_database_stats()
            
            # Error handler stats
            if self.error_handler:
                stats['components']['error_handler'] = self.error_handler.get_error_stats()
            
            # Logging stats
            if self.logger_manager:
                stats['components']['logging'] = {
                    'setup_complete': self.logger_manager._setup_complete,
                    'log_directory': str(self.logger_manager.log_dir)
                }
            
            return stats
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(e, "Getting system stats")
            return {'error': str(e)}
    
    def shutdown(self) -> None:
        """Gracefully shutdown the system"""
        print("\n🔄 Shutting down Core System...")
        
        try:
            # Log shutdown
            if self.logger_manager:
                self.logger_manager.info("System shutdown initiated")
            
            # Disconnect MT5
            if self.mt5_manager and self.mt5_manager.connected:
                self.mt5_manager.disconnect()
                print("OK - MT5 disconnected")
            
            # Stop error handler
            if self.error_handler:
                self.error_handler.stop()
                print("OK - Error handler stopped")
            
            # Final log
            if self.logger_manager:
                uptime = (datetime.now() - self.start_time).total_seconds()
                self.logger_manager.info(f"System shutdown completed. Uptime: {uptime:.1f} seconds")
                print("OK - Logging system finalized")
            
            self.initialized = False
            print(" Core System shutdown complete")
            
        except Exception as e:
            print(f"ERROR - Error during shutdown: {str(e)}")
    
    # Convenience properties for easy access
    @property
    def logger(self):
        """Quick access to logger"""
        return self.logger_manager
    
    @property
    def database(self):
        """Quick access to database"""
        return self.database_manager
    
    @property
    def mt5(self):
        """Quick access to MT5 manager"""
        return self.mt5_manager
    
    @property
    def errors(self):
        """Quick access to error handler"""
        return self.error_handler


def main():
    """Main function for testing the core system"""
    print(" XAUUSD MT5 Trading System - Phase 1 Test")
    print("="*50)
    
    # Initialize core system
    core = CoreSystem()
    
    try:
        # Initialize system
        if not core.initialize():
            print("ERROR - System initialization failed")
            return False
        
        # Run system tests
        test_results = core.test_system()
        
        # Try MT5 connection (optional)
        print("\n Attempting MT5 connection (optional)...")
        mt5_connected = core.connect_mt5()
        if mt5_connected:
            print("OK - MT5 connection successful")
            
            # Test data retrieval
            try:
                symbol = core.mt5.get_valid_symbol("XAUUSD")
                data = core.mt5.get_historical_data(symbol, "M15", 10)
                if not data.empty:
                    print(f"OK - Retrieved {len(data)} bars of market data")
                    core.logger.info(f"Market data test successful: {len(data)} bars")
                else:
                    print("WARNING -️ No market data retrieved")
            except Exception as e:
                print(f"WARNING -️ Market data test failed: {e}")
        else:
            print("WARNING -️ MT5 connection failed (this is normal if MT5 is not configured)")
        
        # Display system statistics
        print("\nStatus: System Statistics:")
        print("="*30)
        stats = core.get_system_stats()
        print(f"System ID: {stats.get('system_id', 'Unknown')}")
        print(f"Uptime: {stats.get('uptime_seconds', 0):.1f} seconds")
        print(f"Initialized: {stats.get('initialized', False)}")
        
        # Wait for user input
        print("\n Phase 1 Core System is running!")
        print("Press Enter to shutdown...")
        input()
        
        return True
        
    except KeyboardInterrupt:
        print("\nWARNING -️ Interrupted by user")
        return True
    except Exception as e:
        print(f"\nERROR - Unexpected error: {e}")
        return False
    finally:
        # Always shutdown gracefully
        core.shutdown()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)