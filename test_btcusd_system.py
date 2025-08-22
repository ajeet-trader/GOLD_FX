#!/usr/bin/env python3
"""
BTCUSD System Test Script
========================

This script provides a comprehensive test of the trading system using BTCUSD configuration.
It tests all major components and provides detailed output for debugging.

Usage:
    python test_btcusd_system.py
"""

import sys
import os
from pathlib import Path
import yaml
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_configuration():
    """Test that BTCUSD configuration loads correctly"""
    print("🔧 Testing Configuration Loading...")
    
    try:
        config_path = Path(__file__).parent / "config" / "master_config.yaml"
        
        if not config_path.exists():
            print("❌ Master configuration file not found!")
            return False
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        symbol = config.get('trading', {}).get('symbol', 'Unknown')
        mode = config.get('mode', 'Unknown')
        magic_number = config.get('execution', {}).get('magic_number', 'Unknown')
        
        print(f"   ✅ Configuration loaded successfully")
        print(f"   📊 Symbol: {symbol}")
        print(f"   🔄 Mode: {mode}")
        print(f"   🎯 Magic Number: {magic_number}")
        
        if symbol == "BTCUSDm":
            print("   ✅ BTCUSD configuration detected")
            return True
        else:
            print(f"   ⚠️  Expected BTCUSDm, found {symbol}")
            return False
            
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_mt5_manager():
    """Test MT5 Manager with BTCUSD"""
    print("\n🔌 Testing MT5 Manager...")
    
    try:
        from core.mt5_manager import MT5Manager
        from utils.logger import LoggerManager
        
        # Load config
        config_path = Path(__file__).parent / "config" / "master_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize logger
        logger_mgr = LoggerManager(config)
        logger_mgr.setup_logging()
        
        # Initialize MT5 Manager
        mt5_manager = MT5Manager(config, logger_mgr)
        mt5_manager.connect()  # Connect in mock mode
        
        print("   ✅ MT5Manager initialized successfully")
        print(f"   📊 Mode: {mt5_manager.mode}")
        
        # Test data retrieval
        print("   🔍 Testing data retrieval...")
        data = mt5_manager.get_historical_data("BTCUSDm", "M15", 100)
        
        if data is not None and len(data) > 0:
            print(f"   ✅ Retrieved {len(data)} bars of BTCUSD data")
            print(f"   💰 Latest price: {data['Close'].iloc[-1]:.2f}")
            print(f"   📈 High: {data['High'].max():.2f}")
            print(f"   📉 Low: {data['Low'].min():.2f}")
            return True
        else:
            print("   ❌ No data retrieved")
            return False
            
    except Exception as e:
        print(f"   ❌ MT5Manager test failed: {e}")
        traceback.print_exc()
        return False

def test_strategies():
    """Test strategy components with BTCUSD"""
    print("\n🧠 Testing Strategy Components...")
    
    try:
        from strategies.technical.ichimoku import IchimokuStrategy
        from core.mt5_manager import MT5Manager
        from utils.logger import LoggerManager
        
        # Load config
        config_path = Path(__file__).parent / "config" / "master_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize components
        logger_mgr = LoggerManager(config)
        logger_mgr.setup_logging()
        mt5_manager = MT5Manager(config, logger_mgr)
        
        # Test Ichimoku strategy
        print("   📊 Testing Ichimoku Strategy...")
        ichimoku = IchimokuStrategy(config, mt5_manager)
        
        signals = ichimoku.generate_signal("BTCUSDm", "M15")
        print(f"   ✅ Ichimoku generated {len(signals)} signals")
        
        for i, signal in enumerate(signals[:3]):  # Show first 3 signals
            print(f"      Signal {i+1}: {signal.signal_type.value} at {signal.price:.2f} "
                  f"(Confidence: {signal.confidence:.2f})")
        
        return len(signals) >= 0  # Success if no errors
        
    except Exception as e:
        print(f"   ❌ Strategy test failed: {e}")
        traceback.print_exc()
        return False

def test_execution_engine():
    """Test execution engine with BTCUSD"""
    print("\n⚡ Testing Execution Engine...")
    
    try:
        from core.execution_engine import ExecutionEngine
        from core.mt5_manager import MT5Manager
        from core.risk_manager import RiskManager
        from utils.logger import LoggerManager
        from utils.database import DatabaseManager
        
        # Load config
        config_path = Path(__file__).parent / "config" / "master_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize components
        logger_mgr = LoggerManager(config)
        logger_mgr.setup_logging()
        
        mt5_manager = MT5Manager(config, logger_mgr)
        mt5_manager.connect()  # Connect in mock mode
        database = DatabaseManager(config)
        database.initialize_database()  # Initialize database tables
        risk_manager = RiskManager(config, mt5_manager, database)
        
        # Initialize execution engine
        execution_engine = ExecutionEngine(config, mt5_manager, risk_manager, database, logger_mgr)
        
        print("   ✅ ExecutionEngine initialized successfully")
        print(f"   🎯 Magic Number: {execution_engine.config.get('execution', {}).get('magic_number', 'Unknown')}")
        print(f"   📊 Symbol: BTCUSDm")
        print(f"   🔄 Mode: {execution_engine.mode}")
        
        # Test signal validation (without actual trading)
        print("   🔍 Testing signal validation...")
        
        # Create a mock signal for testing
        from core.base import TradingSignal, SignalType
        test_signal = TradingSignal(
            symbol="BTCUSDm",
            signal_type=SignalType.BUY,
            confidence=0.75,
            price=45000.0,
            strategy="test",
            timeframe="M15"
        )
        
        # Test signal validation
        is_valid = execution_engine._validate_signal(test_signal)
        print(f"   ✅ Signal validation: {'PASSED' if is_valid else 'FAILED'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ExecutionEngine test failed: {e}")
        traceback.print_exc()
        return False

def test_database():
    """Test database operations with BTCUSD"""
    print("\n💾 Testing Database Operations...")
    
    try:
        from utils.database import DatabaseManager
        from utils.logger import LoggerManager
        
        # Load config
        config_path = Path(__file__).parent / "config" / "master_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize components
        logger_mgr = LoggerManager(config)
        logger_mgr.setup_logging()
        
        database = DatabaseManager(config)
        database.initialize_database()  # Initialize database tables
        
        print("   ✅ Database initialized successfully")
        print(f"   📁 Database path: {config.get('database', {}).get('sqlite', {}).get('path', 'Unknown')}")
        
        # Test account creation/retrieval
        account_data = {
            'login': 123456,
            'balance': 100.0,
            'equity': 100.0,
            'margin': 0.0,
            'free_margin': 100.0,
            'server': 'test_server'
        }
        account_id = database.store_account_info(account_data)
        print(f"   ✅ Account ID: {account_id}")
        
        # Test signal storage
        signal_data = {
            'symbol': 'BTCUSDm',
            'strategy': 'test',
            'signal_type': 'BUY',
            'confidence': 0.75,
            'price': 45000.0,
            'timeframe': 'M15'
        }
        
        signal_id = database.store_signal(signal_data)
        print(f"   ✅ Stored test signal with ID: {signal_id}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("🚀 BTCUSD System Comprehensive Test")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("MT5 Manager", test_mt5_manager),
        ("Strategies", test_strategies),
        ("Execution Engine", test_execution_engine),
        ("Database", test_database)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name:<20} {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! BTCUSD system is ready for testing.")
        print("\n📋 Next Steps:")
        print("   1. Run 'python run_system.py' to start the full system")
        print("   2. Check logs in btcusd_*.log files")
        print("   3. Monitor dashboard on port 8502")
        print("   4. Use 'python switch_to_btcusd.py --rollback' to return to GOLD")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("   Consider running 'python switch_to_btcusd.py --rollback' if needed")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
