#!/usr/bin/env python3
"""
Real Market Data Strategy Testing
=================================
Tests strategies with ACTUAL current market data, not mock data
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
from datetime import datetime
from typing import Dict, Any

try:
    from src.core.mt5_manager import MT5Manager
    from src.strategies.technical.ichimoku import IchimokuStrategy
    from src.strategies.technical.harmonic import HarmonicStrategy
    from src.strategies.technical.elliott_wave import ElliottWaveStrategy
    from src.utils.logger import get_logger_manager
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure the system is properly set up.")
    sys.exit(1)


def test_with_real_data():
    """Test all strategies with real current market data"""
    
    print("=" * 60)
    print("TESTING STRATEGIES WITH REAL MARKET DATA")
    print("=" * 60)
    
    # Check current XAUUSD price first
    print(f"\nTest Time: {datetime.now()}")
    
    # Initialize MT5 Manager
    print("\n1. Connecting to MetaTrader 5...")
    try:
        mt5_manager = MT5Manager("XAUUSDm")  # Use XAUUSDm for micro lots
        
        if not mt5_manager.connect():
            print("FAILED to connect to MT5")
            print("   - Check MT5 credentials in .env file:")
            print("     MT5_LOGIN=your_account")
            print("     MT5_PASSWORD=your_password") 
            print("     MT5_SERVER=your_broker_server")
            print("   - Ensure MT5 terminal is running and logged in")
            return False
    except Exception as e:
        print(f"FAILED to initialize MT5 Manager: {str(e)}")
        print("   - Check if MetaTrader 5 is installed")
        print("   - Verify .env file exists with correct credentials")
        return False
    
    print("SUCCESS: Connected to MT5!")
    
    # Get current price
    current_data = mt5_manager.get_realtime_data("XAUUSDm") 
    if current_data:
        current_price = current_data.get('bid', 'N/A')
        print(f"Current XAUUSD Price: ${current_price}")
    else:
        print("WARNING: Could not get current price, trying historical data...")
    
    # Test configuration for real trading
    real_config = {
        'parameters': {
            'confidence_threshold': 0.55,
            'lookback_period': 200,
            'timeframe_primary': 'M15'
        }
    }
    
    # Initialize strategies with REAL MT5 manager
    strategies = {
        'Ichimoku': IchimokuStrategy(real_config, mt5_manager),
        'Harmonic': HarmonicStrategy(real_config, mt5_manager), 
        'Elliott Wave': ElliottWaveStrategy(real_config, mt5_manager)
    }
    
    print(f"\n2. Testing {len(strategies)} strategies with REAL market data...")
    
    total_signals = 0
    all_results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\nTesting {strategy_name} Strategy:")
        print("-" * 40)
        
        try:
            # Generate signals with REAL data
            signals = strategy.generate_signal("XAUUSDm", "M15")
            signal_count = len(signals)
            total_signals += signal_count
            
            print(f"   Generated {signal_count} signals")
            
            if signals:
                for i, signal in enumerate(signals, 1):
                    print(f"   Signal {i}:")
                    print(f"     Time: {signal.timestamp}")
                    print(f"     Type: {signal.signal_type.value}")
                    print(f"     Price: ${signal.price:.2f}")
                    print(f"     Confidence: {signal.confidence:.1%}")
                    print(f"     Grade: {signal.grade.value}")
                    if signal.stop_loss:
                        print(f"     Stop Loss: ${signal.stop_loss:.2f}")
                    if signal.take_profit:
                        print(f"     Take Profit: ${signal.take_profit:.2f}")
                    if signal.metadata and 'signal_reason' in signal.metadata:
                        print(f"     Reason: {signal.metadata['signal_reason']}")
                    print()
            else:
                print("   WARNING: No signals generated")
                
            # Store results
            all_results[strategy_name] = {
                'signal_count': signal_count,
                'signals': signals,
                'status': 'success'
            }
                
        except Exception as e:
            print(f"   ERROR: {str(e)}")
            all_results[strategy_name] = {
                'signal_count': 0,
                'signals': [],
                'status': 'error',
                'error': str(e)
            }
    
    # Summary
    print("\n" + "=" * 60)
    print("REAL MARKET DATA TEST SUMMARY")
    print("=" * 60)
    print(f"Test completed at: {datetime.now()}")
    print(f"Current XAUUSD: ${current_price}" if current_data else "Current XAUUSD: Unable to fetch")
    print(f"Total signals generated: {total_signals}")
    print()
    
    for strategy_name, results in all_results.items():
        status_symbol = "PASS" if results['status'] == 'success' else "FAIL"
        print(f"{status_symbol} {strategy_name}: {results['signal_count']} signals")
        if results['status'] == 'error':
            print(f"   Error: {results['error']}")
    
    print(f"\nTarget: 5-10 signals per strategy")
    print(f"Achievement: {total_signals} total signals from {len(strategies)} strategies")
    
    # Cleanup
    mt5_manager.disconnect()
    
    if total_signals >= 5:
        print("\nSUCCESS: Strategies are generating signals from real market data!")
        return True
    else:
        print("\nNeed more signals. Consider adjusting strategy parameters.")
        return False


if __name__ == "__main__":
    try:
        success = test_with_real_data()
        exit_code = 0 if success else 1
        print(f"\nExiting with code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
