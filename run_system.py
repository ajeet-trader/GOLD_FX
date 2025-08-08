#!/usr/bin/env python3
"""
Main System Runner
==================
Quick launcher for the XAUUSD Trading System

Usage:
    python run_system.py           # Start system
    python run_system.py --test    # Run tests
    python run_system.py --setup   # Re-run setup
"""



import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from src.core_system import CoreSystem
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure Phase 1 setup is complete.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='XAUUSD Trading System')
    parser.add_argument('--test', action='store_true', help='Run system tests')
    parser.add_argument('--setup', action='store_true', help='Re-run setup')
    parser.add_argument('--connect', action='store_true', help='Test MT5 connection')
    
    args = parser.parse_args()
    
    if args.setup:
        from setup_phase1 import Phase1Setup
        setup = Phase1Setup(include_dev=True, run_tests=True)
        return setup.run_setup()
    
    # Initialize core system
    core = CoreSystem()
    
    try:
        # Initialize
        if not core.initialize():
            print("❌ System initialization failed")
            return False
        
        if args.test:
            # Run tests
            test_results = core.test_system()
            return test_results['summary']['success_rate'] > 0.8
        
        elif args.connect:
            # Test MT5 connection
            return core.connect_mt5()
        
        else:
            # Start interactive mode
            print("🎯 XAUUSD Trading System - Interactive Mode")
            print("Commands:")
            print("  connect  - Connect to MT5")
            print("  test     - Run system tests")
            print("  stats    - Show system statistics")
            print("  quit     - Exit system")
            
            while True:
                try:
                    cmd = input("\n> ").strip().lower()
                    
                    if cmd == 'quit':
                        break
                    elif cmd == 'connect':
                        core.connect_mt5()
                    elif cmd == 'test':
                        core.test_system()
                    elif cmd == 'stats':
                        stats = core.get_system_stats()
                        print(f"System Stats: {stats}")
                    else:
                        print("Unknown command. Try: connect, test, stats, quit")
                        
                except KeyboardInterrupt:
                    break
            
            return True
    
    finally:
        core.shutdown()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
