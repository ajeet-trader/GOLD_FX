#!/usr/bin/env python3
"""
Main System Runner
==================
Quick launcher for the XAUUSD Trading System

Usage:
    python run_system.py           # Start system (interactive with task pump)
    python run_system.py --test    # Run tests
    python run_system.py --setup   # Re-run setup
    python run_system.py --mode mock/live  # Set mode
"""



import sys
import argparse
import time
import threading
import queue
import importlib.util
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import CLI args utility
try:
    from src.utils.cli_args import parse_mode, print_mode_banner
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False
    def parse_mode():
        return "mock"
    def print_mode_banner(mode):
        pass

try:
    from src.phase_1_core_integration import CoreSystem
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure Phase 1 setup is complete.")
    sys.exit(1)

# Phase 2 Execution Engine (for live/mock execution and MT5 main-thread task pump)
try:
    from src.core.execution_engine import ExecutionEngine
except Exception as e:
    ExecutionEngine = None  # Will guard usage below


def main():
    parser = argparse.ArgumentParser(description='XAUUSD Trading System')
    parser.add_argument('--test', action='store_true', help='Run system tests')
    parser.add_argument('--setup', action='store_true', help='Re-run setup')
    parser.add_argument('--connect', action='store_true', help='Test MT5 connection')
    parser.add_argument('--mode', choices=['mock', 'live'], help='Run mode (mock/live)')
    
    args = parser.parse_args()
    
    # Print mode banner if CLI is available
    if CLI_AVAILABLE:
        mode = parse_mode()
        print_mode_banner(mode)
    
    if args.setup:
        # Dynamically load tests/Phase-1/setup_phase1.py (hyphenated dir)
        setup_path = Path(__file__).parent / 'tests' / 'Phase-1' / 'setup_phase1.py'
        if setup_path.exists():
            spec = importlib.util.spec_from_file_location("phase1_setup_loader", str(setup_path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, 'Phase1Setup'):
                    setup = getattr(mod, 'Phase1Setup')(include_dev=True, run_tests=True)
                    return setup.run_setup()
        print("⚠️ Phase 1 setup script not found or failed to load. Skipping setup.")
        return False
    
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
            # Start interactive mode with main-thread task pump
            print("🎯 XAUUSD Trading System - Interactive Mode")
            print("Commands:")
            print("  connect  - Connect to MT5")
            print("  test     - Run system tests")
            print("  stats    - Show system statistics")
            print("  quit     - Exit system")

            # Create ExecutionEngine to manage MT5 operations (if available)
            engine = None
            if ExecutionEngine is not None:
                try:
                    # Pass mode from CoreSystem; execution settings optional
                    engine_config = {'mode': core.mode, 'execution': core.config.get('execution', {})}
                    engine = ExecutionEngine(engine_config)
                    print(f"ExecutionEngine initialized in {engine.mode} mode")
                except Exception as e:
                    print(f"⚠️ Failed to initialize ExecutionEngine: {e}")
                    engine = None

            # Non-blocking input via background thread
            cmd_queue: "queue.Queue[str]" = queue.Queue()
            stop_event = threading.Event()

            def _input_worker():
                while not stop_event.is_set():
                    try:
                        cmd = input("\n> ").strip().lower()
                        cmd_queue.put(cmd)
                        if cmd == 'quit':
                            # allow main loop to observe and exit
                            break
                    except EOFError:
                        break
                    except KeyboardInterrupt:
                        cmd_queue.put('quit')
                        break

            input_thread = threading.Thread(target=_input_worker, name="InputThread", daemon=True)
            input_thread.start()

            try:
                # Main-thread loop pumps MT5 tasks regularly
                while True:
                    # 1) Pump MT5 main-thread tasks and update positions
                    if engine is not None:
                        try:
                            engine.process_pending_trades()
                        except Exception:
                            # Keep interactive shell alive even if engine errors
                            pass

                    # 2) Handle user commands (non-blocking)
                    try:
                        cmd = cmd_queue.get_nowait()
                    except queue.Empty:
                        cmd = None

                    if cmd:
                        if cmd == 'quit':
                            print("Exiting...")
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

                    # Short sleep to yield and set pump cadence
                    time.sleep(0.2)

                return True
            finally:
                # Signal input thread to stop and cleanup
                stop_event.set()
                # Graceful engine shutdown
                if engine is not None:
                    try:
                        engine.stop_engine()
                    except Exception:
                        pass
    
    finally:
        core.shutdown()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
