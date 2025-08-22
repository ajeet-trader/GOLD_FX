# test_main_thread_execution.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import threading
from src.core.execution_engine import ExecutionEngine
from src.core.base import Signal, SignalType, SignalGrade
from datetime import datetime

def main():
    """Ensure execution happens in main thread"""
    
    print(f"Running in main thread: {threading.current_thread() == threading.main_thread()}")
    
    config = {
        'mode': 'live',
        'execution': {
            'min_confidence': 0.6,
            'signal_age_threshold': 300
        }
    }
    
    engine = ExecutionEngine(config)
    
    # Create signal
    signal = Signal(
        timestamp=datetime.now(),
        symbol="XAUUSDm",
        strategy_name="main_thread_test",
        signal_type=SignalType.BUY,
        confidence=0.85,
        price=3328.68,
        timeframe="M15",
        grade=SignalGrade.A
    )
    
    # Process in main thread
    result = engine.process_signal(signal)
    print(f"Result: {result.status.value}")
    
    engine.stop_engine()

if __name__ == "__main__":
    main()
