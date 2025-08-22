# test_complete_execution.py
import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.execution_engine import ExecutionEngine, ExecutionStatus
from src.core.base import Signal, SignalType, SignalGrade
from datetime import datetime
import logging
import time

# Initialize config (use simple dict for mock mode)
config = {
    'mode': 'live',
    'execution': {
        'min_confidence': 0.6,
        'signal_age_threshold': 300
    }
}

# Configure logging to see engine activity
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

print("Initializing ExecutionEngine...")
engine = ExecutionEngine(config)
print(f"Engine mode: {engine.mode}")

# Test signal WITHOUT stop_loss and take_profit
test_signal = Signal(
    timestamp=datetime.now(),
    symbol="XAUUSDm",
    strategy_name="test_strategy",
    signal_type=SignalType.BUY,
    confidence=0.85,
    price=3328.68,
    timeframe="M15",
    strength=0.8,
    grade=SignalGrade.A,
    stop_loss=None,  # Let engine calculate
    take_profit=None  # Let engine calculate
)

print("Testing signal without SL/TP...")
result = engine.process_signal(test_signal)

print(f"\nResult:")
print(f"  Status: {result.status.value}")
print(f"  Ticket: {result.ticket}")
print(f"  Executed Price: {result.executed_price}")
print(f"  Stop Loss: {result.stop_loss}")
print(f"  Take Profit: {result.take_profit}")
print(f"  Error: {result.error_message}")

if result.status == ExecutionStatus.EXECUTED:
    print("\n✅ SUCCESS! Signal executed with auto-calculated SL/TP")
else:
    print(f"\n❌ Failed: {result.error_message}")

engine.stop_engine()
# Give a brief moment for background thread to wind down
time.sleep(0.2)
print("Done.")