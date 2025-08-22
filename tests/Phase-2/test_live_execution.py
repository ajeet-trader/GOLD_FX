# test_live_execution.py
import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.execution_engine import ExecutionEngine, ExecutionStatus
from src.core.base import Signal, SignalType, SignalGrade
from datetime import datetime
import logging
import time

# Initialize config (force live mode)
config = {
    'mode': 'live',
    'execution': {
        'min_confidence': 0.6,
        'signal_age_threshold': 300
    }
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

print("Initializing ExecutionEngine...")
engine = ExecutionEngine(config)
print(f"Engine mode: {engine.mode}")

# Create a signal WITHOUT SL/TP (like your order_flow strategy)
test_signal = Signal(
    timestamp=datetime.now(),
    symbol="XAUUSDm",
    strategy_name="order_flow",
    signal_type=SignalType.BUY,
    confidence=0.95,
    price=3328.68,
    timeframe="M15",
    strength=0.9,
    grade=SignalGrade.A,
    stop_loss=None,   # Let engine calculate
    take_profit=None  # Let engine calculate
)

print("\nProcessing signal without SL/TP...")
print(f"  Strategy: {test_signal.strategy_name}")
print(f"  Type: {test_signal.signal_type.value}")
print(f"  Price: {test_signal.price}")
print(f"  Confidence: {test_signal.confidence}")

# Process the signal
result = engine.process_signal(test_signal)

print(f"\n📊 Execution Result:")
print(f"  Status: {result.status.value}")
print(f"  Ticket: {result.ticket}")
print(f"  Executed Price: {result.executed_price}")
print(f"  Stop Loss: {result.stop_loss}")
print(f"  Take Profit: {result.take_profit}")
print(f"  Error: {result.error_message}")

if result.status == ExecutionStatus.EXECUTED:
    print("\n✅ SUCCESS! Trade executed with auto-calculated SL/TP")
    
    # Wait a moment then close
    time.sleep(3)
    
    if result.ticket:
        print(f"\nClosing position {result.ticket}...")
        close_result = engine.mt5_manager.close_position(result.ticket)
        if close_result.get('success'):
            print(f"✅ Position closed at {close_result.get('price')}")
else:
    print(f"\n❌ Execution failed: {result.error_message}")

# Stop engine
engine.stop_engine()
time.sleep(0.2)
print("\n✅ Test completed")
