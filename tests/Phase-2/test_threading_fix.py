import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.execution_engine import ExecutionEngine
from src.core.base import Signal, SignalType, SignalGrade
from datetime import datetime

print("🧪 Testing Threading Fix")
print("=" * 50)

# Test config
config = {
    'execution': {
        'min_confidence': 0.6,
        'signal_age_threshold': 300,
        'order': {'retry_attempts': 3, 'retry_delay': 1},
        'slippage': {'max_slippage': 3},
        'magic_number': 123456
    },
    'mode': 'live'
}

# Create engine (should have no threading now)
try:
    engine = ExecutionEngine(config)
    print("✅ ExecutionEngine created without threading")
    
    # Verify no threading attributes exist
    threading_attrs = ['monitor_thread', 'execution_lock', 'monitoring_lock', 'monitoring_active']
    for attr in threading_attrs:
        if hasattr(engine, attr):
            print(f"❌ Found threading attribute: {attr}")
        else:
            print(f"✅ No threading attribute: {attr}")
    
    # Create test signal
    signal = Signal(
        timestamp=datetime.now(),
        symbol="XAUUSDm",
        strategy_name="threading_test",
        signal_type=SignalType.BUY,
        confidence=0.85,
        price=3326.00,
        timeframe="M15",
        stop_loss=3320.00,
        take_profit=3335.00
    )
    
    print("\n🔄 Processing test signal...")
    result = engine.process_signal(signal)
    
    print(f"\n📊 Result:")
    print(f"  Status: {result.status.value}")
    print(f"  Ticket: {result.ticket}")
    print(f"  Price: {result.executed_price}")
    print(f"  Error: {result.error_message}")
    
    if result.status.value == "EXECUTED":
        print("\n🎉 SUCCESS! Threading fix worked!")
        print("🚀 Orders should now execute without 'AutoTrading disabled' error")
    elif result.status.value == "FAILED" and "AutoTrading disabled" in result.error_message:
        print("\n❌ Still has threading issue - check MT5 AutoTrading button")
    else:
        print(f"\n⚠️ Different issue: {result.error_message}")
    
    # Clean shutdown
    engine.stop_engine()
    print("\n✅ Engine stopped cleanly (no thread cleanup needed)")
    
except Exception as e:
    print(f"❌ Error during test: {e}")

print("\n" + "=" * 50)
print("Threading fix test completed!")
