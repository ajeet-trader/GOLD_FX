import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.execution_engine import ExecutionEngine
import logging

logging.basicConfig(level=logging.INFO)

print("🧪 Testing Mode Validation Fix")
print("=" * 60)

# Test 1: Force Live Mode
print("\n1. Testing LIVE mode (should fail without MT5)...")
try:
    config_live = {
        'mode': 'live',
        'execution': {'magic_number': 123456}
    }
    engine_live = ExecutionEngine(config_live)
    print(f"✅ Live mode successful: {engine_live.mode}")
    engine_live.stop_engine()
except Exception as e:
    print(f"❌ Live mode failed (expected): {e}")

# Test 2: Mock Mode
print("\n2. Testing MOCK mode...")
try:
    config_mock = {
        'mode': 'mock',
        'execution': {'magic_number': 123456}
    }
    engine_mock = ExecutionEngine(config_mock)
    print(f"✅ Mock mode successful: {engine_mock.mode}")
    print(f"   Mock balance: ${engine_mock.mt5_manager.get_account_balance()}")
    print(f"   Mock gold price: {engine_mock.mt5_manager.price_data['XAUUSDm']}")
    engine_mock.stop_engine()
except Exception as e:
    print(f"❌ Mock mode failed: {e}")

print("\n" + "=" * 60)
print("Mode validation test completed!")
