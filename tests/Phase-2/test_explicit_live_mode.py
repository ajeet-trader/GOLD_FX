import sys
import os
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Force live mode by setting environment variable
os.environ['TRADING_MODE'] = 'live'

from src.core.execution_engine import ExecutionEngine
import logging

logging.basicConfig(level=logging.INFO)

print("🧪 Testing EXPLICIT Live Mode (Should Fail)")
print("=" * 60)

try:
    # Explicitly force live mode
    config = {
        'mode': 'live',  # Config says live
        'execution': {
            'min_confidence': 0.6,
            'magic_number': 123456
        }
    }
    
    print("🔴 Attempting to create ExecutionEngine in LIVE mode...")
    engine = ExecutionEngine(config)
    
    # If we get here, something is wrong
    print(f"❌ PROBLEM: Expected failure but got mode: {engine.mode}")
    
    if engine.mode == 'live':
        print("✅ Live mode working (unexpected - you have MT5 connected)")
    else:
        print("❌ Live mode silently fell back to mock (this is the bug)")
    
    engine.stop_engine()
    
except RuntimeError as e:
    print(f"✅ EXPECTED: Live mode failed correctly: {e}")
except ConnectionError as e:
    print(f"✅ EXPECTED: Connection failed correctly: {e}")
except Exception as e:
    print(f"✅ EXPECTED: Live mode failed with: {e}")

print("\n" + "=" * 60)
