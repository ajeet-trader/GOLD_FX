import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


import MetaTrader5 as mt5
from datetime import datetime

from src.core.mt5_manager import MT5Manager

# Test with your fixed MT5Manager
mt5_mgr = MT5Manager(symbol="XAUUSDm", magic_number=123456)

# Connect to MT5
if not mt5_mgr.connect():
    print("Failed to connect to MT5")
    exit()

print(f"✅ Connected to MT5")
print(f"   Balance: ${mt5_mgr.get_account_balance()}")

# Test order execution
result = mt5_mgr.place_market_order(
    symbol="XAUUSDm",
    order_type="BUY",
    volume=0.01,
    comment="Test fixed execution"
)

print(f"\n📊 Order Result:")
print(f"   Success: {result.get('success')}")
print(f"   RetCode: {result.get('retcode')}")
print(f"   Ticket: {result.get('ticket')}")
print(f"   Price: {result.get('price')}")
print(f"   Comment: {result.get('comment')}")

if result.get('success'):
    print(f"\n✅ ORDER EXECUTED SUCCESSFULLY!")
    
    # Test position close
    import time
    time.sleep(2)  # Wait a bit
    
    ticket = result.get('ticket')
    if ticket:
        close_result = mt5_mgr.close_position(ticket)
        print(f"\n📊 Close Result:")
        print(f"   Success: {close_result.get('success')}")
        print(f"   Price: {close_result.get('price')}")
else:
    print(f"\n❌ Order failed with retcode: {result.get('retcode')}")

mt5_mgr.disconnect()