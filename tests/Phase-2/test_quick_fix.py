# test_quick_fix.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import MetaTrader5 as mt5
from src.core.mt5_manager import MT5Manager

# Test 1: Direct MT5 check
print("=== Direct MT5 Status ===")
if mt5.initialize():
    terminal_info = mt5.terminal_info()
    print(f"AutoTrading: {terminal_info.trade_allowed}")
    account_info = mt5.account_info()
    print(f"Account Trading: {account_info.trade_allowed}")
    mt5.shutdown()

# Test 2: MT5Manager check
print("\n=== MT5Manager Status ===")
mgr = MT5Manager(symbol="XAUUSDm")
if mgr.connect():
    # Check if the issue is in the manager
    result = mgr.place_market_order("XAUUSDm", "BUY", 0.01, comment="Quick test")
    print(f"Order success: {result.get('success')}")
    print(f"Error: {result.get('comment')}")
    mgr.disconnect()
