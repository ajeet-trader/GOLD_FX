"""
Precise AutoTrading Diagnostic Script

This script identifies exactly why AutoTrading is disabled
and provides specific fix instructions.
"""

import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

try:
    import MetaTrader5 as mt5
    print("✅ MetaTrader5 module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MetaTrader5: {e}")
    sys.exit(1)

def check_autotrading_precise():
    """Precise AutoTrading diagnostic to find the exact issue"""
    print("🔍 PRECISE AUTOTRADING DIAGNOSTIC")
    print("="*50)
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ FAILED: Cannot initialize MT5")
        print("   CAUSE: MT5 terminal is not running or not installed properly")
        print("   FIX: Start MT5 terminal manually first")
        return False
    
    print("✅ MT5 initialized successfully")
    
    # Check 1: Terminal Trading Permission
    print("\n🔍 CHECK 1: TERMINAL TRADING PERMISSIONS")
    print("-" * 30)
    
    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        print("❌ FAILED: Cannot get terminal info")
        return False
    
    print(f"Terminal Trade Allowed: {terminal_info.trade_allowed}")
    print(f"Trade API Disabled: {terminal_info.tradeapi_disabled}")
    print(f"Terminal Connected: {terminal_info.connected}")
    
    if not terminal_info.trade_allowed:
        print("🔴 ISSUE FOUND: Terminal AutoTrading is DISABLED")
        print("   EXACT LOCATION: MT5 → Tools → Options → Expert Advisors")
        print("   REQUIRED SETTINGS:")
        print("     ✅ Allow automated trading")
        print("     ✅ Allow DLL imports")
        print("     ✅ Allow imports of external experts")
        print("   ACTION: Enable these settings and restart MT5")
        return False
    
    if terminal_info.tradeapi_disabled:
        print("🔴 ISSUE FOUND: Trade API is disabled")
        print("   FIX: Enable trading API in MT5 settings")
        return False
    
    print("✅ Terminal trading permissions are correct")
    
    # Check 2: Account Trading Permission
    print("\n🔍 CHECK 2: ACCOUNT TRADING PERMISSIONS")
    print("-" * 30)
    
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ FAILED: Cannot get account info")
        print("   CAUSE: Not logged into MT5 account")
        print("   FIX: Login to your MT5 account first")
        return False
    
    print(f"Account: {account_info.login}")
    print(f"Server: {account_info.server}")
    print(f"Account Trade Allowed: {account_info.trade_allowed}")
    print(f"Account Trade Expert: {account_info.trade_expert}")
    print(f"Balance: ${account_info.balance}")
    
    if not account_info.trade_allowed:
        print("🔴 ISSUE FOUND: Account trading is disabled")
        print("   POSSIBLE CAUSES:")
        print("     - Market is closed")
        print("     - Broker disabled trading for this account")
        print("     - Account verification issues")
        print("   ACTION: Contact your broker")
        return False
    
    if not account_info.trade_expert:
        print("🔴 ISSUE FOUND: Expert Advisor trading disabled for account")
        print("   FIX: Enable EA trading in account settings or contact broker")
        return False
    
    print("✅ Account trading permissions are correct")
    
    # Check 3: Symbol Trading Permissions
    print("\n🔍 CHECK 3: SYMBOL TRADING PERMISSIONS")
    print("-" * 30)
    
    symbol = "XAUUSDm"
    symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info is None:
        print(f"⚠️  Symbol {symbol} not found, attempting to select...")
        if mt5.symbol_select(symbol, True):
            symbol_info = mt5.symbol_info(symbol)
            print(f"✅ {symbol} selected successfully")
        else:
            print(f"❌ FAILED: Cannot select {symbol}")
            print("   CAUSE: Symbol not available on this broker")
            print("   FIX: Use a different symbol or contact broker")
            return False
    
    print(f"Symbol: {symbol}")
    print(f"Visible: {symbol_info.visible}")
    print(f"Trade Mode: {symbol_info.trade_mode}")
    print(f"Current Bid: {symbol_info.bid}")
    print(f"Current Ask: {symbol_info.ask}")
    
    if symbol_info.trade_mode == 0:
        print("🔴 ISSUE FOUND: Trading disabled for this symbol")
        print("   CAUSE: Broker has disabled trading for XAUUSDm")
        print("   FIX: Contact broker or use different symbol")
        return False
    
    print("✅ Symbol trading permissions are correct")
    
    # Check 4: Test Order Validation
    print("\n🔍 CHECK 4: ORDER VALIDATION TEST")
    print("-" * 30)
    
    test_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_BUY,
        "price": symbol_info.ask,
        "deviation": 20,
        "magic": 123456,
        "comment": "Diagnostic test",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    check_result = mt5.order_check(test_request)
    if check_result is None:
        print("❌ FAILED: Order check returned None")
        print("   CAUSE: MT5 internal error or connection issue")
        print("   FIX: Restart MT5 terminal")
        return False
    
    print(f"Order Check Retcode: {check_result.retcode}")
    print(f"Balance Available: ${check_result.balance}")
    print(f"Margin Required: ${check_result.margin}")
    print(f"Free Margin: ${check_result.margin_free}")
    
    # Decode the specific error
    if check_result.retcode == 0:
        print("✅ Order validation passed - AutoTrading should work!")
        return True
    elif check_result.retcode == 10027:
        print("🔴 CRITICAL ISSUE: AutoTrading disabled in terminal")
        print("   EXACT PROBLEM: Expert Advisors are disabled")
        print("   EXACT FIX:")
        print("     1. MT5 → Tools → Options → Expert Advisors")
        print("     2. ✅ Check 'Allow automated trading'")
        print("     3. ✅ Check 'Allow DLL imports'")
        print("     4. Click OK")
        print("     5. Restart MT5 completely")
        return False
    elif check_result.retcode == 10018:
        print("🔴 ISSUE: Trading disabled")
        print("   POSSIBLE CAUSES:")
        print("     - Market hours (expected if market closed)")
        print("     - Account restrictions")
        print("     - Server maintenance")
        return False
    elif check_result.retcode == 10015:
        print("🔴 ISSUE: Invalid price")
        print("   CAUSE: Price feeds not working")
        return False
    elif check_result.retcode == 10013:
        print("🔴 ISSUE: Invalid request parameters")
        print("   CAUSE: Order parameters are incorrect")
        return False
    else:
        print(f"🔴 UNKNOWN ISSUE: RetCode {check_result.retcode}")
        print("   ACTION: Research MT5 error codes or contact support")
        return False

def main():
    """Main diagnostic function"""
    try:
        print("Starting AutoTrading diagnostic...")
        print("This will identify exactly why AutoTrading is failing.")
        print()
        
        success = check_autotrading_precise()
        
        print("\n" + "="*50)
        print("📋 DIAGNOSTIC SUMMARY")
        print("="*50)
        
        if success:
            print("🟢 RESULT: AutoTrading is correctly configured")
            print("   Your system should work when market is open")
        else:
            print("🔴 RESULT: AutoTrading has configuration issues")
            print("   Follow the specific FIX instructions above")
            
        print("\n💡 NEXT STEPS:")
        if success:
            print("   1. Your AutoTrading is working correctly")
            print("   2. Test during market hours for full validation")
            print("   3. Market closed errors are normal outside trading hours")
        else:
            print("   1. Apply the specific fixes shown above")
            print("   2. Restart MT5 terminal completely")
            print("   3. Re-run this diagnostic to verify fixes")
            print("   4. Then test your trading system")
        
    except Exception as e:
        print(f"❌ Diagnostic error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()