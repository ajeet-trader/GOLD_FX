"""
AutoTrading Issue Fix Script

This script attempts to resolve common AutoTrading disabled issues
by checking and trying to fix MT5 AutoTrading settings programmatically.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import MetaTrader5 as mt5
import time
import os

def fix_autotrading_issues():
    """Attempt to fix common AutoTrading issues"""
    print("🔧 AUTOTRADING ISSUE FIX UTILITY")
    print("="*50)
    
    # Step 1: Initialize MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5. Ensure MT5 terminal is running.")
        return False
    
    print("✅ MT5 initialized successfully")
    
    # Step 2: Check current status
    terminal_info = mt5.terminal_info()
    account_info = mt5.account_info()
    
    print(f"\n📊 CURRENT STATUS:")
    print(f"Terminal Trade Allowed: {terminal_info.trade_allowed}")
    print(f"Account Trade Allowed: {account_info.trade_allowed}")
    print(f"Account: {account_info.login}")
    print(f"Server: {account_info.server}")
    
    # Step 3: Try to enable symbol
    symbol = "XAUUSDm"
    symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info is None:
        print(f"\n⚠️ Symbol {symbol} not found, trying to select...")
        if mt5.symbol_select(symbol, True):
            print(f"✅ {symbol} selected successfully")
            symbol_info = mt5.symbol_info(symbol)
        else:
            print(f"❌ Failed to select {symbol}")
            return False
    
    print(f"\n📈 SYMBOL INFO ({symbol}):")
    print(f"Visible: {symbol_info.visible}")
    print(f"Trade Mode: {symbol_info.trade_mode}")
    print(f"Current Bid/Ask: {symbol_info.bid}/{symbol_info.ask}")
    
    # Step 4: Test order capability
    print(f"\n🧪 TESTING ORDER CAPABILITY:")
    
    # Create a minimal test order structure (not sent)
    test_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_BUY,
        "price": symbol_info.ask,
        "deviation": 20,
        "magic": 123456,
        "comment": "AutoTrading test",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    print("✅ Test order structure created successfully")
    
    # Check if we can get order check (without actually placing)
    try:
        # This should show us what would happen if we tried to place an order
        check_result = mt5.order_check(test_request)
        if check_result is None:
            print("❌ Order check returned None - MT5 issue")
        else:
            print(f"📋 Order Check Result:")
            print(f"   Retcode: {check_result.retcode}")
            print(f"   Balance: {check_result.balance}")
            print(f"   Equity: {check_result.equity}")
            print(f"   Margin: {check_result.margin}")
            print(f"   Free Margin: {check_result.margin_free}")
            
            if check_result.retcode == mt5.TRADE_RETCODE_DONE:
                print("✅ Order check passed - AutoTrading should work")
            else:
                print(f"⚠️ Order check failed: {check_result.retcode}")
                
                # Common error codes
                if check_result.retcode == 10027:
                    print("   → AutoTrading is disabled in terminal settings")
                elif check_result.retcode == 10018:
                    print("   → Trading is disabled (check account/market hours)")
                elif check_result.retcode == 10015:
                    print("   → Invalid prices")
                elif check_result.retcode == 10013:
                    print("   → Invalid request parameters")
                
    except Exception as e:
        print(f"❌ Error during order check: {e}")
    
    # Step 5: Provide fix recommendations
    print(f"\n💡 AUTOTRADING FIX RECOMMENDATIONS:")
    print("-" * 40)
    
    if not terminal_info.trade_allowed:
        print("🔴 CRITICAL: Terminal AutoTrading is DISABLED")
        print("   FIX: MT5 → Tools → Options → Expert Advisors")
        print("        ✅ Allow automated trading")
        print("        ✅ Allow DLL imports")
        print("        ✅ Allow imports of external experts")
        print("   Then restart MT5 terminal")
        return False
    
    if not account_info.trade_allowed:
        print("🔴 CRITICAL: Account trading is DISABLED")
        print("   FIX: Contact your broker to enable trading")
        print("        OR wait for market hours")
        return False
    
    print("✅ AutoTrading appears to be enabled correctly")
    print("   If orders still fail, try:")
    print("   1. Restart MT5 terminal completely")
    print("   2. Check if it's market hours")
    print("   3. Verify account has sufficient margin")
    print("   4. Check Windows Firewall/Antivirus")
    
    mt5.shutdown()
    return True

def restart_mt5_connection():
    """Restart MT5 connection"""
    print("\n🔄 RESTARTING MT5 CONNECTION...")
    
    # Shutdown current connection
    mt5.shutdown()
    time.sleep(2)
    
    # Reconnect
    if mt5.initialize():
        print("✅ MT5 reconnected successfully")
        return True
    else:
        print("❌ Failed to reconnect to MT5")
        return False

def main():
    """Main fix function"""
    try:
        success = fix_autotrading_issues()
        
        if not success:
            print("\n🔄 Attempting MT5 restart...")
            if restart_mt5_connection():
                print("✅ Try running your trading system again")
            else:
                print("❌ Manual MT5 restart required")
        
    except Exception as e:
        print(f"❌ Error during fix: {e}")
    
    finally:
        mt5.shutdown()
        print(f"\n🏁 AutoTrading fix utility completed")

if __name__ == "__main__":
    main()