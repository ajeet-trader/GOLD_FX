"""
MT5 AutoTrading Detailed Diagnostic Script

This script performs comprehensive checks on MT5 AutoTrading settings
to identify why orders might be failing with "AutoTrading disabled" errors.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

try:
    import MetaTrader5 as mt5
    print("✅ MetaTrader5 module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MetaTrader5: {e}")
    sys.exit(1)

def check_autotrading_comprehensive():
    """Comprehensive AutoTrading diagnostic"""
    print("="*70)
    print("🔍 MT5 AUTOTRADING COMPREHENSIVE DIAGNOSTIC")
    print("="*70)
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return False
    
    print("✅ MT5 initialized successfully")
    
    try:
        # 1. Check terminal info
        print("\n📊 TERMINAL INFORMATION:")
        print("-" * 30)
        terminal_info = mt5.terminal_info()
        if terminal_info:
            print(f"Terminal Build: {terminal_info.build}")
            print(f"Terminal Company: {terminal_info.company}")
            print(f"Terminal Path: {terminal_info.path}")
            print(f"Data Path: {terminal_info.data_path}")
            print(f"Common Path: {terminal_info.commondata_path}")
            print(f"Language: {terminal_info.language}")
            print(f"CPU Cores: {terminal_info.cpu_cores}")
            
            # Critical AutoTrading checks
            print(f"\n🎯 AUTOTRADING STATUS:")
            print(f"Trade Allowed: {terminal_info.trade_allowed}")
            print(f"Tradeapi Disabled: {terminal_info.tradeapi_disabled}")
            print(f"Mqid: {terminal_info.mqid}")
            print(f"Retransmission: {terminal_info.retransmission}")
            print(f"Connected: {terminal_info.connected}")
            
        else:
            print("❌ Could not retrieve terminal info")
        
        # 2. Check account info
        print(f"\n👤 ACCOUNT INFORMATION:")
        print("-" * 30)
        account_info = mt5.account_info()
        if account_info:
            print(f"Account: {account_info.login}")
            print(f"Server: {account_info.server}")
            print(f"Currency: {account_info.currency}")
            print(f"Balance: ${account_info.balance:.2f}")
            print(f"Equity: ${account_info.equity:.2f}")
            print(f"Margin: ${account_info.margin:.2f}")
            print(f"Free Margin: ${account_info.margin_free:.2f}")
            
            # Critical trading permission checks
            print(f"\n🔐 TRADING PERMISSIONS:")
            print(f"Trade Allowed: {account_info.trade_allowed}")
            print(f"Trade Expert: {account_info.trade_expert}")
            print(f"Margin So Mode: {account_info.margin_so_mode}")
            
        else:
            print("❌ Could not retrieve account info")
        
        # 3. Check symbol info for XAUUSDm
        print(f"\n💰 SYMBOL INFORMATION (XAUUSDm):")
        print("-" * 30)
        symbol_info = mt5.symbol_info("XAUUSDm")
        if symbol_info:
            print(f"Symbol: {symbol_info.name}")
            print(f"Visible: {symbol_info.visible}")
            print(f"Select: {symbol_info.select}")
            print(f"Trade Mode: {symbol_info.trade_mode}")
            print(f"Trade Execution: {symbol_info.trade_execution}")
            print(f"Filling Mode: {symbol_info.filling_mode}")
            print(f"Order Mode: {symbol_info.order_mode}")
            print(f"Expiration Mode: {symbol_info.expiration_mode}")
            print(f"Min Volume: {symbol_info.volume_min}")
            print(f"Max Volume: {symbol_info.volume_max}")
            print(f"Volume Step: {symbol_info.volume_step}")
            print(f"Spread: {symbol_info.spread}")
            print(f"Digits: {symbol_info.digits}")
            
        else:
            print("❌ Could not retrieve XAUUSDm symbol info")
            # Try to select the symbol
            if mt5.symbol_select("XAUUSDm", True):
                print("✅ XAUUSDm symbol selected successfully")
                symbol_info = mt5.symbol_info("XAUUSDm")
                if symbol_info:
                    print(f"After selection - Trade Mode: {symbol_info.trade_mode}")
            else:
                print("❌ Failed to select XAUUSDm symbol")
        
        # 4. Test a simple market info request
        print(f"\n📈 MARKET DATA TEST:")
        print("-" * 30)
        tick = mt5.symbol_info_tick("XAUUSDm")
        if tick:
            print(f"Current Bid: {tick.bid}")
            print(f"Current Ask: {tick.ask}")
            print(f"Last: {tick.last}")
            print(f"Time: {tick.time}")
        else:
            print("❌ Could not retrieve tick data for XAUUSDm")
        
        # 5. Check positions and orders
        print(f"\n📋 CURRENT POSITIONS & ORDERS:")
        print("-" * 30)
        positions = mt5.positions_get()
        orders = mt5.orders_get()
        print(f"Open Positions: {len(positions) if positions else 0}")
        print(f"Pending Orders: {len(orders) if orders else 0}")
        
        # 6. Check last error
        print(f"\n⚠️  LAST ERROR:")
        print("-" * 30)
        last_error = mt5.last_error()
        print(f"Error Code: {last_error[0]}")
        print(f"Error Description: {last_error[1]}")
        
        # 7. Test order capabilities
        print(f"\n🧪 ORDER CAPABILITY TEST:")
        print("-" * 30)
        
        # Check if we can create a test request structure
        try:
            test_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": "XAUUSDm",
                "volume": 0.01,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask if tick else 2500.0,
                "deviation": 20,
                "magic": 123456,
                "comment": "AutoTrading test",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            print("✅ Test order request structure created")
            print(f"Test Price: {test_request['price']}")
            print(f"Test Volume: {test_request['volume']}")
            print("⚠️  NOTE: This is just a structure test - no actual order sent")
            
        except Exception as e:
            print(f"❌ Error creating test order structure: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during diagnostic: {e}")
        return False
    
    finally:
        mt5.shutdown()
        print(f"\n🔚 MT5 connection closed")

def check_dll_and_permissions():
    """Check DLL and permission issues"""
    print(f"\n🔧 DLL AND PERMISSIONS CHECK:")
    print("-" * 30)
    
    import os
    
    # Check if running as administrator
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        print(f"Running as Administrator: {is_admin}")
    except:
        print("Could not check administrator status")
    
    # Check MT5 installation path
    import winreg
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                            r"SOFTWARE\MetaQuotes\Terminal\Common")
        mt5_path = winreg.QueryValueEx(key, "Path")[0]
        print(f"MT5 Installation Path: {mt5_path}")
        winreg.CloseKey(key)
        
        # Check if terminal64.exe exists
        terminal_exe = Path(mt5_path) / "terminal64.exe"
        print(f"Terminal executable exists: {terminal_exe.exists()}")
        
    except Exception as e:
        print(f"Could not check MT5 installation: {e}")

def main():
    """Run comprehensive AutoTrading diagnostic"""
    
    check_dll_and_permissions()
    
    success = check_autotrading_comprehensive()
    
    print("\n" + "="*70)
    print("🎯 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    if success:
        print("\n✅ Basic MT5 connection successful")
        print("\n🔍 If AutoTrading errors persist, check:")
        print("   1. MT5 Terminal -> Tools -> Options -> Expert Advisors")
        print("      - ✅ Allow automated trading")
        print("      - ✅ Allow DLL imports")
        print("      - ✅ Allow imports of external experts")
        print("   2. Check symbol-specific trading permissions")
        print("   3. Verify account trading hours and market status")
        print("   4. Check if broker allows automated trading")
        print("   5. Restart MT5 terminal after enabling AutoTrading")
        print("   6. Check Windows Firewall/Antivirus blocking")
    else:
        print("\n❌ MT5 connection failed")
        print("   1. Ensure MT5 terminal is running")
        print("   2. Check MT5 login credentials")
        print("   3. Verify MetaTrader5 Python package installation")
        print("   4. Run as Administrator if needed")

if __name__ == "__main__":
    main()