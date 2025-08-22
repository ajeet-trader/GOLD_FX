# import MetaTrader5 as mt5

# # connect to already running MT5 terminal
# if not mt5.initialize():
#     print("Initialize failed:", mt5.last_error())
# else:
#     print("AutoTrading allowed:", mt5.terminal_info().trade_allowed)
import MetaTrader5 as mt5

# Connect to MT5
if not mt5.initialize():
    print("Failed to initialize MT5:", mt5.last_error())
    exit()

# Show account info
account_info = mt5.account_info()
if account_info is None:
    print("Failed to get account info:", mt5.last_error())
else:
    print(f"Account: {account_info.login}")
    print(f"Balance: {account_info.balance}")

# Symbol to trade
symbol = "XAUUSDm"

# Ensure symbol is available
if not mt5.symbol_select(symbol, True):
    print(f"Failed to select symbol {symbol}")
    exit()

# Prepare order
price = mt5.symbol_info_tick(symbol).ask
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": 0.01,  # lot size
    "type": mt5.ORDER_TYPE_BUY,
    "price": price,
    "deviation": 20,
    "magic": 123456,
    "comment": "Python Test Order",
    "type_time": mt5.ORDER_TIME_GTC,   # Good till canceled
    "type_filling": mt5.ORDER_FILLING_IOC
}

# Send order
result = mt5.order_send(request)

# Print result
if result is None:
    print("Order send failed:", mt5.last_error())
else:
    print("Order result:", result)
    print("Return code:", result.retcode)
