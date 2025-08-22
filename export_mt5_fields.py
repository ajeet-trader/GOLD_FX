# import MetaTrader5 as mt5
# import csv

# # List of known MT5 trade request fields
# # (from MT5 docs for TradeRequest structure)
# mt5_fields = [
#     "action", "symbol", "volume", "type",
#     "price", "sl", "tp", "deviation",
#     "magic", "comment", "type_time", "type_filling"
# ]

# # Write to CSV
# with open("mt5_fields.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Field", "Description"])
#     for field in mt5_fields:
#         writer.writerow([field, "MT5 TradeRequest field"])
        
# print("✅ MT5 fields exported to mt5_fields.csv")

# mt5_dictionary_inspector.py

import sys
from pathlib import Path
import pandas as pd
import MetaTrader5 as mt5
import csv
from datetime import datetime
import inspect

# Add your project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class MT5DictionaryInspector:
    """Comprehensive MT5 API field inspector"""
    
    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def connect_mt5(self):
        """Connect to MT5 for inspection"""
        try:
            if not mt5.initialize():
                print("❌ Failed to initialize MT5")
                return False
                
            # Try to login (use your existing connection)
            from src.core.mt5_manager import MT5Manager
            mgr = MT5Manager()
            if mgr.connect():
                print("✅ Connected to MT5 for inspection")
                return True
            else:
                print("⚠️ Using MT5 without login (limited data)")
                return True
        except Exception as e:
            print(f"❌ MT5 connection error: {e}")
            return False
    
    def inspect_order_request_fields(self):
        """Inspect all valid order request fields"""
        print("\n🔍 Inspecting Order Request Fields...")
        
        order_fields = []
        
        # Known order request fields from MT5 documentation
        known_fields = {
            'action': 'TRADE_ACTION_* constant',
            'magic': 'Expert Advisor ID (integer)',
            'order': 'Order ticket (for modify/delete)',
            'symbol': 'Symbol name (string)',
            'volume': 'Volume in lots (double)',
            'price': 'Price (double)',
            'stoplimit': 'Stop limit price (double)',
            'sl': 'Stop loss price (double)',
            'tp': 'Take profit price (double)', 
            'deviation': 'Maximum price deviation (integer)',
            'type': 'ORDER_TYPE_* constant',
            'type_filling': 'ORDER_FILLING_* constant',
            'type_time': 'ORDER_TIME_* constant',
            'expiration': 'Expiration time (datetime)',
            'comment': 'Order comment (string)',
            'position': 'Position ticket (for close)',
            'position_by': 'Opposite position ticket (for close by)'
        }
        
        for field, description in known_fields.items():
            order_fields.append({
                'field': field,
                'description': description,
                'required': field in ['action', 'symbol', 'volume', 'type'],
                'type': self._extract_type(description),
                'category': 'order_request'
            })
            
        self.results['order_fields'] = order_fields
        print(f"✅ Found {len(order_fields)} order request fields")
    
    def inspect_mt5_constants(self):
        """Inspect all MT5 constants and enums"""
        print("\n🔍 Inspecting MT5 Constants...")
        
        constants = []
        
        # Get all MT5 module attributes
        mt5_attrs = dir(mt5)
        
        # Categorize constants
        categories = {
            'TRADE_ACTION': 'Trade actions',
            'ORDER_TYPE': 'Order types', 
            'ORDER_FILLING': 'Order filling types',
            'ORDER_TIME': 'Order time types',
            'TRADE_RETCODE': 'Return codes',
            'SYMBOL': 'Symbol constants',
            'ACCOUNT': 'Account constants',
            'TIMEFRAME': 'Timeframe constants'
        }
        
        for attr_name in mt5_attrs:
            if attr_name.isupper() and '_' in attr_name:
                try:
                    value = getattr(mt5, attr_name)
                    category = 'other'
                    
                    # Categorize
                    for cat_prefix, cat_desc in categories.items():
                        if attr_name.startswith(cat_prefix):
                            category = cat_desc
                            break
                    
                    constants.append({
                        'constant': attr_name,
                        'value': value,
                        'type': type(value).__name__,
                        'category': category
                    })
                except Exception:
                    pass
        
        self.results['constants'] = constants
        print(f"✅ Found {len(constants)} MT5 constants")
    
    def inspect_error_codes(self):
        """Inspect MT5 error codes"""
        print("\n🔍 Inspecting Error Codes...")
        
        # Common MT5 error codes
        error_codes = {
            10000: "TRADE_RETCODE_REQUOTE - Requote",
            10001: "TRADE_RETCODE_REJECT - Request rejected", 
            10002: "TRADE_RETCODE_CANCEL - Request canceled by trader",
            10003: "TRADE_RETCODE_PLACED - Order placed",
            10004: "TRADE_RETCODE_DONE - Request completed",
            10005: "TRADE_RETCODE_DONE_PARTIAL - Only part of the request was completed",
            10006: "TRADE_RETCODE_ERROR - Request processing error",
            10007: "TRADE_RETCODE_TIMEOUT - Request canceled by timeout",
            10008: "TRADE_RETCODE_INVALID - Invalid request",
            10009: "TRADE_RETCODE_INVALID_VOLUME - Invalid volume in the request",
            10010: "TRADE_RETCODE_INVALID_PRICE - Invalid price in the request",
            10011: "TRADE_RETCODE_INVALID_STOPS - Invalid stops in the request",
            10012: "TRADE_RETCODE_TRADE_DISABLED - Trade is disabled",
            10013: "TRADE_RETCODE_MARKET_CLOSED - Market is closed",
            10014: "TRADE_RETCODE_NO_MONEY - There is not enough money to complete the request",
            10015: "TRADE_RETCODE_PRICE_CHANGED - Prices changed",
            10016: "TRADE_RETCODE_PRICE_OFF - There are no quotes to process the request",
            10017: "TRADE_RETCODE_INVALID_EXPIRATION - Invalid order expiration date",
            10018: "TRADE_RETCODE_ORDER_CHANGED - Order state changed",
            10019: "TRADE_RETCODE_TOO_MANY_REQUESTS - Too frequent requests",
            10020: "TRADE_RETCODE_NO_CHANGES - No changes in request",
            10021: "TRADE_RETCODE_SERVER_DISABLES_AT - Autotrading disabled by server",
            10022: "TRADE_RETCODE_CLIENT_DISABLES_AT - Autotrading disabled by client terminal",
            10023: "TRADE_RETCODE_LOCKED - Request locked for processing",
            10024: "TRADE_RETCODE_FROZEN - Order or position frozen",
            10025: "TRADE_RETCODE_INVALID_FILL - Invalid order filling type",
            10026: "TRADE_RETCODE_CONNECTION - No connection with the trade server",
            10027: "TRADE_RETCODE_ONLY_REAL - Operation is allowed only for live accounts",
            10028: "TRADE_RETCODE_LIMIT_ORDERS - The number of pending orders has reached the limit",
            10029: "TRADE_RETCODE_LIMIT_VOLUME - The volume of orders and positions for the symbol has reached the limit",
            10030: "TRADE_RETCODE_INVALID_ORDER - Incorrect or prohibited order type",
        }
        
        error_list = []
        for code, description in error_codes.items():
            error_list.append({
                'error_code': code,
                'description': description,
                'severity': 'critical' if code in [10021, 10022, 10027] else 'normal'
            })
        
        self.results['error_codes'] = error_list
        print(f"✅ Found {len(error_list)} error codes")
    
    def inspect_symbol_info_fields(self):
        """Inspect symbol info structure fields"""
        print("\n🔍 Inspecting Symbol Info Fields...")
        
        try:
            # Try to get symbol info for a known symbol
            symbol_info = mt5.symbol_info("EURUSD")
            if symbol_info is None:
                symbol_info = mt5.symbol_info("XAUUSD") 
            
            symbol_fields = []
            if symbol_info:
                for field in dir(symbol_info):
                    if not field.startswith('_'):
                        try:
                            value = getattr(symbol_info, field)
                            symbol_fields.append({
                                'field': field,
                                'value_type': type(value).__name__,
                                'sample_value': str(value)[:50],
                                'category': 'symbol_info'
                            })
                        except Exception:
                            pass
            
            self.results['symbol_fields'] = symbol_fields
            print(f"✅ Found {len(symbol_fields)} symbol info fields")
            
        except Exception as e:
            print(f"⚠️ Could not inspect symbol info: {e}")
            self.results['symbol_fields'] = []
    
    def inspect_account_info_fields(self):
        """Inspect account info structure fields"""
        print("\n🔍 Inspecting Account Info Fields...")
        
        try:
            account_info = mt5.account_info()
            account_fields = []
            
            if account_info:
                for field in dir(account_info):
                    if not field.startswith('_'):
                        try:
                            value = getattr(account_info, field)
                            account_fields.append({
                                'field': field,
                                'value_type': type(value).__name__,
                                'sample_value': str(value)[:50],
                                'category': 'account_info'
                            })
                        except Exception:
                            pass
            
            self.results['account_fields'] = account_fields
            print(f"✅ Found {len(account_fields)} account info fields")
            
        except Exception as e:
            print(f"⚠️ Could not inspect account info: {e}")
            self.results['account_fields'] = []
    
    def _extract_type(self, description):
        """Extract data type from description"""
        if 'integer' in description.lower():
            return 'int'
        elif 'double' in description.lower() or 'price' in description.lower():
            return 'float'
        elif 'string' in description.lower():
            return 'str'
        elif 'datetime' in description.lower():
            return 'datetime'
        elif 'constant' in description.lower():
            return 'constant'
        else:
            return 'unknown'
    
    def export_to_csv(self):
        """Export all results to CSV files"""
        print(f"\n📊 Exporting results to CSV files...")
        
        output_dir = Path("mt5_inspection_results")
        output_dir.mkdir(exist_ok=True)
        
        exported_files = []
        
        for result_name, data in self.results.items():
            if data:  # Only export non-empty datasets
                filename = f"mt5_{result_name}_{self.timestamp}.csv"
                filepath = output_dir / filename
                
                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False)
                
                exported_files.append(filepath)
                print(f"✅ Exported {result_name}: {len(data)} records → {filename}")
        
        # Create summary file
        summary_file = output_dir / f"mt5_inspection_summary_{self.timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"MT5 Dictionary Inspection Results\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"=" * 50 + "\n\n")
            
            for result_name, data in self.results.items():
                f.write(f"{result_name.upper()}: {len(data)} records\n")
            
            f.write(f"\n" + "=" * 50 + "\n")
            f.write("FILES GENERATED:\n")
            for file in exported_files:
                f.write(f"- {file.name}\n")
        
        print(f"\n📋 Summary saved to: {summary_file.name}")
        return exported_files
    
    def run_full_inspection(self):
        """Run complete MT5 dictionary inspection"""
        print("🚀 Starting MT5 Dictionary Inspection")
        print("=" * 60)
        
        if not self.connect_mt5():
            print("❌ Cannot proceed without MT5 connection")
            return False
        
        try:
            self.inspect_order_request_fields()
            self.inspect_mt5_constants()
            self.inspect_error_codes()
            self.inspect_symbol_info_fields()
            self.inspect_account_info_fields()
            
            exported_files = self.export_to_csv()
            
            print("\n" + "=" * 60)
            print("🎉 MT5 Dictionary Inspection Complete!")
            print(f"📁 Results saved in: mt5_inspection_results/")
            print(f"📊 Files generated: {len(exported_files)}")
            
            # Show the most important file
            order_fields_file = [f for f in exported_files if 'order_fields' in f.name]
            if order_fields_file:
                print(f"🎯 KEY FILE: {order_fields_file[0].name}")
                print("   ^ This shows exactly what fields MT5 order_send() accepts")
            
            return True
            
        except Exception as e:
            print(f"❌ Inspection failed: {e}")
            return False
        
        finally:
            mt5.shutdown()

if __name__ == "__main__":
    inspector = MT5DictionaryInspector()
    inspector.run_full_inspection()
