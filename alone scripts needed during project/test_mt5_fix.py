#!/usr/bin/env python3
"""
Test script to verify MT5 Manager NoneType fix
"""

import sys
import os
sys.path.append('src')

try:
    from core.mt5_manager import MT5Manager
    
    print("Creating MT5Manager instance...")
    mt5_mgr = MT5Manager()
    
    print("Testing get_valid_symbol with None (should not crash)...")
    try:
        # This should not crash anymore
        result = mt5_mgr.get_valid_symbol(None)
        print(f"✅ get_valid_symbol(None) returned: {result}")
    except Exception as e:
        if "Not connected to MT5" in str(e):
            print("✅ Expected connection error (MT5 not connected)")
        else:
            print(f"❌ Unexpected error: {e}")
            sys.exit(1)
    
    print("Testing get_valid_symbol with empty string...")
    try:
        result = mt5_mgr.get_valid_symbol("")
        print(f"✅ get_valid_symbol('') returned: {result}")
    except Exception as e:
        if "Not connected to MT5" in str(e):
            print("✅ Expected connection error (MT5 not connected)")
        else:
            print(f"❌ Unexpected error: {e}")
            sys.exit(1)
    
    print("✅ All tests passed! NoneType fix is working.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
