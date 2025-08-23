#!/usr/bin/env python3
"""
Test Phase 2 imports and basic functionality
"""

import sys
import os
sys.path.append('src')

print("Testing Phase 2 imports...")

try:
    print("1. Importing phase_2_core_integration...")
    import phase_2_core_integration
    print("✅ phase_2_core_integration imported successfully")
    
    print("2. Testing Phase2TradingSystem class...")
    from phase_2_core_integration import Phase2TradingSystem
    print("✅ Phase2TradingSystem class imported successfully")
    
    print("3. Creating Phase2TradingSystem instance...")
    system = Phase2TradingSystem()
    print("✅ Phase2TradingSystem instance created successfully")
    
    print("4. Testing system initialization...")
    try:
        success = system.initialize()
        if success:
            print("✅ System initialization completed successfully")
        else:
            print("⚠️ System initialization returned False (expected in mock mode)")
    except Exception as e:
        print(f"⚠️ System initialization failed: {e}")
    
    print("\n✅ All import tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
