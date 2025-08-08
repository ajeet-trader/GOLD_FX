# run_simple.py

import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.core_system import CoreSystem

core = CoreSystem()
if core.initialize():
    print("System initialized successfully")
    # Test MT5 connection
    if core.connect_mt5():
        print("MT5 connected")
    core.shutdown()