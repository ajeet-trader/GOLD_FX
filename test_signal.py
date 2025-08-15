#!/usr/bin/env python3
"""
Test individual strategy imports to diagnose signal engine issues
"""

def test_individual_imports():
    try:
        # Test each strategy import individually
        from src.strategies.technical.ichimoku import IchimokuStrategy
        print("✅ Ichimoku import successful")
        
        from src.strategies.technical.harmonic import HarmonicStrategy
        print("✅ Harmonic import successful")
        
        from src.strategies.technical.elliott_wave import ElliottWaveStrategy
        print("✅ Elliott Wave import successful")
        
        from src.strategies.smc.order_blocks import OrderBlocksStrategy
        print("✅ Order Blocks import successful")
        
        from src.strategies.ml.lstm_predictor import LSTMPredictor  # Note: LSTMPredictor, not LSTMStrategy
        print("✅ LSTM import successful")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing individual strategy imports...")
    print("=" * 50)
    test_individual_imports()
    print("=" * 50)
    print("Test completed!")
