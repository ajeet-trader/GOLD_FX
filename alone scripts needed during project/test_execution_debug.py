# test_execution_debug.py
import logging
from datetime import datetime
from dataclasses import dataclass
from src.core.execution_engine import ExecutionEngine, SignalType, SignalGrade

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@dataclass
class TestSignal:
    timestamp: datetime = datetime.now()
    symbol: str = "XAUUSDm"
    strategy_name: str = "debug_test"
    signal_type = SignalType.BUY
    confidence: float = 0.95
    price: float = 2500.0
    timeframe: str = "M15"
    strength: float = 0.8
    grade = SignalGrade.A
    stop_loss: float = 2480.0
    take_profit: float = 2520.0
    metadata: dict = None

# Enhanced mock risk manager
class DebugMockRiskManager:
    def __init__(self):
        self.mode = 'mock'
        
    def calculate_position_size(self, signal, balance, positions):
        print(f"🔍 DEBUG: calculate_position_size called")
        print(f"   Signal: {signal.symbol} {signal.signal_type.value}")
        print(f"   Balance: ${balance}")
        print(f"   Positions: {len(positions)}")
        
        result = {
            'position_size': 0.02,
            'allowed': True,  # ENSURE THIS IS TRUE
            'risk_assessment': {
                'monetary_risk': 20.0,
                'risk_percentage': 0.02
            },
            'reason': 'Debug mock - allowed'
        }
        
        print(f"   Returning: {result}")
        return result
        
    def update_position_closed(self, trade_result):
        pass

# Test configuration
config = {
    'execution': {
        'order': {'retry_attempts': 3, 'retry_delay': 1},
        'slippage': {'max_slippage': 3},
        'signal_age_threshold': 300
    },
    'mode': 'mock'
}

print("🧪 EXECUTION ENGINE DEBUG TEST")
print("=" * 50)

# Create execution engine with debug mock
execution_engine = ExecutionEngine(
    config,
    mt5_manager=None,
    risk_manager=DebugMockRiskManager(),
    database_manager=None,
    logger_manager=None
)

print(f"Engine mode: {execution_engine.mode}")
print(f"MT5 Manager type: {type(execution_engine.mt5_manager).__name__}")
print(f"Risk Manager type: {type(execution_engine.risk_manager).__name__}")

# Create and process test signal
signal = TestSignal()
print(f"\n🔄 Processing signal: {signal.strategy_name} {signal.signal_type.value}")

result = execution_engine.process_signal(signal)

print(f"\n📊 RESULT:")
print(f"Status: {result.status.value}")
print(f"Error: {result.error_message}")
print(f"Position Size: {result.executed_size}")
print(f"Price: {result.executed_price}")

execution_engine.stop_engine()
