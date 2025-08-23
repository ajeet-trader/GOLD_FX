# test_live_order.py
import logging
from datetime import datetime
from dataclasses import dataclass
from src.core.execution_engine import ExecutionEngine, SignalType, SignalGrade

logging.basicConfig(level=logging.INFO)

class MinimalMockDB:
    def get_trades(self, limit=1000):
        return []

@dataclass
class TestSignal:
    timestamp: datetime = datetime.now()
    symbol: str = "XAUUSDm"
    strategy_name: str = "live_test"
    signal_type = SignalType.BUY
    confidence: float = 0.90
    price: float = 3330.0
    timeframe: str = "M15"
    strength: float = 0.9
    grade = SignalGrade.A
    stop_loss: float = 3320.0
    take_profit: float = 3350.0
    metadata: dict = None

# Very permissive config for testing
config = {
    'execution': {
        'order': {'retry_attempts': 3, 'retry_delay': 1},
        'slippage': {'max_slippage': 10},  # Higher slippage tolerance
        'signal_age_threshold': 600,  # 10 minutes
        'min_confidence': 0.5,
        'magic_number': 123456
    },
    'risk_management': {
        'risk_per_trade': 0.005,  # 0.5% risk - very low
        'max_risk_per_trade': 0.01,
        'max_portfolio_risk': 0.03,
        'max_drawdown': 0.20,
        'max_daily_loss': 0.05,
        'max_weekly_loss': 0.15,
        'max_consecutive_losses': 10  # High tolerance
    },
    'position_sizing': {
        'method': 'FIXED',  # Simplest method
        'min_position_size': 0.01,
        'max_position_size': 0.02,
        'max_positions': 5
    },
    'capital': {
        'initial_capital': 149.17,
        'minimum_capital': 30.0,
        'reserve_cash': 0.02
    },
    'mode': 'live'
}

print("🚀 LIVE ORDER EXECUTION TEST")
print("=" * 60)

try:
    # Create execution engine with real risk manager
    execution_engine = ExecutionEngine(
        config,
        database_manager=MinimalMockDB()
    )
    
    print(f"Engine mode: {execution_engine.mode}")
    print(f"MT5 Manager: {type(execution_engine.mt5_manager).__name__}")
    print(f"Risk Manager: {type(execution_engine.risk_manager).__name__}")
    
    # Get current MT5 state
    balance = execution_engine.mt5_manager.get_account_balance()
    positions = execution_engine.mt5_manager.get_open_positions()
    print(f"Account: ${balance:.2f}, Positions: {len(positions)}")
    
    # Check magic number consistency
    ee_magic = getattr(execution_engine, 'MAGIC_NUMBER', 'N/A')
    mt5_magic = getattr(execution_engine.mt5_manager, 'magic_number', 'N/A')
    print(f"Magic Numbers - Engine: {ee_magic}, MT5: {mt5_magic}")
    
    # Create and test signal
    signal = TestSignal()
    print(f"\n📊 Processing Signal:")
    print(f"   {signal.strategy_name}: {signal.signal_type.value} {signal.symbol}")
    print(f"   Price: {signal.price}, Confidence: {signal.confidence:.1%}")
    print(f"   SL: {signal.stop_loss}, TP: {signal.take_profit}")
    
    # Process the signal
    print(f"\n🔄 Executing trade...")
    result = execution_engine.process_signal(signal)
    
    print(f"\n📊 EXECUTION RESULT:")
    print(f"   Status: {result.status.value}")
    print(f"   Ticket: {result.ticket}")
    print(f"   Executed Size: {result.executed_size}")
    print(f"   Executed Price: {result.executed_price}")
    print(f"   Slippage: {result.slippage}")
    print(f"   Error: {result.error_message}")
    
    # If failed, analyze why
    if result.status.value in ['FAILED', 'REJECTED']:
        print(f"\n🔍 FAILURE ANALYSIS:")
        print(f"   Error message: {result.error_message}")
        
        # Check risk manager directly
        rm_result = execution_engine.risk_manager.calculate_position_size(signal, balance, positions)
        print(f"   Risk mgr allowed: {rm_result.get('allowed', False)}")
        print(f"   Risk mgr size: {rm_result.get('position_size', 0)}")
        print(f"   Risk mgr reason: {rm_result.get('reason', 'N/A')}")
    
    execution_engine.stop_engine()
    
except Exception as e:
    print(f"❌ CRITICAL ERROR: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
