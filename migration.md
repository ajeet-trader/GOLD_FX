🔹 Ordered Chunks (Ascending)

Here’s a clean sequence for Phase 3 readiness:

Chunk 1: Live Data Integration

Goal: Allow switching between MockMT5Manager and real MT5Manager.

Add a data_mode flag (mock / live) in master_config.yaml.

Update signal_engine.py to load the right manager based on config.

Add data validation (missing bars, symbol mismatch, decimal precision).

Chunk 2: Strategy Live Testing

Goal: Validate strategies on real bars instead of mock.

Run 1–2 strategies first (e.g., Ichimoku + Order Blocks).

Compare mock outputs vs live outputs for consistency.

Add logs for “data quality warnings.”

Chunk 3: Risk Manager Live Checks

Goal: Dry-run risk control before live execution.

Ensure RiskManager checks capital, drawdown, exposure before any trade.

Add “journal-only mode”: logs trades but doesn’t send them.

Simulate different account balances (50, 100, 1000) and confirm risk caps work.

Chunk 4: Execution Engine Dry Run

Goal: Process live signals into execution logs (no broker trade yet).

Extend ExecutionEngine with dry_run=True flag.

Log execution flow: order sizing, stop-loss/take-profit levels, slippage handling.

Add emergency stop test (max_drawdown trigger).

Chunk 5: End-to-End Live Dry Run

Goal: Full pipeline test with live MT5 bars but no trade placement.

Strategies → Signal Engine → Risk Manager → Execution Engine (dry-run mode).

Verify logs for consistency, no crashes, correct strategy counts.

Build summary dashboard: active strategies, signals generated, risk metrics.

Chunk 6: Controlled Live Trading

Goal: First real trade — low risk.

Set max_positions=1, risk_per_trade=0.01.

Enable only one technical strategy (Ichimoku).

Place single trade, monitor execution, journal results.

Emergency stop should trigger correctly if thresholds hit.

🔹 AI Prompts for Each Chunk

Here are ready-to-use prompts you can feed into AI (like me, or another coding agent).
Each prompt assumes you’re pasting code + config along with it.

Prompt 1: Live Data Integration
You are helping me extend my MT5 manager system.  
Goal: add a "mock/live" switch so strategies can run on either random bars or real MT5 data.  
Files to modify: master_config.yaml, signal_engine.py, mt5_manager.py.  
Steps needed:  
1. Add `data_mode` config option ("mock" / "live").  
2. In signal_engine.py, decide whether to load MockMT5Manager or MT5Manager.  
3. Add data validation: missing bars, wrong symbol, precision mismatch.  
Please generate clean Python code updates for those files, showing me exactly what to insert.

Prompt 2: Strategy Live Testing
You are helping me test my trading strategies with real MT5 bars.  
Goal: run IchimokuStrategy and OrderBlocksStrategy on live data and compare outputs with mock mode.  
Steps needed:  
1. Add command-line flag `--live` to strategies for quick testing.  
2. Fetch last 200 bars from MT5 and run analysis.  
3. Print both mock signals and live signals side by side for comparison.  
Please generate modifications for ichimoku.py and order_blocks.py that enable this dual testing.

Prompt 3: Risk Manager Live Checks
You are helping me upgrade RiskManager for dry-run live mode.  
Goal: enforce capital limits and stop trading when thresholds are breached.  
Steps:  
1. Add a `dry_run` flag to RiskManager.  
2. If dry_run=True, log trades but don’t send them.  
3. Simulate balances of $50, $100, $1000 and confirm risk_per_trade, max_drawdown, and portfolio_risk rules apply.  
Please extend risk_manager.py to handle this and give me testing code to simulate 3 balances.

Prompt 4: Execution Engine Dry Run
You are helping me extend ExecutionEngine for live signals.  
Goal: add dry-run mode where execution is logged but not sent to broker.  
Steps:  
1. Add a dry_run flag to ExecutionEngine.__init__.  
2. If dry_run=True, print order details instead of placing trades.  
3. Log slippage, stop-loss, and take-profit calculations.  
Please generate modifications for execution_engine.py with examples of both normal mode and dry-run mode.

Prompt 5: End-to-End Live Dry Run
You are helping me test the full trading pipeline in dry-run live mode.  
Goal: strategies → signal_engine → risk_manager → execution_engine should all work on live MT5 bars without sending trades.  
Steps:  
1. Add a test harness `test_live_pipeline.py`.  
2. Load config with data_mode=live, dry_run=True.  
3. Run 3 cycles of signal generation + risk check + execution logging.  
4. Print final dashboard of strategies, signals, and risk metrics.  
Please generate this test harness with clean code and console-friendly output.

Prompt 6: Controlled Live Trading
You are helping me transition from dry-run to controlled live trading.  
Goal: place one real trade at very low risk.  
Steps:  
1. In config, set max_positions=1 and risk_per_trade=0.01.  
2. Enable only IchimokuStrategy.  
3. Modify execution_engine.py so that dry_run=False triggers real order placement.  
4. Add emergency stop if equity drawdown exceeds 5%.  
Please update execution_engine.py and show how to configure master_config.yaml for this test.