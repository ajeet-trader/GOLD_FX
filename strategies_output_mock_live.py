# run_all_strategies.py
import sys
import os
import importlib
from pathlib import Path
from datetime import datetime

# Ensure root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.core.base import AbstractStrategy
from src.utils.cli_args import parse_mode, print_mode_banner

# Strategies to include (excluding already migrated references ichimoku & harmonic if you wish)
STRATEGIES = [
    "src.strategies.fusion.confidence_sizing",
    "src.strategies.fusion.regime_detection",
    "src.strategies.fusion.weighted_voting",
    "src.strategies.fusion.adaptive_ensemble",
    "src.strategies.ml.ensemble_nn",
    "src.strategies.ml.lstm_predictor",
    "src.strategies.ml.rl_agent",
    "src.strategies.ml.xgboost_classifier",
    "src.strategies.smc.liquidity_pools",
    "src.strategies.smc.manipulation",
    "src.strategies.smc.market_structure",
    "src.strategies.smc.order_blocks",
    "src.strategies.technical.elliott_wave",
    "src.strategies.technical.fibonacci_advanced",
    "src.strategies.technical.gann",
    "src.strategies.technical.harmonic",   # migrated (kept as example)
    "src.strategies.technical.ichimoku",   # migrated (kept as example)
    "src.strategies.technical.market_profile",
    "src.strategies.technical.momentum_divergence",
    "src.strategies.technical.order_flow",
    "src.strategies.technical.volume_profile",
    "src.strategies.technical.wyckoff",
]

OUTPUT_FILE = Path(__file__).parent / "strategy_run_results.txt"


def run_strategy(module_path: str, mode: str):
    """
    Import a strategy dynamically, run it in the given mode, and return results.
    """
    try:
        module = importlib.import_module(module_path)
        # Assume strategy class = PascalCase of filename
        strategy_class_name = "".join(part.capitalize() for part in module_path.split(".")[-1].split("_"))
        StrategyClass = getattr(module, strategy_class_name)

        config = {
            "symbol": "XAUUSD",
            "timeframe": "H1",
            "risk": 1.0,
            "other_params": {},
            "mode": mode,
        }

        strategy = StrategyClass(config)
        print_mode_banner(mode)

        signals = strategy.generate_signals()
        analysis = strategy.analyze()
        perf = strategy.performance_summary()

        return {
            "status": "success",
            "signals": signals,
            "analysis": analysis,
            "performance": perf,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"Strategy Run Results ({timestamp})\n")
        f.write("=" * 80 + "\n\n")

        for module_path in STRATEGIES:
            f.write(f"### Strategy: {module_path}\n\n")
            for mode in ["mock", "live"]:
                f.write(f"--- Mode: {mode.upper()} ---\n")
                results = run_strategy(module_path, mode)

                if results["status"] == "success":
                    f.write("Signals:\n")
                    f.write(str(results["signals"]) + "\n\n")
                    f.write("Analysis:\n")
                    f.write(str(results["analysis"]) + "\n\n")
                    f.write("Performance:\n")
                    f.write(str(results["performance"]) + "\n\n")
                else:
                    f.write(f"Error running strategy: {results['error']}\n\n")

            f.write("=" * 80 + "\n\n")

    print(f"All strategies executed. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
