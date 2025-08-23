# strategies_output_mock_live.py
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
import pytz

print("DEBUG: Running file ->", __file__)

def get_output_filename(use_timestamp: bool = True):
    """Generate filename with IST timestamp and .md extension"""
    if use_timestamp:
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        timestamp = now_ist.strftime("%Y%m%d_%H%M%S")
        return Path(__file__).parent / f"strategy_results_{timestamp}.md"
    else:
        return Path(__file__).parent / "strategy_results.md"

# Strategy files to run
STRATEGY_FILES = [
    "src/strategies/fusion/confidence_sizing.py",
    "src/strategies/fusion/regime_detection.py",
    "src/strategies/fusion/weighted_voting.py",
    "src/strategies/fusion/adaptive_ensemble.py",
    "src/strategies/ml/ensemble_nn.py",
    "src/strategies/ml/lstm_predictor.py",
    "src/strategies/ml/rl_agent.py",
    "src/strategies/ml/xgboost_classifier.py",
    "src/strategies/smc/liquidity_pools.py",
    "src/strategies/smc/manipulation.py",
    "src/strategies/smc/market_structure.py",
    "src/strategies/smc/order_blocks.py",
    "src/strategies/technical/elliott_wave.py",
    "src/strategies/technical/fibonacci_advanced.py",
    "src/strategies/technical/gann.py",
    "src/strategies/technical/harmonic.py",
    "src/strategies/technical/ichimoku.py",
    "src/strategies/technical/market_profile.py",
    "src/strategies/technical/momentum_divergence.py",
    "src/strategies/technical/order_flow.py",
    "src/strategies/technical/volume_profile.py",
    "src/strategies/technical/wyckoff.py",
]

def run_strategy_file(strategy_file: str, mode: str):
    """Run a strategy file as subprocess and capture output"""
    try:
        cmd = [sys.executable, strategy_file, "--mode", mode]
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = '1'

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(__file__).parent,
            env=env,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode == 0:
            return {"status": "success", "output": result.stdout, "errors": result.stderr}
        else:
            return {"status": "error", "output": result.stdout, "errors": result.stderr, "return_code": result.returncode}

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "output": "", "errors": "⏱️ Strategy execution timed out after 60 seconds"}
    except Exception as e:
        return {"status": "exception", "output": "", "errors": str(e)}

def main():
    output_file = get_output_filename(use_timestamp=True)  # change to False to always overwrite "strategy_results.md"
    ist = pytz.timezone('Asia/Kolkata')
    timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")

    print(f"Starting strategy execution at {timestamp}")
    print(f"Output will be saved to: {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        # Header
        f.write(f"# 📊 Strategy Run Results\n")
        f.write(f"**Run Timestamp:** {timestamp}\n\n")
        f.write("---\n\n")

        total_strategies = len(STRATEGY_FILES)
        for idx, strategy_file in enumerate(STRATEGY_FILES, start=1):
            strategy_name = Path(strategy_file).stem
            print(f"Running strategy {idx}/{total_strategies}: {strategy_name}")

            f.write(f"## 🚀 Strategy: `{strategy_name}`\n")
            f.write(f"**File Path:** `{strategy_file}`\n\n")

            for mode in ["mock", "live"]:
                f.write(f"<details>\n<summary>⚙️ Mode: **{mode.upper()}**</summary>\n\n")
                result = run_strategy_file(strategy_file, mode)

                if result["status"] == "success":
                    f.write("✅ **Execution Successful**\n\n")
                    if result["output"]:
                        f.write("```txt\n")
                        f.write(result["output"])
                        f.write("\n```\n")
                    if result["errors"]:
                        f.write("⚠️ **Warnings / STDERR**\n")
                        f.write("```txt\n")
                        f.write(result["errors"])
                        f.write("\n```\n")

                elif result["status"] == "error":
                    f.write(f"❌ **Execution Failed** (Return Code: {result.get('return_code', 'N/A')})\n\n")
                    if result["output"]:
                        f.write("**STDOUT:**\n```txt\n")
                        f.write(result["output"])
                        f.write("\n```\n")
                    f.write("**STDERR:**\n```txt\n")
                    f.write(result["errors"])
                    f.write("\n```\n")

                elif result["status"] == "timeout":
                    f.write(f"⏱️ **Execution Timeout:** {result['errors']}\n\n")

                else:
                    f.write(f"⚠️ **Exception:** {result['errors']}\n\n")

                f.write("</details>\n\n")

            f.write("---\n\n")

    print(f"All strategies executed. Results saved to {output_file}")

if __name__ == "__main__":
    main()
