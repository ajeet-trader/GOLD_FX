# src_core_outputs.py
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
import pytz

# Base path where core files are located
BASE_PATH = Path("J:/Gold_FX/src/core")

def get_output_filename():
    """Generate filename with IST timestamp"""
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(ist)
    timestamp = now_ist.strftime("%Y%m%d_%H%M%S")
    return Path(__file__).parent / f"core_results_{timestamp}.txt"

# Core files to run
CORE_FILES = [
    BASE_PATH / "execution_engine.py",
    BASE_PATH / "risk_manager.py",
    BASE_PATH / "signal_engine.py",
]

def run_core_file(core_file: Path, mode: str):
    """
    Run a core file as subprocess and capture output
    """
    try:
        cmd = [sys.executable, str(core_file), "--mode", mode]
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = '1'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,
            cwd=BASE_PATH,   # run inside core folder
            env=env,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "output": result.stdout,
                "errors": result.stderr
            }
        else:
            return {
                "status": "error",
                "output": result.stdout,
                "errors": result.stderr,
                "return_code": result.returncode
            }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "output": "",
            "errors": "Execution timed out after 60 seconds"
        }
    except Exception as e:
        return {
            "status": "exception",
            "output": "",
            "errors": str(e)
        }

def main():
    output_file = get_output_filename()
    ist = pytz.timezone('Asia/Kolkata')
    timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    
    print(f"Starting core execution at {timestamp}")
    print(f"Output will be saved to: {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Core Run Results ({timestamp})\n")
        f.write("=" * 80 + "\n\n")
        
        total_files = len(CORE_FILES)
        current_file = 0
        
        for core_file in CORE_FILES:
            current_file += 1
            core_name = core_file.stem
            
            print(f"Running core {current_file}/{total_files}: {core_name}")
            
            f.write(f"### Core File: {core_file}\n")
            f.write(f"### Core Name: {core_name}\n\n")
            
            for mode in ["mock", "live"]:
                f.write(f"--- Mode: {mode.upper()} ---\n")
                
                result = run_core_file(core_file, mode)
                
                if result["status"] == "success":
                    f.write("STDOUT OUTPUT:\n")
                    f.write(result["output"])
                    f.write("\n")
                    if result["errors"]:
                        f.write("STDERR OUTPUT:\n")
                        f.write(result["errors"])
                        f.write("\n")
                        
                elif result["status"] == "error":
                    f.write(f"EXECUTION FAILED (Return Code: {result.get('return_code', 'N/A')}):\n")
                    if result["output"]:
                        f.write("STDOUT:\n")
                        f.write(result["output"])
                        f.write("\n")
                    f.write("STDERR:\n")
                    f.write(result["errors"])
                    f.write("\n")
                    
                elif result["status"] == "timeout":
                    f.write(f"EXECUTION TIMEOUT: {result['errors']}\n\n")
                else:
                    f.write(f"EXCEPTION: {result['errors']}\n\n")
                
                f.write("-" * 60 + "\n\n")
            
            f.write("=" * 80 + "\n\n")
    
    print(f"All core modules executed. Results saved to {output_file}")

if __name__ == "__main__":
    main()
