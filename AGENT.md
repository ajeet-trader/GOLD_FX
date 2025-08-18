# AGENT.md - Gold_FX Trading System Guide

## Build/Lint/Test Commands
```bash
# Run all tests
python tests/Phase-2/test_phase2_complete.py
python -m pytest tests/Phase-2/ -v

# Run single test
python tests/Phase-1/run_simple.py
python src/phase_2_core_integration.py --test

# Code formatting and linting
black src/ tests/
flake8 src/ tests/
pytest tests/ -v --cov=src

# System startup
python run_system.py                    # Interactive mode
python src/phase_2_core_integration.py --mode paper  # Paper trading
python src/phase_2_core_integration.py --mode live   # Live trading
```

## Architecture
- **Entry Point**: `src/phase_2_core_integration.py` (main system) or `run_system.py` (launcher)
- **Core Modules**: `src/core/` (mt5_manager, signal_engine, risk_manager, execution_engine)
- **Strategies**: `src/strategies/technical/` (10), `smc/` (5), `ml/` (4), `fusion/` (4)
- **Utils**: `src/utils/` (logger, database, error_handler, notifications)
- **Database**: SQLite with comprehensive trading schema in `data/trading.db`
- **Config**: Master configuration in `config/master_config.yaml`

## Code Style
- **Python 3.8+** with type hints (Optional, Dict, List, etc.)
- **Class names**: PascalCase (MT5Manager, SignalEngine)
- **Function/variable names**: snake_case (get_historical_data)
- **Constants**: UPPER_CASE (TIMEFRAMES, ORDER_TYPES)
- **Docstrings**: Multi-line with detailed parameter descriptions
- **Imports**: Standard library first, then third-party, then local imports
- **Error handling**: Try-catch with logging, graceful fallbacks for ML dependencies
- **Logging**: Comprehensive logging with colorlog, multiple log files by category
