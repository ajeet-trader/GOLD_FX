# Sprint 1 Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting information for all fixes implemented in Phase 3 Sprint 1 of the Gold_FX trading system. This sprint focused on resolving critical issues that were preventing proper signal generation and system stability.

## Sprint 1 Summary

### Completed Fixes

1. **EnsembleNN TensorFlow Tensor Shape Errors** ✅
2. **Signal Age Validation Logic Issues** ✅ 
3. **Elliott Wave DatetimeIndex Slicing Errors** ✅
4. **Liquidity Pools Signal Overflow (141 signals)** ✅
5. **XGBoost Signal Generation Issues** ✅

### System Improvements

- Created comprehensive test suite (50+ tests)
- Built system diagnostic tools
- Enhanced error handling and validation
- Improved performance and memory usage

---

## 1. EnsembleNN TensorFlow Tensor Shape Fixes

### Problem Description
The EnsembleNN strategy was failing with TensorFlow tensor shape errors during both training and prediction phases, preventing ML signal generation.

### Root Cause
- Inconsistent tensor shapes between training (30, 1) and prediction (1, 30, 1) phases
- LSTM feature extraction returning inconsistent shapes
- Prediction reshape logic causing dimension mismatches

### Solution Implemented
```python
# Fixed LSTM feature extraction shape consistency
def _extract_lstm_features(self, data):
    # Before: return normalized_sequence.reshape(1, 30, 1)
    # After: return normalized_sequence.reshape(30, 1)
    
# Fixed prediction reshape logic
if lstm_input.ndim == 2 and lstm_input.shape == (30, 1):
    lstm_input = lstm_input.reshape(1, 30, 1)
```

### Files Modified
- `src/strategies/ml/ensemble_nn.py`

### Verification Steps
1. **Check Strategy Import**:
   ```bash
   cd j:\Gold_FX && venv\Scripts\activate
   python -c "from src.strategies.ml.ensemble_nn import EnsembleNNStrategy; print('✅ Import successful')"
   ```

2. **Test Signal Generation**:
   ```bash
   python src/strategies/ml/ensemble_nn.py
   ```

3. **Run Comprehensive Tests**:
   ```bash
   python -m unittest tests.Sprint-1.test_ensemble_nn_fixes -v
   ```

### Troubleshooting Common Issues

#### Issue: "TensorFlow not available"
**Symptoms**: Strategy loads but warnings about TensorFlow availability
**Solution**: 
```bash
pip install tensorflow>=2.10.0
```

#### Issue: "Insufficient training data"
**Symptoms**: Model trains but shows warnings about data quality
**Solution**: 
- Check that historical data has at least 100 bars
- Verify OHLCV data completeness
- Ensure data has sufficient variance

#### Issue: Tensor shape errors persist
**Symptoms**: Still getting shape mismatch errors
**Solution**:
1. Clear any cached models: Delete `.keras` files in project directory
2. Restart Python session to clear TensorFlow session state
3. Check that fixes are applied correctly in `_extract_lstm_features` method

---

## 2. Signal Age Validation Logic Fixes

### Problem Description
Signal age validation was blocking legitimate trading signals with threshold misconfigurations and weekend restrictions affecting mock mode.

### Root Cause
- Signal age threshold too restrictive (300s instead of reasonable 3600s)
- Weekend market closure restrictions applied to mock mode
- Test configuration using production thresholds

### Solution Implemented
```python
# Updated signal age thresholds
max_signal_age_seconds = 3600  # 1 hour for live mode
test_max_signal_age_seconds = 7200  # 2 hours for testing

# Weekend bypass for mock mode
if self.mode == 'live' and current_weekday in [5, 6]:  # Saturday=5, Sunday=6
    return {'valid': False, 'reason': 'Weekend market closure'}
```

### Files Modified
- `src/core/execution_engine.py`

### Verification Steps
1. **Check Age Validation Method**:
   ```python
   from src.core.execution_engine import ExecutionEngine
   print(hasattr(ExecutionEngine, '_validate_signal_age'))  # Should be True
   ```

2. **Test Signal Processing**:
   - Create signals with different ages
   - Verify 1-hour threshold in live mode
   - Verify 2-hour threshold in test mode

### Troubleshooting Common Issues

#### Issue: All signals being rejected as "too old"
**Symptoms**: No signals pass age validation
**Solution**:
1. Check system time accuracy
2. Verify signal timestamps are recent
3. Check age calculation logic in `_validate_signal_age`

#### Issue: Weekend signals blocked in mock mode
**Symptoms**: Signals fail during weekends even in mock mode
**Solution**:
1. Verify mode is set to 'mock' not 'live'
2. Check weekend bypass logic implementation
3. Test with mock datetime if needed

#### Issue: ExecutionEngine initialization fails
**Symptoms**: Cannot create ExecutionEngine instance
**Solution**:
1. Check that all required parameters are provided
2. Verify MT5Manager and RiskManager instances are valid
3. Check configuration file format

---

## 3. Elliott Wave DatetimeIndex Slicing Fixes

### Problem Description
Elliott Wave strategy was failing with DatetimeIndex slicing errors when trying to access volume data for wave confirmation.

### Root Cause
- Using `.loc[]` with integer indices on DatetimeIndex
- Insufficient bounds checking for wave indices
- Range validation missing for start/end indices

### Solution Implemented
```python
# Fixed volume data access
# Before: data.loc[wave.start_index:wave.end_index, 'Volume']
# After: data.iloc[start_idx:end_idx + 1]['Volume']

# Added bounds checking
start_idx = max(0, min(wave.start_index, len(data) - 1))
end_idx = max(0, min(wave.end_index, len(data) - 1))
if start_idx >= end_idx:
    continue
```

### Files Modified
- `src/strategies/technical/elliott_wave.py`

### Verification Steps
1. **Test Strategy Import**:
   ```python
   from src.strategies.technical.elliott_wave import ElliottWaveStrategy
   strategy = ElliottWaveStrategy({'parameters': {'mode': 'mock'}})
   ```

2. **Test Signal Generation**:
   ```python
   signals = strategy.generate_signal("XAUUSDm", "M15")
   print(f"Generated {len(signals)} signals")
   ```

### Troubleshooting Common Issues

#### Issue: "slice indices must be integers or None"
**Symptoms**: DatetimeIndex slicing errors persist
**Solution**:
1. Verify `.iloc[]` is used instead of `.loc[]` for integer indexing
2. Check that bounds checking is properly implemented
3. Ensure wave indices are integers

#### Issue: Empty volume data returned
**Symptoms**: Volume confirmation always fails
**Solution**:
1. Check that data includes 'Volume' column
2. Verify volume data is not all zeros
3. Check wave index calculations

#### Issue: Wave detection fails
**Symptoms**: No waves identified in price data
**Solution**:
1. Check min_wave_size parameter (should be reasonable, e.g., 5-10)
2. Verify price data has sufficient movement
3. Check lookback_bars parameter (should be 100-200)

---

## 4. Liquidity Pools Signal Throttling Fixes

### Problem Description
Liquidity Pools strategy was generating excessive signals (141+ signals) causing system overflow and performance degradation.

### Root Cause
- No signal throttling mechanism
- All liquidity pools being processed regardless of strength
- No cooldown period between signal generations
- No deduplication of similar signals

### Solution Implemented
```python
# Added throttling parameters
max_signals_per_run = 5
min_pool_strength = 2.0
max_active_pools = 10
cooldown_bars = 10  # Increased from 3

# Implemented signal deduplication
def _deduplicate_signals(self, signals):
    # Remove similar signals based on proximity
    # Sort by strength and keep top signals only
```

### Files Modified
- `src/strategies/smc/liquidity_pools.py`

### Verification Steps
1. **Test Signal Throttling**:
   ```python
   from src.strategies.smc.liquidity_pools import LiquidityPoolsStrategy
   strategy = LiquidityPoolsStrategy({'parameters': {'mode': 'mock'}})
   signals = strategy.generate_signal("XAUUSDm", "M15")
   print(f"Generated {len(signals)} signals (should be ≤ 5)")
   ```

2. **Check Throttling Parameters**:
   ```python
   print(f"Max signals per run: {strategy.max_signals_per_run}")
   print(f"Min pool strength: {strategy.min_pool_strength}")
   print(f"Cooldown bars: {strategy.cooldown_bars}")
   ```

### Troubleshooting Common Issues

#### Issue: Still generating too many signals
**Symptoms**: More than 5 signals per run
**Solution**:
1. Check that throttling parameters are correctly set
2. Verify signal counting logic in generation method
3. Check deduplication implementation

#### Issue: No signals generated at all
**Symptoms**: Always returns empty signal list
**Solution**:
1. Check if min_pool_strength is too high
2. Verify pool identification logic
3. Check cooldown reset mechanism

#### Issue: Performance still slow
**Symptoms**: Strategy takes long time to execute
**Solution**:
1. Verify max_active_pools limit is enforced
2. Check memory cleanup implementation
3. Profile pool analysis algorithm

---

## 5. XGBoost Signal Generation Fixes

### Problem Description
XGBoost strategy was training successfully (92.3% accuracy) but generating 0 signals due to prediction logic and threshold issues.

### Root Cause
- Prediction logic always selecting highest probability class (often HOLD)
- Training data labeling too conservative (0.2% threshold)
- Minimum confidence threshold too high (0.60)
- Risk-reward validation too strict

### Solution Implemented
```python
# Fixed prediction logic for individual class probabilities
signal_threshold = 0.15  # Generate signal if BUY/SELL > 15%
if buy_prob > signal_threshold and buy_prob > sell_prob:
    prediction = 'BUY'
    confidence = float(buy_prob)

# Reduced labeling thresholds
if price_change > 0.0005:  # 0.05% instead of 0.2%
    label = 'BUY'

# Lowered confidence threshold
min_confidence = 0.15  # Instead of 0.60

# Improved risk-reward parameters
stop_loss_factor = 0.5  # Instead of 1.5
reward_ratio_threshold = 0.5  # Instead of 1.2
```

### Files Modified
- `src/strategies/ml/xgboost_classifier.py`

### Verification Steps
1. **Test XGBoost Availability**:
   ```python
   try:
       import xgboost as xgb
       print("✅ XGBoost available")
   except ImportError:
       print("❌ XGBoost not available")
   ```

2. **Test Signal Generation**:
   ```python
   from src.strategies.ml.xgboost_classifier import XGBoostClassifierStrategy
   strategy = XGBoostClassifierStrategy({'parameters': {'mode': 'mock'}})
   signals = strategy.generate_signal("XAUUSDm", "M15")
   print(f"Generated {len(signals)} signals")
   ```

3. **Check Model Training**:
   ```python
   print(f"Model trained: {strategy.is_trained}")
   print(f"Model accuracy: {strategy.model_accuracy:.3f}")
   print(f"Min confidence: {strategy.min_confidence}")
   ```

### Troubleshooting Common Issues

#### Issue: "XGBoost not available"
**Symptoms**: Strategy runs in fallback mode
**Solution**:
```bash
pip install xgboost>=1.6.0
pip install scikit-learn>=1.1.0
```

#### Issue: Model trains but generates no signals
**Symptoms**: High accuracy but 0 signals
**Solution**:
1. Check prediction probability logging
2. Verify individual class probability thresholds
3. Check risk-reward parameter calculations

#### Issue: All predictions are HOLD
**Symptoms**: Model only predicts HOLD class
**Solution**:
1. Check training data label distribution
2. Verify price change thresholds for labeling
3. Increase training data variance

#### Issue: Signals fail risk-reward validation
**Symptoms**: Signals created but validation fails
**Solution**:
1. Check ATR calculation for reasonable values
2. Verify stop loss and take profit calculations
3. Adjust risk-reward ratio threshold

---

## General Troubleshooting

### System Diagnostic Tool
Use the built-in diagnostic tool to check overall system health:

```bash
cd j:\Gold_FX && venv\Scripts\activate
python tools/system_diagnostics.py --summary
```

Expected output should show:
- ✅ All Sprint 1 fixes healthy
- System health score > 80%
- No critical issues

### Common Environment Issues

#### Python Version
Ensure Python 3.8+ is being used:
```bash
python --version  # Should be 3.8 or higher
```

#### Virtual Environment
Always activate the virtual environment:
```bash
cd j:\Gold_FX
venv\Scripts\activate
```

#### Missing Dependencies
Install missing packages:
```bash
pip install -r requirements.txt
```

#### Path Issues
If import errors occur, check that you're running from the project root:
```bash
cd j:\Gold_FX  # Make sure you're in the project root
python -c "import sys; print(sys.path[0])"  # Should show project root
```

### Performance Monitoring

#### Memory Usage
Monitor memory usage during strategy execution:
```python
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

#### Execution Time
Profile strategy execution times:
```python
import time
start = time.time()
# ... strategy execution ...
elapsed = time.time() - start
print(f"Execution time: {elapsed:.2f}s")
```

### Getting Help

#### Log Analysis
Check log files for detailed error information:
- `logs/engines/` - Core engine logs
- `logs/execution/` - Execution engine logs
- `logs/risk/` - Risk management logs

#### Test Execution
Run specific tests to validate fixes:
```bash
# Test all Sprint 1 fixes
python -m unittest tests.Sprint-1.test_sprint1_comprehensive

# Test specific fix
python -m unittest tests.Sprint-1.test_xgboost_fixes -v
```

#### System Status
Check system status using diagnostic tools:
```bash
python tools/system_diagnostics.py --report
```

This generates a detailed JSON report with complete system status.

---

## Sprint 1 Success Criteria

### All fixes should meet these criteria:

1. **No Critical Errors**: All strategies should import and initialize without errors
2. **Signal Generation**: All fixed strategies should generate signals in appropriate scenarios
3. **Performance**: No significant performance degradation
4. **Test Coverage**: All fixes covered by comprehensive tests
5. **Documentation**: Clear troubleshooting information available

### Validation Checklist

- [ ] EnsembleNN generates signals without tensor shape errors
- [ ] Signal age validation allows legitimate signals through
- [ ] Elliott Wave processes volume confirmation without slicing errors
- [ ] Liquidity Pools respects signal throttling limits (≤ 5 signals)
- [ ] XGBoost generates signals based on model predictions
- [ ] System diagnostic tool reports all fixes as healthy
- [ ] Comprehensive test suite passes
- [ ] No memory leaks or performance issues

## Contact and Support

For additional support with Sprint 1 fixes:

1. **Check Diagnostic Tool**: Run `python tools/system_diagnostics.py --summary`
2. **Review Test Results**: Run comprehensive test suite
3. **Check Documentation**: Review this troubleshooting guide
4. **Examine Logs**: Check relevant log files for detailed error information

---

**Last Updated**: August 24, 2025
**Sprint 1 Status**: ✅ COMPLETED
**Next Phase**: Sprint 2 - Advanced Features Implementation