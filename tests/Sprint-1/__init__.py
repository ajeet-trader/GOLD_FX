"""
Sprint 1 Test Suite Package
===========================

Comprehensive test suite for Phase 3 Sprint 1 fixes in the Gold_FX trading system.

This package contains tests for all critical fixes implemented in Sprint 1:

1. EnsembleNN TensorFlow tensor shape fixes
2. Signal age validation logic improvements  
3. Elliott Wave DatetimeIndex slicing fixes
4. Liquidity Pools signal throttling implementation
5. XGBoost signal generation optimization

Test Modules:
- test_ensemble_nn_fixes: Tests for EnsembleNN TensorFlow fixes
- test_signal_age_fixes: Tests for signal age validation improvements
- test_elliott_wave_fixes: Tests for Elliott Wave DatetimeIndex fixes
- test_liquidity_pools_fixes: Tests for Liquidity Pools throttling
- test_xgboost_fixes: Tests for XGBoost signal generation fixes
- test_sprint1_comprehensive: Master test runner for all fixes

Usage:
    # Run all Sprint 1 tests
    python -m tests.Sprint-1.test_sprint1_comprehensive
    
    # Run specific test module
    python -m pytest tests/Sprint-1/test_ensemble_nn_fixes.py -v
    
    # Run with meta-validation
    python -m tests.Sprint-1.test_sprint1_comprehensive --meta-test
"""

__version__ = "1.0.0"
__author__ = "Gold_FX Development Team"

# Test module imports for convenience
from .test_ensemble_nn_fixes import TestEnsembleNNFixes, TestEnsembleNNIntegration
from .test_signal_age_fixes import TestSignalAgeValidationFixes, TestSignalAgeIntegration
from .test_elliott_wave_fixes import TestElliottWaveDatetimeIndexFixes, TestElliottWaveIntegration
from .test_liquidity_pools_fixes import TestLiquidityPoolsThrottlingFixes, TestLiquidityPoolsIntegration
from .test_xgboost_fixes import TestXGBoostSignalGenerationFixes, TestXGBoostIntegration
from .test_sprint1_comprehensive import Sprint1TestRunner, TestSprint1Comprehensive

__all__ = [
    'TestEnsembleNNFixes',
    'TestEnsembleNNIntegration',
    'TestSignalAgeValidationFixes', 
    'TestSignalAgeIntegration',
    'TestElliottWaveDatetimeIndexFixes',
    'TestElliottWaveIntegration',
    'TestLiquidityPoolsThrottlingFixes',
    'TestLiquidityPoolsIntegration',
    'TestXGBoostSignalGenerationFixes',
    'TestXGBoostIntegration',
    'Sprint1TestRunner',
    'TestSprint1Comprehensive'
]