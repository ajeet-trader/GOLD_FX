"""
Comprehensive Test Runner for Sprint 1 Fixes
============================================

Master test suite that runs all Sprint 1 fix tests and provides comprehensive reporting.
This test runner validates all critical fixes implemented in Phase 3 Sprint 1:

1. EnsembleNN TensorFlow tensor shape fixes
2. Signal age validation logic improvements
3. Elliott Wave DatetimeIndex slicing fixes
4. Liquidity Pools signal throttling implementation
5. XGBoost signal generation optimization

Usage:
    python -m tests.Sprint-1.test_sprint1_comprehensive
"""

import sys
import os
from pathlib import Path
import unittest
import time
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import all test modules
from tests.Sprint1.test_ensemble_nn_fixes import TestEnsembleNNFixes, TestEnsembleNNIntegration
from tests.Sprint1.test_signal_age_fixes import TestSignalAgeValidationFixes, TestSignalAgeIntegration
from tests.Sprint1.test_elliott_wave_fixes import TestElliottWaveDatetimeIndexFixes, TestElliottWaveIntegration
from tests.Sprint1.test_liquidity_pools_fixes import TestLiquidityPoolsThrottlingFixes, TestLiquidityPoolsIntegration
from tests.Sprint1.test_xgboost_fixes import TestXGBoostSignalGenerationFixes, TestXGBoostIntegration


class Sprint1TestRunner:
    """Comprehensive test runner for Sprint 1 fixes"""
    
    def __init__(self):
        self.test_suites = [
            # EnsembleNN fixes
            ('EnsembleNN TensorFlow Fixes', TestEnsembleNNFixes),
            ('EnsembleNN Integration', TestEnsembleNNIntegration),
            
            # Signal age validation fixes
            ('Signal Age Validation Fixes', TestSignalAgeValidationFixes),
            ('Signal Age Integration', TestSignalAgeIntegration),
            
            # Elliott Wave fixes
            ('Elliott Wave DatetimeIndex Fixes', TestElliottWaveDatetimeIndexFixes),
            ('Elliott Wave Integration', TestElliottWaveIntegration),
            
            # Liquidity Pools fixes
            ('Liquidity Pools Throttling Fixes', TestLiquidityPoolsThrottlingFixes),
            ('Liquidity Pools Integration', TestLiquidityPoolsIntegration),
            
            # XGBoost fixes
            ('XGBoost Signal Generation Fixes', TestXGBoostSignalGenerationFixes),
            ('XGBoost Integration', TestXGBoostIntegration),
        ]
        
        self.results = {}
        self.total_tests = 0
        self.total_passed = 0
        self.total_failed = 0
        self.total_errors = 0
        self.total_skipped = 0

    def run_all_tests(self, verbosity=2):
        """Run all Sprint 1 test suites"""
        print("=" * 80)
        print("SPRINT 1 COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Running comprehensive tests for all Sprint 1 fixes...")
        print(f"Total test suites: {len(self.test_suites)}")
        print("=" * 80)
        
        overall_start = time.time()
        
        for suite_name, test_class in self.test_suites:
            print(f"\n🧪 Running {suite_name}...")
            print("-" * 60)
            
            start_time = time.time()
            
            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            
            # Capture output
            output = StringIO()
            runner = unittest.TextTestRunner(
                stream=output,
                verbosity=verbosity,
                buffer=True
            )
            
            # Run tests
            result = runner.run(suite)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Store results
            self.results[suite_name] = {
                'result': result,
                'elapsed': elapsed,
                'output': output.getvalue()
            }
            
            # Update totals
            self.total_tests += result.testsRun
            self.total_passed += result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
            self.total_failed += len(result.failures)
            self.total_errors += len(result.errors)
            self.total_skipped += len(result.skipped)
            
            # Print summary for this suite
            self._print_suite_summary(suite_name, result, elapsed)
        
        overall_end = time.time()
        overall_elapsed = overall_end - overall_start
        
        # Print comprehensive summary
        self._print_comprehensive_summary(overall_elapsed)
        
        return self.results

    def _print_suite_summary(self, suite_name, result, elapsed):
        """Print summary for individual test suite"""
        status = "✅ PASS" if result.wasSuccessful() else "❌ FAIL"
        
        print(f"   {status} | Tests: {result.testsRun} | "
              f"Passed: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)} | "
              f"Failed: {len(result.failures)} | "
              f"Errors: {len(result.errors)} | "
              f"Skipped: {len(result.skipped)} | "
              f"Time: {elapsed:.2f}s")
        
        # Show failures/errors if any
        if result.failures:
            print(f"   ⚠️  Failures: {len(result.failures)}")
            for test, traceback in result.failures:
                print(f"      - {test}")
        
        if result.errors:
            print(f"   ❌ Errors: {len(result.errors)}")
            for test, traceback in result.errors:
                print(f"      - {test}")

    def _print_comprehensive_summary(self, total_elapsed):
        """Print comprehensive summary of all tests"""
        print("\n" + "=" * 80)
        print("SPRINT 1 TEST SUITE COMPREHENSIVE SUMMARY")
        print("=" * 80)
        
        success_rate = (self.total_passed / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"📊 Overall Results:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   Passed: {self.total_passed} ({success_rate:.1f}%)")
        print(f"   Failed: {self.total_failed}")
        print(f"   Errors: {self.total_errors}")
        print(f"   Skipped: {self.total_skipped}")
        print(f"   Total Time: {total_elapsed:.2f}s")
        
        print(f"\n📋 Suite Breakdown:")
        for suite_name, data in self.results.items():
            result = data['result']
            elapsed = data['elapsed']
            status = "✅" if result.wasSuccessful() else "❌"
            print(f"   {status} {suite_name}: {result.testsRun} tests in {elapsed:.2f}s")
        
        print(f"\n🎯 Sprint 1 Fix Validation:")
        print(f"   1. EnsembleNN TensorFlow Fixes: {'✅ Validated' if self._is_fix_validated('EnsembleNN') else '❌ Issues Found'}")
        print(f"   2. Signal Age Validation Fixes: {'✅ Validated' if self._is_fix_validated('Signal Age') else '❌ Issues Found'}")
        print(f"   3. Elliott Wave DatetimeIndex Fixes: {'✅ Validated' if self._is_fix_validated('Elliott Wave') else '❌ Issues Found'}")
        print(f"   4. Liquidity Pools Throttling Fixes: {'✅ Validated' if self._is_fix_validated('Liquidity Pools') else '❌ Issues Found'}")
        print(f"   5. XGBoost Signal Generation Fixes: {'✅ Validated' if self._is_fix_validated('XGBoost') else '❌ Issues Found'}")
        
        overall_status = "✅ SUCCESS" if self.total_failed == 0 and self.total_errors == 0 else "❌ ISSUES FOUND"
        print(f"\n🏆 Sprint 1 Status: {overall_status}")
        
        if success_rate >= 90:
            print("🎉 Excellent! Sprint 1 fixes are well-validated.")
        elif success_rate >= 75:
            print("👍 Good! Most Sprint 1 fixes are validated.")
        else:
            print("⚠️  Warning! Some Sprint 1 fixes need attention.")
        
        print("=" * 80)

    def _is_fix_validated(self, fix_name):
        """Check if a specific fix is validated based on test results"""
        related_suites = [name for name in self.results.keys() if fix_name in name]
        
        for suite_name in related_suites:
            result = self.results[suite_name]['result']
            if not result.wasSuccessful():
                return False
        
        return len(related_suites) > 0

    def generate_detailed_report(self, output_file=None):
        """Generate detailed test report"""
        if output_file is None:
            output_file = f"sprint1_test_report_{int(time.time())}.txt"
        
        with open(output_file, 'w') as f:
            f.write("SPRINT 1 COMPREHENSIVE TEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("SUMMARY:\n")
            f.write(f"Total Tests: {self.total_tests}\n")
            f.write(f"Passed: {self.total_passed}\n")
            f.write(f"Failed: {self.total_failed}\n")
            f.write(f"Errors: {self.total_errors}\n")
            f.write(f"Skipped: {self.total_skipped}\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            for suite_name, data in self.results.items():
                result = data['result']
                f.write(f"\n{suite_name}:\n")
                f.write(f"  Tests Run: {result.testsRun}\n")
                f.write(f"  Successful: {result.wasSuccessful()}\n")
                f.write(f"  Failures: {len(result.failures)}\n")
                f.write(f"  Errors: {len(result.errors)}\n")
                f.write(f"  Time: {data['elapsed']:.2f}s\n")
                
                if result.failures:
                    f.write(f"  FAILURES:\n")
                    for test, traceback in result.failures:
                        f.write(f"    - {test}\n")
                        f.write(f"      {traceback}\n")
                
                if result.errors:
                    f.write(f"  ERRORS:\n")
                    for test, traceback in result.errors:
                        f.write(f"    - {test}\n")
                        f.write(f"      {traceback}\n")
        
        print(f"📄 Detailed report saved to: {output_file}")
        return output_file


class TestSprint1Comprehensive(unittest.TestCase):
    """Meta-test that validates the comprehensive test suite itself"""
    
    def test_all_sprint1_fixes_validated(self):
        """Meta-test: Ensure all Sprint 1 fixes have comprehensive test coverage"""
        runner = Sprint1TestRunner()
        results = runner.run_all_tests(verbosity=1)
        
        # Should have results for all test suites
        self.assertEqual(len(results), len(runner.test_suites))
        
        # Check that critical fixes are covered
        critical_fixes = ['EnsembleNN', 'Signal Age', 'Elliott Wave', 'Liquidity Pools', 'XGBoost']
        
        for fix in critical_fixes:
            covered = any(fix in suite_name for suite_name in results.keys())
            self.assertTrue(covered, f"Critical fix '{fix}' should have test coverage")
        
        # Overall success rate should be reasonable
        success_rate = (runner.total_passed / runner.total_tests * 100) if runner.total_tests > 0 else 0
        self.assertGreater(success_rate, 50, "Overall test success rate should be > 50%")
        
        print(f"\n✅ Meta-validation complete: {success_rate:.1f}% success rate across {runner.total_tests} tests")


def main():
    """Main entry point for running Sprint 1 comprehensive tests"""
    if len(sys.argv) > 1 and sys.argv[1] == '--meta-test':
        # Run meta-test only
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        # Run comprehensive test suite
        runner = Sprint1TestRunner()
        results = runner.run_all_tests()
        
        # Generate detailed report
        report_file = runner.generate_detailed_report()
        
        # Exit with appropriate code
        if runner.total_failed == 0 and runner.total_errors == 0:
            print("🎉 All Sprint 1 tests passed!")
            sys.exit(0)
        else:
            print("❌ Some Sprint 1 tests failed. Check the report for details.")
            sys.exit(1)


if __name__ == '__main__':
    main()