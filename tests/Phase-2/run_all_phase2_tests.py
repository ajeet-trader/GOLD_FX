"""
Phase 2 Complete Test Suite Runner
=================================

Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-08-24

Comprehensive test runner for all Phase 2 components:
- Signal Engine Tests
- Execution Engine Tests  
- Risk Manager Tests
- Phase 2 Integration Tests
- Performance and reliability validation

USAGE:
======
1. Run all tests (mock mode): python run_all_phase2_tests.py
2. Run all tests (live mode): python run_all_phase2_tests.py --mode live
3. Quick tests only: python run_all_phase2_tests.py --quick
4. Verbose output: python run_all_phase2_tests.py -v
5. Specific component: python run_all_phase2_tests.py --component signal_engine
"""

import unittest
import sys
import argparse
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

# Add project root to PYTHONPATH
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Complete Phase 2 Test Suite')
    parser.add_argument('--mode', choices=['mock', 'live'], default='mock',
                       help='Trading mode: mock (safe) or live (connects to MT5)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose test output')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick tests only')
    parser.add_argument('--component', 
                       choices=['signal_engine', 'execution_engine', 'risk_manager', 'integration', 'all'],
                       default='all',
                       help='Run specific component tests')
    parser.add_argument('--no-integration', action='store_true',
                       help='Skip integration tests')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed test report')
    return parser.parse_args()


class Phase2TestRunner:
    """Comprehensive test runner for Phase 2 components"""
    
    def __init__(self, args):
        self.args = args
        self.test_results = {}
        self.start_time = datetime.now()
        self.tests_dir = Path(__file__).parent
        
        # Define test components
        self.test_components = {
            'signal_engine': {
                'file': 'test_signal_engine.py',
                'description': 'Signal Engine - Strategy loading and signal generation',
                'critical': True
            },
            'execution_engine': {
                'file': 'test_execution_engine.py', 
                'description': 'Execution Engine - Order processing and execution',
                'critical': True
            },
            'risk_manager': {
                'file': 'test_risk_manager.py',
                'description': 'Risk Manager - Position sizing and risk controls',
                'critical': True
            },
            'integration': {
                'file': 'test_phase2_integration.py',
                'description': 'Integration Tests - Complete system workflow',
                'critical': False
            }
        }

    def print_banner(self):
        """Print test suite banner"""
        print("="*80)
        print("[TEST] PHASE 2 COMPLETE TEST SUITE")
        print("="*80)
        print(f"[MODE] Mode: {self.args.mode.upper()}")
        print(f"[COMP] Component: {self.args.component.upper()}")
        print(f"[FAST] Quick Mode: {'Enabled' if self.args.quick else 'Disabled'}")
        print(f"[VERB] Verbose: {'Enabled' if self.args.verbose else 'Disabled'}")
        print(f"[INTG] Integration: {'Disabled' if self.args.no_integration else 'Enabled'}")
        print(f"[REPT] Report: {'Enabled' if self.args.report else 'Disabled'}")
        print("="*80)

    def run_component_test(self, component_name: str, test_info: Dict) -> Dict:
        """Run a specific component test"""
        print(f"\n[RUN] Running {component_name} tests...")
        print(f"[DESC] {test_info['description']}")
        
        test_file = self.tests_dir / test_info['file']
        
        if not test_file.exists():
            print(f"[!] Test file not found: {test_file}")
            return {
                'component': component_name,
                'status': 'SKIPPED',
                'reason': 'Test file not found',
                'tests_run': 0,
                'failures': 0,
                'errors': 0,
                'duration': 0
            }
        
        # Build command
        cmd = [sys.executable, str(test_file)]
        
        if self.args.mode:
            cmd.extend(['--mode', self.args.mode])
        if self.args.verbose:
            cmd.append('-v')
        if self.args.quick and component_name == 'integration':
            cmd.append('--quick')
        
        # Run test
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300 if self.args.quick else 600  # 5-10 minute timeout
            )
            duration = time.time() - start_time
            
            # Parse output for test results
            test_result = self.parse_test_output(result.stdout, result.stderr, result.returncode)
            test_result.update({
                'component': component_name,
                'duration': duration,
                'command': ' '.join(cmd)
            })
            
            # Print summary
            if test_result['status'] == 'PASSED':
                print(f"[+] {component_name}: {test_result['tests_run']} tests passed in {duration:.1f}s")
            else:
                print(f"[-] {component_name}: {test_result['failures']} failures, {test_result['errors']} errors")
                
            if self.args.verbose and result.stderr:
                print(f"[STDERR] Stderr output:\n{result.stderr}")
                
            return test_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"[TIMEOUT] {component_name}: Test timed out after {duration:.1f}s")
            return {
                'component': component_name,
                'status': 'TIMEOUT',
                'reason': f'Timed out after {duration:.1f}s',
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'duration': duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"[ERROR] {component_name}: Test failed with error: {str(e)}")
            return {
                'component': component_name,
                'status': 'ERROR',
                'reason': str(e),
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'duration': duration
            }

    def parse_test_output(self, stdout: str, stderr: str, returncode: int) -> Dict:
        """Parse test output to extract results"""
        result = {
            'status': 'UNKNOWN',
            'tests_run': 0,
            'failures': 0,
            'errors': 0,
            'success_rate': 0.0,
            'stdout': stdout,
            'stderr': stderr,
            'returncode': returncode
        }
        
        # Look for various unittest output patterns
        stdout_lines = stdout.split('\n')
        
        # Try to find test count patterns
        for line in stdout_lines:
            line = line.strip()
            
            # Pattern: "Ran X tests in Y.Zs"
            if 'Ran ' in line and ' test' in line and ' in ' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'Ran' and i + 1 < len(parts):
                            result['tests_run'] = int(parts[i + 1])
                            break
                except (ValueError, IndexError):
                    pass
            
            # Pattern: "[TESTS] Total Tests Run: X" or "[TESTS] Total Tests: X" (custom format)
            elif '[TESTS] Total Tests' in line:
                try:
                    number = line.split(':')[-1].strip()
                    result['tests_run'] = int(number)
                except (ValueError, IndexError):
                    pass
            
            # Pattern: "[RATE] Success Rate: X.X" (custom format)
            elif '[RATE] Success Rate:' in line:
                try:
                    rate_str = line.split(':')[-1].strip().replace('%', '')
                    rate = float(rate_str)
                    result['success_rate'] = rate / 100.0
                    if rate == 100.0:
                        result['status'] = 'PASSED'
                except (ValueError, IndexError):
                    pass
            
            # Pattern: "FAILED (failures=X, errors=Y)"
            elif 'FAILED (' in line:
                try:
                    if 'failures=' in line:
                        failures_part = line.split('failures=')[1]
                        if ',' in failures_part:
                            failures_str = failures_part.split(',')[0]
                        else:
                            failures_str = failures_part.split(')')[0]
                        result['failures'] = int(failures_str)
                    
                    if 'errors=' in line:
                        errors_part = line.split('errors=')[1]
                        errors_str = errors_part.split(')')[0]
                        result['errors'] = int(errors_str)
                except (ValueError, IndexError):
                    pass
            
            # Pattern: "OK" at end of successful test run
            elif line == 'OK' and result['tests_run'] > 0:
                result['status'] = 'PASSED'
            
            # Pattern: "Success Rate: X.X%" (custom format)
            elif 'Success Rate:' in line:
                try:
                    rate_str = line.split(':')[-1].strip().replace('%', '')
                    rate = float(rate_str)
                    if rate == 100.0:
                        result['status'] = 'PASSED'
                except (ValueError, IndexError):
                    pass
        
        # Final status determination with multiple checks
        if result['tests_run'] > 0:
            if result['failures'] == 0 and result['errors'] == 0:
                if returncode == 0 or result['success_rate'] == 1.0:
                    result['status'] = 'PASSED'
                else:
                    result['status'] = 'ERROR'
            else:
                result['status'] = 'FAILED'
        elif returncode == 0:
            # Check for success indicators without explicit test count
            success_indicators = [
                'ALL TESTS PASSED',
                'Success Rate: 100.0',
                'OK\n' in stdout and 'Ran ' in stdout
            ]
            if any(indicator in stdout if isinstance(indicator, str) else indicator for indicator in success_indicators):
                # Try to extract test count from "Ran X tests" pattern
                for line in stdout_lines:
                    if 'Ran ' in line and ' test' in line:
                        try:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == 'Ran' and i + 1 < len(parts):
                                    result['tests_run'] = int(parts[i + 1])
                                    result['status'] = 'PASSED'
                                    break
                        except (ValueError, IndexError):
                            pass
        else:
            result['status'] = 'ERROR'
        
        # Calculate success rate
        if result['tests_run'] > 0:
            passed = result['tests_run'] - result['failures'] - result['errors']
            result['success_rate'] = passed / result['tests_run']
        
        return result

    def run_all_tests(self) -> Dict:
        """Run all selected tests"""
        self.print_banner()
        
        components_to_run = []
        
        if self.args.component == 'all':
            components_to_run = list(self.test_components.keys())
            if self.args.no_integration:
                components_to_run = [c for c in components_to_run if c != 'integration']
        else:
            components_to_run = [self.args.component]
        
        print(f"\n[TARGET] Running tests for: {', '.join(components_to_run)}")
        
        # Run tests
        for component in components_to_run:
            if component in self.test_components:
                test_info = self.test_components[component]
                self.test_results[component] = self.run_component_test(component, test_info)
            else:
                print(f"[!] Unknown component: {component}")
        
        return self.test_results

    def print_summary(self):
        """Print comprehensive test summary"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*80)
        print("[SUMMARY] PHASE 2 TEST SUITE SUMMARY")
        print("="*80)
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        passed_components = 0
        
        for component, result in self.test_results.items():
            status_emoji = "[+]" if result['status'] == 'PASSED' else "[-]"
            critical_marker = "[!]" if self.test_components[component]['critical'] and result['status'] != 'PASSED' else ""
            
            print(f"{status_emoji} {critical_marker}{component.upper()}: "
                  f"{result['tests_run']} tests, "
                  f"{result['failures']} failures, "
                  f"{result['errors']} errors "
                  f"({result['duration']:.1f}s)")
            
            total_tests += result['tests_run']
            total_failures += result['failures']
            total_errors += result['errors']
            
            if result['status'] == 'PASSED':
                passed_components += 1
        
        # Overall statistics
        total_passed = total_tests - total_failures - total_errors
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        component_success_rate = (passed_components / len(self.test_results) * 100) if self.test_results else 0
        
        print(f"\n[STATS] OVERALL STATISTICS:")
        print(f"   [TESTS] Total Tests: {total_tests}")
        print(f"   [PASS] Passed: {total_passed}")
        print(f"   [FAIL] Failed: {total_failures}")
        print(f"   [ERROR] Errors: {total_errors}")
        print(f"   [RATE] Test Success Rate: {overall_success_rate:.1f}%")
        print(f"   [COMP] Component Success Rate: {component_success_rate:.1f}%")
        print(f"   [TIME] Total Duration: {total_duration:.1f}s")
        
        # Critical component status
        critical_components = [c for c, info in self.test_components.items() if info['critical']]
        critical_failures = [c for c in critical_components if c in self.test_results and self.test_results[c]['status'] != 'PASSED']
        
        if critical_failures:
            print(f"\n[CRITICAL] CRITICAL COMPONENT FAILURES:")
            for component in critical_failures:
                result = self.test_results[component]
                print(f"   [!] {component}: {result.get('reason', 'Tests failed')}")
        
        # Recommendations
        print(f"\n[RECOMMEND] RECOMMENDATIONS:")
        if overall_success_rate >= 95:
            print(f"  [+] Excellent! System is ready for Phase 3.")
        elif overall_success_rate >= 85:
            print(f"  [+] Good! Minor issues to address before Phase 3.")
        elif overall_success_rate >= 70:
            print(f"  [!] Moderate issues found. Address failures before proceeding.")
        else:
            print(f"  [-] Significant issues found. System needs attention.")
        
        if critical_failures:
            print(f"  [!] Critical components failed - must be fixed before Phase 3!")
        
        print("="*80)

    def generate_report(self):
        """Generate detailed test report"""
        if not self.args.report:
            return
            
        report_file = self.tests_dir / f"phase2_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            'test_suite': 'Phase 2 Complete Test Suite',
            'timestamp': self.start_time.isoformat(),
            'duration': (datetime.now() - self.start_time).total_seconds(),
            'configuration': {
                'mode': self.args.mode,
                'component': self.args.component,
                'quick_mode': self.args.quick,
                'verbose': self.args.verbose,
                'no_integration': self.args.no_integration
            },
            'results': self.test_results,
            'summary': {
                'total_components': len(self.test_results),
                'passed_components': sum(1 for r in self.test_results.values() if r['status'] == 'PASSED'),
                'total_tests': sum(r['tests_run'] for r in self.test_results.values()),
                'total_failures': sum(r['failures'] for r in self.test_results.values()),
                'total_errors': sum(r['errors'] for r in self.test_results.values())
            }
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n[REPORT] Test report saved: {report_file}")
        except Exception as e:
            print(f"\n[!] Failed to save report: {str(e)}")

    def get_exit_code(self) -> int:
        """Get appropriate exit code based on test results"""
        # Check for critical component failures
        critical_components = [c for c, info in self.test_components.items() if info['critical']]
        critical_failures = [c for c in critical_components if c in self.test_results and self.test_results[c]['status'] != 'PASSED']
        
        if critical_failures:
            return 1  # Critical failure
        
        # Check overall success rate
        total_tests = sum(r['tests_run'] for r in self.test_results.values())
        total_failures = sum(r['failures'] for r in self.test_results.values())
        total_errors = sum(r['errors'] for r in self.test_results.values())
        
        if total_tests == 0:
            return 1  # No tests run
        
        success_rate = (total_tests - total_failures - total_errors) / total_tests
        
        if success_rate < 0.8:  # Less than 80% success rate
            return 1
        
        return 0  # Success


def main():
    """Main function"""
    args = parse_cli_args()
    
    # Force mock mode in CI/CD
    if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
        args.mode = 'mock'
        print("[SETUP] CI/CD detected - forcing mock mode")
    
    # Create and run test suite
    runner = Phase2TestRunner(args)
    results = runner.run_all_tests()
    runner.print_summary()
    runner.generate_report()
    
    # Exit with appropriate code
    exit_code = runner.get_exit_code()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()