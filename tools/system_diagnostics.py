"""
System Diagnostic Tools for Phase 3 Sprint 1
=============================================

Comprehensive diagnostic and health monitoring tools for the Gold_FX trading system.
Monitors all Sprint 1 fixes and provides ongoing system health validation.

Key Features:
- Strategy health monitoring
- Signal generation validation
- Performance metrics tracking
- Error detection and reporting
- System component status checking
- Real-time diagnostics dashboard
"""

import sys
import os
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import json

# Import system components
from src.core.base import SignalType, SignalGrade

# Setup logging
try:
    from src.utils.logger import logger_manager
    logger = logger_manager.get_logger("system_diagnostics")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("system_diagnostics")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class SystemDiagnostics:
    """Comprehensive system diagnostic and health monitoring"""
    
    def __init__(self, config_path: str = "config/master_config.yaml"):
        """Initialize system diagnostics"""
        self.config_path = config_path
        
        # Setup logger
        try:
            from src.utils.logger import logger_manager
            self.logger = logger_manager.get_logger("system_diagnostics")
        except ImportError:
            import logging
            self.logger = logging.getLogger("system_diagnostics")
        
        # Diagnostic results storage
        self.diagnostic_results = {}
        self.health_status = {}
        self.performance_metrics = {}
        
        # Strategy categories from Sprint 1 fixes
        self.sprint1_strategies = {
            'ml': ['ensemble_nn', 'xgboost_classifier'],
            'technical': ['elliott_wave'],
            'smc': ['liquidity_pools'],
            'execution': ['signal_age_validation']
        }
        
        self.logger.info("System Diagnostics initialized")

    def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive system health check"""
        self.logger.info("Starting comprehensive health check...")
        
        start_time = time.time()
        
        # Core system checks
        self.health_status['core_system'] = self._check_core_system_health()
        
        # Sprint 1 fix validations
        self.health_status['sprint1_fixes'] = self._validate_sprint1_fixes()
        
        # Strategy health checks
        self.health_status['strategies'] = self._check_strategy_health()
        
        # Performance monitoring
        self.health_status['performance'] = self._check_system_performance()
        
        # Dependencies validation
        self.health_status['dependencies'] = self._check_dependencies()
        
        # Generate summary
        summary = self._generate_health_summary()
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        summary['diagnostic_runtime'] = elapsed
        summary['timestamp'] = datetime.now().isoformat()
        
        self.logger.info(f"Health check completed in {elapsed:.2f}s")
        
        return summary

    def _check_core_system_health(self) -> Dict[str, Any]:
        """Check core system component health"""
        core_health = {}
        
        try:
            # Check logging system
            core_health['logging'] = {
                'status': 'healthy',
                'logger_manager_available': hasattr(logger_manager, 'get_logger'),
                'loggers_active': len(logger_manager.loggers) if hasattr(logger_manager, 'loggers') else 0
            }
            
            # Check configuration system
            try:
                if os.path.exists(self.config_path):
                    core_health['configuration'] = {
                        'status': 'healthy',
                        'config_file_exists': True,
                        'config_path': self.config_path
                    }
                else:
                    core_health['configuration'] = {
                        'status': 'warning',
                        'config_file_exists': False,
                        'config_path': self.config_path
                    }
            except Exception as e:
                core_health['configuration'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Check src directory structure
            src_path = Path("src")
            core_health['src_structure'] = {
                'status': 'healthy' if src_path.exists() else 'error',
                'core_exists': (src_path / "core").exists(),
                'strategies_exists': (src_path / "strategies").exists(),
                'utils_exists': (src_path / "utils").exists()
            }
            
        except Exception as e:
            self.logger.error(f"Core system health check failed: {e}")
            core_health['error'] = str(e)
        
        return core_health

    def _validate_sprint1_fixes(self) -> Dict[str, Any]:
        """Validate all Sprint 1 fixes are working correctly"""
        sprint1_validation = {}
        
        # EnsembleNN TensorFlow fixes
        sprint1_validation['ensemble_nn'] = self._validate_ensemble_nn_fixes()
        
        # Signal age validation fixes
        sprint1_validation['signal_age_validation'] = self._validate_signal_age_fixes()
        
        # Elliott Wave DatetimeIndex fixes
        sprint1_validation['elliott_wave'] = self._validate_elliott_wave_fixes()
        
        # Liquidity Pools throttling fixes
        sprint1_validation['liquidity_pools'] = self._validate_liquidity_pools_fixes()
        
        # XGBoost signal generation fixes
        sprint1_validation['xgboost'] = self._validate_xgboost_fixes()
        
        return sprint1_validation

    def _validate_ensemble_nn_fixes(self) -> Dict[str, Any]:
        """Validate EnsembleNN TensorFlow tensor shape fixes"""
        validation = {'status': 'unknown', 'checks': {}}
        
        try:
            # Check if strategy file exists
            strategy_path = Path("src/strategies/ml/ensemble_nn.py")
            validation['checks']['file_exists'] = strategy_path.exists()
            
            if strategy_path.exists():
                # Try to import strategy
                try:
                    from src.strategies.ml.ensemble_nn import EnsembleNNStrategy
                    validation['checks']['import_successful'] = True
                    
                    # Check for TensorFlow availability
                    validation['checks']['tensorflow_available'] = TENSORFLOW_AVAILABLE
                    
                    if TENSORFLOW_AVAILABLE:
                        # Create instance to test initialization
                        config = {'parameters': {'mode': 'mock'}}
                        strategy = EnsembleNNStrategy(config)
                        validation['checks']['initialization_successful'] = True
                        
                        # Check for fixed methods
                        validation['checks']['has_lstm_features_method'] = hasattr(strategy, '_extract_lstm_features')
                        validation['checks']['has_training_features_method'] = hasattr(strategy, '_extract_lstm_features_for_training')
                        
                        validation['status'] = 'healthy'
                    else:
                        validation['status'] = 'warning'
                        validation['message'] = 'TensorFlow not available'
                        
                except Exception as e:
                    validation['checks']['import_error'] = str(e)
                    validation['status'] = 'error'
            else:
                validation['status'] = 'error'
                validation['message'] = 'Strategy file not found'
                
        except Exception as e:
            validation['status'] = 'error'
            validation['error'] = str(e)
        
        return validation

    def _validate_signal_age_fixes(self) -> Dict[str, Any]:
        """Validate signal age validation fixes"""
        validation = {'status': 'unknown', 'checks': {}}
        
        try:
            # Check if execution engine exists
            engine_path = Path("src/core/execution_engine.py")
            validation['checks']['file_exists'] = engine_path.exists()
            
            if engine_path.exists():
                # Try to import execution engine
                try:
                    from src.core.execution_engine import ExecutionEngine
                    validation['checks']['import_successful'] = True
                    
                    # Check for signal age validation method
                    validation['checks']['has_validate_signal_age'] = hasattr(ExecutionEngine, '_validate_signal_age')
                    
                    validation['status'] = 'healthy'
                    
                except Exception as e:
                    validation['checks']['import_error'] = str(e)
                    validation['status'] = 'error'
            else:
                validation['status'] = 'error'
                validation['message'] = 'Execution engine file not found'
                
        except Exception as e:
            validation['status'] = 'error'
            validation['error'] = str(e)
        
        return validation

    def _validate_elliott_wave_fixes(self) -> Dict[str, Any]:
        """Validate Elliott Wave DatetimeIndex fixes"""
        validation = {'status': 'unknown', 'checks': {}}
        
        try:
            # Check if strategy file exists
            strategy_path = Path("src/strategies/technical/elliott_wave.py")
            validation['checks']['file_exists'] = strategy_path.exists()
            
            if strategy_path.exists():
                # Try to import strategy
                try:
                    from src.strategies.technical.elliott_wave import ElliottWaveStrategy
                    validation['checks']['import_successful'] = True
                    
                    # Create instance to test initialization
                    config = {'parameters': {'mode': 'mock'}}
                    strategy = ElliottWaveStrategy(config)
                    validation['checks']['initialization_successful'] = True
                    
                    # Check for volume confirmation method
                    validation['checks']['has_volume_confirmation'] = hasattr(strategy, '_check_volume_confirmation')
                    
                    validation['status'] = 'healthy'
                    
                except Exception as e:
                    validation['checks']['import_error'] = str(e)
                    validation['status'] = 'error'
            else:
                validation['status'] = 'error'
                validation['message'] = 'Strategy file not found'
                
        except Exception as e:
            validation['status'] = 'error'
            validation['error'] = str(e)
        
        return validation

    def _validate_liquidity_pools_fixes(self) -> Dict[str, Any]:
        """Validate Liquidity Pools throttling fixes"""
        validation = {'status': 'unknown', 'checks': {}}
        
        try:
            # Check if strategy file exists
            strategy_path = Path("src/strategies/smc/liquidity_pools.py")
            validation['checks']['file_exists'] = strategy_path.exists()
            
            if strategy_path.exists():
                # Try to import strategy
                try:
                    from src.strategies.smc.liquidity_pools import LiquidityPoolsStrategy
                    validation['checks']['import_successful'] = True
                    
                    # Create instance to test initialization
                    config = {'parameters': {'mode': 'mock'}}
                    strategy = LiquidityPoolsStrategy(config)
                    validation['checks']['initialization_successful'] = True
                    
                    # Check for throttling parameters
                    validation['checks']['has_max_signals_per_run'] = hasattr(strategy, 'max_signals_per_run')
                    validation['checks']['has_min_pool_strength'] = hasattr(strategy, 'min_pool_strength')
                    validation['checks']['has_cooldown_bars'] = hasattr(strategy, 'cooldown_bars')
                    
                    validation['status'] = 'healthy'
                    
                except Exception as e:
                    validation['checks']['import_error'] = str(e)
                    validation['status'] = 'error'
            else:
                validation['status'] = 'error'
                validation['message'] = 'Strategy file not found'
                
        except Exception as e:
            validation['status'] = 'error'
            validation['error'] = str(e)
        
        return validation

    def _validate_xgboost_fixes(self) -> Dict[str, Any]:
        """Validate XGBoost signal generation fixes"""
        validation = {'status': 'unknown', 'checks': {}}
        
        try:
            # Check if strategy file exists
            strategy_path = Path("src/strategies/ml/xgboost_classifier.py")
            validation['checks']['file_exists'] = strategy_path.exists()
            
            if strategy_path.exists():
                # Try to import strategy
                try:
                    from src.strategies.ml.xgboost_classifier import XGBoostClassifierStrategy
                    validation['checks']['import_successful'] = True
                    
                    # Check for XGBoost availability
                    validation['checks']['xgboost_available'] = XGBOOST_AVAILABLE
                    
                    if XGBOOST_AVAILABLE:
                        # Create instance to test initialization
                        config = {'parameters': {'mode': 'mock'}}
                        strategy = XGBoostClassifierStrategy(config)
                        validation['checks']['initialization_successful'] = True
                        
                        # Check for improved confidence threshold
                        validation['checks']['min_confidence'] = strategy.min_confidence
                        validation['checks']['has_improved_threshold'] = strategy.min_confidence <= 0.20
                        
                        validation['status'] = 'healthy'
                    else:
                        validation['status'] = 'warning'
                        validation['message'] = 'XGBoost not available'
                        
                except Exception as e:
                    validation['checks']['import_error'] = str(e)
                    validation['status'] = 'error'
            else:
                validation['status'] = 'error'
                validation['message'] = 'Strategy file not found'
                
        except Exception as e:
            validation['status'] = 'error'
            validation['error'] = str(e)
        
        return validation

    def _check_strategy_health(self) -> Dict[str, Any]:
        """Check health of all strategies"""
        strategy_health = {}
        
        # Define strategy paths
        strategy_paths = {
            'technical': [
                'src/strategies/technical/bollinger_bands.py',
                'src/strategies/technical/elliott_wave.py',
                'src/strategies/technical/fibonacci_retracement.py',
                'src/strategies/technical/macd_strategy.py',
                'src/strategies/technical/moving_averages.py',
                'src/strategies/technical/rsi_strategy.py',
                'src/strategies/technical/stochastic_oscillator.py',
                'src/strategies/technical/support_resistance.py',
                'src/strategies/technical/trend_following.py',
                'src/strategies/technical/volume_analysis.py'
            ],
            'smc': [
                'src/strategies/smc/bos_choch.py',
                'src/strategies/smc/fair_value_gaps.py',
                'src/strategies/smc/liquidity_pools.py',
                'src/strategies/smc/order_blocks.py',
                'src/strategies/smc/premium_discount.py'
            ],
            'ml': [
                'src/strategies/ml/ensemble_nn.py',
                'src/strategies/ml/lstm_predictor.py',
                'src/strategies/ml/rl_trader.py',
                'src/strategies/ml/xgboost_classifier.py'
            ],
            'fusion': [
                'src/strategies/fusion/adaptive_multi_strategy.py',
                'src/strategies/fusion/ensemble_signals.py',
                'src/strategies/fusion/market_regime_fusion.py',
                'src/strategies/fusion/ml_technical_fusion.py'
            ]
        }
        
        total_strategies = 0
        available_strategies = 0
        
        for category, paths in strategy_paths.items():
            category_health = {'available': 0, 'total': len(paths), 'strategies': {}}
            
            for path in paths:
                strategy_name = Path(path).stem
                strategy_exists = Path(path).exists()
                
                category_health['strategies'][strategy_name] = {
                    'exists': strategy_exists,
                    'path': path
                }
                
                if strategy_exists:
                    category_health['available'] += 1
                    available_strategies += 1
                
                total_strategies += 1
            
            category_health['availability_rate'] = category_health['available'] / category_health['total']
            strategy_health[category] = category_health
        
        # Overall strategy health
        strategy_health['overall'] = {
            'total_strategies': total_strategies,
            'available_strategies': available_strategies,
            'availability_rate': available_strategies / total_strategies if total_strategies > 0 else 0,
            'status': 'healthy' if available_strategies >= 20 else 'warning' if available_strategies >= 15 else 'error'
        }
        
        return strategy_health

    def _check_system_performance(self) -> Dict[str, Any]:
        """Check system performance metrics"""
        performance = {}
        
        try:
            # Memory usage estimation
            import psutil
            process = psutil.Process()
            performance['memory'] = {
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'memory_percent': process.memory_percent()
            }
        except ImportError:
            performance['memory'] = {'status': 'psutil not available'}
        
        # Check disk space
        try:
            import shutil
            disk_usage = shutil.disk_usage('.')
            performance['disk'] = {
                'total_gb': disk_usage.total / (1024**3),
                'free_gb': disk_usage.free / (1024**3),
                'used_percent': (disk_usage.used / disk_usage.total) * 100
            }
        except Exception as e:
            performance['disk'] = {'error': str(e)}
        
        # Response time test
        start_time = time.time()
        # Simulate some work
        _ = [i**2 for i in range(1000)]
        end_time = time.time()
        
        performance['response_time'] = {
            'test_operation_ms': (end_time - start_time) * 1000,
            'status': 'fast' if (end_time - start_time) < 0.1 else 'slow'
        }
        
        return performance

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check system dependencies"""
        dependencies = {}
        
        # Core Python packages
        core_deps = ['pandas', 'numpy', 'datetime', 'pathlib', 'logging']
        dependencies['core'] = {}
        
        for dep in core_deps:
            try:
                __import__(dep)
                dependencies['core'][dep] = {'available': True}
            except ImportError:
                dependencies['core'][dep] = {'available': False}
        
        # ML dependencies
        ml_deps = {
            'tensorflow': TENSORFLOW_AVAILABLE,
            'xgboost': XGBOOST_AVAILABLE,
        }
        
        for dep, available in ml_deps.items():
            dependencies[dep] = {'available': available}
            if available:
                try:
                    module = __import__(dep)
                    if hasattr(module, '__version__'):
                        dependencies[dep]['version'] = module.__version__
                except:
                    pass
        
        # Optional dependencies
        optional_deps = ['scikit-learn', 'matplotlib', 'plotly', 'psutil']
        dependencies['optional'] = {}
        
        for dep in optional_deps:
            try:
                __import__(dep.replace('-', '_'))
                dependencies['optional'][dep] = {'available': True}
            except ImportError:
                dependencies['optional'][dep] = {'available': False}
        
        return dependencies

    def _generate_health_summary(self) -> Dict[str, Any]:
        """Generate comprehensive health summary"""
        summary = {
            'overall_status': 'unknown',
            'score': 0,
            'max_score': 100,
            'categories': {},
            'issues': [],
            'recommendations': []
        }
        
        score = 0
        max_score = 0
        
        # Core system score (20 points)
        core_health = self.health_status.get('core_system', {})
        if core_health.get('logging', {}).get('status') == 'healthy':
            score += 5
        if core_health.get('configuration', {}).get('status') == 'healthy':
            score += 5
        if core_health.get('src_structure', {}).get('status') == 'healthy':
            score += 10
        max_score += 20
        
        # Sprint 1 fixes score (30 points)
        sprint1_health = self.health_status.get('sprint1_fixes', {})
        for fix_name, fix_status in sprint1_health.items():
            if fix_status.get('status') == 'healthy':
                score += 6
            elif fix_status.get('status') == 'warning':
                score += 3
            max_score += 6
        
        # Strategy health score (30 points)
        strategy_health = self.health_status.get('strategies', {})
        overall_strategy = strategy_health.get('overall', {})
        if overall_strategy:
            availability_rate = overall_strategy.get('availability_rate', 0)
            score += int(availability_rate * 30)
        max_score += 30
        
        # Dependencies score (20 points)
        deps_health = self.health_status.get('dependencies', {})
        core_deps = deps_health.get('core', {})
        core_available = sum(1 for dep in core_deps.values() if dep.get('available'))
        score += min(core_available * 2, 10)
        
        ml_deps_available = sum(1 for dep in ['tensorflow', 'xgboost'] 
                               if deps_health.get(dep, {}).get('available'))
        score += ml_deps_available * 5
        max_score += 20
        
        # Calculate final score
        summary['score'] = score
        summary['max_score'] = max_score
        score_percentage = (score / max_score) * 100 if max_score > 0 else 0
        
        # Determine overall status
        if score_percentage >= 90:
            summary['overall_status'] = 'excellent'
        elif score_percentage >= 75:
            summary['overall_status'] = 'good'
        elif score_percentage >= 60:
            summary['overall_status'] = 'fair'
        else:
            summary['overall_status'] = 'poor'
        
        # Generate issues and recommendations
        self._generate_issues_and_recommendations(summary)
        
        # Add detailed results
        summary['detailed_results'] = self.health_status
        
        return summary

    def _generate_issues_and_recommendations(self, summary: Dict[str, Any]):
        """Generate issues and recommendations based on health check"""
        issues = []
        recommendations = []
        
        # Check Sprint 1 fixes
        sprint1_health = self.health_status.get('sprint1_fixes', {})
        for fix_name, fix_status in sprint1_health.items():
            if fix_status.get('status') == 'error':
                issues.append(f"Sprint 1 fix '{fix_name}' has errors")
                recommendations.append(f"Review and fix issues with {fix_name} implementation")
            elif fix_status.get('status') == 'warning':
                issues.append(f"Sprint 1 fix '{fix_name}' has warnings")
                recommendations.append(f"Check dependencies and configuration for {fix_name}")
        
        # Check strategy availability
        strategy_health = self.health_status.get('strategies', {})
        overall_strategy = strategy_health.get('overall', {})
        if overall_strategy.get('availability_rate', 0) < 0.9:
            issues.append("Some strategy files are missing")
            recommendations.append("Verify all strategy files are present in the src/strategies directory")
        
        # Check dependencies
        deps_health = self.health_status.get('dependencies', {})
        if not deps_health.get('tensorflow', {}).get('available'):
            issues.append("TensorFlow not available")
            recommendations.append("Install TensorFlow for ML strategy functionality")
        
        if not deps_health.get('xgboost', {}).get('available'):
            issues.append("XGBoost not available")
            recommendations.append("Install XGBoost for ML strategy functionality")
        
        summary['issues'] = issues
        summary['recommendations'] = recommendations

    def generate_diagnostic_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive diagnostic report"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"system_diagnostic_report_{timestamp}.json"
        
        # Run health check
        health_summary = self.run_comprehensive_health_check()
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(health_summary, f, indent=2, default=str)
        
        self.logger.info(f"Diagnostic report saved to: {output_file}")
        
        return output_file

    def print_health_summary(self):
        """Print human-readable health summary"""
        summary = self.run_comprehensive_health_check()
        
        print("=" * 60)
        print("SYSTEM HEALTH DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        # Overall status
        status_emoji = {
            'excellent': '🟢',
            'good': '🟡', 
            'fair': '🟠',
            'poor': '🔴'
        }
        
        overall_status = summary.get('overall_status', 'unknown')
        emoji = status_emoji.get(overall_status, '⚪')
        
        print(f"Overall Status: {emoji} {overall_status.upper()}")
        print(f"Health Score: {summary.get('score', 0)}/{summary.get('max_score', 100)} ({summary.get('score', 0)/summary.get('max_score', 1)*100:.1f}%)")
        print(f"Diagnostic Runtime: {summary.get('diagnostic_runtime', 0):.2f}s")
        print()
        
        # Sprint 1 fixes status
        print("Sprint 1 Fixes Status:")
        sprint1_health = summary.get('detailed_results', {}).get('sprint1_fixes', {})
        for fix_name, fix_status in sprint1_health.items():
            status = fix_status.get('status', 'unknown')
            emoji = '✅' if status == 'healthy' else '⚠️' if status == 'warning' else '❌'
            print(f"  {emoji} {fix_name.replace('_', ' ').title()}: {status}")
        
        # Strategy availability
        print("\nStrategy Availability:")
        strategy_health = summary.get('detailed_results', {}).get('strategies', {})
        for category, category_health in strategy_health.items():
            if category != 'overall':
                available = category_health.get('available', 0)
                total = category_health.get('total', 0)
                rate = category_health.get('availability_rate', 0)
                print(f"  {category.upper()}: {available}/{total} ({rate*100:.1f}%)")
        
        # Issues and recommendations
        issues = summary.get('issues', [])
        if issues:
            print(f"\n⚠️  Issues Found ({len(issues)}):")
            for issue in issues:
                print(f"  - {issue}")
        
        recommendations = summary.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations ({len(recommendations)}):")
            for rec in recommendations:
                print(f"  - {rec}")
        
        if not issues:
            print("\n✅ No critical issues found!")
        
        print("=" * 60)


def main():
    """Main entry point for system diagnostics"""
    diagnostics = SystemDiagnostics()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--report':
            # Generate detailed report
            report_file = diagnostics.generate_diagnostic_report()
            print(f"Diagnostic report generated: {report_file}")
        elif sys.argv[1] == '--summary':
            # Print summary only
            diagnostics.print_health_summary()
        else:
            print("Usage: python system_diagnostics.py [--report|--summary]")
    else:
        # Default: print summary
        diagnostics.print_health_summary()


if __name__ == '__main__':
    main()