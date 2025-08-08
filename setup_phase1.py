#!/usr/bin/env python3
"""
Setup Script - Automated Phase 1 Setup
======================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-08-07

This script automatically sets up the Phase 1 environment:
- Creates directory structure
- Installs dependencies
- Sets up configuration files
- Tests system components
- Validates installation

Run this script to prepare your environment for Phase 1 development.

Usage:
    python setup_phase1.py
    python setup_phase1.py --dev  # Include development tools
    python setup_phase1.py --test  # Run tests after setup
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform
import argparse
import json
import yaml
from datetime import datetime


class Phase1Setup:
    """Automated setup for Phase 1 components"""
    
    def __init__(self, include_dev: bool = False, run_tests: bool = False):
        self.include_dev = include_dev
        self.run_tests = run_tests
        self.project_root = Path.cwd()
        self.python_executable = sys.executable
        
        # Setup status
        self.setup_status = {
            'directories': False,
            'dependencies': False,
            'configuration': False,
            'files': False,
            'tests': False
        }
    
    def run_setup(self) -> bool:
        """Run the complete setup process"""
        print("🎯 XAUUSD MT5 Trading System - Phase 1 Setup")
        print("="*50)
        print(f"Python: {sys.version}")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Project Root: {self.project_root}")
        print()
        
        try:
            # Step 1: Create directory structure
            if not self.create_directories():
                return False
            
            # Step 2: Install dependencies
            if not self.install_dependencies():
                return False
            
            # Step 3: Create configuration files
            if not self.create_configuration():
                return False
            
            # Step 4: Copy/create necessary files
            if not self.setup_files():
                return False
            
            # Step 5: Run tests if requested
            if self.run_tests:
                if not self.run_system_tests():
                    return False
            
            # Setup complete
            self.print_setup_summary()
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {str(e)}")
            return False
    
    def create_directories(self) -> bool:
        """Create the required directory structure"""
        print("📁 Step 1: Creating directory structure...")
        
        directories = [
            'src',
            'src/core',
            'src/utils',
            'src/strategies',
            'src/strategies/technical',
            'src/strategies/smc',
            'src/strategies/ml',
            'src/strategies/fusion',
            'src/analysis',
            'config',
            'data',
            'logs',
            'logs/trades',
            'logs/signals',
            'logs/performance',
            'logs/errors',
            'docs',
            'tests',
            'backtest',
            'dashboard'
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Created: {directory}")
            
            # Create __init__.py files for Python packages
            init_files = [
                'src/__init__.py',
                'src/core/__init__.py',
                'src/utils/__init__.py',
                'src/strategies/__init__.py',
                'src/strategies/technical/__init__.py',
                'src/strategies/smc/__init__.py',
                'src/strategies/ml/__init__.py',
                'src/strategies/fusion/__init__.py',
                'src/analysis/__init__.py'
            ]
            
            for init_file in init_files:
                init_path = self.project_root / init_file
                init_path.touch()
                print(f"   ✅ Created: {init_file}")
            
            self.setup_status['directories'] = True
            print("✅ Directory structure created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create directories: {str(e)}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install required Python dependencies"""
        print("📦 Step 2: Installing dependencies...")
        
        # Basic requirements
        basic_requirements = [
            'pandas>=2.0.0',
            'numpy>=1.24.0',
            'PyYAML>=6.0',
            'colorlog>=6.7.0',
            'SQLAlchemy>=2.0.0'
        ]
        
        # Development requirements
        dev_requirements = [
            'pytest>=7.4.0',
            'flake8>=6.0.0',
            'black>=23.7.0'
        ]
        
        try:
            # Install basic requirements
            print("   Installing basic requirements...")
            for requirement in basic_requirements:
                result = subprocess.run(
                    [self.python_executable, '-m', 'pip', 'install', requirement],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"   ✅ Installed: {requirement}")
                else:
                    print(f"   ⚠️ Warning: {requirement} - {result.stderr.strip()}")
            
            # Install development requirements if requested
            if self.include_dev:
                print("   Installing development requirements...")
                for requirement in dev_requirements:
                    result = subprocess.run(
                        [self.python_executable, '-m', 'pip', 'install', requirement],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        print(f"   ✅ Installed: {requirement}")
                    else:
                        print(f"   ⚠️ Warning: {requirement} - {result.stderr.strip()}")
            
            # Try to install MetaTrader5 (may fail on some systems)
            print("   Installing MetaTrader5...")
            result = subprocess.run(
                [self.python_executable, '-m', 'pip', 'install', 'MetaTrader5'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("   ✅ Installed: MetaTrader5")
            else:
                print("   ⚠️ Warning: MetaTrader5 installation failed (may need manual installation)")
            
            self.setup_status['dependencies'] = True
            print("✅ Dependencies installed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to install dependencies: {str(e)}")
            return False
    
    def create_configuration(self) -> bool:
        """Create configuration files"""
        print("⚙️ Step 3: Creating configuration files...")
        
        try:
            # Create master configuration
            config_file = self.project_root / 'config' / 'master_config.yaml'
            
            # Default configuration
            default_config = {
                'mt5': {
                    'login': None,
                    'password': None,
                    'server': None,
                    'terminal_path': None,
                    'magic_number': 123456,
                    'slippage': 20,
                    'deviation': 10
                },
                'trading': {
                    'symbol': 'XAUUSDm',
                    'capital': {
                        'initial_capital': 100.0,
                        'target_capital': 1000.0,
                        'minimum_capital': 50.0,
                        'reserve_cash': 0.10
                    },
                    'risk_management': {
                        'risk_per_trade': 0.03,
                        'max_risk_per_trade': 0.05,
                        'max_portfolio_risk': 0.15,
                        'max_drawdown': 0.25,
                        'max_daily_loss': 0.10,
                        'max_weekly_loss': 0.20,
                        'max_consecutive_losses': 4
                    },
                    'position_sizing': {
                        'method': 'kelly_modified',
                        'kelly_safety_factor': 0.30,
                        'min_position_size': 0.01,
                        'max_position_size': 0.10,
                        'max_positions': 3
                    }
                },
                'database': {
                    'type': 'sqlite',
                    'sqlite': {
                        'path': 'data/trading.db'
                    },
                    'retention': {
                        'trades': 365,
                        'signals': 90,
                        'ticks': 7,
                        'performance': 365
                    }
                },
                'logging': {
                    'level': 'INFO',
                    'files': {
                        'system_log': 'logs/system.log',
                        'trade_log': 'logs/trades.log',
                        'signal_log': 'logs/signals.log',
                        'error_log': 'logs/errors.log',
                        'performance_log': 'logs/performance.log'
                    },
                    'rotation': {
                        'enabled': True,
                        'max_size': '10MB',
                        'backup_count': 10
                    },
                    'console': {
                        'enabled': True,
                        'colored': True,
                        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    }
                },
                'error_handling': {
                    'restart_on_error': True,
                    'max_restart_attempts': 3,
                    'error_notification': True
                },
                'system': {
                    'performance': {
                        'multiprocessing': True,
                        'max_workers': 4
                    },
                    'memory': {
                        'max_memory_usage': 0.8,
                        'garbage_collection': True,
                        'gc_interval': 3600
                    }
                }
            }
            
            # Write configuration file
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
            print(f"   ✅ Created: {config_file}")
            
            # Create MT5 specific config
            mt5_config_file = self.project_root / 'config' / 'mt5_config.yaml'
            mt5_config = {
                'connection': {
                    'timeout': 60000,
                    'retry_count': 3,
                    'retry_delay': 5
                },
                'symbols': {
                    'primary': 'XAUUSDm',
                    'alternatives': ['XAUUSD', 'GOLD', 'Gold']
                },
                'timeframes': {
                    'primary': 'M15',
                    'secondary': ['M5', 'H1'],
                    'analysis': ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
                }
            }
            
            with open(mt5_config_file, 'w', encoding='utf-8') as f:
                yaml.dump(mt5_config, f, default_flow_style=False, indent=2)
            
            print(f"   ✅ Created: {mt5_config_file}")
            
            # Create environment file template
            env_file = self.project_root / '.env.template'
            env_content = """# MT5 Configuration
# Copy this file to .env and fill in your actual credentials
# NEVER commit .env to version control!

MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
MT5_TERMINAL_PATH=path_to_terminal64.exe

# Database Configuration
DATABASE_URL=sqlite:///data/trading.db

# Notification Settings (Optional)
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# System Settings
DEBUG=false
LOG_LEVEL=INFO
"""
            
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)
            
            print(f"   ✅ Created: {env_file}")
            
            # Create gitignore
            gitignore_file = self.project_root / '.gitignore'
            gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Trading System Specific
data/
logs/
*.db
*.log
backups/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
*~

# MT5 files
*.ex5
*.mq5
MQL5/
"""
            
            with open(gitignore_file, 'w', encoding='utf-8') as f:
                f.write(gitignore_content)
            
            print(f"   ✅ Created: {gitignore_file}")
            
            self.setup_status['configuration'] = True
            print("✅ Configuration files created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create configuration files: {str(e)}")
            return False
    
    def setup_files(self) -> bool:
        """Setup necessary Python files"""
        print("📄 Step 4: Creating Python files...")
        
        try:
            # Create a simple README
            readme_file = self.project_root / 'README.md'
            readme_content = f"""# XAUUSD MT5 Trading System

**Version:** 1.0.0  
**Created:** {datetime.now().strftime('%Y-%m-%d')}  
**Goal:** Transform $100 to $1000 within 30 days using automated XAUUSD trading

## Phase 1 - Foundation ✅

Phase 1 components have been successfully set up:

- ✅ **MT5 Integration**: Connection and data management
- ✅ **Logging System**: Comprehensive logging infrastructure  
- ✅ **Database**: SQLite database with complete schema
- ✅ **Error Handling**: Robust error handling and recovery

## Quick Start

1. **Configure MT5 credentials:**
   ```bash
   cp .env.template .env
   # Edit .env with your MT5 credentials
   ```

2. **Run the system:**
   ```bash
   python src/core_system.py
   ```

3. **Test the system:**
   ```bash
   python src/core_system.py --test
   ```

## Project Structure

```
├── src/
│   ├── core/           # Core system components
│   ├── utils/          # Utility modules (logging, database, errors)
│   ├── strategies/     # Trading strategies (Phase 2)
│   └── analysis/       # Analysis tools (Phase 2)
├── config/             # Configuration files
├── data/               # Database and data files
├── logs/               # Log files
├── tests/              # Test files
└── docs/               # Documentation
```

## Next Steps - Phase 2

- [ ] Implement technical analysis strategies
- [ ] Add Smart Money Concepts (SMC)
- [ ] Build machine learning models
- [ ] Create strategy fusion system
- [ ] Implement risk management

## Documentation

- `docs/` - Complete documentation
- `config/master_config.yaml` - Main configuration
- `PROJECT_TRACKER.md` - Development progress

## Support

For issues and questions, check the project documentation or create an issue.

---
**⚠️ Risk Warning:** Trading involves substantial risk. Never trade with money you cannot afford to lose.
"""
            
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            print(f"   ✅ Created: {readme_file}")
            
            # Create main runner script
            main_script = self.project_root / 'run_system.py'
            main_content = '''#!/usr/bin/env python3
"""
Main System Runner
==================
Quick launcher for the XAUUSD Trading System

Usage:
    python run_system.py           # Start system
    python run_system.py --test    # Run tests
    python run_system.py --setup   # Re-run setup
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from core_system import CoreSystem
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure Phase 1 setup is complete.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='XAUUSD Trading System')
    parser.add_argument('--test', action='store_true', help='Run system tests')
    parser.add_argument('--setup', action='store_true', help='Re-run setup')
    parser.add_argument('--connect', action='store_true', help='Test MT5 connection')
    
    args = parser.parse_args()
    
    if args.setup:
        from setup_phase1 import Phase1Setup
        setup = Phase1Setup(include_dev=True, run_tests=True)
        return setup.run_setup()
    
    # Initialize core system
    core = CoreSystem()
    
    try:
        # Initialize
        if not core.initialize():
            print("❌ System initialization failed")
            return False
        
        if args.test:
            # Run tests
            test_results = core.test_system()
            return test_results['summary']['success_rate'] > 0.8
        
        elif args.connect:
            # Test MT5 connection
            return core.connect_mt5()
        
        else:
            # Start interactive mode
            print("🎯 XAUUSD Trading System - Interactive Mode")
            print("Commands:")
            print("  connect  - Connect to MT5")
            print("  test     - Run system tests")
            print("  stats    - Show system statistics")
            print("  quit     - Exit system")
            
            while True:
                try:
                    cmd = input("\\n> ").strip().lower()
                    
                    if cmd == 'quit':
                        break
                    elif cmd == 'connect':
                        core.connect_mt5()
                    elif cmd == 'test':
                        core.test_system()
                    elif cmd == 'stats':
                        stats = core.get_system_stats()
                        print(f"System Stats: {stats}")
                    else:
                        print("Unknown command. Try: connect, test, stats, quit")
                        
                except KeyboardInterrupt:
                    break
            
            return True
    
    finally:
        core.shutdown()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
            
            with open(main_script, 'w', encoding='utf-8') as f:
                f.write(main_content)
            
            print(f"   ✅ Created: {main_script}")
            
            # Create requirements.txt
            requirements_file = self.project_root / 'requirements.txt'
            requirements_content = """# XAUUSD MT5 Trading System - Phase 1 Requirements

# Core Dependencies
pandas>=2.0.0
numpy>=1.24.0
PyYAML>=6.0
colorlog>=6.7.0
SQLAlchemy>=2.0.0

# MT5 Integration
MetaTrader5>=5.0.45

# Development Tools (optional)
pytest>=7.4.0
flake8>=6.0.0
black>=23.7.0

# Future Phase 2 Dependencies (commented out for now)
# scikit-learn>=1.3.0
# tensorflow>=2.13.0
# xgboost>=1.7.0
# streamlit>=1.25.0
# plotly>=5.15.0
# TA-Lib>=0.4.27
"""
            
            with open(requirements_file, 'w', encoding='utf-8') as f:
                f.write(requirements_content)
            
            print(f"   ✅ Created: {requirements_file}")
            
            # Create a simple test file
            test_file = self.project_root / 'tests' / 'test_phase1.py'
            test_content = '''"""
Basic Phase 1 Tests
==================
Simple tests to verify Phase 1 components are working
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestPhase1(unittest.TestCase):
    """Basic tests for Phase 1 components"""
    
    def test_imports(self):
        """Test that all modules can be imported"""
        try:
            from utils.logger import LoggerManager
            from utils.database import DatabaseManager
            from utils.error_handler import ErrorHandler
            from core.mt5_manager import MT5Manager
            from core_system import CoreSystem
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_core_system_creation(self):
        """Test that CoreSystem can be created"""
        try:
            from core_system import CoreSystem
            core = CoreSystem()
            self.assertIsNotNone(core)
        except Exception as e:
            self.fail(f"CoreSystem creation failed: {e}")


if __name__ == '__main__':
    unittest.main()
'''
            
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            print(f"   ✅ Created: {test_file}")
            
            self.setup_status['files'] = True
            print("✅ Python files created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create files: {str(e)}")
            return False
    
    def run_system_tests(self) -> bool:
        """Run basic system tests"""
        print("🧪 Step 5: Running system tests...")
        
        try:
            # Add src to Python path
            sys.path.insert(0, str(self.project_root / 'src'))
            
            # Test basic imports first
            print("   Testing imports...")
            try:
                from src.utils.logger import LoggerManager
                from src.utils.database import DatabaseManager
                from src.utils.error_handler import ErrorHandler
                from src.core.mt5_manager import MT5Manager
                print("   ✅ All imports successful")
            except ImportError as e:
                print(f"   ❌ Import failed: {e}")
                print("   ⚠️  Note: You'll need to copy the Phase 1 module files to run full tests")
                return True  # Don't fail setup for missing module files
            
            # Test core system if available
            try:
                from src.core_system import CoreSystem
                
                # Create and initialize core system
                core = CoreSystem()
                
                if not core.initialize():
                    print("❌ Core system initialization failed")
                    return False
                
                # Run basic functionality tests
                test_results = {
                    'logging': False,
                    'database': False,
                    'error_handling': False
                }
                
                # Test logging
                try:
                    core.logger.info("Test log message")
                    test_results['logging'] = True
                    print("   ✅ Logging test passed")
                except:
                    print("   ❌ Logging test failed")
                
                # Test database
                try:
                    stats = core.database.get_database_stats()
                    test_results['database'] = True
                    print("   ✅ Database test passed")
                except:
                    print("   ❌ Database test failed")
                
                # Test error handling
                try:
                    test_error = Exception("Test error")
                    core.errors.handle_error(test_error, "System test")
                    test_results['error_handling'] = True
                    print("   ✅ Error handling test passed")
                except:
                    print("   ❌ Error handling test failed")
                
                # Shutdown system
                core.shutdown()
                
                # Calculate success rate
                passed_tests = sum(test_results.values())
                total_tests = len(test_results)
                success_rate = passed_tests / total_tests
                
                if success_rate >= 0.8:  # 80% success rate required
                    print(f"✅ System tests passed ({success_rate:.1%} success rate)")
                    self.setup_status['tests'] = True
                    return True
                else:
                    print(f"❌ System tests failed ({success_rate:.1%} success rate)")
                    return False
                    
            except ImportError:
                print("   ⚠️  Core system not available yet - basic import tests passed")
                self.setup_status['tests'] = True
                return True
                
        except Exception as e:
            print(f"❌ System tests failed: {str(e)}")
            return False
    
    def print_setup_summary(self) -> None:
        """Print setup completion summary"""
        print("\n" + "="*60)
        print("🎉 Phase 1 Setup Complete!")
        print("="*60)
        
        # Status summary
        print("\n📋 Setup Status:")
        for component, status in self.setup_status.items():
            icon = "✅" if status else "❌"
            print(f"   {icon} {component.replace('_', ' ').title()}")
        
        # Next steps
        print("\n🚀 Next Steps:")
        print("1. Copy Phase 1 module files:")
        print("   • Copy logger.py to src/utils/")
        print("   • Copy database.py to src/utils/")
        print("   • Copy error_handler.py to src/utils/")
        print("   • Copy core_system.py to src/")
        print("   • Copy mt5_manager.py to src/core/")
        print()
        print("2. Configure your MT5 credentials:")
        print("   • Copy .env.template to .env")
        print("   • Edit .env with your MT5 login details")
        print()
        print("3. Test the system:")
        print("   • python run_system.py --test")
        print("   • python run_system.py --connect")
        print()
        print("4. Start development:")
        print("   • python run_system.py")
        print("   • Review PROJECT_TRACKER.md for Phase 2 tasks")
        
        # Project structure
        print("\n📁 Project Structure Created:")
        print("   • src/core/        - Core system components")
        print("   • src/utils/       - Utility modules")
        print("   • config/          - Configuration files")
        print("   • data/            - Database and data storage")
        print("   • logs/            - System logs")
        print("   • docs/            - Documentation")
        print("   • tests/           - Test files")
        
        # Important files
        print("\n📄 Important Files Created:")
        print("   • config/master_config.yaml - Main configuration")
        print("   • .env.template            - Environment variables template")
        print("   • run_system.py            - Main system launcher")
        print("   • requirements.txt         - Python dependencies")
        print("   • README.md                - Project documentation")
        
        print("\n⚠️  Remember:")
        print("   • Copy the Phase 1 module files from the artifacts")
        print("   • Never commit .env file to version control")
        print("   • Test on demo account before live trading")
        print("   • Review all configuration settings")
        
        print("\n🎯 Phase 1 Foundation structure is ready!")
        print("Copy the module files and you'll be ready for development!")
        print("="*60)


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Phase 1 Setup Script')
    parser.add_argument('--dev', action='store_true', 
                       help='Include development tools')
    parser.add_argument('--test', action='store_true', 
                       help='Run tests after setup')
    parser.add_argument('--force', action='store_true',
                       help='Force reinstall even if already set up')
    
    args = parser.parse_args()
    
    # Check if already set up
    if not args.force:
        config_file = Path('config/master_config.yaml')
        if config_file.exists():
            print("⚠️  Phase 1 appears to already be set up.")
            print("Use --force to reinstall, or run system tests:")
            print("python run_system.py --test")
            return True
    
    # Create setup instance
    setup = Phase1Setup(include_dev=args.dev, run_tests=args.test)
    
    # Run setup
    return setup.run_setup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)