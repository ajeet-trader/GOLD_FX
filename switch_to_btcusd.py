#!/usr/bin/env python3
"""
BTCUSD Configuration Switcher
============================

This script helps switch the trading system from GOLD (XAUUSDm) to BTCUSD (BTCUSDm)
for testing purposes when the gold market is closed.

Features:
- Backs up current configuration
- Switches to BTCUSD configuration
- Updates all relevant files with BTCUSD parameters
- Provides easy rollback functionality

Usage:
    python switch_to_btcusd.py [--rollback]
    
Options:
    --rollback    Restore original GOLD configuration
"""

import os
import sys
import shutil
import yaml
import argparse
from datetime import datetime
from pathlib import Path

class BTCUSDSwitcher:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "config"
        self.master_config = self.config_dir / "master_config.yaml"
        self.btcusd_config = self.config_dir / "btcusd_config.yaml"
        self.backup_dir = self.project_root / "config_backups"
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(exist_ok=True)
        
    def backup_current_config(self):
        """Backup the current master configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"master_config_backup_{timestamp}.yaml"
        
        if self.master_config.exists():
            shutil.copy2(self.master_config, backup_file)
            print(f"✅ Backed up current configuration to: {backup_file}")
            return backup_file
        else:
            print("⚠️  No master configuration found to backup")
            return None
    
    def switch_to_btcusd(self):
        """Switch the system to use BTCUSD configuration"""
        print("🔄 Switching to BTCUSD configuration...")
        
        # Backup current config
        backup_file = self.backup_current_config()
        
        # Copy BTCUSD config to master config
        if self.btcusd_config.exists():
            shutil.copy2(self.btcusd_config, self.master_config)
            print("✅ Successfully switched to BTCUSD configuration")
            
            # Update any hardcoded references in test files
            self.update_test_files()
            
            print("\n📊 BTCUSD Configuration Active:")
            print("   - Symbol: BTCUSDm")
            print("   - Mode: mock")
            print("   - Risk per trade: 2%")
            print("   - Max positions: 2")
            print("   - Magic number: 654321")
            print("   - Dashboard port: 8502")
            print("   - Logs: btcusd_*.log")
            
            if backup_file:
                print(f"\n🔙 To rollback: python {__file__} --rollback")
                print(f"   Backup saved as: {backup_file.name}")
                
        else:
            print("❌ BTCUSD configuration file not found!")
            return False
            
        return True
    
    def update_test_files(self):
        """Update test files to use BTCUSD instead of XAUUSDm"""
        test_files = [
            "tests/Phase-2/test_complete_execution.py",
            "tests/Phase-1/run_simple.py",
            "run_system.py"
        ]
        
        replacements = {
            "XAUUSDm": "BTCUSDm",
            "XAUUSD": "BTCUSD",
            "GOLD": "BTC",
            "Gold": "Bitcoin"
        }
        
        updated_files = []
        
        for test_file in test_files:
            file_path = self.project_root / test_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    for old_val, new_val in replacements.items():
                        content = content.replace(old_val, new_val)
                    
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        updated_files.append(test_file)
                        
                except Exception as e:
                    print(f"⚠️  Warning: Could not update {test_file}: {e}")
        
        if updated_files:
            print(f"📝 Updated {len(updated_files)} test files for BTCUSD:")
            for file in updated_files:
                print(f"   - {file}")
    
    def rollback_to_gold(self):
        """Rollback to the most recent GOLD configuration"""
        print("🔄 Rolling back to GOLD configuration...")
        
        # Find the most recent backup
        backup_files = list(self.backup_dir.glob("master_config_backup_*.yaml"))
        if not backup_files:
            print("❌ No backup files found to rollback to!")
            return False
        
        # Get the most recent backup
        latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
        
        # Restore the backup
        shutil.copy2(latest_backup, self.master_config)
        print(f"✅ Restored configuration from: {latest_backup.name}")
        
        # Revert test files
        self.revert_test_files()
        
        print("\n📊 GOLD Configuration Restored:")
        print("   - Symbol: XAUUSDm")
        print("   - Mode: mock")
        print("   - Risk per trade: 3%")
        print("   - Max positions: 3")
        print("   - Magic number: 123456")
        print("   - Dashboard port: 8501")
        print("   - Logs: system.log, trades.log, etc.")
        
        return True
    
    def revert_test_files(self):
        """Revert test files back to GOLD/XAUUSDm"""
        test_files = [
            "tests/Phase-2/test_complete_execution.py",
            "tests/Phase-1/run_simple.py", 
            "run_system.py"
        ]
        
        replacements = {
            "BTCUSDm": "XAUUSDm",
            "BTCUSD": "XAUUSD",
            "BTC": "GOLD",
            "Bitcoin": "Gold"
        }
        
        reverted_files = []
        
        for test_file in test_files:
            file_path = self.project_root / test_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    for old_val, new_val in replacements.items():
                        content = content.replace(old_val, new_val)
                    
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        reverted_files.append(test_file)
                        
                except Exception as e:
                    print(f"⚠️  Warning: Could not revert {test_file}: {e}")
        
        if reverted_files:
            print(f"📝 Reverted {len(reverted_files)} test files to GOLD:")
            for file in reverted_files:
                print(f"   - {file}")
    
    def show_status(self):
        """Show current configuration status"""
        print("📊 Current Configuration Status:")
        
        if self.master_config.exists():
            try:
                with open(self.master_config, 'r') as f:
                    config = yaml.safe_load(f)
                
                symbol = config.get('trading', {}).get('symbol', 'Unknown')
                mode = config.get('mode', 'Unknown')
                magic_number = config.get('execution', {}).get('magic_number', 'Unknown')
                
                print(f"   - Symbol: {symbol}")
                print(f"   - Mode: {mode}")
                print(f"   - Magic Number: {magic_number}")
                
                if symbol == "BTCUSDm":
                    print("   - Status: 🟡 BTCUSD Configuration Active")
                elif symbol == "XAUUSDm":
                    print("   - Status: 🟢 GOLD Configuration Active")
                else:
                    print("   - Status: ❓ Unknown Configuration")
                    
            except Exception as e:
                print(f"   - Error reading config: {e}")
        else:
            print("   - Status: ❌ No master configuration found")

def main():
    parser = argparse.ArgumentParser(description="Switch trading system between GOLD and BTCUSD")
    parser.add_argument("--rollback", action="store_true", 
                       help="Rollback to GOLD configuration")
    parser.add_argument("--status", action="store_true",
                       help="Show current configuration status")
    
    args = parser.parse_args()
    
    switcher = BTCUSDSwitcher()
    
    print("🚀 BTCUSD Configuration Switcher")
    print("=" * 40)
    
    if args.status:
        switcher.show_status()
    elif args.rollback:
        success = switcher.rollback_to_gold()
        if success:
            print("\n✅ Successfully rolled back to GOLD configuration")
        else:
            print("\n❌ Rollback failed")
            sys.exit(1)
    else:
        success = switcher.switch_to_btcusd()
        if success:
            print("\n✅ Successfully switched to BTCUSD configuration")
            print("\n🧪 Ready for testing with BTCUSD!")
            print("\nNext steps:")
            print("1. Run your test files (they've been updated for BTCUSD)")
            print("2. Check logs in btcusd_*.log files")
            print("3. Dashboard available on port 8502")
            print("4. Use --rollback to switch back to GOLD when done")
        else:
            print("\n❌ Switch failed")
            sys.exit(1)

if __name__ == "__main__":
    main()
