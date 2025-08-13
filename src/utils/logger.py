"""
Logger Module - Complete Logging Infrastructure

==============================================

Author: XAUUSD Trading System

Version: 1.0.0

Date: 2025-08-07

This module provides comprehensive logging functionality for the trading system:

- Structured logging with multiple levels
- File rotation and management
- Console and file output
- Trade-specific logging
- Performance logging
- Error tracking

Features:

- Automatic log rotation
- Colored console output
- JSON structured logs
- Multiple log files for different purposes
- Performance monitoring
- Error tracking and notifications

Dependencies:

- logging
- colorlog
- json
- pathlib
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import json
import sys
import traceback
from typing import Dict, Any, Optional
import colorlog
from enum import Enum

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LoggerManager:
    """
    Comprehensive logging manager for the trading system

    This class manages all logging operations including:
    - System logs
    - Trade logs
    - Signal logs
    - Error logs
    - Performance logs

    Example:
    >>> logger_mgr = LoggerManager(config)
    >>> logger_mgr.setup_logging()
    >>> logger_mgr.log_trade("BUY", "XAUUSDm", 0.01, 1950.0)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the logging manager

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.logging_config = config.get('logging', {})
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.log_dir / "trades").mkdir(exist_ok=True)
        (self.log_dir / "signals").mkdir(exist_ok=True)
        (self.log_dir / "performance").mkdir(exist_ok=True)
        (self.log_dir / "errors").mkdir(exist_ok=True)

        # Logger instances
        self.system_logger = None
        self.trade_logger = None
        self.signal_logger = None
        self.error_logger = None
        self.performance_logger = None

        # Setup flag
        self._setup_complete = False

    def setup_logging(self) -> bool:
        """
        Set up all logging components

        Returns:
            bool: True if setup successful
        """
        try:
            # Get log level
            log_level = getattr(logging, self.logging_config.get('level', 'INFO'))

            # Setup system logger
            self._setup_system_logger(log_level)

            # Setup specialized loggers
            self._setup_trade_logger()
            self._setup_signal_logger()
            self._setup_error_logger()
            self._setup_performance_logger()

            # Setup console logger if enabled
            if self.logging_config.get('console', {}).get('enabled', True):
                self._setup_console_logger(log_level)

            self._setup_complete = True
            self.system_logger.info("Logging system initialized successfully")
            return True

        except Exception as e:
            print(f"Failed to setup logging: {str(e)}")
            return False

    def _setup_system_logger(self, log_level: int) -> None:
        """Setup the main system logger"""
        self.system_logger = logging.getLogger('xau_system')
        self.system_logger.setLevel(log_level)

        # ✅ Close old handlers before clearing
        for handler in list(self.system_logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
        self.system_logger.handlers.clear()

        # File handler with rotation
        log_file = self.log_dir / "system.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=int(self._parse_size(self.logging_config.get('rotation', {}).get('max_size', '10MB'))),
            backupCount=self.logging_config.get('rotation', {}).get('backup_count', 10)
        )

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.system_logger.addHandler(file_handler)

    def _setup_trade_logger(self) -> None:
        """Setup trade-specific logger"""
        self.trade_logger = logging.getLogger('xau_trades')
        self.trade_logger.setLevel(logging.INFO)

        # ✅ Close old handlers before clearing
        for handler in list(self.trade_logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
        self.trade_logger.handlers.clear()

        log_file = self.log_dir / "trades" / "trades.log"
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when='midnight', interval=1, backupCount=30
        )

        formatter = JsonFormatter()
        file_handler.setFormatter(formatter)
        self.trade_logger.addHandler(file_handler)
        self.trade_logger.propagate = False

    def _setup_signal_logger(self) -> None:
        """Setup signal-specific logger"""
        self.signal_logger = logging.getLogger('xau_signals')
        self.signal_logger.setLevel(logging.INFO)

        # ✅ Close old handlers before clearing
        for handler in list(self.signal_logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
        self.signal_logger.handlers.clear()

        log_file = self.log_dir / "signals" / "signals.log"
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when='midnight', interval=1, backupCount=30
        )

        formatter = JsonFormatter()
        file_handler.setFormatter(formatter)
        self.signal_logger.addHandler(file_handler)
        self.signal_logger.propagate = False

    def _setup_error_logger(self) -> None:
        """Setup error-specific logger"""
        self.error_logger = logging.getLogger('xau_errors')
        self.error_logger.setLevel(logging.ERROR)

        # ✅ Close old handlers before clearing
        for handler in list(self.error_logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
        self.error_logger.handlers.clear()

        log_file = self.log_dir / "errors" / "errors.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5*1024*1024, backupCount=10
        )

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n'
            'Exception: %(exc_info)s\n' + '='*80,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.error_logger.addHandler(file_handler)
        self.error_logger.propagate = False

    def _setup_performance_logger(self) -> None:
        """Setup performance-specific logger"""
        self.performance_logger = logging.getLogger('xau_performance')
        self.performance_logger.setLevel(logging.INFO)

        # ✅ Close old handlers before clearing
        for handler in list(self.performance_logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
        self.performance_logger.handlers.clear()

        log_file = self.log_dir / "performance" / "performance.log"
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when='midnight', interval=1, backupCount=365
        )

        formatter = JsonFormatter()
        file_handler.setFormatter(formatter)
        self.performance_logger.addHandler(file_handler)
        self.performance_logger.propagate = False

    def _setup_console_logger(self, log_level: int) -> None:
        """Setup colored console output"""
        console_config = self.logging_config.get('console', {})
        if console_config.get('colored', True):
            console_handler = colorlog.StreamHandler()
            console_handler.setLevel(log_level)
            color_formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(color_formatter)
            self.system_logger.addHandler(console_handler)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            self.system_logger.addHandler(console_handler)

    def _parse_size(self, size_str: str) -> int:
        """Parse size string (e.g., '10MB') to bytes"""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    # Logging methods
    def log_system(self, level: str, message: str, **kwargs) -> None:
        """Log system message"""
        if not self._setup_complete:
            return
        
        log_func = getattr(self.system_logger, level.lower())
        log_func(message, **kwargs)
    
    def log_trade(self, action: str, symbol: str, volume: float, price: float, 
                  ticket: Optional[int] = None, sl: Optional[float] = None, 
                  tp: Optional[float] = None, **kwargs) -> None:
        """Log trade execution"""
        if not self._setup_complete:
            return
        
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'symbol': symbol,
            'volume': volume,
            'price': price,
            'ticket': ticket,
            'stop_loss': sl,
            'take_profit': tp,
            **kwargs
        }
        
        self.trade_logger.info("trade_executed", extra={'trade_data': trade_data})
    
    def log_signal(self, strategy: str, signal_type: str, symbol: str, 
                   confidence: float, price: float, **kwargs) -> None:
        """Log trading signal"""
        if not self._setup_complete:
            return
        
        signal_data = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'signal_type': signal_type,
            'symbol': symbol,
            'confidence': confidence,
            'price': price,
            **kwargs
        }
        
        self.signal_logger.info("signal_generated", extra={'signal_data': signal_data})
    
    def log_error(self, error: Exception, context: str = "", **kwargs) -> None:
        """Log error with full context"""
        if not self._setup_complete:
            print(f"Error (logging not setup): {error}")
            return
        
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc(),
            **kwargs
        }
        
        self.error_logger.error(f"Error in {context}: {error}", exc_info=True, 
                               extra={'error_data': error_data})
    
    def log_performance(self, metric_name: str, value: float, period: str = "daily", 
                       **kwargs) -> None:
        """Log performance metrics"""
        if not self._setup_complete:
            return
        
        perf_data = {
            'timestamp': datetime.now().isoformat(),
            'metric_name': metric_name,
            'value': value,
            'period': period,
            **kwargs
        }
        
        self.performance_logger.info("performance_metric", extra={'perf_data': perf_data})
    
    # Convenience methods
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        self.log_system("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        self.log_system("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        self.log_system("warning", message, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log error message"""
        if error:
            self.log_error(error, message, **kwargs)
        else:
            self.log_system("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message"""
        self.log_system("critical", message, **kwargs)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a specific logger by name"""
        loggers = {
            'system': self.system_logger,
            'trade': self.trade_logger,
            'signal': self.signal_logger,
            'error': self.error_logger,
            'performance': self.performance_logger
        }
        return loggers.get(name, self.system_logger)
    
    def cleanup_old_logs(self, days: int = 30) -> None:
        """Clean up old log files"""
        try:
            cutoff_date = datetime.now().timestamp() - (days * 24 * 3600)
            
            for log_file in self.log_dir.rglob("*.log*"):
                if log_file.stat().st_mtime < cutoff_date:
                    log_file.unlink()
                    self.info(f"Deleted old log file: {log_file}")
                    
        except Exception as e:
            self.error("Failed to cleanup old logs", e)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra data if present
        if hasattr(record, 'trade_data'):
            log_obj.update(record.trade_data)
        elif hasattr(record, 'signal_data'):
            log_obj.update(record.signal_data)
        elif hasattr(record, 'error_data'):
            log_obj.update(record.error_data)
        elif hasattr(record, 'perf_data'):
            log_obj.update(record.perf_data)
        
        return json.dumps(log_obj)


# Global logger instance
_logger_manager = None


def get_logger_manager(config: Optional[Dict] = None) -> LoggerManager:
    """Get or create the global logger manager"""
    global _logger_manager
    
    if _logger_manager is None:
        if config is None:
            # Default config if none provided
            config = {
                'logging': {
                    'level': 'INFO',
                    'console': {'enabled': True, 'colored': True},
                    'rotation': {'max_size': '10MB', 'backup_count': 10}
                }
            }
        _logger_manager = LoggerManager(config)
        _logger_manager.setup_logging()
    
    return _logger_manager


def setup_logging(config: Dict) -> LoggerManager:
    """Setup logging with configuration"""
    logger_mgr = LoggerManager(config)
    logger_mgr.setup_logging()
    return logger_mgr


# Testing function
if __name__ == "__main__":
    """Test the logging system"""
    
    # Test configuration
    test_config = {
        'logging': {
            'level': 'DEBUG',
            'console': {
                'enabled': True,
                'colored': True
            },
            'rotation': {
                'max_size': '1MB',
                'backup_count': 5
            }
        }
    }
    
    # Create logger manager
    logger_mgr = LoggerManager(test_config)
    logger_mgr.setup_logging()
    
    # Test all logging methods
    logger_mgr.info("System started")
    logger_mgr.debug("Debug message")
    logger_mgr.warning("Warning message")
    
    # Test trade logging
    logger_mgr.log_trade("BUY", "XAUUSDm", 0.01, 1950.0, ticket=12345, 
                        sl=1945.0, tp=1960.0, strategy="test")
    
    # Test signal logging
    logger_mgr.log_signal("ichimoku", "BUY", "XAUUSDm", 0.85, 1950.0, 
                         timeframe="M15")
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger_mgr.log_error(e, "testing error logging")
    
    # Test performance logging
    logger_mgr.log_performance("daily_return", 2.5, "daily", account_balance=100.0)
    
    print("Logging test completed. Check the logs/ directory for output files.")
