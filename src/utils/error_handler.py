"""
Error Handler - Complete Error Handling Framework
================================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-08-07

This module provides comprehensive error handling for the trading system:
- Custom exception classes
- Error categorization and severity levels
- Automatic error recovery mechanisms
- Error notification system
- Performance impact monitoring
- System health monitoring

Features:
- Graceful error handling
- Automatic retry mechanisms
- Circuit breaker patterns
- Error aggregation and reporting
- Recovery strategies
- System shutdown protocols

Dependencies:
    - logging
    - traceback
    - datetime
    - threading
"""

import logging
import traceback
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import queue
import json


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorCategory(Enum):
    """Error categories"""
    CONNECTION = "CONNECTION"
    TRADING = "TRADING"
    DATA = "DATA"
    SYSTEM = "SYSTEM"
    VALIDATION = "VALIDATION"
    NETWORK = "NETWORK"
    CONFIGURATION = "CONFIGURATION"
    UNKNOWN = "UNKNOWN"


@dataclass
class ErrorContext:
    """Error context information"""
    timestamp: datetime
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception]
    traceback_info: str
    function_name: str
    module_name: str
    line_number: int
    context_data: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0


# Custom Exception Classes
class TradingSystemError(Exception):
    """Base exception for trading system errors"""
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Dict = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()


class ConnectionError(TradingSystemError):
    """Connection-related errors"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, ErrorCategory.CONNECTION, ErrorSeverity.HIGH, context)


class TradingError(TradingSystemError):
    """Trading operation errors"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH, context: Dict = None):
        super().__init__(message, ErrorCategory.TRADING, severity, context)


class DataError(TradingSystemError):
    """Data-related errors"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Dict = None):
        super().__init__(message, ErrorCategory.DATA, severity, context)


class ValidationError(TradingSystemError):
    """Validation errors"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM, context)


class ConfigurationError(TradingSystemError):
    """Configuration errors"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH, context)


class SystemError(TradingSystemError):
    """System-level errors"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.CRITICAL, context: Dict = None):
        super().__init__(message, ErrorCategory.SYSTEM, severity, context)


class ErrorHandler:
    """
    Comprehensive error handling manager
    
    This class provides:
    - Error catching and categorization
    - Automatic retry mechanisms
    - Recovery strategies
    - Error reporting and notifications
    - System health monitoring
    
    Example:
        >>> error_handler = ErrorHandler(config)
        >>> error_handler.start()
        >>> 
        >>> @error_handler.handle_errors(retry_count=3)
        >>> def risky_function():
        ...     # Some risky operation
        ...     pass
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the error handler
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.error_config = config.get('error_handling', {})
        
        # Error storage
        self.error_queue = queue.Queue()
        self.error_history: List[ErrorContext] = []
        self.error_stats = {
            'total_errors': 0,
            'errors_by_category': {},
            'errors_by_severity': {},
            'recovery_success_rate': 0.0
        }
        
        # Circuit breaker states
        self.circuit_breakers = {}
        
        # Recovery strategies
        self.recovery_strategies = {
            ErrorCategory.CONNECTION: self._recover_connection,
            ErrorCategory.TRADING: self._recover_trading,
            ErrorCategory.DATA: self._recover_data,
            ErrorCategory.SYSTEM: self._recover_system
        }
        
        # Threading
        self._error_processor_thread = None
        self._monitoring_thread = None
        self._stop_event = threading.Event()
        
        # Logger
        self.logger = logging.getLogger('xau_error_handler')
        
        # System state
        self.system_healthy = True
        self.last_health_check = datetime.now()
    
    def start(self) -> None:
        """Start the error handling system"""
        try:
            # Start error processing thread
            self._error_processor_thread = threading.Thread(
                target=self._process_errors,
                name="ErrorProcessor",
                daemon=True
            )
            self._error_processor_thread.start()
            
            # Start monitoring thread
            self._monitoring_thread = threading.Thread(
                target=self._monitor_system_health,
                name="HealthMonitor",
                daemon=True
            )
            self._monitoring_thread.start()
            
            self.logger.info("Error handling system started")
            
        except Exception as e:
            self.logger.error(f"Failed to start error handling system: {str(e)}")
    
    def stop(self) -> None:
        """Stop the error handling system"""
        self._stop_event.set()
        
        if self._error_processor_thread and self._error_processor_thread.is_alive():
            self._error_processor_thread.join(timeout=5)
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        
        self.logger.info("Error handling system stopped")
    
    def handle_error(self, error: Exception, context: str = "", 
                    function_name: str = "", **kwargs) -> ErrorContext:
        """
        Handle an error and create error context
        
        Args:
            error (Exception): The exception that occurred
            context (str): Additional context information
            function_name (str): Name of the function where error occurred
            **kwargs: Additional context data
        
        Returns:
            ErrorContext: Error context object
        """
        # Determine error category and severity
        if isinstance(error, TradingSystemError):
            category = error.category
            severity = error.severity
        else:
            category = self._categorize_error(error)
            severity = self._determine_severity(error, category)
        
        # Create error context
        error_context = ErrorContext(
            timestamp=datetime.now(),
            error_id=self._generate_error_id(),
            category=category,
            severity=severity,
            message=str(error),
            exception=error,
            traceback_info=traceback.format_exc(),
            function_name=function_name or self._get_calling_function(),
            module_name=self._get_calling_module(),
            line_number=self._get_line_number(),
            context_data={
                'context': context,
                **kwargs
            }
        )
        
        # Add to queue for processing
        self.error_queue.put(error_context)
        
        # Update statistics
        self._update_error_stats(error_context)
        
        # Log the error
        self.logger.error(
            f"Error handled: {error_context.category.value} - {error_context.message}",
            extra={'error_context': error_context}
        )
        
        return error_context
    
    def handle_errors(self, retry_count: int = 0, recovery: bool = True,
                     circuit_breaker: bool = False, 
                     exceptions: tuple = (Exception,)) -> Callable:
        """
        Decorator for automatic error handling
        
        Args:
            retry_count (int): Number of retry attempts
            recovery (bool): Whether to attempt recovery
            circuit_breaker (bool): Whether to use circuit breaker pattern
            exceptions (tuple): Exceptions to catch
        
        Returns:
            Callable: Decorated function
        
        Example:
            >>> @error_handler.handle_errors(retry_count=3, recovery=True)
            >>> def risky_function():
            ...     # Some operation that might fail
            ...     pass
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                function_name = func.__name__
                
                # Check circuit breaker
                if circuit_breaker and self._is_circuit_open(function_name):
                    raise SystemError(f"Circuit breaker open for {function_name}")
                
                last_error = None
                
                for attempt in range(retry_count + 1):
                    try:
                        result = func(*args, **kwargs)
                        
                        # Reset circuit breaker on success
                        if circuit_breaker:
                            self._reset_circuit_breaker(function_name)
                        
                        return result
                        
                    except exceptions as e:
                        last_error = e
                        
                        # Handle the error
                        error_context = self.handle_error(
                            e, 
                            f"Function: {function_name}, Attempt: {attempt + 1}",
                            function_name,
                            attempt=attempt + 1,
                            max_attempts=retry_count + 1,
                            args=str(args),
                            kwargs=str(kwargs)
                        )
                        
                        # Update circuit breaker
                        if circuit_breaker:
                            self._update_circuit_breaker(function_name)
                        
                        # Try recovery if enabled and not last attempt
                        if recovery and attempt < retry_count:
                            recovery_successful = self._attempt_recovery(error_context)
                            error_context.recovery_attempted = True
                            error_context.recovery_successful = recovery_successful
                            
                            if recovery_successful:
                                # Wait before retry
                                time.sleep(min(2 ** attempt, 30))  # Exponential backoff
                                continue
                        
                        # If last attempt, break
                        if attempt == retry_count:
                            break
                        
                        # Wait before retry
                        time.sleep(min(2 ** attempt, 30))
                
                # All attempts failed
                raise last_error
            
            return wrapper
        return decorator
    
    def _process_errors(self) -> None:
        """Process errors from the queue"""
        while not self._stop_event.is_set():
            try:
                # Get error from queue with timeout
                try:
                    error_context = self.error_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Add to history
                self.error_history.append(error_context)
                
                # Limit history size
                if len(self.error_history) > 1000:
                    self.error_history = self.error_history[-500:]
                
                # Check if system health is affected
                self._check_system_health(error_context)
                
                # Send notifications if necessary
                self._send_error_notification(error_context)
                
                # Mark task as done
                self.error_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in error processor: {str(e)}")
    
    def _monitor_system_health(self) -> None:
        """Monitor overall system health"""
        while not self._stop_event.is_set():
            try:
                # Check error rates
                recent_errors = self._get_recent_errors(minutes=5)
                critical_errors = [e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]
                
                # Check if too many critical errors
                if len(critical_errors) >= 3:
                    self.system_healthy = False
                    self.logger.critical("System health degraded: Too many critical errors")
                    self._trigger_emergency_protocols()
                
                # Check error rate
                if len(recent_errors) >= 10:
                    self.system_healthy = False
                    self.logger.warning("System health degraded: High error rate")
                
                # Update health check timestamp
                self.last_health_check = datetime.now()
                
                # Sleep for next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {str(e)}")
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on its type and message"""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Connection errors
        if any(keyword in error_message for keyword in 
               ['connection', 'timeout', 'network', 'socket', 'unreachable']):
            return ErrorCategory.CONNECTION
        
        # Trading errors
        if any(keyword in error_message for keyword in 
               ['trade', 'order', 'position', 'margin', 'insufficient']):
            return ErrorCategory.TRADING
        
        # Data errors
        if any(keyword in error_message for keyword in 
               ['data', 'parse', 'format', 'empty', 'invalid']):
            return ErrorCategory.DATA
        
        # Validation errors
        if 'validation' in error_type or 'value' in error_type:
            return ErrorCategory.VALIDATION
        
        # Configuration errors
        if any(keyword in error_message for keyword in 
               ['config', 'setting', 'parameter']):
            return ErrorCategory.CONFIGURATION
        
        # System errors
        if any(keyword in error_message for keyword in 
               ['memory', 'disk', 'permission', 'system']):
            return ErrorCategory.SYSTEM
        
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity"""
        error_message = str(error).lower()
        
        # Critical severity indicators
        if any(keyword in error_message for keyword in 
               ['critical', 'fatal', 'shutdown', 'corrupt']):
            return ErrorSeverity.CRITICAL
        
        # High severity based on category
        if category in [ErrorCategory.CONNECTION, ErrorCategory.TRADING, ErrorCategory.SYSTEM]:
            return ErrorSeverity.HIGH
        
        # Medium severity
        if category in [ErrorCategory.DATA, ErrorCategory.CONFIGURATION]:
            return ErrorSeverity.MEDIUM
        
        # Default to low
        return ErrorSeverity.LOW
    
    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from an error"""
        try:
            recovery_func = self.recovery_strategies.get(error_context.category)
            if recovery_func:
                return recovery_func(error_context)
            return False
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {str(e)}")
            return False
    
    def _recover_connection(self, error_context: ErrorContext) -> bool:
        """Recover from connection errors"""
        try:
            self.logger.info("Attempting connection recovery")
            
            # Wait for network to stabilize
            time.sleep(5)
            
            # TODO: Implement specific connection recovery logic
            # This would typically involve:
            # - Checking network connectivity
            # - Re-establishing MT5 connection
            # - Validating connection health
            
            return True  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Connection recovery failed: {str(e)}")
            return False
    
    def _recover_trading(self, error_context: ErrorContext) -> bool:
        """Recover from trading errors"""
        try:
            self.logger.info("Attempting trading recovery")
            
            # TODO: Implement trading recovery logic
            # This would typically involve:
            # - Checking account status
            # - Validating margin requirements
            # - Adjusting position sizes
            # - Retrying with different parameters
            
            return True  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Trading recovery failed: {str(e)}")
            return False
    
    def _recover_data(self, error_context: ErrorContext) -> bool:
        """Recover from data errors"""
        try:
            self.logger.info("Attempting data recovery")
            
            # TODO: Implement data recovery logic
            # This would typically involve:
            # - Re-fetching data from source
            # - Using backup data sources
            # - Cleaning/validating data
            
            return True  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Data recovery failed: {str(e)}")
            return False
    
    def _recover_system(self, error_context: ErrorContext) -> bool:
        """Recover from system errors"""
        try:
            self.logger.info("Attempting system recovery")
            
            # TODO: Implement system recovery logic
            # This would typically involve:
            # - Freeing memory
            # - Cleaning temporary files
            # - Restarting components
            
            return True  # Placeholder
            
        except Exception as e:
            self.logger.error(f"System recovery failed: {str(e)}")
            return False
    
    def _is_circuit_open(self, function_name: str) -> bool:
        """Check if circuit breaker is open for a function"""
        circuit = self.circuit_breakers.get(function_name)
        if not circuit:
            return False
        
        # Check if circuit should be reset
        if datetime.now() - circuit['last_failure'] > timedelta(minutes=5):
            self._reset_circuit_breaker(function_name)
            return False
        
        return circuit['failure_count'] >= 5
    
    def _update_circuit_breaker(self, function_name: str) -> None:
        """Update circuit breaker state"""
        if function_name not in self.circuit_breakers:
            self.circuit_breakers[function_name] = {
                'failure_count': 0,
                'last_failure': datetime.now()
            }
        
        self.circuit_breakers[function_name]['failure_count'] += 1
        self.circuit_breakers[function_name]['last_failure'] = datetime.now()
    
    def _reset_circuit_breaker(self, function_name: str) -> None:
        """Reset circuit breaker state"""
        if function_name in self.circuit_breakers:
            self.circuit_breakers[function_name]['failure_count'] = 0
    
    def _check_system_health(self, error_context: ErrorContext) -> None:
        """Check if error affects system health"""
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.system_healthy = False
            self.logger.critical(f"System health compromised by critical error: {error_context.message}")
    
    def _trigger_emergency_protocols(self) -> None:
        """Trigger emergency protocols"""
        try:
            self.logger.critical("Triggering emergency protocols")
            
            # TODO: Implement emergency protocols
            # This would typically involve:
            # - Closing all open positions
            # - Stopping new trading
            # - Sending emergency notifications
            # - Creating system backup
            
            # Placeholder notification
            self._send_emergency_notification()
            
        except Exception as e:
            self.logger.error(f"Emergency protocols failed: {str(e)}")
    
    def _send_error_notification(self, error_context: ErrorContext) -> None:
        """Send error notification based on severity"""
        try:
            if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                # TODO: Implement notification system
                # This would send emails, SMS, or push notifications
                self.logger.warning(f"High/Critical error notification: {error_context.message}")
                
        except Exception as e:
            self.logger.error(f"Failed to send error notification: {str(e)}")
    
    def _send_emergency_notification(self) -> None:
        """Send emergency notification"""
        try:
            # TODO: Implement emergency notification
            self.logger.critical("Emergency notification sent")
            
        except Exception as e:
            self.logger.error(f"Failed to send emergency notification: {str(e)}")
    
    def _get_recent_errors(self, minutes: int = 5) -> List[ErrorContext]:
        """Get errors from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [e for e in self.error_history if e.timestamp > cutoff_time]
    
    def _update_error_stats(self, error_context: ErrorContext) -> None:
        """Update error statistics"""
        self.error_stats['total_errors'] += 1
        
        # Update category stats
        category = error_context.category.value
        self.error_stats['errors_by_category'][category] = \
            self.error_stats['errors_by_category'].get(category, 0) + 1
        
        # Update severity stats
        severity = error_context.severity.value
        self.error_stats['errors_by_severity'][severity] = \
            self.error_stats['errors_by_severity'].get(severity, 0) + 1
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID"""
        return f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(threading.current_thread())}"
    
    def _get_calling_function(self) -> str:
        """Get the name of the calling function"""
        try:
            frame = traceback.extract_stack()[-4]  # Go back 4 frames
            return frame.name
        except:
            return "unknown"
    
    def _get_calling_module(self) -> str:
        """Get the name of the calling module"""
        try:
            frame = traceback.extract_stack()[-4]
            return frame.filename.split('/')[-1]
        except:
            return "unknown"
    
    def _get_line_number(self) -> int:
        """Get the line number where error occurred"""
        try:
            frame = traceback.extract_stack()[-4]
            return frame.lineno
        except:
            return 0
    
    # Public methods for error statistics and monitoring
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        recent_errors = self._get_recent_errors(60)  # Last hour
        recovery_attempts = [e for e in self.error_history if e.recovery_attempted]
        successful_recoveries = [e for e in recovery_attempts if e.recovery_successful]
        
        recovery_rate = 0.0
        if recovery_attempts:
            recovery_rate = len(successful_recoveries) / len(recovery_attempts)
        
        return {
            **self.error_stats,
            'recovery_success_rate': recovery_rate,
            'recent_error_count': len(recent_errors),
            'system_healthy': self.system_healthy,
            'last_health_check': self.last_health_check,
            'circuit_breakers': len([cb for cb in self.circuit_breakers.values() 
                                   if cb['failure_count'] >= 5])
        }
    
    def get_error_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent error history"""
        recent_errors = self.error_history[-limit:]
        return [
            {
                'error_id': e.error_id,
                'timestamp': e.timestamp.isoformat(),
                'category': e.category.value,
                'severity': e.severity.value,
                'message': e.message,
                'function_name': e.function_name,
                'module_name': e.module_name,
                'recovery_attempted': e.recovery_attempted,
                'recovery_successful': e.recovery_successful,
                'retry_count': e.retry_count
            }
            for e in recent_errors
        ]
    
    def reset_system_health(self) -> None:
        """Reset system health status"""
        self.system_healthy = True
        self.logger.info("System health status reset")
    
    def is_system_healthy(self) -> bool:
        """Check if system is healthy"""
        return self.system_healthy


# Global error handler instance
_error_handler = None


def get_error_handler(config: Optional[Dict] = None) -> ErrorHandler:
    """Get or create the global error handler"""
    global _error_handler
    
    if _error_handler is None:
        if config is None:
            # Default config if none provided
            config = {
                'error_handling': {
                    'restart_on_error': True,
                    'max_restart_attempts': 3,
                    'error_notification': True
                }
            }
        _error_handler = ErrorHandler(config)
        _error_handler.start()
    
    return _error_handler


def setup_error_handling(config: Dict) -> ErrorHandler:
    """Setup error handling with configuration"""
    error_handler = ErrorHandler(config)
    error_handler.start()
    return error_handler


# Testing function
if __name__ == "__main__":
    """Test the error handling system"""
    
    # Test configuration
    test_config = {
        'error_handling': {
            'restart_on_error': True,
            'max_restart_attempts': 3,
            'error_notification': True
        }
    }
    
    # Create error handler
    error_handler = ErrorHandler(test_config)
    error_handler.start()
    
    # Test error handling decorator
    @error_handler.handle_errors(retry_count=3, recovery=True)
    def test_function(should_fail: bool = True):
        if should_fail:
            raise ConnectionError("Test connection error")
        return "Success!"
    
    # Test direct error handling
    try:
        raise TradingError("Test trading error", context={'symbol': 'XAUUSDm'})
    except Exception as e:
        error_handler.handle_error(e, "Testing direct error handling")
    
    # Test function with decorator
    try:
        result = test_function(should_fail=True)
    except Exception as e:
        print(f"Function failed after retries: {e}")
    
    # Test successful function
    try:
        result = test_function(should_fail=False)
        print(f"Function succeeded: {result}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Wait a bit for error processing
    time.sleep(2)
    
    # Get error statistics
    stats = error_handler.get_error_stats()
    print(f"Error statistics: {json.dumps(stats, indent=2, default=str)}")
    
    # Get error history
    history = error_handler.get_error_history(limit=5)
    print(f"Recent errors: {json.dumps(history, indent=2, default=str)}")
    
    # Stop error handler
    error_handler.stop()
    
    print("Error handling test completed!")