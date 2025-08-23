"""
Data Validator Module - Comprehensive OHLCV Data Validation
=========================================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-08-21

This module provides comprehensive validation for trading data to ensure:
- Time series integrity (no gaps, proper sequencing)
- OHLCV consistency (Low <= Open, High, Close <= High)
- Price validation (no negative values, extreme outliers)
- Data quality checks for reliable trading decisions

Classes:
    DataValidator: Main validation class with comprehensive checks
    ValidationResult: Result container for validation outcomes
    
Functions:
    validate_ohlcv_data: Quick validation function for OHLCV DataFrames
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

# Try to use LoggerManager, fallback to standard logging
try:
    from src.utils.logger import get_logger_manager
    logger_manager = get_logger_manager()
    logger = logger_manager.get_logger('data_validator')
except ImportError:
    # Fallback to standard logging if LoggerManager not available
    logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for data validation results"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Union[int, float]]
    cleaned_data: Optional[pd.DataFrame] = None

class DataValidator:
    """
    Comprehensive data validator for OHLCV trading data
    
    Validates:
    - Time series gaps and consistency
    - OHLCV price relationships
    - Outlier detection and removal
    - Data integrity checks
    """
    
    def __init__(self, 
                 max_gap_multiplier: float = 3.0,
                 outlier_threshold: float = 5.0,
                 min_price: float = 0.001,
                 max_price_change: float = 0.20):
        """
        Initialize validator with configurable thresholds
        
        Args:
            max_gap_multiplier: Maximum allowed gap as multiple of timeframe
            outlier_threshold: Standard deviations for outlier detection
            min_price: Minimum valid price value
            max_price_change: Maximum allowed price change between bars (20%)
        """
        self.max_gap_multiplier = max_gap_multiplier
        self.outlier_threshold = outlier_threshold
        self.min_price = min_price
        self.max_price_change = max_price_change
    
    def validate_ohlcv_data(self, 
                           df: pd.DataFrame, 
                           symbol: str, 
                           timeframe: str,
                           auto_clean: bool = False) -> ValidationResult:
        """
        Comprehensive validation of OHLCV data
        
        Args:
            df: DataFrame with OHLCV data (time index required)
            symbol: Trading symbol for context
            timeframe: Timeframe (M1, M5, H1, etc.)
            auto_clean: Whether to automatically clean/fix issues
            
        Returns:
            ValidationResult with validation outcome and optional cleaned data
        """
        errors = []
        warnings = []
        stats = {}
        cleaned_df = df.copy() if auto_clean else None
        
        try:
            # Basic structure validation
            structure_valid, structure_errors = self._validate_structure(df)
            errors.extend(structure_errors)
            
            if not structure_valid:
                return ValidationResult(False, errors, warnings, stats)
            
            # Time series validation
            time_valid, time_errors, time_warnings, time_stats = self._validate_time_series(
                df, timeframe
            )
            errors.extend(time_errors)
            warnings.extend(time_warnings)
            stats.update(time_stats)
            
            # OHLCV consistency validation
            ohlcv_valid, ohlcv_errors, ohlcv_warnings = self._validate_ohlcv_consistency(df)
            errors.extend(ohlcv_errors)
            warnings.extend(ohlcv_warnings)
            
            # Price validation
            price_valid, price_errors, price_warnings = self._validate_prices(df)
            errors.extend(price_errors)
            warnings.extend(price_warnings)
            
            # Outlier detection
            outlier_warnings, outlier_stats = self._detect_outliers(df)
            warnings.extend(outlier_warnings)
            stats.update(outlier_stats)
            
            # Auto-cleaning if requested
            if auto_clean and (errors or warnings):
                cleaned_df = self._clean_data(df, errors, warnings)
                stats['rows_cleaned'] = len(df) - len(cleaned_df) if cleaned_df is not None else 0
            
            # Overall validation result
            is_valid = len(errors) == 0
            stats['total_rows'] = len(df)
            stats['error_count'] = len(errors)
            stats['warning_count'] = len(warnings)
            
            if is_valid:
                logger.info(f"✅ Data validation passed for {symbol} {timeframe}: {len(df)} rows")
            else:
                logger.warning(f"❌ Data validation failed for {symbol} {timeframe}: {len(errors)} errors")
            
            return ValidationResult(is_valid, errors, warnings, stats, cleaned_df)
            
        except Exception as e:
            logger.error(f"Validation error for {symbol}: {str(e)}")
            return ValidationResult(False, [f"Validation exception: {str(e)}"], [], {})
    
    def _validate_structure(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate basic DataFrame structure"""
        errors = []
        
        if df is None or df.empty:
            errors.append("DataFrame is empty or None")
            return False, errors
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check index type
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Index must be DatetimeIndex")
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            errors.append("Duplicate timestamps found")
        
        # Check data types
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column {col} must be numeric")
        
        return len(errors) == 0, errors
    
    def _validate_time_series(self, df: pd.DataFrame, timeframe: str) -> Tuple[bool, List[str], List[str], Dict]:
        """Validate time series consistency"""
        errors = []
        warnings = []
        stats = {}
        
        if len(df) < 2:
            return True, errors, warnings, stats
        
        # Get expected time delta
        timeframe_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080
        }
        
        expected_delta_minutes = timeframe_minutes.get(timeframe, 60)
        expected_delta = pd.Timedelta(minutes=expected_delta_minutes)
        
        # Check time ordering
        if not df.index.is_monotonic_increasing:
            errors.append("Time series is not properly ordered")
        
        # Analyze gaps
        time_diffs = df.index.to_series().diff().dropna()
        gaps = time_diffs[time_diffs > expected_delta * self.max_gap_multiplier]
        
        stats['expected_delta_minutes'] = expected_delta_minutes
        stats['total_gaps'] = len(gaps)
        stats['max_gap_minutes'] = int(time_diffs.max().total_seconds() / 60) if len(time_diffs) > 0 else 0
        
        if len(gaps) > 0:
            total_missing_bars = sum((gap / expected_delta - 1) for gap in gaps)
            stats['estimated_missing_bars'] = int(total_missing_bars)
            
            if total_missing_bars > len(df) * 0.2:  # More than 20% missing (increased threshold)
                errors.append(f"Significant data gaps detected: ~{int(total_missing_bars)} missing bars")
            elif total_missing_bars > len(df) * 0.05:  # Only warn if > 5% missing
                warnings.append(f"Minor data gaps detected: ~{int(total_missing_bars)} missing bars")
        
        return len(errors) == 0, errors, warnings, stats
    
    def _validate_ohlcv_consistency(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """Validate OHLCV price relationships"""
        errors = []
        warnings = []
        
        # Check OHLC relationships: Low <= Open, Close <= High and High >= Open, Close
        invalid_low = (df['Low'] > df['Open']) | (df['Low'] > df['Close']) | (df['Low'] > df['High'])
        invalid_high = (df['High'] < df['Open']) | (df['High'] < df['Close']) | (df['High'] < df['Low'])
        
        if invalid_low.any():
            count = invalid_low.sum()
            errors.append(f"Invalid Low prices in {count} bars (Low > Open/Close/High)")
        
        if invalid_high.any():
            count = invalid_high.sum()
            errors.append(f"Invalid High prices in {count} bars (High < Open/Close/Low)")
        
        # Check for zero or negative prices
        zero_prices = (df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
        if zero_prices.any():
            count = zero_prices.sum()
            errors.append(f"Zero or negative prices in {count} bars")
        
        # Check for unrealistic spreads (High-Low > 50% of Close)
        spread_ratio = (df['High'] - df['Low']) / df['Close']
        extreme_spreads = spread_ratio > 0.5
        if extreme_spreads.any():
            count = extreme_spreads.sum()
            warnings.append(f"Extremely wide spreads in {count} bars (>50% of close price)")
        
        return len(errors) == 0, errors, warnings
    
    def _validate_prices(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """Validate price values and changes"""
        errors = []
        warnings = []
        
        # Check minimum price threshold
        too_low = (df[['Open', 'High', 'Low', 'Close']] < self.min_price).any(axis=1)
        if too_low.any():
            count = too_low.sum()
            errors.append(f"Prices below minimum threshold ({self.min_price}) in {count} bars")
        
        # Check for extreme price changes between consecutive bars
        if len(df) > 1:
            price_changes = df['Close'].pct_change().abs()
            extreme_changes = price_changes > self.max_price_change
            
            if extreme_changes.any():
                count = extreme_changes.sum()
                max_change = price_changes.max()
                warnings.append(f"Extreme price changes in {count} bars (max: {max_change:.2%})")
        
        return len(errors) == 0, errors, warnings
    
    def _detect_outliers(self, df: pd.DataFrame) -> Tuple[List[str], Dict]:
        """Detect statistical outliers in price data"""
        warnings = []
        stats = {}
        
        try:
            # Calculate z-scores for close prices
            close_mean = df['Close'].mean()
            close_std = df['Close'].std()
            
            if close_std > 0:
                z_scores = np.abs((df['Close'] - close_mean) / close_std)
                outliers = z_scores > self.outlier_threshold
                
                outlier_count = outliers.sum()
                stats['outlier_count'] = outlier_count
                stats['outlier_threshold'] = self.outlier_threshold
                
                if outlier_count > 0:
                    max_z_score = z_scores.max()
                    warnings.append(f"Statistical outliers detected: {outlier_count} bars "
                                  f"(max z-score: {max_z_score:.2f})")
        
        except Exception as e:
            warnings.append(f"Outlier detection failed: {str(e)}")
        
        return warnings, stats
    
    def _clean_data(self, df: pd.DataFrame, errors: List[str], warnings: List[str]) -> pd.DataFrame:
        """Attempt to clean/fix data issues"""
        cleaned_df = df.copy()
        
        try:
            # Remove rows with invalid OHLC relationships
            valid_ohlc = (
                (cleaned_df['Low'] <= cleaned_df['Open']) &
                (cleaned_df['Low'] <= cleaned_df['Close']) &
                (cleaned_df['Low'] <= cleaned_df['High']) &
                (cleaned_df['High'] >= cleaned_df['Open']) &
                (cleaned_df['High'] >= cleaned_df['Close']) &
                (cleaned_df['High'] >= cleaned_df['Low'])
            )
            
            # Remove rows with zero/negative prices
            positive_prices = (cleaned_df[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)
            
            # Combine filters
            valid_rows = valid_ohlc & positive_prices
            cleaned_df = cleaned_df[valid_rows]
            
            # Remove statistical outliers
            if len(cleaned_df) > 10:  # Only if enough data remains
                close_mean = cleaned_df['Close'].mean()
                close_std = cleaned_df['Close'].std()
                
                if close_std > 0:
                    z_scores = np.abs((cleaned_df['Close'] - close_mean) / close_std)
                    non_outliers = z_scores <= self.outlier_threshold
                    cleaned_df = cleaned_df[non_outliers]
            
            logger.info(f"Data cleaning: {len(df)} -> {len(cleaned_df)} rows")
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            return df
        
        return cleaned_df

# Convenience function for quick validation
def validate_ohlcv_data(df: pd.DataFrame, 
                       symbol: str = "Unknown", 
                       timeframe: str = "Unknown",
                       auto_clean: bool = False) -> ValidationResult:
    """
    Quick validation function for OHLCV data
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading symbol (for logging)
        timeframe: Timeframe (for gap analysis)
        auto_clean: Whether to attempt automatic cleaning
        
    Returns:
        ValidationResult with validation outcome
    """
    validator = DataValidator()
    return validator.validate_ohlcv_data(df, symbol, timeframe, auto_clean)

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2025-01-01', periods=100, freq='5T')
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    base_price = 1950.0
    price_changes = np.random.normal(0, 0.001, 100).cumsum()
    closes = base_price + price_changes * base_price
    
    # Create OHLCV with proper relationships
    opens = np.roll(closes, 1)
    opens[0] = base_price
    
    highs = np.maximum(opens, closes) + np.random.uniform(0, 0.002, 100) * closes
    lows = np.minimum(opens, closes) - np.random.uniform(0, 0.002, 100) * closes
    volumes = np.random.randint(1000, 5000, 100)
    
    # Create DataFrame
    test_df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=dates)
    
    # Add some intentional issues for testing
    test_df.loc[test_df.index[10], 'Low'] = test_df.loc[test_df.index[10], 'High'] + 1  # Invalid low
    test_df.loc[test_df.index[20], 'Close'] = -100  # Negative price
    test_df.loc[test_df.index[30], 'High'] = test_df.loc[test_df.index[30], 'Close'] * 2  # Extreme spread
    
    # Test validation
    print("Testing Data Validator...")
    print("=" * 50)
    
    result = validate_ohlcv_data(test_df, "XAUUSD", "M5", auto_clean=True)
    
    print(f"Validation Result: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")
    print(f"Errors: {len(result.errors)}")
    for error in result.errors:
        print(f"  - {error}")
    
    print(f"Warnings: {len(result.warnings)}")
    for warning in result.warnings:
        print(f"  - {warning}")
    
    print(f"Statistics: {result.stats}")
    
    if result.cleaned_data is not None:
        print(f"Original rows: {len(test_df)}, Cleaned rows: {len(result.cleaned_data)}")
