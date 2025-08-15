"""
Risk Manager - Advanced Risk Management System
=============================================
Author: XAUUSD Trading System
Version: 2.0.0
Date: 2025-08-08

Comprehensive risk management for aggressive 10x returns target:
- Kelly Criterion position sizing with safety factors
- Dynamic drawdown protection
- Correlation-based risk limits
- Portfolio heat monitoring
- Emergency stop mechanisms
- Recovery mode activation

Features:
- Multi-level risk controls
- Adaptive position sizing
- Real-time risk monitoring
- Circuit breaker mechanisms
- Performance-based adjustments
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json

# Import base classes
try:
    from ..signal_engine import Signal, SignalType, SignalGrade
except ImportError:
    # Fallback for testing
    class SignalType(Enum):
        BUY = "BUY"
        SELL = "SELL"
        HOLD = "HOLD"
    
    class SignalGrade(Enum):
        A = "A"
        B = "B" 
        C = "C"
        D = "D"
    
    @dataclass
    class Signal:
        timestamp: datetime
        symbol: str
        strategy_name: str
        signal_type: SignalType
        confidence: float
        price: float
        timeframe: str
        strength: float = 0.0
        grade: Optional[SignalGrade] = None
        stop_loss: Optional[float] = None
        take_profit: Optional[float] = None
        metadata: Dict[str, Any] = None


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class PositionSizingMethod(Enum):
    """Position sizing methods"""
    FIXED = "FIXED"
    KELLY = "KELLY"
    KELLY_MODIFIED = "kelly_modified"
    VOLATILITY_BASED = "VOLATILITY_BASED"
    CONFIDENCE_BASED = "CONFIDENCE_BASED"

@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    timestamp: datetime
    account_balance: float
    equity: float
    unrealized_pnl: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    
    # Drawdown metrics
    current_drawdown: float
    max_drawdown: float
    drawdown_duration: int  # days
    
    # Position metrics
    total_exposure: float
    position_count: int
    largest_position: float
    
    # Risk ratios
    risk_per_trade: float
    portfolio_risk: float
    var_95: float  # Value at Risk 95%
    
    # Performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float


@dataclass
class PositionRisk:
    """Individual position risk assessment"""
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: Optional[float]
    
    # Risk metrics
    risk_amount: float
    risk_percentage: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    # Position characteristics
    time_in_position: timedelta
    correlation_risk: float
    liquidity_risk: float


class RiskManager:
    """
    Advanced Risk Management System for 10x Trading Target
    
    This system manages all aspects of trading risk:
    - Position sizing using Kelly Criterion with safety factors
    - Portfolio-level risk monitoring
    - Dynamic drawdown protection
    - Correlation analysis between positions
    - Emergency stop mechanisms
    - Recovery mode for losing streaks
    
    Designed for aggressive 10x returns while protecting capital.
    """
    
    def __init__(self, config: Dict[str, Any], mt5_manager, database_manager):
        """Initialize Risk Manager"""
        self.config = config
        self.mt5_manager = mt5_manager
        self.database_manager = database_manager
        
        # Risk configuration
        self.risk_config = config.get('risk_management', {})
        self.capital_config = config.get('capital', {})
        
        # Risk parameters
        self.risk_per_trade = self.risk_config.get('risk_per_trade', 0.03)
        self.max_risk_per_trade = self.risk_config.get('max_risk_per_trade', 0.05)
        self.max_portfolio_risk = self.risk_config.get('max_portfolio_risk', 0.15)
        self.max_drawdown = self.risk_config.get('max_drawdown', 0.25)
        self.max_daily_loss = self.risk_config.get('max_daily_loss', 0.10)
        self.max_weekly_loss = self.risk_config.get('max_weekly_loss', 0.20)
        self.max_consecutive_losses = self.risk_config.get('max_consecutive_losses', 4)
        
        # Position sizing
        self.sizing_config = config.get('position_sizing', {})
        self.sizing_method = PositionSizingMethod(self.sizing_config.get('method', 'kelly_modified'))
        self.kelly_safety_factor = self.sizing_config.get('kelly_safety_factor', 0.30)
        self.min_position_size = self.sizing_config.get('min_position_size', 0.01)
        self.max_position_size = self.sizing_config.get('max_position_size', 0.10)
        self.max_positions = self.sizing_config.get('max_positions', 3)
        
        # Capital management
        self.initial_capital = self.capital_config.get('initial_capital', 100.0)
        self.target_capital = self.capital_config.get('target_capital', 1000.0)
        self.minimum_capital = self.capital_config.get('minimum_capital', 50.0)
        self.reserve_cash = self.capital_config.get('reserve_cash', 0.10)
        
        # Risk tracking
        self.equity_peak = self.initial_capital
        self.consecutive_losses = 0
        self.recovery_mode = False
        self.emergency_stop = False
        self.last_trade_time = None
        
        # Performance tracking
        self.performance_history = []
        self.trade_history = []
        self.daily_returns = []
        
        # Risk state
        self.current_risk_level = RiskLevel.LOW
        self.risk_metrics_history = []
        
        # Logger
        self.logger = logging.getLogger('risk_manager')
        
        # Initialize risk monitoring
        self._initialize_risk_monitoring()
    
    def _initialize_risk_monitoring(self) -> None:
        """Initialize risk monitoring systems"""
        try:
            # Load historical performance if available
            self._load_performance_history()
            
            # Calculate initial risk metrics
            self._update_risk_metrics()
            
            self.logger.info("Risk management system initialized")
            self.logger.info(f"Target: ${self.initial_capital} → ${self.target_capital} (10x)")
            self.logger.info(f"Max risk per trade: {self.max_risk_per_trade:.1%}")
            self.logger.info(f"Max portfolio risk: {self.max_portfolio_risk:.1%}")
            self.logger.info(f"Max drawdown: {self.max_drawdown:.1%}")
            
        except Exception as e:
            self.logger.error(f"Risk monitoring initialization failed: {str(e)}")
    
    def calculate_position_size(self, signal: Signal, account_balance: float, 
                              open_positions: List[Dict]) -> Dict[str, Any]:
        """
        Calculate optimal position size based on risk management rules
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            open_positions: List of current open positions
            
        Returns:
            Dict with position size and risk assessment
        """
        try:
            # Check if trading is allowed
            if not self._is_trading_allowed(account_balance, open_positions):
                return {
                    'position_size': 0.0,
                    'allowed': False,
                    'reason': 'Trading not allowed due to risk limits',
                    'risk_assessment': self._get_current_risk_level()
                }
            
            # Calculate base position size using selected method
            base_size = self._calculate_base_position_size(signal, account_balance)
            
            # Apply risk adjustments
            adjusted_size = self._apply_risk_adjustments(
                base_size, signal, account_balance, open_positions
            )
            
            # Apply position limits
            final_size = self._apply_position_limits(adjusted_size, account_balance)
            
            # Calculate risk metrics for this position
            risk_assessment = self._assess_position_risk(
                signal, final_size, account_balance, open_positions
            )
            
            return {
                'position_size': final_size,
                'allowed': final_size > 0,
                'base_size': base_size,
                'adjusted_size': adjusted_size,
                'risk_assessment': risk_assessment,
                'sizing_method': self.sizing_method.value,
                'risk_percentage': risk_assessment.get('risk_percentage', 0),
                'correlation_impact': risk_assessment.get('correlation_impact', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Position size calculation failed: {str(e)}")
            return {
                'position_size': 0.0,
                'allowed': False,
                'reason': f'Calculation error: {str(e)}',
                'risk_assessment': {}
            }
    
    def _is_trading_allowed(self, account_balance: float, open_positions: List[Dict]) -> bool:
        """Check if trading is currently allowed"""
        try:
            # Emergency stop check
            if self.emergency_stop:
                return False
            
            # Capital preservation check
            if account_balance <= self.minimum_capital:
                self.logger.warning(f"Account balance ${account_balance} below minimum ${self.minimum_capital}")
                return False
            
            # Daily loss limit check
            daily_pnl = self._calculate_daily_pnl()
            if daily_pnl <= -self.max_daily_loss * self.equity_peak:
                self.logger.warning(f"Daily loss limit reached: {daily_pnl:.2f}")
                return False
            
            # Weekly loss limit check
            weekly_pnl = self._calculate_weekly_pnl()
            if weekly_pnl <= -self.max_weekly_loss * self.equity_peak:
                self.logger.warning(f"Weekly loss limit reached: {weekly_pnl:.2f}")
                return False
            
            # Maximum positions check
            if len(open_positions) >= self.max_positions:
                return False
            
            # Consecutive losses check
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.logger.warning(f"Max consecutive losses reached: {self.consecutive_losses}")
                return False
            
            # Drawdown check
            current_drawdown = self._calculate_current_drawdown(account_balance)
            if current_drawdown >= self.max_drawdown:
                self.logger.warning(f"Max drawdown reached: {current_drawdown:.2%}")
                self._activate_emergency_stop()
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Trading allowance check failed: {str(e)}")
            return False
    
    def _calculate_base_position_size(self, signal: Signal, account_balance: float) -> float:
        """Calculate base position size using selected method"""
        try:
            if self.sizing_method == PositionSizingMethod.FIXED:
                return self._calculate_fixed_size(account_balance)
            
            elif self.sizing_method == PositionSizingMethod.KELLY:
                return self._calculate_kelly_size(signal, account_balance)
            
            elif self.sizing_method == PositionSizingMethod.KELLY_MODIFIED:
                return self._calculate_kelly_modified_size(signal, account_balance)
            
            elif self.sizing_method == PositionSizingMethod.VOLATILITY_BASED:
                return self._calculate_volatility_based_size(signal, account_balance)
            
            elif self.sizing_method == PositionSizingMethod.CONFIDENCE_BASED:
                return self._calculate_confidence_based_size(signal, account_balance)
            
            else:
                # Default to Kelly Modified
                return self._calculate_kelly_modified_size(signal, account_balance)
                
        except Exception as e:
            self.logger.error(f"Base position size calculation failed: {str(e)}")
            return self.min_position_size
    
    def _calculate_fixed_size(self, account_balance: float) -> float:
        """Calculate fixed position size"""
        # Fixed percentage of account balance
        fixed_percentage = 0.02  # 2% of account
        position_value = account_balance * fixed_percentage
        
        # Convert to lot size (assuming $100 per lot for gold)
        lot_size = position_value / 10000  # Approximate lot value
        
        return max(self.min_position_size, min(lot_size, self.max_position_size))
    
    def _calculate_kelly_size(self, signal: Signal, account_balance: float) -> float:
        """Calculate Kelly Criterion position size"""
        try:
            # Get historical win rate and average win/loss
            win_rate, avg_win, avg_loss = self._get_strategy_performance(signal.strategy_name)
            
            if win_rate == 0 or avg_loss == 0:
                # No historical data, use conservative size
                return self.min_position_size
            
            # Kelly formula: f = (bp - q) / b
            # where: b = odds received (avg_win/avg_loss), p = win_rate, q = 1-p
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Apply to account balance
            if kelly_fraction > 0:
                position_value = account_balance * kelly_fraction
                lot_size = position_value / 10000
                return max(self.min_position_size, min(lot_size, self.max_position_size))
            else:
                return self.min_position_size
                
        except Exception as e:
            self.logger.error(f"Kelly calculation failed: {str(e)}")
            return self.min_position_size
    
    def _calculate_kelly_modified_size(self, signal: Signal, account_balance: float) -> float:
        """Calculate Kelly Criterion with safety factor"""
        try:
            # Get Kelly size
            kelly_size = self._calculate_kelly_size(signal, account_balance)
            
            # Apply safety factor
            modified_size = kelly_size * self.kelly_safety_factor
            
            # Adjust based on signal confidence
            confidence_multiplier = signal.confidence  # 0.6-1.0 range
            final_size = modified_size * confidence_multiplier
            
            # Adjust based on signal grade
            grade_multipliers = {
                SignalGrade.A: 1.5,
                SignalGrade.B: 1.0,
                SignalGrade.C: 0.7,
                SignalGrade.D: 0.4
            }
            
            if signal.grade in grade_multipliers:
                final_size *= grade_multipliers[signal.grade]
            
            return max(self.min_position_size, min(final_size, self.max_position_size))
            
        except Exception as e:
            self.logger.error(f"Modified Kelly calculation failed: {str(e)}")
            return self.min_position_size
    
    def _calculate_volatility_based_size(self, signal: Signal, account_balance: float) -> float:
        """Calculate position size based on market volatility"""
        try:
            # Get recent volatility
            symbol_data = self.mt5_manager.get_historical_data(signal.symbol, signal.timeframe, 100)
            if symbol_data is None or len(symbol_data) < 20:
                return self.min_position_size
            
            # Calculate ATR-based volatility
            atr = self._calculate_atr(symbol_data, 14)
            current_price = symbol_data['Close'].iloc[-1]
            
            # Volatility as percentage
            volatility_pct = atr / current_price
            
            # Inverse relationship: higher volatility = smaller position
            base_risk = self.risk_per_trade
            volatility_adjusted_risk = base_risk / (1 + volatility_pct * 10)
            
            # Calculate position size
            risk_amount = account_balance * volatility_adjusted_risk
            
            if signal.stop_loss:
                stop_distance = abs(signal.price - signal.stop_loss)
                if stop_distance > 0:
                    # Position size = Risk Amount / Stop Loss Distance
                    # Converted to lots
                    lot_size = risk_amount / (stop_distance * 100)  # Approximate conversion
                    return max(self.min_position_size, min(lot_size, self.max_position_size))
            
            # Fallback calculation
            position_value = risk_amount * 10  # 10x leverage assumption
            lot_size = position_value / 10000
            
            return max(self.min_position_size, min(lot_size, self.max_position_size))
            
        except Exception as e:
            self.logger.error(f"Volatility-based calculation failed: {str(e)}")
            return self.min_position_size
    
    def _calculate_confidence_based_size(self, signal: Signal, account_balance: float) -> float:
        """Calculate position size based on signal confidence"""
        try:
            # Base risk scaled by confidence
            confidence_risk = self.risk_per_trade * signal.confidence
            
            # Additional scaling based on signal strength
            strength_multiplier = min(signal.strength * 2, 2.0) if signal.strength else 1.0
            adjusted_risk = confidence_risk * strength_multiplier
            
            # Ensure within limits
            final_risk = max(0.005, min(adjusted_risk, self.max_risk_per_trade))
            
            # Calculate position size
            risk_amount = account_balance * final_risk
            
            if signal.stop_loss:
                stop_distance = abs(signal.price - signal.stop_loss)
                if stop_distance > 0:
                    lot_size = risk_amount / (stop_distance * 100)
                    return max(self.min_position_size, min(lot_size, self.max_position_size))
            
            # Fallback
            position_value = risk_amount * 15  # Higher leverage for high confidence
            lot_size = position_value / 10000
            
            return max(self.min_position_size, min(lot_size, self.max_position_size))
            
        except Exception as e:
            self.logger.error(f"Confidence-based calculation failed: {str(e)}")
            return self.min_position_size
    
    def _apply_risk_adjustments(self, base_size: float, signal: Signal, 
                              account_balance: float, open_positions: List[Dict]) -> float:
        """Apply various risk adjustments to base position size"""
        try:
            adjusted_size = base_size
            
            # Recovery mode adjustment
            if self.recovery_mode:
                adjusted_size *= 0.5  # Reduce size in recovery mode
                self.logger.info("Recovery mode: reducing position size by 50%")
            
            # Correlation adjustment
            correlation_factor = self._calculate_correlation_factor(signal, open_positions)
            adjusted_size *= correlation_factor
            
            # Time-based adjustment (avoid overtrading)
            time_factor = self._calculate_time_factor()
            adjusted_size *= time_factor
            
            # Market session adjustment
            session_factor = self._calculate_session_factor(signal)
            adjusted_size *= session_factor
            
            # Portfolio heat adjustment
            heat_factor = self._calculate_portfolio_heat_factor(open_positions, account_balance)
            adjusted_size *= heat_factor
            
            return max(0, adjusted_size)
            
        except Exception as e:
            self.logger.error(f"Risk adjustment failed: {str(e)}")
            return base_size
    
    def _calculate_correlation_factor(self, signal: Signal, open_positions: List[Dict]) -> float:
        """Calculate correlation factor to avoid overexposure"""
        try:
            if not open_positions:
                return 1.0
            
            # Check for same symbol exposure
            same_symbol_exposure = 0
            same_direction_exposure = 0
            
            for position in open_positions:
                if position.get('symbol') == signal.symbol:
                    same_symbol_exposure += abs(position.get('volume', 0))
                
                # Check direction correlation (simplified)
                if position.get('type') == signal.signal_type.value:
                    same_direction_exposure += abs(position.get('volume', 0))
            
            # Reduce size if high correlation
            if same_symbol_exposure > 0.05:  # More than 0.05 lots in same symbol
                return 0.5
            elif same_direction_exposure > 0.1:  # More than 0.1 lots in same direction
                return 0.7
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Correlation factor calculation failed: {str(e)}")
            return 1.0
    
    def _calculate_time_factor(self) -> float:
        """Calculate time-based factor to prevent overtrading"""
        try:
            if self.last_trade_time is None:
                return 1.0
            
            time_since_last = datetime.now() - self.last_trade_time
            minutes_since = time_since_last.total_seconds() / 60
            
            # Reduce size if trading too frequently
            if minutes_since < 15:  # Less than 15 minutes
                return 0.3
            elif minutes_since < 30:  # Less than 30 minutes
                return 0.6
            elif minutes_since < 60:  # Less than 1 hour
                return 0.8
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Time factor calculation failed: {str(e)}")
            return 1.0
    
    def _calculate_session_factor(self, signal: Signal) -> float:
        """Calculate session-based risk factor"""
        try:
            current_hour = datetime.now().hour
            
            # Define session multipliers (based on volatility)
            session_multipliers = {
                'asian': 0.8,      # 00:00-09:00 GMT (lower volatility)
                'london': 1.2,     # 09:00-17:00 GMT (higher volatility)
                'newyork': 1.0,    # 14:00-23:00 GMT (standard)
                'overlap': 1.5     # 14:00-17:00 GMT (highest volatility)
            }
            
            # Determine current session
            if 0 <= current_hour < 9:
                return session_multipliers['asian']
            elif 14 <= current_hour < 17:
                return session_multipliers['overlap']
            elif 9 <= current_hour < 17:
                return session_multipliers['london']
            elif 14 <= current_hour < 23:
                return session_multipliers['newyork']
            else:
                return 0.6  # Off-hours trading (reduced size)
                
        except Exception as e:
            self.logger.error(f"Session factor calculation failed: {str(e)}")
            return 1.0
    
    def _calculate_portfolio_heat_factor(self, open_positions: List[Dict], account_balance: float) -> float:
        """Calculate portfolio heat factor"""
        try:
            if not open_positions:
                return 1.0
            
            # Calculate total portfolio risk
            total_risk = 0
            for position in open_positions:
                # Estimate risk based on position size and current P&L
                position_risk = abs(position.get('profit', 0))
                total_risk += position_risk
            
            portfolio_risk_pct = total_risk / account_balance
            
            # Reduce size if portfolio risk is high
            if portfolio_risk_pct > self.max_portfolio_risk:
                return 0.3  # Severely reduce
            elif portfolio_risk_pct > self.max_portfolio_risk * 0.7:
                return 0.6
            elif portfolio_risk_pct > self.max_portfolio_risk * 0.5:
                return 0.8
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Portfolio heat calculation failed: {str(e)}")
            return 1.0
    
    def _apply_position_limits(self, size: float, account_balance: float) -> float:
        """Apply final position size limits"""
        try:
            # Ensure within min/max limits
            limited_size = max(self.min_position_size, min(size, self.max_position_size))
            
            # Ensure position doesn't exceed account capacity
            max_affordable = account_balance * 0.5 / 10000  # Conservative limit
            limited_size = min(limited_size, max_affordable)
            
            # Round to appropriate precision
            return round(limited_size, 2)
            
        except Exception as e:
            self.logger.error(f"Position limit application failed: {str(e)}")
            return self.min_position_size
    
    def _assess_position_risk(self, signal: Signal, position_size: float, 
                            account_balance: float, open_positions: List[Dict]) -> Dict[str, Any]:
        """Assess risk for the proposed position"""
        try:
            risk_assessment = {}
            
            # Calculate monetary risk
            if signal.stop_loss:
                stop_distance = abs(signal.price - signal.stop_loss)
                monetary_risk = stop_distance * position_size * 100  # Approximate
                risk_percentage = monetary_risk / account_balance
            else:
                # Estimate risk as 2% of position value
                position_value = position_size * signal.price * 100
                monetary_risk = position_value * 0.02
                risk_percentage = monetary_risk / account_balance
            
            # Calculate reward if take profit is set
            reward_risk_ratio = 0
            if signal.take_profit and signal.stop_loss:
                if signal.signal_type == SignalType.BUY:
                    reward = (signal.take_profit - signal.price) * position_size * 100
                else:
                    reward = (signal.price - signal.take_profit) * position_size * 100
                
                if monetary_risk > 0:
                    reward_risk_ratio = reward / monetary_risk
            
            # Portfolio impact
            total_portfolio_risk = monetary_risk
            for position in open_positions:
                total_portfolio_risk += abs(position.get('profit', 0))
            
            portfolio_risk_pct = total_portfolio_risk / account_balance
            
            risk_assessment = {
                'monetary_risk': monetary_risk,
                'risk_percentage': risk_percentage,
                'reward_risk_ratio': reward_risk_ratio,
                'portfolio_risk_pct': portfolio_risk_pct,
                'position_value': position_size * signal.price * 100,
                'correlation_impact': len([p for p in open_positions if p.get('symbol') == signal.symbol]),
                'risk_level': self._determine_risk_level(risk_percentage, portfolio_risk_pct)
            }
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {str(e)}")
            return {'risk_level': 'HIGH'}
    
    def _determine_risk_level(self, risk_percentage: float, portfolio_risk_pct: float) -> str:
        """Determine overall risk level"""
        max_risk = max(risk_percentage, portfolio_risk_pct)
        
        if max_risk <= 0.01:
            return 'LOW'
        elif max_risk <= 0.03:
            return 'MODERATE'
        elif max_risk <= 0.05:
            return 'HIGH'
        else:
            return 'EXTREME'
    
    def update_position_closed(self, trade_result: Dict[str, Any]) -> None:
        """Update risk metrics when a position is closed"""
        try:
            profit = trade_result.get('profit', 0)
            
            # Update consecutive losses counter
            if profit < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Check for recovery mode
            if self.consecutive_losses >= 3:
                self._activate_recovery_mode()
            elif profit > 0 and self.recovery_mode:
                self._deactivate_recovery_mode()
            
            # Update equity peak
            current_equity = self.mt5_manager.get_account_equity()
            if current_equity > self.equity_peak:
                self.equity_peak = current_equity
            
            # Store trade result
            self.trade_history.append({
                'timestamp': datetime.now(),
                'profit': profit,
                'symbol': trade_result.get('symbol', ''),
                'strategy': trade_result.get('strategy', ''),
                'equity': current_equity
            })
            
            # Update risk metrics
            self._update_risk_metrics()
            
            # Update last trade time
            self.last_trade_time = datetime.now()
            
            self.logger.info(f"Position closed: P&L ${profit:.2f}, Consecutive losses: {self.consecutive_losses}")
            
        except Exception as e:
            self.logger.error(f"Position close update failed: {str(e)}")
    
    def _activate_recovery_mode(self) -> None:
        """Activate recovery mode after losses"""
        if not self.recovery_mode:
            self.recovery_mode = True
            self.logger.warning("Recovery mode ACTIVATED - Reducing position sizes")
    
    def _deactivate_recovery_mode(self) -> None:
        """Deactivate recovery mode after profit"""
        if self.recovery_mode:
            self.recovery_mode = False
            self.logger.info("Recovery mode DEACTIVATED - Normal position sizing resumed")
    
    def _activate_emergency_stop(self) -> None:
        """Activate emergency stop mechanism"""
        self.emergency_stop = True
        self.logger.critical("EMERGENCY STOP ACTIVATED - All trading halted")
        
        # Close all open positions if possible
        try:
            self.mt5_manager.close_all_positions()
        except Exception as e:
            self.logger.error(f"Failed to close positions during emergency stop: {str(e)}")
    
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop (manual intervention required)"""
        try:
            self.emergency_stop = False
            self.consecutive_losses = 0
            self.recovery_mode = False
            
            self.logger.info("Emergency stop RESET - Trading can resume")
            return True
            
        except Exception as e:
            self.logger.error(f"Emergency stop reset failed: {str(e)}")
            return False
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        try:
            today = datetime.now().date()
            daily_trades = [t for t in self.trade_history 
                          if t['timestamp'].date() == today]
            
            return sum(t['profit'] for t in daily_trades)
            
        except Exception as e:
            self.logger.error(f"Daily P&L calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_weekly_pnl(self) -> float:
        """Calculate this week's P&L"""
        try:
            week_start = datetime.now() - timedelta(days=7)
            weekly_trades = [t for t in self.trade_history 
                           if t['timestamp'] >= week_start]
            
            return sum(t['profit'] for t in weekly_trades)
            
        except Exception as e:
            self.logger.error(f"Weekly P&L calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_current_drawdown(self, current_equity: float) -> float:
        """Calculate current drawdown from peak"""
        try:
            if self.equity_peak <= 0:
                return 0.0
            
            drawdown = (self.equity_peak - current_equity) / self.equity_peak
            return max(0, drawdown)
            
        except Exception as e:
            self.logger.error(f"Drawdown calculation failed: {str(e)}")
            return 0.0
            
    def _update_risk_metrics(self) -> None:
        """Update comprehensive risk metrics"""
        try:
            # Get current account info
            account_balance = self.mt5_manager.get_account_balance()
            equity = self.mt5_manager.get_account_equity()
            open_positions = self.mt5_manager.get_open_positions()
            
            # Calculate metrics
            current_drawdown = self._calculate_current_drawdown(equity)
            daily_pnl = self._calculate_daily_pnl()
            weekly_pnl = self._calculate_weekly_pnl()
            monthly_pnl = self._calculate_monthly_pnl()
            
            # Position metrics
            total_exposure = sum(abs(pos.get('volume', 0) * pos.get('price_current', 0) * 100) 
                               for pos in open_positions)
            unrealized_pnl = sum(pos.get('profit', 0) for pos in open_positions)
            
            # Performance metrics
            win_rate = self._calculate_win_rate()
            profit_factor = self._calculate_profit_factor()
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # Create risk metrics object
            risk_metrics = RiskMetrics(
                timestamp=datetime.now(),
                account_balance=account_balance,
                equity=equity,
                unrealized_pnl=unrealized_pnl,
                daily_pnl=daily_pnl,
                weekly_pnl=weekly_pnl,
                monthly_pnl=monthly_pnl,
                current_drawdown=current_drawdown,
                max_drawdown=max(current_drawdown, getattr(self.risk_metrics_history[-1], 'max_drawdown', 0) 
                               if self.risk_metrics_history else 0),
                drawdown_duration=self._calculate_drawdown_duration(),
                total_exposure=total_exposure,
                position_count=len(open_positions),
                largest_position=max((abs(pos.get('volume', 0)) for pos in open_positions), default=0),
                risk_per_trade=self.risk_per_trade,
                portfolio_risk=total_exposure / account_balance if account_balance > 0 else 0,
                var_95=self._calculate_var_95(),
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=self._calculate_sortino_ratio(),
                calmar_ratio=self._calculate_calmar_ratio(),
                win_rate=win_rate,
                profit_factor=profit_factor
            )
            
            # Store metrics
            self.risk_metrics_history.append(risk_metrics)
            
            # Keep only recent metrics (last 1000 records)
            if len(self.risk_metrics_history) > 1000:
                self.risk_metrics_history = self.risk_metrics_history[-1000:]
            
            # Update current risk level
            self._update_risk_level(risk_metrics)
            
        except Exception as e:
            self.logger.error(f"Risk metrics update failed: {str(e)}")
    
    def _calculate_monthly_pnl(self) -> float:
        """Calculate this month's P&L"""
        try:
            month_start = datetime.now().replace(day=1)
            monthly_trades = [t for t in self.trade_history 
                            if t['timestamp'] >= month_start]
            
            return sum(t['profit'] for t in monthly_trades)
            
        except Exception as e:
            self.logger.error(f"Monthly P&L calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_win_rate(self) -> float:
        """Calculate overall win rate"""
        try:
            if not self.trade_history:
                return 0.0
            
            winning_trades = [t for t in self.trade_history if t['profit'] > 0]
            return len(winning_trades) / len(self.trade_history)
            
        except Exception as e:
            self.logger.error(f"Win rate calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        try:
            if not self.trade_history:
                return 0.0
            
            gross_profit = sum(t['profit'] for t in self.trade_history if t['profit'] > 0)
            gross_loss = abs(sum(t['profit'] for t in self.trade_history if t['profit'] < 0))
            
            return gross_profit / gross_loss if gross_loss > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Profit factor calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.daily_returns) < 30:
                return 0.0
            
            returns = np.array(self.daily_returns)
            excess_returns = returns - 0.02/252  # Assuming 2% risk-free rate
            
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Sharpe ratio calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(self.daily_returns) < 30:
                return 0.0
            
            returns = np.array(self.daily_returns)
            excess_returns = returns - 0.02/252
            negative_returns = returns[returns < 0]
            
            downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0.01
            
            return np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Sortino ratio calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio"""
        try:
            if not self.risk_metrics_history:
                return 0.0
            
            annual_return = self._calculate_annual_return()
            max_drawdown = max((rm.max_drawdown for rm in self.risk_metrics_history), default=0.01)
            
            return annual_return / max_drawdown if max_drawdown > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Calmar ratio calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_annual_return(self) -> float:
        """Calculate annualized return"""
        try:
            if not self.trade_history:
                return 0.0
            
            total_return = sum(t['profit'] for t in self.trade_history)
            days_trading = (datetime.now() - self.trade_history[0]['timestamp']).days
            
            if days_trading > 0:
                return (total_return / self.initial_capital) * (365 / days_trading)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Annual return calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_drawdown_duration(self) -> int:
        """Calculate current drawdown duration in days"""
        try:
            if not self.risk_metrics_history:
                return 0
            
            # Find when current drawdown started
            drawdown_start = None
            for i in range(len(self.risk_metrics_history) - 1, -1, -1):
                if self.risk_metrics_history[i].current_drawdown == 0:
                    drawdown_start = self.risk_metrics_history[i].timestamp
                    break
            
            if drawdown_start:
                return (datetime.now() - drawdown_start).days
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"Drawdown duration calculation failed: {str(e)}")
            return 0
    
    def _calculate_var_95(self) -> float:
        """Calculate 95% Value at Risk"""
        try:
            if len(self.daily_returns) < 30:
                return 0.0
            
            returns = np.array(self.daily_returns)
            return np.percentile(returns, 5)  # 5th percentile for 95% VaR
            
        except Exception as e:
            self.logger.error(f"VaR calculation failed: {str(e)}")
            return 0.0
    
    def _update_risk_level(self, risk_metrics: RiskMetrics) -> None:
        """Update current risk level based on metrics"""
        try:
            risk_score = 0
            
            # Drawdown factor
            if risk_metrics.current_drawdown > 0.20:
                risk_score += 3
            elif risk_metrics.current_drawdown > 0.15:
                risk_score += 2
            elif risk_metrics.current_drawdown > 0.10:
                risk_score += 1
            
            # Portfolio risk factor
            if risk_metrics.portfolio_risk > 0.15:
                risk_score += 3
            elif risk_metrics.portfolio_risk > 0.10:
                risk_score += 2
            elif risk_metrics.portfolio_risk > 0.05:
                risk_score += 1
            
            # Consecutive losses factor
            if self.consecutive_losses >= 4:
                risk_score += 3
            elif self.consecutive_losses >= 3:
                risk_score += 2
            elif self.consecutive_losses >= 2:
                risk_score += 1
            
            # Daily loss factor
            daily_loss_pct = abs(risk_metrics.daily_pnl) / risk_metrics.account_balance
            if daily_loss_pct > 0.10:
                risk_score += 3
            elif daily_loss_pct > 0.05:
                risk_score += 2
            elif daily_loss_pct > 0.02:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 8:
                self.current_risk_level = RiskLevel.EXTREME
            elif risk_score >= 5:
                self.current_risk_level = RiskLevel.HIGH
            elif risk_score >= 3:
                self.current_risk_level = RiskLevel.MODERATE
            else:
                self.current_risk_level = RiskLevel.LOW
                
        except Exception as e:
            self.logger.error(f"Risk level update failed: {str(e)}")
            self.current_risk_level = RiskLevel.MODERATE
    
    def _get_strategy_performance(self, strategy_name: str) -> Tuple[float, float, float]:
        """Get historical performance for a strategy"""
        try:
            strategy_trades = [t for t in self.trade_history 
                             if t.get('strategy') == strategy_name]
            
            if not strategy_trades:
                return 0.5, 0.02, 0.02  # Default values
            
            winning_trades = [t for t in strategy_trades if t['profit'] > 0]
            losing_trades = [t for t in strategy_trades if t['profit'] < 0]
            
            win_rate = len(winning_trades) / len(strategy_trades)
            avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0.02
            avg_loss = abs(np.mean([t['profit'] for t in losing_trades])) if losing_trades else 0.02
            
            return win_rate, avg_win, avg_loss
            
        except Exception as e:
            self.logger.error(f"Strategy performance calculation failed: {str(e)}")
            return 0.5, 0.02, 0.02
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(data) < period + 1:
                return data['Close'].iloc[-1] * 0.01
            
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else data['Close'].iloc[-1] * 0.01
            
        except Exception as e:
            self.logger.error(f"ATR calculation failed: {str(e)}")
            return data['Close'].iloc[-1] * 0.01 if len(data) > 0 else 10.0
    
    def _load_performance_history(self) -> None:
        """Load historical performance data"""
        try:
            if self.database_manager:
                # Load trade history from database
                trades = self.database_manager.get_trades(limit=1000)
                
                for trade in trades:
                    if trade.get('status') == 'CLOSED':
                        self.trade_history.append({
                            'timestamp': trade.get('close_time', datetime.now()),
                            'profit': trade.get('profit', 0),
                            'symbol': trade.get('symbol', ''),
                            'strategy': trade.get('strategy', ''),
                            'equity': trade.get('profit', 0)  # Simplified
                        })
                
                self.logger.info(f"Loaded {len(self.trade_history)} historical trades")
                
        except Exception as e:
            self.logger.error(f"Performance history loading failed: {str(e)}")
    
    def _get_current_risk_level(self) -> Dict[str, Any]:
        """Get current risk assessment"""
        return {
            'risk_level': self.current_risk_level.value,
            'recovery_mode': self.recovery_mode,
            'emergency_stop': self.emergency_stop,
            'consecutive_losses': self.consecutive_losses,
            'equity_peak': self.equity_peak
        }
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            current_equity = self.mt5_manager.get_account_equity()
            open_positions = self.mt5_manager.get_open_positions()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'risk_level': self.current_risk_level.value,
                'account_status': {
                    'balance': self.mt5_manager.get_account_balance(),
                    'equity': current_equity,
                    'equity_peak': self.equity_peak,
                    'current_drawdown': self._calculate_current_drawdown(current_equity),
                    'daily_pnl': self._calculate_daily_pnl(),
                    'weekly_pnl': self._calculate_weekly_pnl(),
                    'monthly_pnl': self._calculate_monthly_pnl()
                },
                'position_metrics': {
                    'open_positions': len(open_positions),
                    'max_positions': self.max_positions,
                    'total_exposure': sum(abs(pos.get('volume', 0) * pos.get('price_current', 0) * 100) 
                                        for pos in open_positions),
                    'unrealized_pnl': sum(pos.get('profit', 0) for pos in open_positions)
                },
                'risk_controls': {
                    'recovery_mode': self.recovery_mode,
                    'emergency_stop': self.emergency_stop,
                    'consecutive_losses': self.consecutive_losses,
                    'max_consecutive_losses': self.max_consecutive_losses,
                    'risk_per_trade': self.risk_per_trade,
                    'max_drawdown_limit': self.max_drawdown
                },
                'performance_metrics': {
                    'total_trades': len(self.trade_history),
                    'win_rate': self._calculate_win_rate(),
                    'profit_factor': self._calculate_profit_factor(),
                    'sharpe_ratio': self._calculate_sharpe_ratio(),
                    'max_drawdown': max((rm.max_drawdown for rm in self.risk_metrics_history), default=0)
                },
                'position_sizing': {
                    'method': self.sizing_method.value,
                    'kelly_safety_factor': self.kelly_safety_factor,
                    'min_size': self.min_position_size,
                    'max_size': self.max_position_size
                }
            }
            
        except Exception as e:
            self.logger.error(f"Risk summary generation failed: {str(e)}")
            return {'error': str(e)}


# Testing function
if __name__ == "__main__":
    """Test the Risk Manager"""
    
    # Test configuration
    test_config = {
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
        },
        'capital': {
            'initial_capital': 100.0,
            'target_capital': 1000.0,
            'minimum_capital': 50.0,
            'reserve_cash': 0.10
        }
    }
    
    # Mock managers for testing
    class MockMT5Manager:
        def get_account_balance(self):
            return 150.0
        
        def get_account_equity(self):
            return 145.0
        
        def get_open_positions(self):
            return [
                {'symbol': 'XAUUSDm', 'type': 'BUY', 'volume': 0.02, 
                 'price_current': 1960.0, 'profit': 25.0},
                {'symbol': 'XAUUSDm', 'type': 'SELL', 'volume': 0.01, 
                 'price_current': 1955.0, 'profit': -10.0}
            ]
        
        def get_historical_data(self, symbol, timeframe, bars):
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            dates = pd.date_range(start=datetime.now() - timedelta(days=5), 
                                 end=datetime.now(), freq='15Min')
            
            close_prices = 1950 + np.cumsum(np.random.randn(len(dates)) * 2)
            
            return pd.DataFrame({
                'Open': close_prices + np.random.randn(len(dates)) * 0.5,
                'High': close_prices + np.abs(np.random.randn(len(dates)) * 3),
                'Low': close_prices - np.abs(np.random.randn(len(dates)) * 3),
                'Close': close_prices,
                'Volume': np.random.randint(100, 1000, len(dates))
            }, index=dates)
        
        def close_all_positions(self):
            pass
    
    class MockDatabaseManager:
        def get_trades(self, limit=1000):
            return []
    
    # Create mock signal
    from dataclasses import dataclass
    from enum import Enum
    
    @dataclass
    class MockSignal:
        symbol: str = "XAUUSDm"
        strategy_name: str = "test_strategy"
        signal_type: SignalType = SignalType.BUY
        confidence: float = 0.85
        price: float = 1960.0
        timeframe: str = "M15"
        strength: float = 0.8
        grade: SignalGrade = SignalGrade.A
        stop_loss: float = 1950.0
        take_profit: float = 1980.0
    
    # Create risk manager
    mock_mt5 = MockMT5Manager()
    mock_db = MockDatabaseManager()
    risk_manager = RiskManager(test_config, mock_mt5, mock_db)
    
    # Test position sizing
    signal = MockSignal()
    sizing_result = risk_manager.calculate_position_size(
        signal, 150.0, mock_mt5.get_open_positions()
    )
    
    print("Position Sizing Result:")
    print(f"Position Size: {sizing_result['position_size']}")
    print(f"Allowed: {sizing_result['allowed']}")
    print(f"Risk Assessment: {sizing_result['risk_assessment']}")
    
    # Test risk summary
    risk_summary = risk_manager.get_risk_summary()
    print(f"\nRisk Summary:")
    print(f"Risk Level: {risk_summary['risk_level']}")
    print(f"Account Status: {risk_summary['account_status']}")
    print(f"Position Metrics: {risk_summary['position_metrics']}")
    
    # Test trade update
    trade_result = {
        'profit': 50.0,
        'symbol': 'XAUUSDm',
        'strategy': 'test_strategy'
    }
    risk_manager.update_position_closed(trade_result)
    
    print("Risk Manager test completed!")
