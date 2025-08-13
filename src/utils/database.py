"""
Database Module - Complete Database Schema and Management
========================================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-08-07

This module handles all database operations for the trading system:
- SQLite database with comprehensive schema
- Trade storage and retrieval
- Signal logging and analysis
- Performance tracking
- Configuration storage
- Data export/import functionality

Features:
- Automated schema creation
- Data validation and integrity
- Performance optimization
- Backup and restore
- Data retention policies
- Query optimization

Dependencies:
    - sqlite3
    - sqlalchemy
    - pandas
    - pathlib
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text, ForeignKey
#from sqlalchemy.ext.declarative import declarative_base #MovedIn20Warning: The ``declarative_base()`` function is now available as sqlalchemy.orm.declarative_base()
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool
import json
import logging
from contextlib import contextmanager

# Create base class for models
Base = declarative_base()

# Database Models
class Account(Base):
    """Account information model"""
    __tablename__ = 'accounts'
    
    id = Column(Integer, primary_key=True)
    login = Column(Integer, unique=True, nullable=False)
    server = Column(String(100), nullable=False)
    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    margin = Column(Float, nullable=False)
    free_margin = Column(Float, nullable=False)
    margin_level = Column(Float, nullable=False)
    profit = Column(Float, nullable=False)
    leverage = Column(Integer, nullable=False)
    currency = Column(String(10), nullable=False)
    # Add these fields to match MT5 account info
    trade_allowed = Column(Boolean, default=True)
    trade_expert = Column(Boolean, default=True)
    limit_orders = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    trades = relationship("Trade", back_populates="account")
    performance_records = relationship("Performance", back_populates="account")

    
class Trade(Base):
    """Trade execution model"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)
    ticket = Column(Integer, nullable=False)
    symbol = Column(String(20), nullable=False)
    action = Column(String(10), nullable=False)  # BUY/SELL
    volume = Column(Float, nullable=False)
    price_open = Column(Float, nullable=False)
    price_close = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    profit = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    comment = Column(String(200))
    magic_number = Column(Integer)
    strategy = Column(String(50))
    timeframe = Column(String(10))
    signal_id = Column(Integer, ForeignKey('signals.id'))
    
    # Timestamps
    open_time = Column(DateTime, nullable=False)
    close_time = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)
    
    # Status
    status = Column(String(20), default='OPEN')  # OPEN, CLOSED, CANCELLED
    
    # Relationships
    account = relationship("Account", back_populates="trades")
    signal = relationship("Signal", back_populates="trades")


class Signal(Base):
    """Trading signal model"""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    symbol = Column(String(20), nullable=False)
    strategy = Column(String(50), nullable=False)
    signal_type = Column(String(10), nullable=False)  # BUY/SELL
    confidence = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timeframe = Column(String(10), nullable=False)
    
    # Technical analysis data
    rsi = Column(Float)
    macd = Column(Float)
    bollinger_position = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    atr = Column(Float)
    volume = Column(Float)
    
    # Signal quality metrics
    strength = Column(Float)
    quality_grade = Column(String(1))  # A, B, C
    risk_reward_ratio = Column(Float)
    
    # Execution data
    executed = Column(Boolean, default=False)
    execution_price = Column(Float)
    execution_time = Column(DateTime)
    
    # Additional metadata
    #metadata = Column(Text)  # JSON string for additional data
    signal_metadata = Column(Text)  # JSON string for additional data
    
    # Relationships
    trades = relationship("Trade", back_populates="signal")


class Performance(Base):
    """Performance tracking model"""
    __tablename__ = 'performance'
    
    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)
    date = Column(DateTime, nullable=False)
    period = Column(String(20), nullable=False)  # daily, weekly, monthly
    
    # Financial metrics
    balance_start = Column(Float, nullable=False)
    balance_end = Column(Float, nullable=False)
    equity_start = Column(Float, nullable=False)
    equity_end = Column(Float, nullable=False)
    profit_loss = Column(Float, nullable=False)
    profit_loss_percent = Column(Float, nullable=False)
    
    # Trading metrics
    trades_total = Column(Integer, default=0)
    trades_won = Column(Integer, default=0)
    trades_lost = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    avg_win = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    
    # Risk metrics
    max_drawdown = Column(Float, default=0.0)
    max_drawdown_percent = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    sortino_ratio = Column(Float, default=0.0)
    calmar_ratio = Column(Float, default=0.0)
    
    # Volume metrics
    volume_traded = Column(Float, default=0.0)
    commission_paid = Column(Float, default=0.0)
    swap_total = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    account = relationship("Account", back_populates="performance_records")


class MarketData(Base):
    """Historical market data model"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    
    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Additional data
    spread = Column(Float)
    real_volume = Column(Float)
    
    created_at = Column(DateTime, default=datetime.now)


class Configuration(Base):
    """System configuration model"""
    __tablename__ = 'configurations'
    
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    description = Column(String(500))
    category = Column(String(50))
    data_type = Column(String(20))  # string, int, float, bool, json
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class DatabaseManager:
    """
    Comprehensive database manager for the trading system
    
    This class handles all database operations including:
    - Database initialization and schema creation
    - CRUD operations for all models
    - Data validation and integrity
    - Performance optimization
    - Backup and restore operations
    
    Example:
        >>> db_mgr = DatabaseManager(config)
        >>> db_mgr.initialize_database()
        >>> db_mgr.store_trade(trade_data)
    """
    
    def __init__(self, config: Union[Dict[str, Any], str]):
        if isinstance(config, str):
            config = {"database": {"sqlite": {"path": config}}}
        """
        Initialize the database manager
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.db_config = config.get('database', {})
        
        # Database path
        self.db_path = Path(self.db_config.get('sqlite', {}).get('path', 'data/trading.db'))
        self.db_path.parent.mkdir(exist_ok=True)
        
        # SQLAlchemy setup
        self.engine = None
        self.Session = None
        self.initialized = False
        
        # Logger
        self.logger = logging.getLogger('xau_database')
    
    def initialize_database(self) -> bool:
        """
        Initialize the database and create all tables
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Create engine
            self.engine = create_engine(
                f'sqlite:///{self.db_path}',
                poolclass=StaticPool,
                connect_args={
                    'check_same_thread': False,
                    'timeout': 30
                },
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            # Create all tables
            Base.metadata.create_all(self.engine)
            
            # Initialize default configurations
            self._init_default_configs()
            
            self.initialized = True
            self.logger.info("Database initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            return False
    
    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions
        
        Example:
            >>> with db_mgr.get_session() as session:
            ...     session.add(trade)
            ...     session.commit()
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()
    
    def _init_default_configs(self) -> None:
        """Initialize default configuration values"""
        default_configs = [
            {
                'key': 'system.version',
                'value': '1.0.0',
                'description': 'System version',
                'category': 'system',
                'data_type': 'string'
            },
            {
                'key': 'system.initialized_at',
                'value': datetime.now().isoformat(),
                'description': 'System initialization timestamp',
                'category': 'system',
                'data_type': 'string'
            }
        ]
        
        with self.get_session() as session:
            for config in default_configs:
                existing = session.query(Configuration).filter_by(key=config['key']).first()
                if not existing:
                    session.add(Configuration(**config))
    
    # Account operations
    def store_account_info(self, account_data: Dict[str, Any]) -> Optional[int]:
        """
        Store or update account information
        
        Args:
            account_data (dict): Account information dictionary
            
        Returns:
            int: Account ID if successful, None otherwise
        """
        try:
            with self.get_session() as session:
                # Check if account exists
                existing = session.query(Account).filter_by(login=account_data['login']).first()
                
                if existing:
                    # Update existing account
                    for key, value in account_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.updated_at = datetime.now()
                    account_id = existing.id
                else:
                    # Create new account
                    account = Account(**account_data)
                    session.add(account)
                    session.flush()
                    account_id = account.id
                
                return account_id
                
        except Exception as e:
            self.logger.error(f"Failed to store account info: {str(e)}")
            return None
    
    def get_account_info(self, login: int) -> Optional[Dict[str, Any]]:
        """Get account information by login"""
        try:
            with self.get_session() as session:
                account = session.query(Account).filter_by(login=login).first()
                if account:
                    return {
                        'id': account.id,
                        'login': account.login,
                        'server': account.server,
                        'balance': account.balance,
                        'equity': account.equity,
                        'margin': account.margin,
                        'free_margin': account.free_margin,
                        'margin_level': account.margin_level,
                        'profit': account.profit,
                        'leverage': account.leverage,
                        'currency': account.currency,
                        'created_at': account.created_at,
                        'updated_at': account.updated_at
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get account info: {str(e)}")
            return None
    
    # Trade operations
    def store_trade(self, trade_data: Dict[str, Any]) -> Optional[int]:
        """
        Store a new trade
        
        Args:
            trade_data (dict): Trade information dictionary
            
        Returns:
            int: Trade ID if successful, None otherwise
        """
        try:
            with self.get_session() as session:
                trade = Trade(**trade_data)
                session.add(trade)
                session.flush()
                return trade.id
                
        except Exception as e:
            self.logger.error(f"Failed to store trade: {str(e)}")
            return None
    
    def update_trade(self, trade_id: int, update_data: Dict[str, Any]) -> bool:
        """Update an existing trade"""
        try:
            with self.get_session() as session:
                trade = session.query(Trade).filter_by(id=trade_id).first()
                if trade:
                    for key, value in update_data.items():
                        if hasattr(trade, key):
                            setattr(trade, key, value)
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update trade: {str(e)}")
            return False
    
    def close_trade(self, ticket: int, close_price: float, close_time: datetime, 
                    profit: float, swap: float = 0.0, commission: float = 0.0) -> bool:
        """Close a trade and update its information"""
        try:
            with self.get_session() as session:
                trade = session.query(Trade).filter_by(ticket=ticket, status='OPEN').first()
                if trade:
                    trade.price_close = close_price
                    trade.close_time = close_time
                    trade.profit = profit
                    trade.swap = swap
                    trade.commission = commission
                    trade.status = 'CLOSED'
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to close trade: {str(e)}")
            return False
    
    def get_trades(self, account_id: Optional[int] = None, status: Optional[str] = None,
                   symbol: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """Get trades with optional filters"""
        try:
            with self.get_session() as session:
                query = session.query(Trade)
                
                if account_id:
                    query = query.filter(Trade.account_id == account_id)
                if status:
                    query = query.filter(Trade.status == status)
                if symbol:
                    query = query.filter(Trade.symbol == symbol)
                
                query = query.order_by(Trade.open_time.desc())
                
                if limit:
                    query = query.limit(limit)
                
                trades = query.all()
                return [self._trade_to_dict(trade) for trade in trades]
                
        except Exception as e:
            self.logger.error(f"Failed to get trades: {str(e)}")
            return []
    
    def get_open_trades(self, account_id: Optional[int] = None) -> List[Dict]:
        """Get all open trades"""
        return self.get_trades(account_id=account_id, status='OPEN')
    
    # Signal operations
    def store_signal(self, signal_data: Dict[str, Any]) -> Optional[int]:
        """Store a new trading signal"""
        try:
            with self.get_session() as session:
                # Convert metadata to JSON if it's a dict
                if 'metadata' in signal_data and isinstance(signal_data['metadata'], dict):
                    signal_data['signal_metadata'] = json.dumps(signal_data['metadata'])
                
                signal = Signal(**signal_data)
                session.add(signal)
                session.flush()
                return signal.id
                
        except Exception as e:
            self.logger.error(f"Failed to store signal: {str(e)}")
            return None
    
    def update_signal_execution(self, signal_id: int, execution_price: float, 
                               execution_time: datetime) -> bool:
        """Update signal with execution information"""
        try:
            with self.get_session() as session:
                signal = session.query(Signal).filter_by(id=signal_id).first()
                if signal:
                    signal.executed = True
                    signal.execution_price = execution_price
                    signal.execution_time = execution_time
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update signal execution: {str(e)}")
            return False
    
    def get_signals(self, strategy: Optional[str] = None, symbol: Optional[str] = None,
                    executed: Optional[bool] = None, limit: Optional[int] = 100) -> List[Dict]:
        """Get signals with optional filters"""
        try:
            with self.get_session() as session:
                query = session.query(Signal)
                
                if strategy:
                    query = query.filter(Signal.strategy == strategy)
                if symbol:
                    query = query.filter(Signal.symbol == symbol)
                if executed is not None:
                    query = query.filter(Signal.executed == executed)
                
                query = query.order_by(Signal.timestamp.desc()).limit(limit)
                
                signals = query.all()
                return [self._signal_to_dict(signal) for signal in signals]
                
        except Exception as e:
            self.logger.error(f"Failed to get signals: {str(e)}")
            return []
    
    # Performance operations
    def store_performance(self, performance_data: Dict[str, Any]) -> Optional[int]:
        """Store performance metrics"""
        try:
            with self.get_session() as session:
                performance = Performance(**performance_data)
                session.add(performance)
                session.flush()
                return performance.id
                
        except Exception as e:
            self.logger.error(f"Failed to store performance: {str(e)}")
            return None
    
    def get_performance(self, account_id: int, period: str = 'daily', 
                       days: int = 30) -> List[Dict]:
        """Get performance metrics for a period"""
        try:
            with self.get_session() as session:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                performance = session.query(Performance).filter(
                    Performance.account_id == account_id,
                    Performance.period == period,
                    Performance.date >= cutoff_date
                ).order_by(Performance.date.desc()).all()
                
                return [self._performance_to_dict(perf) for perf in performance]
                
        except Exception as e:
            self.logger.error(f"Failed to get performance: {str(e)}")
            return []
    
    # Market data operations
    def store_market_data(self, market_data: List[Dict[str, Any]]) -> bool:
        """Store market data (OHLCV bars)"""
        try:
            with self.get_session() as session:
                for bar_data in market_data:
                    # Check if data already exists
                    existing = session.query(MarketData).filter(
                        MarketData.symbol == bar_data['symbol'],
                        MarketData.timeframe == bar_data['timeframe'],
                        MarketData.timestamp == bar_data['timestamp']
                    ).first()
                    
                    if not existing:
                        bar = MarketData(**bar_data)
                        session.add(bar)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store market data: {str(e)}")
            return False
    
    def get_market_data(self, symbol: str, timeframe: str, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       limit: int = 1000) -> pd.DataFrame:
        """Get market data as pandas DataFrame"""
        try:
            with self.get_session() as session:
                query = session.query(MarketData).filter(
                    MarketData.symbol == symbol,
                    MarketData.timeframe == timeframe
                )
                
                if start_date:
                    query = query.filter(MarketData.timestamp >= start_date)
                if end_date:
                    query = query.filter(MarketData.timestamp <= end_date)
                
                query = query.order_by(MarketData.timestamp.desc()).limit(limit)
                
                data = query.all()
                
                if data:
                    df = pd.DataFrame([{
                        'timestamp': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                        'spread': bar.spread,
                        'real_volume': bar.real_volume
                    } for bar in data])
                    
                    df.set_index('timestamp', inplace=True)
                    return df.sort_index()
                
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to get market data: {str(e)}")
            return pd.DataFrame()
    
    # Configuration operations
    def set_config(self, key: str, value: Any, description: str = "", 
                   category: str = "general") -> bool:
        """Set a configuration value"""
        try:
            with self.get_session() as session:
                # Determine data type
                data_type = type(value).__name__
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                    data_type = 'json'
                
                # Check if config exists
                existing = session.query(Configuration).filter_by(key=key).first()
                
                if existing:
                    existing.value = str(value)
                    existing.description = description
                    existing.category = category
                    existing.data_type = data_type
                    existing.updated_at = datetime.now()
                else:
                    config = Configuration(
                        key=key,
                        value=str(value),
                        description=description,
                        category=category,
                        data_type=data_type
                    )
                    session.add(config)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to set config: {str(e)}")
            return False
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        try:
            with self.get_session() as session:
                config = session.query(Configuration).filter_by(key=key).first()
                
                if config:
                    value = config.value
                    
                    # Convert based on data type
                    if config.data_type == 'int':
                        return int(value)
                    elif config.data_type == 'float':
                        return float(value)
                    elif config.data_type == 'bool':
                        return value.lower() in ('true', '1', 'yes')
                    elif config.data_type == 'json':
                        return json.loads(value)
                    else:
                        return value
                
                return default
                
        except Exception as e:
            self.logger.error(f"Failed to get config: {str(e)}")
            return default
    
    # Utility methods
    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """Convert Trade object to dictionary"""
        return {
            'id': trade.id,
            'account_id': trade.account_id,
            'ticket': trade.ticket,
            'symbol': trade.symbol,
            'action': trade.action,
            'volume': trade.volume,
            'price_open': trade.price_open,
            'price_close': trade.price_close,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profit,
            'profit': trade.profit,
            'swap': trade.swap,
            'commission': trade.commission,
            'comment': trade.comment,
            'magic_number': trade.magic_number,
            'strategy': trade.strategy,
            'timeframe': trade.timeframe,
            'signal_id': trade.signal_id,
            'open_time': trade.open_time,
            'close_time': trade.close_time,
            'status': trade.status,
            'created_at': trade.created_at
        }
    
    def _signal_to_dict(self, signal: Signal) -> Dict[str, Any]:
        """Convert Signal object to dictionary"""
        metadata = None
        if signal.signal_metadata:
            try:
                metadata = json.loads(signal.signal.metadata)
            except:
                metadata = signal.signal.metadata
        
        return {
            'id': signal.id,
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'strategy': signal.strategy,
            'signal_type': signal.signal_type,
            'confidence': signal.confidence,
            'price': signal.price,
            'timeframe': signal.timeframe,
            'rsi': signal.rsi,
            'macd': signal.macd,
            'bollinger_position': signal.bollinger_position,
            'sma_20': signal.sma_20,
            'sma_50': signal.sma_50,
            'atr': signal.atr,
            'volume': signal.volume,
            'strength': signal.strength,
            'quality_grade': signal.quality_grade,
            'risk_reward_ratio': signal.risk_reward_ratio,
            'executed': signal.executed,
            'execution_price': signal.execution_price,
            'execution_time': signal.execution_time,
            'metadata': metadata
        }
    
    def _performance_to_dict(self, performance: Performance) -> Dict[str, Any]:
        """Convert Performance object to dictionary"""
        return {
            'id': performance.id,
            'account_id': performance.account_id,
            'date': performance.date,
            'period': performance.period,
            'balance_start': performance.balance_start,
            'balance_end': performance.balance_end,
            'equity_start': performance.equity_start,
            'equity_end': performance.equity_end,
            'profit_loss': performance.profit_loss,
            'profit_loss_percent': performance.profit_loss_percent,
            'trades_total': performance.trades_total,
            'trades_won': performance.trades_won,
            'trades_lost': performance.trades_lost,
            'win_rate': performance.win_rate,
            'avg_win': performance.avg_win,
            'avg_loss': performance.avg_loss,
            'profit_factor': performance.profit_factor,
            'max_drawdown': performance.max_drawdown,
            'max_drawdown_percent': performance.max_drawdown_percent,
            'sharpe_ratio': performance.sharpe_ratio,
            'sortino_ratio': performance.sortino_ratio,
            'calmar_ratio': performance.calmar_ratio,
            'volume_traded': performance.volume_traded,
            'commission_paid': performance.commission_paid,
            'swap_total': performance.swap_total,
            'created_at': performance.created_at
        }
    
    # Data export/import
    def export_data(self, table_name: str, start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Export data from a table to DataFrame"""
        try:
            with self.get_session() as session:
                if table_name == 'trades':
                    query = session.query(Trade)
                    if start_date:
                        query = query.filter(Trade.open_time >= start_date)
                    if end_date:
                        query = query.filter(Trade.open_time <= end_date)
                    
                    trades = query.all()
                    return pd.DataFrame([self._trade_to_dict(t) for t in trades])
                
                elif table_name == 'signals':
                    query = session.query(Signal)
                    if start_date:
                        query = query.filter(Signal.timestamp >= start_date)
                    if end_date:
                        query = query.filter(Signal.timestamp <= end_date)
                    
                    signals = query.all()
                    return pd.DataFrame([self._signal_to_dict(s) for s in signals])
                
                elif table_name == 'performance':
                    query = session.query(Performance)
                    if start_date:
                        query = query.filter(Performance.date >= start_date)
                    if end_date:
                        query = query.filter(Performance.date <= end_date)
                    
                    perf = query.all()
                    return pd.DataFrame([self._performance_to_dict(p) for p in perf])
                
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to export data: {str(e)}")
            return pd.DataFrame()
    
    def backup_database(self, backup_path: Optional[Path] = None) -> bool:
        """Create a backup of the database"""
        try:
            if not backup_path:
                backup_path = self.db_path.parent / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            
            # Simple file copy for SQLite
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            self.logger.info(f"Database backed up to: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Database backup failed: {str(e)}")
            return False
    
    def cleanup_old_data(self, days: int = 90) -> bool:
        """Clean up old data based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.get_session() as session:
                # Clean up old market data
                deleted_market = session.query(MarketData).filter(
                    MarketData.created_at < cutoff_date
                ).delete()
                
                # Clean up old signals (keep trade-related signals)
                deleted_signals = session.query(Signal).filter(
                    Signal.timestamp < cutoff_date,
                    Signal.executed == False
                ).delete()
                
                self.logger.info(f"Cleaned up {deleted_market} market data records and {deleted_signals} signals")
                return True
                
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {str(e)}")
            return False
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        try:
            with self.get_session() as session:
                stats = {
                    'accounts': session.query(Account).count(),
                    'trades': session.query(Trade).count(),
                    'open_trades': session.query(Trade).filter(Trade.status == 'OPEN').count(),
                    'signals': session.query(Signal).count(),
                    'executed_signals': session.query(Signal).filter(Signal.executed == True).count(),
                    'performance_records': session.query(Performance).count(),
                    'market_data_records': session.query(MarketData).count(),
                    'configurations': session.query(Configuration).count()
                }
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {str(e)}")
            return {}


# Testing function
if __name__ == "__main__":
    """Test the database system"""
    import yaml
    
    # Test configuration
    test_config = {
        'database': {
            'type': 'sqlite',
            'sqlite': {
                'path': 'test_trading.db'
            }
        }
    }
    
    # Create database manager
    db_mgr = DatabaseManager(test_config)
    db_mgr.initialize_database()
    
    # Test account storage
    account_data = {
        'login': 12345678,
        'server': 'TestServer',
        'balance': 1000.0,
        'equity': 1000.0,
        'margin': 0.0,
        'free_margin': 1000.0,
        'margin_level': 0.0,
        'profit': 0.0,
        'leverage': 500,
        'currency': 'USD'
    }
    
    account_id = db_mgr.store_account_info(account_data)
    print(f"Account stored with ID: {account_id}")
    
    # Test trade storage
    trade_data = {
        'account_id': account_id,
        'ticket': 123456,
        'symbol': 'XAUUSDm',
        'action': 'BUY',
        'volume': 0.01,
        'price_open': 1950.0,
        'stop_loss': 1945.0,
        'take_profit': 1960.0,
        'magic_number': 123456,
        'strategy': 'test',
        'timeframe': 'M15',
        'open_time': datetime.now(),
        'status': 'OPEN'
    }
    
    trade_id = db_mgr.store_trade(trade_data)
    print(f"Trade stored with ID: {trade_id}")
    
    # Test signal storage
    signal_data = {
        'symbol': 'XAUUSDm',
        'strategy': 'ichimoku',
        'signal_type': 'BUY',
        'confidence': 0.85,
        'price': 1950.0,
        'timeframe': 'M15',
        'rsi': 65.0,
        'strength': 0.8,
        'quality_grade': 'A',
        'metadata': {'test': True}
    }
    
    signal_id = db_mgr.store_signal(signal_data)
    print(f"Signal stored with ID: {signal_id}")
    
    # Test configuration
    db_mgr.set_config('test.parameter', 'test_value', 'Test configuration')
    config_value = db_mgr.get_config('test.parameter')
    print(f"Config value: {config_value}")
    
    # Get database stats
    stats = db_mgr.get_database_stats()
    print(f"Database stats: {stats}")
    
    print("Database test completed successfully!")