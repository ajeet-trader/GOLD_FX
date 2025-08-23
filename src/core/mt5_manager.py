"""
MT5 Manager - Core MetaTrader 5 Integration Module
==================================================
Author: XAUUSD Trading System
Version: 1.1.0
Date: 2025-01-08

This module handles all interactions with MetaTrader 5:
- Connection management using environment variables
- Historical data fetching
- Real-time data streaming
- Order execution
- Account management
- Symbol information

Dependencies:
    - MetaTrader5
    - pandas
    - numpy
    - datetime
    - python-dotenv
    
Environment Variables (.env file):
    MT5_LOGIN=your_account_number
    MT5_PASSWORD=your_password
    MT5_SERVER=your_broker_server
    MT5_PATH=path_to_terminal64.exe (optional)    
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
import sys

# Add project root to path if running standalone
if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger_manager

# No longer using basic logging configuration - using LoggerManager instead


class MT5Manager:
    """
    Comprehensive MT5 Manager for XAUUSD Trading System
    
    This class provides a complete interface to MetaTrader 5, handling:
    - Platform initialization and connection using environment variables
    - Data retrieval (historical and real-time)
    - Order management (market, pending, modify, close)
    - Account information
    - Symbol specifications
    
    Environment Variables Required:
        MT5_LOGIN: MT5 account number
        MT5_PASSWORD: MT5 account password
        MT5_SERVER: Broker server name
        MT5_TERMINAL_PATH: Path to MT5 terminal (optional)
    
    Attributes:
        connected (bool): Connection status to MT5
        symbol (str): Trading symbol (default: XAUUSD)
        account_info (dict): Current account information
        symbol_info (dict): Symbol specifications
        magic_number (int): Magic number for orders
    
    Example:
        >>> mt5_mgr = MT5Manager()
        >>> mt5_mgr.connect()
        >>> data = mt5_mgr.get_historical_data("XAUUSDm", "M5", 1000)
        >>> mt5_mgr.place_market_order("XAUUSDm", "BUY", 0.01)
    """
    
    # MT5 Timeframe mappings
    TIMEFRAMES = {
        'M1': mt5.TIMEFRAME_M1,     # 1 minute
        'M5': mt5.TIMEFRAME_M5,     # 5 minutes
        'M15': mt5.TIMEFRAME_M15,   # 15 minutes
        'M30': mt5.TIMEFRAME_M30,   # 30 minutes
        'H1': mt5.TIMEFRAME_H1,     # 1 hour
        'H4': mt5.TIMEFRAME_H4,     # 4 hours
        'D1': mt5.TIMEFRAME_D1,     # 1 day
        'W1': mt5.TIMEFRAME_W1,     # 1 week
        'MN1': mt5.TIMEFRAME_MN1    # 1 month
    }
    
    # Order type mappings
    ORDER_TYPES = {
        'BUY': mt5.ORDER_TYPE_BUY,
        'SELL': mt5.ORDER_TYPE_SELL,
        'BUY_LIMIT': mt5.ORDER_TYPE_BUY_LIMIT,
        'SELL_LIMIT': mt5.ORDER_TYPE_SELL_LIMIT,
        'BUY_STOP': mt5.ORDER_TYPE_BUY_STOP,
        'SELL_STOP': mt5.ORDER_TYPE_SELL_STOP
    }
    
    def __init__(self, symbol: str = "XAUUSD", magic_number: int = 123456):
        """
        Initialize MT5 Manager with environment variables
        
        Args:
            symbol (str): Default trading symbol (default: "XAUUSD")
            magic_number (int): Magic number for orders (default: 123456)
        """
        self.connected = False
        self.symbol = symbol
        self.account_info = {}
        self.symbol_info = {}
        self.magic_number = magic_number
        
        # Load credentials from environment
        self.mt5_login = os.getenv('MT5_LOGIN')
        self.mt5_password = os.getenv('MT5_PASSWORD')
        self.mt5_server = os.getenv('MT5_SERVER')
        self.mt5_terminal_path = os.getenv('MT5_TERMINAL_PATH')
        
        # Logger Manager for structured logging
        self.logger_manager = get_logger_manager()
        self.logger = self.logger_manager.get_logger('mt5')
        
        # Validate required environment variables
        if not all([self.mt5_login, self.mt5_password, self.mt5_server]):
            self.logger.warning("Missing required environment variables: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER")
            self.logger.info("Please set these variables in your .env file")
        
        # Convert login to integer
        try:
            self.mt5_login = int(self.mt5_login) if self.mt5_login else None
        except (ValueError, TypeError):
            self.mt5_login = None
            self.logger.error("MT5_LOGIN must be a valid integer")
        
        # Load available symbols if CSV exists
        self.available_symbols = self._load_available_symbols()
        
        self.logger.info(f"MT5Manager initialized for symbol: {self.symbol}")
        self.logger.info(f"Magic number: {self.magic_number}")
    
    def connect(self, login: Optional[int] = None, password: Optional[str] = None, 
                server: Optional[str] = None, path: Optional[str] = None) -> bool:
        """
        Establish connection to MT5 terminal using environment variables or provided credentials
        
        Args:
            login (int, optional): Account login. Uses env var if not provided
            password (str, optional): Account password. Uses env var if not provided
            server (str, optional): Broker server. Uses env var if not provided
            path (str, optional): Path to MT5 terminal. Uses env var if not provided
        
        Returns:
            bool: True if connection successful, False otherwise
        
        Raises:
            ConnectionError: If unable to connect to MT5
        
        Example:
            >>> mt5_mgr.connect()
            True
        """
        try:
            # Use provided credentials or fall back to environment variables
            login = login or self.mt5_login
            password = password or self.mt5_password
            server = server or self.mt5_server
            path = path or self.mt5_terminal_path
            
            # Validate required credentials
            if not all([login, password, server]):
                raise ConnectionError(
                    "Missing MT5 credentials. Please set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER "
                    "in your .env file or provide them as parameters."
                )
            
            # Initialize MT5 connection
            if path and Path(path).exists():
                self.logger.info(f"Initializing MT5 with custom path: {path}")
                if not mt5.initialize(path):
                    raise ConnectionError(f"Failed to initialize MT5 with path: {path}")
            else:
                self.logger.info("Initializing MT5 with default path")
                if not mt5.initialize():
                    raise ConnectionError("Failed to initialize MT5")
            
            # Login to account
            self.logger.info(f"Attempting to login to account {login} on server {server}")
            authorized = mt5.login(login, password=password, server=server)
            if not authorized:
                error = mt5.last_error()
                raise ConnectionError(f"Failed to login to MT5: {error}")
            
            # Verify connection and get account info
            self.account_info = self._get_account_info()
            if not self.account_info:
                raise ConnectionError("Failed to retrieve account information")
            
            # Get symbol info
            self.symbol_info = self._get_symbol_info(self.symbol)
            if not self.symbol_info:
                self.logger.warning(f"Symbol {self.symbol} not found, trying alternative symbols")
                # Try alternative symbols for Gold
                alternative_symbols = ['XAUUSDm', 'XAUUSD', 'GOLD', 'Gold', 'XAUUSD.', 'XAUUSDpro']
                for alt_symbol in alternative_symbols:
                    self.symbol_info = self._get_symbol_info(alt_symbol)
                    if self.symbol_info:
                        self.symbol = alt_symbol
                        self.logger.info(f"Using alternative symbol: {self.symbol}")
                        break
                
                if not self.symbol_info:
                    self.logger.error("No valid Gold symbol found")
                    raise ConnectionError("Gold symbol not available")
            
            self.connected = True
            self.logger.info(f"[SUCCESS] Successfully connected to MT5")
            self.logger.info(f"   Account: {self.account_info['login']}")
            self.logger.info(f"   Balance: ${self.account_info['balance']:,.2f}")
            self.logger.info(f"   Server: {self.account_info['server']}")
            self.logger.info(f"   Symbol: {self.symbol}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Connection failed: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """
        Disconnect from MT5 terminal
        
        Example:
            >>> mt5_mgr.disconnect()
        """
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Disconnected from MT5")
    
    def _load_available_symbols(self) -> Dict:
        """
        Load available trading symbols from CSV file
        
        Returns:
            dict: Dictionary of available symbols with their specifications
        
        Private method - used internally
        """
        symbols = {}
        
        # Try to load from CSV if exists
        csv_path = Path("tradable_exness_instruments.csv")
        if csv_path.exists():
            try:
                import csv
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        symbols[row['symbol']] = {
                            'description': row['description'],
                            'min_lot': float(row['min_lot']),
                            'max_lot': float(row['max_lot']),
                            'lot_step': float(row['lot_step']),
                            'spread': int(row['spread']),
                            'digits': int(row['digits']),
                            'contract_size': float(row['contract_size'])
                        }
                self.logger.info(f"Loaded {len(symbols)} tradable symbols from CSV")
            except Exception as e:
                self.logger.warning(f"Could not load symbols from CSV: {e}")
        
        return symbols
    
    def _get_account_info(self) -> Dict:
        """
        Get current account information
        
        Returns:
            dict: Account information including balance, equity, margin, etc.
        
        Private method - used internally
        """
        account_info = mt5.account_info()
        if account_info is None:
            return self._get_account_info()
        
        return {
            'login': account_info.login,
            'server': account_info.server,
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'margin_level': account_info.margin_level,
            'profit': account_info.profit,
            'leverage': account_info.leverage,
            'currency': account_info.currency,
            'trade_allowed': account_info.trade_allowed,
            'trade_expert': account_info.trade_expert,
            'limit_orders': account_info.limit_orders
        }
    
    def _get_symbol_info(self, symbol: str) -> Dict:
        """
        Get symbol specifications
        
        Args:
            symbol (str): Trading symbol
        
        Returns:
            dict: Symbol specifications including pip value, lot sizes, etc.
        
        Private method - used internally
        """
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {}
        
        return {
            'symbol': symbol_info.name,
            'bid': symbol_info.bid,
            'ask': symbol_info.ask,
            'spread': symbol_info.spread,
            'digits': symbol_info.digits,
            'point': symbol_info.point,
            'trade_tick_value': symbol_info.trade_tick_value,
            'trade_tick_size': symbol_info.trade_tick_size,
            'trade_contract_size': symbol_info.trade_contract_size,
            'volume_min': symbol_info.volume_min,
            'volume_max': symbol_info.volume_max,
            'volume_step': symbol_info.volume_step,
            'trade_stops_level': symbol_info.trade_stops_level,
            'trade_freeze_level': symbol_info.trade_freeze_level,
            'margin_initial': symbol_info.margin_initial,
            'margin_maintenance': symbol_info.margin_maintenance,
            'session_deals': symbol_info.session_deals,
            'session_buy_orders': symbol_info.session_buy_orders,
            'session_sell_orders': symbol_info.session_sell_orders,
            'volume': symbol_info.volume,
            'volumehigh': symbol_info.volumehigh,
            'volumelow': symbol_info.volumelow
        }
    
    def get_valid_symbol(self, base_symbol: str = "XAUUSD") -> str:
        """
        Get the valid MT5 symbol name for a given base symbol
        
        Args:
            base_symbol (str): Base symbol name (e.g., 'XAUUSD')
        
        Returns:
            str: Valid MT5 symbol name (e.g., 'XAUUSDm')
        
        Example:
            >>> symbol = mt5_mgr.get_valid_symbol("XAUUSD")
            >>> print(symbol)  # Output: "XAUUSDm"
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
        # Common suffix variations for different brokers
        suffixes = ['m', '', '.', '_m', '.m', 'pro', '.pro', '.raw']
        
        for suffix in suffixes:
            test_symbol = f"{base_symbol}{suffix}"
            symbol_info = mt5.symbol_info(test_symbol)
            if symbol_info is not None:
                self.logger.info(f"Found valid symbol: {test_symbol}")
                return test_symbol
        
        # Check if symbol exists in loaded symbols
        if self.available_symbols:
            for symbol in self.available_symbols.keys():
                if base_symbol.upper() in symbol.upper():
                    self.logger.info(f"Found symbol in available list: {symbol}")
                    return symbol
        
        self.logger.warning(f"No valid symbol found for {base_symbol}")
        return base_symbol

    # ----- Public accessor used by signal engine validation -----
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Public wrapper to retrieve symbol info as a dict for external modules.
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        return self._get_symbol_info(symbol)
    
    def get_all_symbols(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get all available symbols from MT5
        
        Args:
            pattern (str, optional): Filter pattern (e.g., 'USD', 'GOLD', 'XAU')
        
        Returns:
            list: List of available symbol names
        
        Example:
            >>> gold_symbols = mt5_mgr.get_all_symbols("XAU")
            >>> print(gold_symbols)  # ['XAUUSDm', 'XAUEURm', etc.]
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        
        symbol_names = [s.name for s in symbols]
        
        if pattern:
            # Filter symbols by pattern
            filtered = [s for s in symbol_names if pattern.upper() in s.upper()]
            return filtered
        
        return symbol_names
    
    def validate_symbol_tradable(self, symbol: str) -> Dict:
        """
        Validate if a symbol is tradable and get its specifications
        
        Args:
            symbol (str): Symbol to validate
        
        Returns:
            dict: Symbol validation results and specifications
        
        Example:
            >>> info = mt5_mgr.validate_symbol_tradable("XAUUSDm")
            >>> print(f"Tradable: {info['tradable']}, Spread: {info['spread']}")
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
        symbol_info = mt5.symbol_info(symbol)
        
        if symbol_info is None:
            return {
                'tradable': False,
                'reason': 'Symbol not found',
                'symbol': symbol
            }
        
        result = {
            'tradable': symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL,
            'symbol': symbol,
            'bid': symbol_info.bid,
            'ask': symbol_info.ask,
            'spread': symbol_info.spread,
            'min_lot': symbol_info.volume_min,
            'max_lot': symbol_info.volume_max,
            'lot_step': symbol_info.volume_step,
            'digits': symbol_info.digits,
            'contract_size': symbol_info.trade_contract_size,
            'margin_required': symbol_info.margin_initial,
            'swap_long': symbol_info.swap_long,
            'swap_short': symbol_info.swap_short,
            'session_deals': symbol_info.session_deals,
            'session_volume': symbol_info.session_volume
        }
        
        if not result['tradable']:
            result['reason'] = 'Trading disabled for this symbol'
        
        return result
    
    def get_historical_data(self, symbol: str, timeframe: str, 
                           bars: int = 1000, start_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from MT5
        
        Args:
            symbol (str): Trading symbol (e.g., 'XAUUSD')
            timeframe (str): Timeframe (e.g., 'M5', 'H1', 'D1')
            bars (int): Number of bars to fetch (default: 1000)
            start_date (datetime, optional): Start date for historical data
        
        Returns:
            pd.DataFrame: DataFrame with columns [time, open, high, low, close, volume]
        
        Raises:
            ValueError: If invalid timeframe or symbol
        
        Example:
            >>> df = mt5_mgr.get_historical_data("XAUUSDm", "M5", 500)
            >>> print(df.head())
        """
        if not self.connected:
            # Provide mock data when not connected instead of throwing error
            logger.info(f"Not connected to MT5, providing mock data for {symbol} {timeframe}")
            return self._generate_mock_historical_data(symbol, timeframe, bars, start_date)
        
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {timeframe}. Use one of {list(self.TIMEFRAMES.keys())}")
        
        # Select symbol
        if not mt5.symbol_select(symbol, True):
            raise ValueError(f"Failed to select symbol: {symbol}")
        
        # Get the MT5 timeframe constant
        tf = self.TIMEFRAMES[timeframe]
        
        try:
            if start_date:
                # Fetch from specific date
                rates = mt5.copy_rates_from(symbol, tf, start_date, bars)
            else:
                # Fetch most recent bars
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data retrieved for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time from Unix timestamp to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Set time as index
            df.set_index('time', inplace=True)
            
            # Rename columns to standard names
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume',
                'spread': 'Spread',
                'real_volume': 'Real_Volume'
            }, inplace=True)
            
            # Keep only OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            self.logger.info(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
            # Add lowercase aliases to standardize downstream usage
            try:
                df['open'] = df['Open']
                df['high'] = df['High']
                df['low'] = df['Low']
                df['close'] = df['Close']
                df['volume'] = df['Volume']
            except Exception:
                pass
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbol: str) -> Dict:
        """
        Get real-time tick data for a symbol
        
        Args:
            symbol (str): Trading symbol
        
        Returns:
            dict: Current tick data including bid, ask, last price, volume
        
        Example:
            >>> tick = mt5_mgr.get_realtime_data("XAUUSDm")
            >>> print(f"Bid: {tick['bid']}, Ask: {tick['ask']}")
        """
        if not self.connected:
            # Provide mock tick data when not connected
            import numpy as np
            np.random.seed(42)
            
            base_prices = {
                'XAUUSD': 1950.0, 'XAUUSDm': 1950.0, 'GOLD': 1950.0,
                'BTCUSD': 45000.0, 'BTCUSDm': 45000.0,
                'EURUSD': 1.0850, 'EURUSDm': 1.0850
            }
            base_price = base_prices.get(symbol, 1950.0)
            
            # Generate realistic bid/ask spread
            spread = base_price * 0.0001  # 0.01% spread
            mid_price = base_price * np.random.uniform(0.998, 1.002)
            
            return {
                'time': datetime.now(),
                'bid': round(mid_price - spread/2, 5),
                'ask': round(mid_price + spread/2, 5),
                'last': round(mid_price, 5),
                'volume': np.random.randint(100, 1000),
                'time_msc': int(datetime.now().timestamp() * 1000),
                'flags': 0,
                'volume_real': np.random.randint(10, 100)
            }
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {}
        
        return {
            'time': datetime.fromtimestamp(tick.time),
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'time_msc': tick.time_msc,
            'flags': tick.flags,
            'volume_real': tick.volume_real
        }
    
    def place_market_order(self, symbol: str, order_type: str, volume: float,
                          sl: Optional[float] = None, tp: Optional[float] = None,
                          comment: str = "", deviation: int = 20, **kwargs) -> Dict:
        """
        Place a market order
        
        Args:
            symbol (str): Trading symbol
            order_type (str): 'BUY' or 'SELL'
            volume (float): Position size in lots
            sl (float, optional): Stop loss price
            tp (float, optional): Take profit price
            comment (str): Order comment (default: "")
            deviation (int): Maximum price deviation (default: 20)
        
        Returns:
            dict: Order result with ticket number, price, status
        
        Example:
            >>> result = mt5_mgr.place_market_order("XAUUSDm", "BUY", 0.01, 
            ...                                     sl=1950.00, tp=1970.00)
            >>> print(f"Order placed: {result['ticket']}")
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
        if order_type not in ['BUY', 'SELL']:
            raise ValueError("order_type must be 'BUY' or 'SELL'")
        
        # Get symbol info for price
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found")
        
        # Enable symbol for trading
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                raise ValueError(f"Failed to select symbol {symbol}")
        
        # Determine price based on order type
        price = symbol_info.ask if order_type == 'BUY' else symbol_info.bid
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": self.ORDER_TYPES[order_type],
            "price": price,
            "deviation": deviation,
            "magic": self.magic_number,
            "comment": comment or f"XAU_System_{order_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add SL/TP if provided
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            return {"success": False, "error": "Order send returned None"}
        
        # Parse result
        response = {
            "success": result.retcode == mt5.TRADE_RETCODE_DONE,
            "ticket": result.order if result.retcode == mt5.TRADE_RETCODE_DONE else 0,
            "deal": result.deal if hasattr(result, 'deal') else 0,
            "price": result.price if hasattr(result, 'price') else 0,
            "volume": result.volume if hasattr(result, 'volume') else 0,
            "retcode": result.retcode,
            "comment": result.comment if hasattr(result, 'comment') else "",
            "request_id": result.request_id if hasattr(result, 'request_id') else 0
        }
        
        if response["success"]:
            self.logger.info(f"Market order placed successfully: {order_type} {volume} {symbol} "
                       f"at {response['price']}, Ticket: {response['ticket']}")
        else:
            self.logger.error(f"Market order failed: {response['comment']}, RetCode: {response['retcode']}")
        
        return response
    
    def place_pending_order(self, symbol: str, order_type: str, volume: float, 
                           price: float, sl: Optional[float] = None, 
                           tp: Optional[float] = None, comment: str = "") -> Dict:
        """
        Place a pending order (limit or stop)
        
        Args:
            symbol (str): Trading symbol
            order_type (str): 'BUY_LIMIT', 'SELL_LIMIT', 'BUY_STOP', 'SELL_STOP'
            volume (float): Position size in lots
            price (float): Order price
            sl (float, optional): Stop loss price
            tp (float, optional): Take profit price
            comment (str): Order comment
        
        Returns:
            dict: Order result with ticket number and status
        
        Example:
            >>> result = mt5_mgr.place_pending_order("XAUUSDm", "BUY_LIMIT", 
            ...                                      0.01, 1950.00)
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
        if order_type not in ['BUY_LIMIT', 'SELL_LIMIT', 'BUY_STOP', 'SELL_STOP']:
            raise ValueError("Invalid pending order type")
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": self.ORDER_TYPES[order_type],
            "price": price,
            "magic": self.magic_number,
            "comment": comment or f"XAU_Pending_{order_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        # Add SL/TP if provided
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            return {"success": False, "error": "Order send returned None"}
        
        response = {
            "success": result.retcode == mt5.TRADE_RETCODE_DONE,
            "ticket": result.order if result.retcode == mt5.TRADE_RETCODE_DONE else 0,
            "retcode": result.retcode,
            "comment": result.comment if hasattr(result, 'comment') else ""
        }
        
        if response["success"]:
            self.logger.info(f"Pending order placed: {order_type} {volume} {symbol} at {price}")
        else:
            self.logger.error(f"Pending order failed: {response['comment']}")
        
        return response
    
    def modify_position(self, ticket: int, sl: Optional[float] = None, 
                       tp: Optional[float] = None) -> Dict:
        """
        Modify an existing position (update SL/TP)
        
        Args:
            ticket (int): Position ticket number
            sl (float, optional): New stop loss price
            tp (float, optional): New take profit price
        
        Returns:
            dict: Modification result
        
        Example:
            >>> result = mt5_mgr.modify_position(12345678, sl=1950.00, tp=1970.00)
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return {"success": False, "error": f"Position {ticket} not found"}
        
        position = position[0]
        symbol = position.symbol
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "magic": self.magic_number,
        }
        
        # Add new SL/TP
        if sl is not None:
            request["sl"] = sl
        else:
            request["sl"] = position.sl
            
        if tp is not None:
            request["tp"] = tp
        else:
            request["tp"] = position.tp
        
        # Send modification
        result = mt5.order_send(request)
        
        if result is None:
            return {"success": False, "error": "Modification returned None"}
        
        response = {
            "success": result.retcode == mt5.TRADE_RETCODE_DONE,
            "retcode": result.retcode,
            "comment": result.comment if hasattr(result, 'comment') else ""
        }
        
        if response["success"]:
            self.logger.info(f"Position {ticket} modified: SL={sl}, TP={tp}")
        else:
            self.logger.error(f"Position modification failed: {response['comment']}")
        
        return response
    
    def close_position(self, ticket: int, volume: Optional[float] = None) -> Dict:
        """
        Close an existing position
        
        Args:
            ticket (int): Position ticket number
            volume (float, optional): Partial volume to close. If None, closes entire position
        
        Returns:
            dict: Close result
        
        Example:
            >>> result = mt5_mgr.close_position(12345678)
            >>> print(f"Position closed: {result['success']}")
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return {"success": False, "error": f"Position {ticket} not found"}
        
        position = position[0]
        symbol = position.symbol
        
        # Get current price
        symbol_info = mt5.symbol_info(symbol)
        if position.type == mt5.ORDER_TYPE_BUY:
            price = symbol_info.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:
            price = symbol_info.ask
            order_type = mt5.ORDER_TYPE_BUY
        
        # Determine volume
        close_volume = volume if volume else position.volume
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": close_volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": f"Close_{ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send close order
        result = mt5.order_send(request)
        
        if result is None:
            return {"success": False, "error": "Close order returned None"}
        
        response = {
            "success": result.retcode == mt5.TRADE_RETCODE_DONE,
            "deal": result.deal if hasattr(result, 'deal') else 0,
            "price": result.price if hasattr(result, 'price') else 0,
            "retcode": result.retcode,
            "comment": result.comment if hasattr(result, 'comment') else ""
        }
        
        if response["success"]:
            self.logger.info(f"Position {ticket} closed at {response['price']}")
        else:
            self.logger.error(f"Position close failed: {response['comment']}")
        
        return response
    
    def close_all_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Close all open positions
        
        Args:
            symbol (str, optional): Close only positions for specific symbol
        
        Returns:
            list: List of close results for each position
        
        Example:
            >>> results = mt5_mgr.close_all_positions("XAUUSDm")
            >>> print(f"Closed {len(results)} positions")
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
        # Get all positions
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if not positions:
            self.logger.info("No open positions to close")
            return []
        
        results = []
        for position in positions:
            result = self.close_position(position.ticket)
            results.append({
                "ticket": position.ticket,
                "symbol": position.symbol,
                "volume": position.volume,
                "closed": result["success"]
            })
        
        self.logger.info(f"Closed {sum(1 for r in results if r['closed'])} out of {len(results)} positions")
        return results
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get all open positions
        
        Args:
            symbol (str, optional): Filter by symbol
        
        Returns:
            list: List of open positions with details
        
        Example:
            >>> positions = mt5_mgr.get_open_positions("XAUUSDm")
            >>> for pos in positions:
            ...     print(f"Ticket: {pos['ticket']}, P/L: {pos['profit']}")
        """
        if not self.connected:
            # Return empty list when not connected
            return []
        
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if not positions:
            return []
        
        result = []
        for pos in positions:
            result.append({
                'ticket': pos.ticket,
                'time': datetime.fromtimestamp(pos.time),
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'sl': pos.sl,
                'tp': pos.tp,
                'price_current': pos.price_current,
                'profit': pos.profit,
                'swap': pos.swap,
                #'commission': pos.commission,
                'commission': getattr(pos, 'commission', 0.0),
                'comment': pos.comment,
                'magic': pos.magic
            })
        
        return result
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get all pending orders
        
        Args:
            symbol (str, optional): Filter by symbol
        
        Returns:
            list: List of pending orders
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
        if symbol:
            orders = mt5.orders_get(symbol=symbol)
        else:
            orders = mt5.orders_get()
        
        if not orders:
            return []
        
        result = []
        for order in orders:
            result.append({
                'ticket': order.ticket,
                'time_setup': datetime.fromtimestamp(order.time_setup),
                'symbol': order.symbol,
                'type': order.type,
                'volume': order.volume_initial,
                'price': order.price_open,
                'sl': order.sl,
                'tp': order.tp,
                'comment': order.comment,
                'magic': order.magic
            })
        
        return result
    
    def get_account_balance(self) -> float:
        """
        Get current account balance
        
        Returns:
            float: Account balance
        
        Example:
            >>> balance = mt5_mgr.get_account_balance()
            >>> print(f"Balance: ${balance:.2f}")
        """
        if not self.connected:
            # Return mock balance when not connected
            return 1000.0
        
        account_info = mt5.account_info()
        return account_info.balance if account_info else 0.0

    # ----- Data validation helpers -------------------------------------------------
    def validate_bars(self, df: pd.DataFrame, symbol: str, timeframe: str,
                      max_gap_bars: int = 5) -> Dict[str, Union[bool, int, str]]:
        """
        Validate OHLCV bars for gaps and precision consistency.

        Returns a dict with:
          - ok (bool)
          - missing_bars (int)
          - message (str)
        """
        try:
            if df is None or df.empty:
                return {"ok": False, "missing_bars": 0, "message": "Empty dataframe"}

            if not isinstance(df.index, pd.DatetimeIndex):
                return {"ok": False, "missing_bars": 0, "message": "Index is not DatetimeIndex"}

            tf_minutes = {
                'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                'H1': 60, 'H4': 240, 'D1': 1440
            }.get(timeframe, 15)

            expected_delta = pd.Timedelta(minutes=tf_minutes)
            gaps = df.index.to_series().diff().dropna()
            missing_bars = int(sum(max(int((g / expected_delta)) - 1, 0) for g in gaps))

            ok = missing_bars <= max_gap_bars
            msg = f"Missing bars: {missing_bars}" if missing_bars else "OK"
            return {"ok": ok, "missing_bars": missing_bars, "message": msg}
        except Exception as e:
            return {"ok": False, "missing_bars": 0, "message": f"Validation error: {str(e)}"}
    
    def get_account_equity(self) -> float:
        """
        Get current account equity
        
        Returns:
            float: Account equity
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
        account_info = mt5.account_info()
        return account_info.equity if account_info else 0.0
    
    def get_account_margin(self) -> Dict:
        """
        Get account margin information
        
        Returns:
            dict: Margin details including used, free, and level
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
        account_info = mt5.account_info()
        if not account_info:
            return {}
        
        return {
            'margin': account_info.margin,
            'margin_free': account_info.margin_free,
            'margin_level': account_info.margin_level,
            'margin_so_call': account_info.margin_so_call,
            'margin_so_so': account_info.margin_so_so
        }
    
    def calculate_position_size(self, symbol: str, risk_amount: float, 
                               stop_loss_pips: float) -> float:
        """
        Calculate position size based on risk
        
        Args:
            symbol (str): Trading symbol
            risk_amount (float): Amount to risk in account currency
            stop_loss_pips (float): Stop loss distance in pips
        
        Returns:
            float: Position size in lots
        
        Example:
            >>> size = mt5_mgr.calculate_position_size("XAUUSDm", 100, 50)
            >>> print(f"Position size: {size} lots")
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return 0.0
        
        # Calculate pip value
        pip_value = symbol_info.trade_tick_value
        
        # Calculate position size
        if stop_loss_pips > 0 and pip_value > 0:
            position_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Round to valid lot size
            lot_step = symbol_info.volume_step
            position_size = round(position_size / lot_step) * lot_step
            
            # Apply min/max constraints
            position_size = max(symbol_info.volume_min, 
                              min(position_size, symbol_info.volume_max))
            
            return position_size
        
        return symbol_info.volume_min
    
    def _generate_mock_historical_data(self, symbol: str, timeframe: str, 
                                     bars: int, start_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate mock historical data when MT5 is not connected
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            bars (int): Number of bars to generate
            start_date (datetime, optional): Start date for data
        
        Returns:
            pd.DataFrame: Mock OHLCV data
        """
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Determine timeframe in minutes
        tf_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440
        }.get(timeframe, 15)
        
        # Generate date range
        if start_date:
            end_date = start_date + timedelta(minutes=tf_minutes * bars)
            dates = pd.date_range(start=start_date, end=end_date, freq=f'{tf_minutes}min')[:bars]
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(minutes=tf_minutes * bars)
            dates = pd.date_range(start=start_date, end=end_date, freq=f'{tf_minutes}min')[:bars]
        
        # Set random seed for consistent data
        np.random.seed(42)
        
        # Base prices for different symbols
        base_prices = {
            'XAUUSD': 1950.0, 'XAUUSDm': 1950.0, 'GOLD': 1950.0,
            'BTCUSD': 45000.0, 'BTCUSDm': 45000.0,
            'EURUSD': 1.0850, 'EURUSDm': 1.0850,
            'GBPUSD': 1.2650, 'GBPUSDm': 1.2650
        }
        base_price = base_prices.get(symbol, 1950.0)
        
        # Generate realistic price series with trend and volatility
        returns = np.random.normal(0, 0.002, len(dates))  # 0.2% daily volatility
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, price_series)):
            # Generate realistic OHLC from close price
            volatility = abs(np.random.normal(0, 0.001)) + 0.0005  # Min 0.05% volatility
            
            high = close * (1 + volatility * np.random.uniform(0.3, 1.0))
            low = close * (1 - volatility * np.random.uniform(0.3, 1.0))
            
            # Open is influenced by previous close
            if i == 0:
                open_price = close * np.random.uniform(0.999, 1.001)
            else:
                prev_close = data[-1]['Close']
                gap = np.random.normal(0, 0.0002)  # Small gap
                open_price = prev_close * (1 + gap)
            
            # Ensure OHLC logic is correct
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume
            base_volume = 1000
            volume = int(base_volume * np.random.uniform(0.5, 2.0))
            
            data.append({
                'Open': round(open_price, 3),
                'High': round(high, 3),
                'Low': round(low, 3),
                'Close': round(close, 3),
                'Volume': volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(data, index=dates)
        
        # Add lowercase aliases for compatibility
        df['open'] = df['Open']
        df['high'] = df['High']
        df['low'] = df['Low']
        df['close'] = df['Close']
        df['volume'] = df['Volume']
        
        return df
    
    def get_trade_history(self, days: int = 30, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Get trade history for specified period
        
        Args:
            days (int): Number of days to look back (default: 30)
            symbol (str, optional): Filter by symbol
        
        Returns:
            pd.DataFrame: Trade history with details
        
        Example:
            >>> history = mt5_mgr.get_trade_history(7, "XAUUSDm")
            >>> print(f"Total trades: {len(history)}")
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # Get deals
        if symbol:
            deals = mt5.history_deals_get(from_date, to_date, group=symbol)
        else:
            deals = mt5.history_deals_get(from_date, to_date)
        
        if not deals:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
        
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Add readable type
        df['type_str'] = df['type'].apply(lambda x: 'BUY' if x == mt5.DEAL_TYPE_BUY else 'SELL')
        
        return df
    
    def check_environment_variables(self) -> Dict:
        """
        Check if all required environment variables are set
        
        Returns:
            dict: Status of each environment variable
        
        Example:
            >>> status = mt5_mgr.check_environment_variables()
            >>> print(status)
        """
        env_vars = {
            'MT5_LOGIN': self.mt5_login,
            'MT5_PASSWORD': self.mt5_password,
            'MT5_SERVER': self.mt5_server,
            'MT5_TERMINAL_PATH': self.mt5_terminal_path
        }
        
        status = {}
        for var, value in env_vars.items():
            if var == 'MT5_TERMINAL_PATH':  # Optional
                status[var] = {
                    'set': value is not None and value != '',
                    'required': False,
                    'value': '***HIDDEN***' if value else None
                }
            else:  # Required
                status[var] = {
                    'set': value is not None and value != '',
                    'required': True,
                    'value': '***HIDDEN***' if value else None
                }
        
        all_required_set = all(
            status[var]['set'] for var in ['MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER']
        )
        
        status['all_required_set'] = all_required_set
        
        return status
    
    def test_connection(self) -> bool:
        """
        Test MT5 connection and display account info
        
        Returns:
            bool: True if connection test successful
        
        Example:
            >>> mt5_mgr.test_connection()
            True
        """
        try:
            print("\n" + "="*60)
            print("MT5 CONNECTION TEST")
            print("="*60)
            
            # Check environment variables first
            print("\n🔧 ENVIRONMENT VARIABLES:")
            env_status = self.check_environment_variables()
            
            for var, info in env_status.items():
                if var == 'all_required_set':
                    continue
                status_icon = "✅" if info['set'] else "❌"
                required_text = "(Required)" if info['required'] else "(Optional)"
                print(f"{status_icon} {var}: {'SET' if info['set'] else 'NOT SET'} {required_text}")
            
            if not env_status['all_required_set']:
                print("\n❌ Missing required environment variables!")
                print("Please set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER in your .env file")
                return False
            
            # Attempt connection
            print(f"\n🔌 CONNECTING TO MT5...")
            if not self.connected:
                if not self.connect():
                    return False
            
            # Account info
            print("\n📊 ACCOUNT INFORMATION:")
            print(f"Login: {self.account_info['login']}")
            print(f"Server: {self.account_info['server']}")
            print(f"Balance: ${self.account_info['balance']:,.2f}")
            print(f"Equity: ${self.account_info['equity']:,.2f}")
            print(f"Free Margin: ${self.account_info['free_margin']:,.2f}")
            print(f"Leverage: 1:{self.account_info['leverage']}")
            print(f"Currency: {self.account_info['currency']}")
            print(f"Trading Allowed: {'Yes' if self.account_info['trade_allowed'] else 'No'}")
            
            # Symbol info
            print(f"\n📈 SYMBOL INFORMATION ({self.symbol}):")
            tick = self.get_realtime_data(self.symbol)
            if tick:
                print(f"Bid: {tick['bid']}")
                print(f"Ask: {tick['ask']}")
                print(f"Spread: {self.symbol_info['spread']} points")
                print(f"Min Lot: {self.symbol_info['volume_min']}")
                print(f"Max Lot: {self.symbol_info['volume_max']}")
                print(f"Lot Step: {self.symbol_info['volume_step']}")
            
            # Test data fetch
            print("\n📊 DATA FETCH TEST:")
            df = self.get_historical_data(self.symbol, "M5", 10)
            if not df.empty:
                print(f"✅ Successfully fetched {len(df)} bars")
                print(f"Latest bar time: {df.index[-1]}")
                print(f"Close price: {df['Close'].iloc[-1]}")
                print(f"Date range: {df.index[0]} to {df.index[-1]}")
            else:
                print("❌ Failed to fetch historical data")
            
            # Check open positions
            positions = self.get_open_positions()
            print(f"\n💼 OPEN POSITIONS: {len(positions)}")
            if positions:
                for pos in positions[:3]:  # Show first 3 positions
                    print(f"  Ticket: {pos['ticket']}, {pos['type']} {pos['volume']} {pos['symbol']}, "
                          f"P/L: ${pos['profit']:.2f}")
                if len(positions) > 3:
                    print(f"  ... and {len(positions) - 3} more")
            
            print("\n✅ CONNECTION TEST SUCCESSFUL!")
            print("="*60)
            return True
            
        except Exception as e:
            print(f"\n❌ Connection test failed: {str(e)}")
            print("="*60)
            return False


# Command-line interface for testing
    
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Import CLI utilities
    from src.utils.cli_args import parse_mode, print_mode_banner
    
    # Parse CLI arguments and mode
    mode = parse_mode()
    print_mode_banner(mode)
    
    import argparse
    
    parser = argparse.ArgumentParser(description='MT5 Manager CLI')
    parser.add_argument('--mode', choices=['mock', 'live'], default=mode, help='Execution mode')
    parser.add_argument('--test', action='store_true', help='Test MT5 connection')
    parser.add_argument('--fetch-history', action='store_true', help='Fetch historical data')
    parser.add_argument('--account-info', action='store_true', help='Display account info')
    parser.add_argument('--check-env', action='store_true', help='Check environment variables')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe for data')
    parser.add_argument('--bars', type=int, default=100, help='Number of bars to fetch')
    
    args = parser.parse_args()
    
    # Create manager based on mode
    if args.mode == 'mock':
        print("📝 Running MT5 Manager in MOCK mode")
        print("⚠️  No real MT5 connection will be made")
        # Create manager but don't attempt connection
        mt5_mgr = MT5Manager(symbol=args.symbol)
        mt5_mgr.connected = False  # Explicitly set to mock mode
    else:
        print("🚀 Running MT5 Manager in LIVE mode")
        mt5_mgr = MT5Manager(symbol=args.symbol)
    
    if args.test:
        if args.mode == 'mock':
            print("📝 MOCK MODE - Simulating MT5 connection test")
            print("\n" + "="*60)
            print("MT5 CONNECTION TEST (MOCK MODE)")
            print("="*60)
            print("📝 Environment: Mock/Test")
            print("✅ Mock connection: SUCCESS")
            print("💰 Mock Balance: $1,000.00")
            print("📈 Mock Symbol: XAUUSDm")
            print("✅ Mock data fetch: SUCCESS")
            print("="*60)
        else:
            mt5_mgr.test_connection()
    
    elif args.check_env:
        print("\n" + "="*50)
        print("ENVIRONMENT VARIABLES CHECK")
        print("="*50)
        status = mt5_mgr.check_environment_variables()
        
        for var, info in status.items():
            if var == 'all_required_set':
                continue
            status_icon = "✅" if info['set'] else "❌"
            required_text = "(Required)" if info['required'] else "(Optional)"
            print(f"{status_icon} {var}: {'SET' if info['set'] else 'NOT SET'} {required_text}")
        
        print(f"\n{'✅' if status['all_required_set'] else '❌'} All required variables: "
              f"{'SET' if status['all_required_set'] else 'NOT SET'}")
        
        if not status['all_required_set']:
            print("\nTo fix this, create a .env file with:")
            print("MT5_LOGIN=your_account_number")
            print("MT5_PASSWORD=your_password") 
            print("MT5_SERVER=your_broker_server")
            print("MT5_TERMINAL_PATH=path_to_terminal64.exe  # Optional")
    
    elif args.fetch_history:
        if args.mode == 'mock':
            print("📝 MOCK MODE - Simulating historical data fetch")
            print(f"Mock data for {args.symbol} {args.timeframe} - {args.bars} bars")
            print("\nMock OHLCV Data:")
            print("Time                 Open     High     Low      Close    Volume")
            print("-" * 65)
            for i in range(5):
                price = 1950 + i * 2
                print(f"2025-08-19 {10+i:02d}:00:00  {price:.2f}   {price+5:.2f}   {price-3:.2f}   {price+2:.2f}   1500")
        elif mt5_mgr.connect():
            # Get valid symbol
            valid_symbol = mt5_mgr.get_valid_symbol(args.symbol)
            print(f"Using symbol: {valid_symbol}")
            
            df = mt5_mgr.get_historical_data(valid_symbol, args.timeframe, args.bars)
            if not df.empty:
                print(f"\nFetched {len(df)} bars for {valid_symbol} {args.timeframe}")
                print(f"Date range: {df.index[0]} to {df.index[-1]}")
                print("\nLatest 5 bars:")
                print(df.tail().round(2))
            else:
                print("❌ No data retrieved")
            mt5_mgr.disconnect()
    
    elif args.account_info:
        if args.mode == 'mock':
            print("\n" + "="*40)
            print("ACCOUNT INFORMATION (MOCK)")
            print("="*40)
            print("Login: 12345678")
            print("Server: MockServer")
            print("Balance: 1,000.00")
            print("Equity: 1,000.00")
            print("Free Margin: 1,000.00")
            print("Leverage: 1:500")
            print("Currency: USD")
            print("Trading Allowed: Yes")
        elif mt5_mgr.connect():
            info = mt5_mgr._get_account_info()
            print("\n" + "="*40)
            print("ACCOUNT INFORMATION")
            print("="*40)
            for key, value in info.items():
                if isinstance(value, float):
                    print(f"{key.replace('_', ' ').title()}: {value:,.2f}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")
            mt5_mgr.disconnect()
    
    else:
        if args.mode == 'mock':
            print("📝 Mock mode initialized successfully")
            print("Use --test, --fetch-history, or --account-info with mock data")
        parser.print_help()