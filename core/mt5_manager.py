
"""
MT5 Manager - Core MetaTrader 5 Integration Module
==================================================
Author: XAUUSD Trading System
Version: 1.0.0
Date: 2025-01-07

This module handles all interactions with MetaTrader 5:
- Connection management
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
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
import time
import json
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MT5Manager:
    """
    Comprehensive MT5 Manager for XAUUSD Trading System
    
    This class provides a complete interface to MetaTrader 5, handling:
    - Platform initialization and connection
    - Data retrieval (historical and real-time)
    - Order management (market, pending, modify, close)
    - Account information
    - Symbol specifications
    
    Attributes:
        config (dict): Configuration dictionary from master_config.yaml
        connected (bool): Connection status to MT5
        symbol (str): Trading symbol (default: XAUUSD)
        account_info (dict): Current account information
        symbol_info (dict): Symbol specifications
    
    Example:
        >>> mt5_mgr = MT5Manager(config)
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
    
    def __init__(self, config: dict):
        """
        Initialize MT5 Manager with configuration
        
        Args:
            config (dict): Configuration dictionary containing MT5 settings
        """
        self.config = config
        self.mt5_config = config.get('mt5', {})
        self.connected = False
        self.symbol = config.get('trading', {}).get('symbol', 'XAUUSDm')  # Updated default symbol
        self.account_info = {}
        self.symbol_info = {}
        self.magic_number = self.mt5_config.get('magic_number', 123456)
        
        # Load available symbols if provided
        self.available_symbols = self._load_available_symbols()
        
        logger.info(f"MT5Manager initialized for symbol: {self.symbol}")
    
    def connect(self, login: Optional[int] = None, password: Optional[str] = None, 
                server: Optional[str] = None, path: Optional[str] = None) -> bool:
        """
        Establish connection to MT5 terminal
        
        Args:
            login (int, optional): Account login. Uses config if not provided
            password (str, optional): Account password. Uses config if not provided
            server (str, optional): Broker server. Uses config if not provided
            path (str, optional): Path to MT5 terminal. Uses config if not provided
        
        Returns:
            bool: True if connection successful, False otherwise
        
        Raises:
            ConnectionError: If unable to connect to MT5
        
        Example:
            >>> mt5_mgr.connect()
            True
        """
        try:
            # Use provided credentials or fall back to config
            login = login or self.mt5_config.get('login')
            password = password or self.mt5_config.get('password')
            server = server or self.mt5_config.get('server')
            path = path or self.mt5_config.get('terminal_path')
            
            # Initialize MT5 connection
            if path:
                if not mt5.initialize(path):
                    raise ConnectionError(f"Failed to initialize MT5 with path: {path}")
            else:
                if not mt5.initialize():
                    raise ConnectionError("Failed to initialize MT5")
            
            # Login to account if credentials provided
            if login and password and server:
                authorized = mt5.login(login, password=password, server=server)
                if not authorized:
                    error = mt5.last_error()
                    raise ConnectionError(f"Failed to login: {error}")
            
            # Verify connection and get account info
            self.account_info = self._get_account_info()
            if not self.account_info:
                raise ConnectionError("Failed to retrieve account information")
            
            # Get symbol info
            self.symbol_info = self._get_symbol_info(self.symbol)
            if not self.symbol_info:
                logger.warning(f"Symbol {self.symbol} not found, trying alternative symbols")
                # Try alternative symbols for Gold
                alternative_symbols = ['XAUUSDm', 'XAUUSD', 'GOLD', 'Gold']
                for alt_symbol in alternative_symbols:
                    self.symbol_info = self._get_symbol_info(alt_symbol)
                    if self.symbol_info:
                        self.symbol = alt_symbol
                        logger.info(f"Using alternative symbol: {self.symbol}")
                        break
                
                if not self.symbol_info:
                    logger.error("No valid Gold symbol found")
                    raise ConnectionError("Gold symbol not available")
            
            self.connected = True
            logger.info(f"Successfully connected to MT5. Account: {self.account_info['login']}, "
                       f"Balance: {self.account_info['balance']}, Server: {self.account_info['server']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
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
            logger.info("Disconnected from MT5")
    
    def _load_available_symbols(self) -> Dict:
        """
        Load available trading symbols from configuration or CSV
        
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
                logger.info(f"Loaded {len(symbols)} tradable symbols from CSV")
            except Exception as e:
                logger.warning(f"Could not load symbols from CSV: {e}")
        
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
            return {}
        
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
        suffixes = ['m', '', '.', '_m', '.m', 'pro', '.pro']
        
        for suffix in suffixes:
            test_symbol = f"{base_symbol}{suffix}"
            symbol_info = mt5.symbol_info(test_symbol)
            if symbol_info is not None:
                logger.info(f"Found valid symbol: {test_symbol}")
                return test_symbol
        
        # Check if symbol exists in loaded symbols
        if self.available_symbols:
            for symbol in self.available_symbols.keys():
                if base_symbol.upper() in symbol.upper():
                    logger.info(f"Found symbol in available list: {symbol}")
                    return symbol
        
        logger.warning(f"No valid symbol found for {base_symbol}")
        return base_symbol
    
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
            ConnectionError: If not connected to MT5
        
        Example:
            >>> df = mt5_mgr.get_historical_data("XAUUSDm", "M5", 500)
            >>> print(df.head())
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
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
            
            logger.info(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
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
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
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
                          comment: str = "", deviation: int = 20) -> Dict:
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
            logger.info(f"Market order placed successfully: {order_type} {volume} {symbol} "
                       f"at {response['price']}, Ticket: {response['ticket']}")
        else:
            logger.error(f"Market order failed: {response['comment']}, RetCode: {response['retcode']}")
        
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
            logger.info(f"Pending order placed: {order_type} {volume} {symbol} at {price}")
        else:
            logger.error(f"Pending order failed: {response['comment']}")
        
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
            logger.info(f"Position {ticket} modified: SL={sl}, TP={tp}")
        else:
            logger.error(f"Position modification failed: {response['comment']}")
        
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
            logger.info(f"Position {ticket} closed at {response['price']}")
        else:
            logger.error(f"Position close failed: {response['comment']}")
        
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
            logger.info("No open positions to close")
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
        
        logger.info(f"Closed {sum(1 for r in results if r['closed'])} out of {len(results)} positions")
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
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
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
                'commission': pos.commission,
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
            raise ConnectionError("Not connected to MT5. Call connect() first.")
        
        account_info = mt5.account_info()
        return account_info.balance if account_info else 0.0
    
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
            if not self.connected:
                logger.info("Not connected. Attempting to connect...")
                if not self.connect():
                    return False
            
            print("\n" + "="*50)
            print("MT5 CONNECTION TEST")
            print("="*50)
            
            # Account info
            print("\n📊 ACCOUNT INFORMATION:")
            print(f"Login: {self.account_info['login']}")
            print(f"Server: {self.account_info['server']}")
            print(f"Balance: ${self.account_info['balance']:.2f}")
            print(f"Equity: ${self.account_info['equity']:.2f}")
            print(f"Leverage: 1:{self.account_info['leverage']}")
            print(f"Currency: {self.account_info['currency']}")
            
            # Symbol info
            print(f"\n📈 SYMBOL INFORMATION ({self.symbol}):")
            tick = self.get_realtime_data(self.symbol)
            if tick:
                print(f"Bid: {tick['bid']}")
                print(f"Ask: {tick['ask']}")
                print(f"Spread: {self.symbol_info['spread']} points")
            
            # Test data fetch
            print("\n📊 DATA FETCH TEST:")
            df = self.get_historical_data(self.symbol, "M5", 10)
            if not df.empty:
                print(f"Successfully fetched {len(df)} bars")
                print(f"Latest bar: {df.index[-1]}")
                print(f"Close price: {df['Close'].iloc[-1]}")
            
            print("\n✅ Connection test successful!")
            print("="*50)
            return True
            
        except Exception as e:
            print(f"\n❌ Connection test failed: {str(e)}")
            return False


# Command-line interface for testing
if __name__ == "__main__":
    """
    Command-line interface for MT5Manager testing
    
    Usage:
        python mt5_manager.py --test              # Test connection
        python mt5_manager.py --fetch-history     # Fetch historical data
        python mt5_manager.py --account-info      # Display account info
    """
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='MT5 Manager CLI')
    parser.add_argument('--test', action='store_true', help='Test MT5 connection')
    parser.add_argument('--fetch-history', action='store_true', help='Fetch historical data')
    parser.add_argument('--account-info', action='store_true', help='Display account info')
    parser.add_argument('--config', type=str, default='config/master_config.yaml', 
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {config_path}")
        config = {}
    
    # Create manager
    mt5_mgr = MT5Manager(config)
    
    if args.test:
        mt5_mgr.test_connection()
    
    elif args.fetch_history:
        if mt5_mgr.connect():
            # First validate the symbol
            symbol = mt5_mgr.get_valid_symbol("XAUUSD")
            print(f"Using symbol: {symbol}")
            df = mt5_mgr.get_historical_data(symbol, "H1", 100)
            print(f"Fetched {len(df)} bars")
            print(df.tail())
            mt5_mgr.disconnect()
    
    elif args.account_info:
        if mt5_mgr.connect():
            info = mt5_mgr._get_account_info()
            for key, value in info.items():
                print(f"{key}: {value}")
            mt5_mgr.disconnect()
    
    else:
        parser.print_help()