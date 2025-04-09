#!/usr/bin/env python3
import os
import logging
import time
import signal
import schedule
import argparse
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from modules.binance_client import BinanceClient
from modules.risk_manager import RiskManager
from modules.strategies import get_strategy
from modules.backtest import Backtester
from modules.websocket_handler import BinanceWebSocketManager
from modules.config import (
    TRADING_SYMBOL, TIMEFRAME, STRATEGY, LOG_LEVEL,
    USE_TELEGRAM, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    SEND_DAILY_REPORT, DAILY_REPORT_TIME, AUTO_COMPOUND,
    # Add these to your config.py:
    # BACKTEST_BEFORE_LIVE = True
    # BACKTEST_MIN_PROFIT_PCT = 5.0
    # BACKTEST_MIN_WIN_RATE = 50.0
    # BACKTEST_PERIOD = "30 days"
)

# Try to import the new config variables, use defaults if not available
try:
    from modules.config import (
        BACKTEST_BEFORE_LIVE, 
        BACKTEST_MIN_PROFIT_PCT, 
        BACKTEST_MIN_WIN_RATE,
        BACKTEST_PERIOD
    )
except ImportError:
    # Default values if not in config
    BACKTEST_BEFORE_LIVE = True
    BACKTEST_MIN_PROFIT_PCT = 5.0
    BACKTEST_MIN_WIN_RATE = 50.0
    BACKTEST_PERIOD = "30 days"

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
running = True
binance_client = None
risk_manager = None
strategy = None
websocket_manager = None
klines_data = {}
new_candle_received = {}
stats = {
    'total_trades': 0,
    'winning_trades': 0,
    'losing_trades': 0,
    'total_profit': 0,
    'start_balance': 0,
    'current_balance': 0,
    'daily_profit': 0,
    'last_trade_time': None,
    'last_report_time': None
}

class TelegramNotifier:
    def __init__(self):
        self.enabled = USE_TELEGRAM and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
        
    def send_message(self, message):
        """Send message to Telegram"""
        if not self.enabled:
            return
            
        try:
            import requests
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            
            # Escape Markdown special characters to avoid parsing errors
            # Replace characters that could cause Markdown parsing errors
            for char in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
                message = message.replace(char, '\\' + char)
                
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "MarkdownV2"
            }
            response = requests.post(url, json=payload)
            if response.status_code != 200:
                logger.error(f"Failed to send Telegram message: {response.text}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False
            
    def send_photo(self, photo_path, caption=None):
        """Send photo to Telegram with optional caption"""
        if not self.enabled:
            return
            
        try:
            import requests
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            
            files = {'photo': open(photo_path, 'rb')}
            data = {"chat_id": TELEGRAM_CHAT_ID}
            
            if caption:
                # Escape Markdown special characters
                for char in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
                    caption = caption.replace(char, '\\' + char)
                data["caption"] = caption
                data["parse_mode"] = "MarkdownV2"
                
            response = requests.post(url, files=files, data=data)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram photo: {e}")
            return False
            
    def send_plain_message(self, message):
        """Send message to Telegram without markdown parsing"""
        if not self.enabled:
            return
            
        try:
            import requests
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message
            }
            response = requests.post(url, json=payload)
            if response.status_code != 200:
                logger.error(f"Failed to send Telegram message: {response.text}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False


def setup():
    """Initialize the trading bot"""
    global binance_client, risk_manager, strategy, stats, websocket_manager
    
    logger.info("Setting up trading bot...")
    
    # Create necessary directories
    state_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state')
    os.makedirs(state_dir, exist_ok=True)
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Initialize Binance client
    try:
        binance_client = BinanceClient()
        logger.info("Binance client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Binance client: {e}")
        exit(1)
    
    # Initialize risk manager
    risk_manager = RiskManager(binance_client)
    
    # Get the selected trading strategy
    strategy = get_strategy(STRATEGY)
    logger.info(f"Using trading strategy: {strategy.strategy_name}")
    
    # Initialize futures settings for the trading symbol
    try:
        binance_client.initialize_futures(TRADING_SYMBOL)
    except Exception as e:
        logger.error(f"Failed to initialize futures: {e}")
        exit(1)
    
    # Initialize WebSocket manager
    websocket_manager = BinanceWebSocketManager()
    websocket_manager.add_symbol(TRADING_SYMBOL)
    
    # Register WebSocket callbacks
    websocket_manager.register_callback('kline', on_kline_closed)
    websocket_manager.register_callback('kline_update', on_kline_update)
    websocket_manager.register_callback('book_ticker', on_book_ticker)
    websocket_manager.register_callback('account_update', on_account_update)
    websocket_manager.register_callback('order_update', on_order_update)
    websocket_manager.register_callback('trade', on_trade)
    
    # Start WebSocket connections
    websocket_manager.start()
    logger.info("WebSocket connections started")
    
    # Initialize klines_data with initial data from REST API 
    initialize_klines_data()
    
    # Record starting balance - try multiple times if needed
    for attempt in range(3):
        try:
            account_balance = binance_client.get_account_balance()
            if account_balance > 0:
                stats['start_balance'] = account_balance
                stats['current_balance'] = account_balance
                logger.info(f"Starting balance: {stats['start_balance']} USDT")
                break
            else:
                logger.warning(f"Got zero balance on attempt {attempt+1}, retrying...")
                time.sleep(2)
        except Exception as e:
            logger.error(f"Error getting balance on attempt {attempt+1}: {e}")
            time.sleep(2)
            
    # If we still have zero balance, try loading from config
    if stats['start_balance'] <= 0:
        # Use the initial balance from config as fallback
        stats['start_balance'] = INITIAL_BALANCE
        stats['current_balance'] = INITIAL_BALANCE
        logger.warning(f"Failed to get balance from API, using configured balance: {INITIAL_BALANCE} USDT")
    
    stats['last_report_time'] = datetime.now()
    
    # Set new_candle_received flag to False initially
    new_candle_received[TRADING_SYMBOL] = False
    
    # Send startup notification
    notifier = TelegramNotifier()
    notifier.send_message(f"🤖 *Trading Bot Started*\n"
                         f"Symbol: {TRADING_SYMBOL}\n"
                         f"Strategy: {strategy.strategy_name}\n"
                         f"Starting Balance: {stats['start_balance']} USDT\n"
                         f"Auto-Compound: {'Enabled' if AUTO_COMPOUND else 'Disabled'}\n"
                         f"Using WebSocket for real-time data!")


def initialize_state_file(force=False):
    """Initialize or repair the state file with proper balance values"""
    global stats
    
    state_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state')
    os.makedirs(state_dir, exist_ok=True)
    
    state_file = os.path.join(state_dir, 'trading_state.json')
    
    # Only initialize if the file doesn't exist or force=True
    if not os.path.exists(state_file) or force:
        logger.info("Initializing state file with current settings")
        
        # Try to get current balance from API
        current_balance = 0.0
        if binance_client:
            try:
                current_balance = binance_client.get_account_balance()
                logger.info(f"Got balance from API: {current_balance} USDT")
            except Exception as e:
                logger.error(f"Could not get balance from API: {e}")
        
        # If API balance failed, use the INITIAL_BALANCE from config
        if current_balance <= 0:
            try:
                from modules.config import INITIAL_BALANCE
                current_balance = INITIAL_BALANCE
                logger.info(f"Using configured initial balance: {INITIAL_BALANCE} USDT")
            except ImportError:
                current_balance = 50.0  # Default fallback
                logger.info(f"Using default initial balance: 50.0 USDT")
        
        # Update stats
        stats['start_balance'] = current_balance
        stats['current_balance'] = current_balance
        stats['last_report_time'] = datetime.now()
        
        # Save to file
        json_stats = stats.copy()
        if json_stats['last_report_time']:
            json_stats['last_report_time'] = json_stats['last_report_time'].isoformat()
        
        with open(state_file, 'w') as f:
            json.dump(json_stats, f)
            
        logger.info(f"State file initialized with balance: {current_balance} USDT")
        return True
    
    return False


def initialize_klines_data():
    """Initialize klines data with historical data from REST API"""
    global klines_data
    
    try:
        klines = binance_client.get_historical_klines(
            symbol=TRADING_SYMBOL,
            interval=TIMEFRAME,
            start_str="2 days ago",
            limit=200
        )
        
        if not klines or len(klines) < 30:
            logger.warning(f"Not enough historical data to initialize (got {len(klines) if klines else 0} candles)")
            return
            
        klines_data[TRADING_SYMBOL] = klines
        logger.info(f"Initialized historical data with {len(klines)} candles")
    except Exception as e:
        logger.error(f"Error initializing klines data: {e}")


def on_kline_closed(symbol, kline_data):
    """Callback for when a kline (candlestick) closes"""
    global klines_data, new_candle_received
    
    logger.info(f"Kline closed for {symbol}: {kline_data['close_time']}")
    
    try:
        candle = [
            kline_data['open_time'],
            str(kline_data['open']),
            str(kline_data['high']),
            str(kline_data['low']),
            str(kline_data['close']),
            str(kline_data['volume']),
            kline_data['close_time'],
            "0",
            "0",
            "0",
            "0",
            "0"
        ]
        
        if symbol in klines_data:
            klines_data[symbol].append(candle)
            if len(klines_data[symbol]) > 200:
                klines_data[symbol].pop(0)
        else:
            klines_data[symbol] = [candle]
            
        new_candle_received[symbol] = True
        check_for_signals(symbol)
        
    except Exception as e:
        logger.error(f"Error processing kline close: {e}")


def on_kline_update(symbol, kline_data):
    """Callback for real-time kline updates (includes unclosed candles)"""
    # Format for easy reading - real-time price updates with colored output
    price = kline_data['close']
    formatted_time = datetime.fromtimestamp(kline_data['close_time']/1000).strftime('%H:%M:%S')
    
    # Calculate price change percentage from open
    price_change = ((kline_data['close'] - kline_data['open']) / kline_data['open']) * 100
    direction = "▲" if price_change >= 0 else "▼"
    
    # Log real-time price updates
    logger.info(f"📊 {symbol} | Price: {price:.2f} | {direction} {abs(price_change):.2f}% | O: {kline_data['open']:.2f} H: {kline_data['high']:.2f} L: {kline_data['low']:.2f} | Vol: {kline_data['volume']:.2f} | {formatted_time}")
    
    # Store data for later use
    global klines_data
    if symbol in klines_data and klines_data[symbol]:
        # Update the latest candle with real-time data until it's closed
        if not kline_data['is_closed'] and len(klines_data[symbol]) > 0:
            candle = [
                kline_data['open_time'],
                str(kline_data['open']),
                str(kline_data['high']),
                str(kline_data['low']),
                str(kline_data['close']),
                str(kline_data['volume']),
                kline_data['close_time'],
                "0",
                "0",
                "0",
                "0",
                "0"
            ]
            klines_data[symbol][-1] = candle


def on_book_ticker(symbol, ticker_data):
    """Callback for book ticker updates (best bid/ask)"""
    # Only log significant changes to avoid flooding
    # Store last values to track changes
    if not hasattr(on_book_ticker, 'last_values'):
        on_book_ticker.last_values = {}
    
    # Store last values per symbol
    last = on_book_ticker.last_values.get(symbol, {})
    bid_price = ticker_data['bid_price']
    ask_price = ticker_data['ask_price']
    
    # Only log if price changed significantly (0.05% or more)
    significant_change = False
    if symbol not in on_book_ticker.last_values:
        significant_change = True
    else:
        last_bid = last.get('bid_price', 0)
        last_ask = last.get('ask_price', 0)
        
        if last_bid > 0 and last_ask > 0:
            bid_change_pct = abs((bid_price - last_bid) / last_bid * 100)
            ask_change_pct = abs((ask_price - last_ask) / last_ask * 100)
            if bid_change_pct >= 0.05 or ask_change_pct >= 0.05:
                significant_change = True
    
    # Update last values
    on_book_ticker.last_values[symbol] = ticker_data
    
    if significant_change:
        spread = ask_price - bid_price
        spread_pct = (spread / ask_price) * 100
        
        logger.info(f"💹 {symbol} | Bid: {bid_price:.2f} ({ticker_data['bid_qty']:.4f}) | Ask: {ask_price:.2f} ({ticker_data['ask_qty']:.4f}) | Spread: {spread:.2f} ({spread_pct:.2f}%)")


def on_trade(symbol, trade_data):
    """Callback for real-time trades"""
    # Register this callback in main setup function
    price = trade_data['price']
    qty = trade_data['quantity']
    value = price * qty
    side = "BUY" if trade_data['buyer_maker'] else "SELL"
    
    # Only log trades above a certain value to avoid flooding
    if value > 10000:  # Only log significant trades (>$10,000)
        trade_time = datetime.fromtimestamp(trade_data['time']/1000).strftime('%H:%M:%S')
        logger.info(f"💰 Large {side} Trade | {symbol} | Price: {price:.2f} | Qty: {qty:.4f} | Value: ${value:.2f} | {trade_time}")


def on_account_update(balance_updates, position_updates):
    """Callback for account updates"""
    global stats
    
    logger.info(f"Account update received: {len(balance_updates)} balances, {len(position_updates)} positions")
    
    try:
        if 'USDT' in balance_updates:
            new_balance = balance_updates['USDT']
            
            if 'current_balance' in stats:
                balance_change = new_balance - stats['current_balance']
                stats['daily_profit'] += balance_change
            
            stats['current_balance'] = new_balance
            logger.info(f"Balance updated: {new_balance} USDT")
            
            risk_manager.update_balance_for_compounding()
            
        for symbol, position in position_updates.items():
            logger.info(f"Position update for {symbol}: {position['position_amount']} @ {position['entry_price']}, PnL: {position['unrealized_pnl']}")
            
    except Exception as e:
        logger.error(f"Error processing account update: {e}")


def on_order_update(order_data):
    """Callback for order updates"""
    global stats
    
    try:
        symbol = order_data['symbol']
        status = order_data['order_status']
        side = order_data['side']
        order_type = order_data['type']
        filled_qty = order_data['filled_quantity']
        price = order_data['last_filled_price']
        
        # Create a more visually informative log message
        if status == 'FILLED':
            # Highlight completed orders with visual indicators
            if order_type == 'MARKET':
                if side == 'BUY':
                    logger.info(f"🟢🟢🟢 EXECUTED {side} ORDER: {filled_qty} {symbol} @ {price} 🟢🟢🟢")
                else:
                    logger.info(f"🔴🔴🔴 EXECUTED {side} ORDER: {filled_qty} {symbol} @ {price} 🔴🔴🔴")
            elif order_type == 'STOP_MARKET' or order_type == 'TAKE_PROFIT_MARKET':
                logger.info(f"⚠️⚠️⚠️ EXECUTED {order_type} {side} ORDER: {filled_qty} {symbol} @ {price} ⚠️⚠️⚠️")
            else:
                logger.info(f"✅✅✅ EXECUTED {order_type} {side} ORDER: {filled_qty} {symbol} @ {price} ✅✅✅")
                
            # For filled orders, update trade statistics
            if order_type == 'MARKET':
                stats['total_trades'] += 1
                stats['last_trade_time'] = datetime.now()
                
                realized_profit = order_data['realized_profit']
                if realized_profit > 0:
                    stats['winning_trades'] += 1
                    logger.info(f"💲💲💲 PROFIT: +{realized_profit:.2f} USDT 💲💲💲")
                elif realized_profit < 0:
                    stats['losing_trades'] += 1
                    logger.info(f"💸💸💸 LOSS: {realized_profit:.2f} USDT 💸💸💸")
                    
                trade_data = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': filled_qty, 
                    'price': price,
                    'realized_profit': realized_profit,
                    'commission': order_data['commission'],
                    'commission_asset': order_data['commission_asset'],
                    'timestamp': datetime.now().isoformat(),
                    'balance': stats['current_balance']
                }
                
                save_trade(trade_data)
                
                # Show current account balance after trade
                try:
                    current_balance = binance_client.get_account_balance()
                    logger.info(f"💰 ACCOUNT BALANCE: {current_balance:.2f} USDT 💰")
                except:
                    pass
                
                # Send notification
                notifier = TelegramNotifier()
                
                if side == 'BUY':
                    message = f"🟢 *Position Opened*\n"
                else:
                    message = f"🔴 *Position Closed*\n"
                    
                message += f"Symbol: {symbol}\n" \
                          f"Side: {side}\n" \
                          f"Quantity: {filled_qty}\n" \
                          f"Price: {price}\n"
                          
                if realized_profit != 0:
                    message += f"Realized P/L: {realized_profit:.2f} USDT\n"
                    
                notifier.send_message(message)
        else:
            # For other order statuses, just log basic info
            logger.info(f"Order Update: {symbol} {side} {order_type} {status} - Qty: {filled_qty}, Price: {price}")
                
    except Exception as e:
        logger.error(f"Error processing order update: {e}")
        import traceback
        logger.error(traceback.format_exc())


def check_for_signals(symbol=None):
    """Check for trading signals and execute trades"""
    global klines_data, new_candle_received
    
    if not symbol:
        symbol = TRADING_SYMBOL
    
    if not new_candle_received.get(symbol, False):
        return
    
    new_candle_received[symbol] = False
    
    logger.info(f"Checking for trading signals for {symbol}")
    
    try:
        klines = klines_data.get(symbol, [])
        
        if not klines or len(klines) < 30:
            logger.warning(f"Not enough historical data to generate signals (got {len(klines) if klines else 0} candles)")
            return
            
        current_price = websocket_manager.get_last_kline(symbol).get('close', None)
        if not current_price:
            logger.error("Failed to get current price from WebSocket")
            return
            
        position = binance_client.get_position_info(symbol)
        position_amount = position['position_amount'] if position else 0
        
        signal = strategy.get_signal(klines)
        
        if signal == "BUY" and position_amount <= 0:
            if position_amount < 0:
                logger.info("Closing existing short position before going long")
                binance_client.place_market_order(symbol, "BUY", abs(position_amount))
                
            if risk_manager.should_open_position(symbol):
                stop_loss_price = risk_manager.calculate_stop_loss(symbol, "BUY", current_price)
                
                quantity = risk_manager.calculate_position_size(
                    symbol, "BUY", current_price, stop_loss_price
                )
                
                if quantity > 0:
                    order = binance_client.place_market_order(symbol, "BUY", quantity)
                    if order:
                        logger.info(f"Opened long position: {quantity} {symbol} at {current_price}")
                        
                        if stop_loss_price:
                            binance_client.place_stop_loss_order(
                                symbol, "SELL", quantity, stop_loss_price
                            )
                            
                        take_profit_price = risk_manager.calculate_take_profit(symbol, "BUY", current_price)
                        if take_profit_price:
                            binance_client.place_take_profit_order(
                                symbol, "SELL", quantity, take_profit_price
                            )
                    
        elif signal == "SELL" and position_amount >= 0:
            if position_amount > 0:
                logger.info("Closing existing long position before going short")
                binance_client.place_market_order(symbol, "SELL", position_amount)
                
            if risk_manager.should_open_position(symbol):
                stop_loss_price = risk_manager.calculate_stop_loss(symbol, "SELL", current_price)
                
                quantity = risk_manager.calculate_position_size(
                    symbol, "SELL", current_price, stop_loss_price
                )
                
                if quantity > 0:
                    order = binance_client.place_market_order(symbol, "SELL", quantity)
                    if order:
                        logger.info(f"Opened short position: {quantity} {symbol} at {current_price}")
                        
                        if stop_loss_price:
                            binance_client.place_stop_loss_order(
                                symbol, "BUY", quantity, new_stop
                            )
                            
                        take_profit_price = risk_manager.calculate_take_profit(symbol, "SELL", current_price)
                        if take_profit_price:
                            binance_client.place_take_profit_order(
                                symbol, "BUY", quantity, take_profit_price
                            )
        
        if position and abs(position['position_amount']) > 0:
            side = "BUY" if position['position_amount'] > 0 else "SELL"
            opposite_side = "SELL" if side == "BUY" else "BUY"
            
            new_stop = risk_manager.adjust_stop_loss_for_trailing(
                symbol, side, current_price, position
            )
            
            if new_stop:
                binance_client.cancel_all_open_orders(symbol)
                
                binance_client.place_stop_loss_order(
                    symbol, opposite_side, abs(position['position_amount']), new_stop
                )
                
                logger.info(f"Updated trailing stop loss to {new_stop}")
    
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}")


def generate_performance_report():
    global stats
    
    logger.info("Generating performance report...")
    
    try:
        # Force refresh the current balance from the API
        if binance_client:
            current_balance = binance_client.get_account_balance()
            logger.info(f"Raw balance from API: {current_balance} USDT")
            
            # Only update if we got a valid balance (including very small amounts)
            if current_balance > 0:
                # Display with more precision for very small balances
                stats['current_balance'] = current_balance
                
                # If we didn't have a start balance before, set it now
                if stats['start_balance'] <= 0:
                    stats['start_balance'] = current_balance
                    logger.info(f"Setting initial balance to {current_balance} USDT")
        else:
            logger.error("Binance client not initialized, cannot get balance")
            current_balance = stats['current_balance']  # Use last known balance
            
    except Exception as e:
        logger.error(f"Error getting current balance: {e}")
        current_balance = stats['current_balance']  # Use last known balance
    
    # Calculate profit metrics with proper handling of small values
    profit_loss = current_balance - stats['start_balance']
    profit_pct = (profit_loss / stats['start_balance']) * 100 if stats['start_balance'] > 0 else 0
    
    daily_profit_pct = (stats['daily_profit'] / (current_balance - stats['daily_profit'])) * 100 if (current_balance - stats['daily_profit']) > 0 else 0
    
    # Load state from file to ensure we have the most recent data
    state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state', 'trading_state.json')
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                saved_stats = json.load(f)
                # Update our stats with any values from the file that might be more accurate
                if saved_stats.get('total_trades', 0) > stats['total_trades']:
                    stats['total_trades'] = saved_stats['total_trades']
                if saved_stats.get('winning_trades', 0) > stats['winning_trades']:
                    stats['winning_trades'] = saved_stats['winning_trades']
                if saved_stats.get('losing_trades', 0) > stats['losing_trades']:
                    stats['losing_trades'] = saved_stats['losing_trades']
        except Exception as e:
            logger.error(f"Error loading state file for report: {e}")
    
    # Use more decimal places for very small balances (< 1.0 USDT)
    balance_format = '.6f' if current_balance < 1.0 else '.2f'
    start_balance_format = '.6f' if stats['start_balance'] < 1.0 else '.2f'
    
    report = f"""
    ===== PERFORMANCE REPORT =====
    Time: {datetime.now()}
    
    Starting Balance: {stats['start_balance']:{start_balance_format}} USDT
    Current Balance: {current_balance:{balance_format}} USDT
    
    Total Profit/Loss: {profit_loss:{balance_format}} USDT ({profit_pct:.2f}%)
    Daily Profit/Loss: {stats['daily_profit']:{balance_format}} USDT ({daily_profit_pct:.2f}%)
    
    Total Trades: {stats['total_trades']}
    Winning Trades: {stats['winning_trades']}
    Losing Trades: {stats['losing_trades']}
    
    Win Rate: {(stats['winning_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0:.2f}%
    """
    
    logger.info(report)
    
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    report_file = os.path.join(report_dir, f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(report_file, 'w') as f:
        f.write(report)
        
    try:
        chart_path = None
        if stats['total_trades'] > 0:
            chart_path = generate_equity_chart(report_dir)
    except Exception as e:
        logger.error(f"Failed to generate chart: {e}")
        chart_path = None
        
    notifier = TelegramNotifier()
    
    # Use different precision for Telegram message too
    telegram_report = f"📊 *Daily Performance Report*\n\n" \
                     f"*Starting Balance:* {stats['start_balance']:{start_balance_format}} USDT\n" \
                     f"*Current Balance:* {current_balance:{balance_format}} USDT\n\n" \
                     f"*Total Profit/Loss:* {profit_loss:{balance_format}} USDT ({profit_pct:.2f}%)\n" \
                     f"*Daily Profit/Loss:* {stats['daily_profit']:{balance_format}} USDT ({daily_profit_pct:.2f}%)\n\n" \
                     f"*Total Trades:* {stats['total_trades']}\n" \
                     f"*Win Rate:* {(stats['winning_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0:.2f}%"
    
    notifier.send_message(telegram_report)
    
    if chart_path and os.path.exists(chart_path):
        notifier.send_photo(chart_path, "Equity Curve")
        
    stats['daily_profit'] = 0.0
    stats['last_report_time'] = datetime.now()
    
    return report_file


def generate_equity_chart(output_dir):
    try:
        trades_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state')
        trades_file = os.path.join(trades_dir, 'trades.json')
        
        if not os.path.exists(trades_file):
            logger.warning("No trade history found for chart generation")
            return None
            
        with open(trades_file, 'r') as f:
            trades = json.load(f)
            
        df = pd.DataFrame(trades)
        
        df['date'] = pd.to_datetime(df['timestamp'])
        df.set_index('date', inplace=True)
        df = df.sort_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['balance'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Balance (USDT)')
        plt.grid(True)
        
        chart_path = os.path.join(output_dir, f'equity_{datetime.now().strftime("%Y%m%d")}.png')
        plt.savefig(chart_path)
        plt.close()
        
        return chart_path
    except Exception as e:
        logger.error(f"Failed to generate equity chart: {e}")
        return None


def send_daily_report():
    """Send daily performance report"""
    if not SEND_DAILY_REPORT:
        return
        
    logger.info("Sending daily performance report...")
    generate_performance_report()


def send_status_report():
    """Send a status report with API connection, WebSocket status, and account balance"""
    
    try:
        logger.info("Generating connection status report")
        
        # Check if websocket is connected
        ws_status = "Connected" if websocket_manager and websocket_manager.is_connected() else "Disconnected"
        
        # Check API connection by attempting to get server time
        api_status = "Connected"
        try:
            binance_client.client.get_server_time()
        except Exception as e:
            api_status = f"Error: {str(e)[:50]}..."
        
        # Get current balance
        balance = binance_client.get_account_balance()
        
        # Get current positions
        positions = []
        try:
            position_info = binance_client.get_position_info(TRADING_SYMBOL)
            if position_info and abs(position_info.get('position_amount', 0)) > 0:
                side = "LONG" if position_info['position_amount'] > 0 else "SHORT"
                positions.append(f"{TRADING_SYMBOL}: {position_info['position_amount']} ({side})")
        except Exception as e:
            positions.append(f"Error getting positions: {str(e)[:30]}...")
        
        # Get current price
        price = "Unknown"
        try:
            price = binance_client.get_current_price(TRADING_SYMBOL)
        except:
            pass
        
        # Create status report
        status_message = (
            f"🔄 *Status Report*\n\n"
            f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"*Connection Status:*\n"
            f"API: {api_status}\n"
            f"WebSocket: {ws_status}\n\n"
            f"*Account:*\n"
            f"Balance: {balance:.4f} USDT\n"
            f"Current {TRADING_SYMBOL} Price: {price}\n\n"
        )
        
        if positions:
            status_message += f"*Active Positions:*\n"
            for pos in positions:
                status_message += f"{pos}\n"
        else:
            status_message += "*No active positions*\n"
        
        status_message += f"\n*Trading Stats:*\n"
        status_message += f"Total Trades: {stats['total_trades']}\n"
        status_message += f"Win Rate: {(stats['winning_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0:.1f}%\n"
        
        # Send the message using plaintext to avoid formatting issues
        notifier = TelegramNotifier()
        notifier.send_plain_message(status_message)
        
        logger.info("Status report sent successfully")
        
    except Exception as e:
        logger.error(f"Error generating status report: {e}")


def handle_exit(signal, frame):
    """Handle exit gracefully"""
    global running, websocket_manager
    logger.info("Shutdown signal received. Cleaning up...")
    running = False
    
    if websocket_manager:
        websocket_manager.stop()
        logger.info("WebSocket connections closed")


def save_state():
    """Save the current state to a file for possible restart"""
    state_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state')
    os.makedirs(state_dir, exist_ok=True)
    
    json_stats = stats.copy()
    if json_stats['last_trade_time']:
        json_stats['last_trade_time'] = json_stats['last_trade_time'].isoformat()
    if json_stats['last_report_time']:
        json_stats['last_report_time'] = json_stats['last_report_time'].isoformat()
    
    state_file = os.path.join(state_dir, 'trading_state.json')
    with open(state_file, 'w') as f:
        json.dump(json_stats, f)
        
    logger.info("Trading state saved")


def save_trade(trade_data):
    """Save trade to trade history"""
    trade_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state')
    os.makedirs(trade_dir, exist_ok=True)
    
    if 'timestamp' not in trade_data:
        trade_data['timestamp'] = datetime.now().isoformat()
    
    trades_file = os.path.join(trade_dir, 'trades.json')
    
    trades = []
    if os.path.exists(trades_file):
        with open(trades_file, 'r') as f:
            try:
                trades = json.load(f)
            except json.JSONDecodeError:
                trades = []
    
    trades.append(trade_data)
    
    with open(trades_file, 'w') as f:
        json.dump(trades, f)


def load_state():
    """Load saved state if exists"""
    state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state', 'trading_state.json')
    if os.path.exists(state_file) and os.path.getsize(state_file) > 0:
        try:
            with open(state_file, 'r') as f:
                loaded_stats = json.load(f)
                
            if 'last_trade_time' in loaded_stats and loaded_stats['last_trade_time']:
                loaded_stats['last_trade_time'] = datetime.fromisoformat(loaded_stats['last_trade_time'])
            if 'last_report_time' in loaded_stats and loaded_stats['last_report_time']:
                loaded_stats['last_report_time'] = datetime.fromisoformat(loaded_stats['last_report_time'])
                
            logger.info("Loaded previous trading state")
            return loaded_stats
        except json.JSONDecodeError as e:
            logger.error(f"Error loading state file (corrupted JSON): {e}")
            logger.info("Creating a new state file")
            # Create a backup of the corrupted file
            if os.path.exists(state_file):
                backup_file = f"{state_file}.bak.{int(time.time())}"
                try:
                    os.rename(state_file, backup_file)
                    logger.info(f"Backed up corrupted state file to {backup_file}")
                except Exception as rename_error:
                    logger.error(f"Failed to backup corrupted state file: {rename_error}")
            return None
    
    logger.info("No previous state found or file is empty")
    return None


def run_backtest(symbol, timeframe, strategy_name, start_date, end_date=None, save_results=True):
    """Run backtest using historical data"""
    logger.info(f"Starting backtest for {symbol} with {strategy_name} strategy")
    
    try:
        binance = BinanceClient()
        
        # Handle relative date strings like "30 days" or "1 year ago"
        if isinstance(start_date, str) and any(word in start_date for word in ['day', 'week', 'month', 'year']):
            # For Binance API, we can use relative dates directly
            api_start_date = start_date
            
            # For the Backtester, convert to YYYY-MM-DD format
            # Extract the numeric value and time unit
            parts = start_date.split()
            
            # Default values
            num = 30  # Default to 30 days if parsing fails
            unit = 'days'
            
            if len(parts) >= 2 and parts[0].isdigit():
                num = int(parts[0])
                unit = parts[1].lower()
                
                # Calculate actual date
                if unit.startswith('day'):
                    backtest_start_date = (datetime.now() - timedelta(days=num)).strftime('%Y-%m-%d')
                elif unit.startswith('week'):
                    backtest_start_date = (datetime.now() - timedelta(weeks=num)).strftime('%Y-%m-%d')
                elif unit.startswith('month'):
                    backtest_start_date = (datetime.now() - timedelta(days=num*30)).strftime('%Y-%m-%d')
                elif unit.startswith('year'):
                    backtest_start_date = (datetime.now() - timedelta(days=num*365)).strftime('%Y-%m-%d')
                else:
                    # Default fallback
                    backtest_start_date = (datetime.now() - timedelta(days=num)).strftime('%Y-%m-%d')
            else:
                # Default fallback if we can't parse the input
                backtest_start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                logger.warning(f"Couldn't parse date format '{start_date}', using past 30 days as default")
        else:
            # If it's already in YYYY-MM-DD format, use it directly for both
            api_start_date = start_date
            backtest_start_date = start_date
            
        logger.info(f"Fetching historical data from: {api_start_date} (as {backtest_start_date})")
        
        try:
            klines = binance.get_historical_klines(
                symbol=symbol,
                interval=timeframe,
                start_str=api_start_date,
                end_str=end_date,
                limit=1000
            )
        except Exception as api_error:
            logger.error(f"API Error fetching klines: {api_error}")
            return None
        
        if not klines or len(klines) < 100:
            logger.error(f"Not enough historical data for backtest. Got {len(klines) if klines else 0} candles.")
            return None
            
        backtester = Backtester(strategy_name, symbol, timeframe, backtest_start_date, end_date)
        
        df = backtester.load_historical_data(klines)
        
        results = backtester.run(df)
        
        if save_results and results:
            output_dir = backtester.save_results(results)
            
            summary = backtester.generate_summary_report(results)
            
            summary_path = os.path.join(output_dir, 'summary.md')
            with open(summary_path, 'w') as f:
                f.write(summary)
                
            print("\n" + summary + "\n")
            
            if USE_TELEGRAM:
                notifier = TelegramNotifier()
                notifier.send_message(f"🔍 *Backtest Completed*\n\n{summary}")
                
                equity_chart = os.path.join(output_dir, 'plots', 'equity_curve.png')
                if os.path.exists(equity_chart):
                    notifier.send_photo(equity_chart, f"Equity Curve - {symbol} {strategy_name}")
            
        return results
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        return None


def validate_backtest_results(results):
    """
    Validate backtest results against minimum performance criteria
    Returns (is_valid, message)
    """
    if not results:
        return False, "Backtest failed to produce results"
    
    # Extract key performance metrics
    total_return_pct = results.get('total_return', 0)
    win_rate = results.get('win_rate', 0)
    total_trades = results.get('total_trades', 0)
    
    # Check if strategy meets minimum criteria
    reasons = []
    
    if total_trades < 5:
        reasons.append(f"Too few trades ({total_trades} < 5)")
    
    if total_return_pct < BACKTEST_MIN_PROFIT_PCT:
        reasons.append(f"Profit too low ({total_return_pct:.2f}% < {BACKTEST_MIN_PROFIT_PCT}%)")
    
    if win_rate < BACKTEST_MIN_WIN_RATE:
        reasons.append(f"Win rate too low ({win_rate:.2f}% < {BACKTEST_MIN_WIN_RATE}%)")
    
    if reasons:
        return False, "Strategy validation failed: " + ", ".join(reasons)
    
    return True, "Strategy passed validation"

def run_safety_backtest(symbol, timeframe, strategy_name):
    """
    Run a backtest to check strategy performance before live trading
    """
    logger.info("Running safety backtest before starting live trading...")
    
    # Use past period defined in config for backtest
    start_date = BACKTEST_PERIOD
    
    # Run the backtest
    results = run_backtest(
        symbol=symbol,
        timeframe=timeframe,
        strategy_name=strategy_name,
        start_date=start_date,
        save_results=True
    )
    
    if not results:
        return False, "Backtest failed to complete"
    
    # Validate the results
    is_valid, message = validate_backtest_results(results)
    
    if is_valid:
        logger.info(f"Strategy validation passed: {symbol} with {strategy_name}")
    else:
        logger.warning(f"Strategy validation failed: {message}")
        
        # Send alert if Telegram is enabled
        notifier = TelegramNotifier()
        notifier.send_message(f"⚠️ *Strategy Validation Failed*\n\n"
                              f"Symbol: {symbol}\n"
                              f"Strategy: {strategy_name}\n"
                              f"Reason: {message}\n\n"
                              f"Bot will not start live trading unless you use --skip-validation")
    
    return is_valid, message

def perform_test_trade(symbol=TRADING_SYMBOL):
    """
    Perform a small test trade to verify trading functionality
    Returns True if successful, False otherwise
    """
    logger.info(f"Performing test trade on {symbol} to verify trading functionality")
    
    try:
        # Get current price
        current_price = binance_client.get_current_price(symbol)
        if not current_price:
            logger.error("Could not get current price for test trade")
            return False
            
        # Get current balance
        initial_balance = binance_client.get_account_balance()
        logger.info(f"Balance before test trade: {initial_balance} USDT")
        
        # Get symbol info for precision
        symbol_info = binance_client.get_symbol_info(symbol)
        if not symbol_info:
            logger.error("Could not get symbol info for test trade")
            return False
        
        # For a $50 account, use a much smaller test trade (minimum value needed for Binance)
        # Aim for just above the minimum required (100 USDT)
        qty_precision = symbol_info.get('quantity_precision', 3)
        
        # Calculate quantity - For a $50 account, we want to ensure our test trade
        # is small enough but also meets Binance's minimum order value
        # Aim for 0.001 BTC (~$75) or minimum required
        if symbol == "BTCUSDT":
            # Use the smallest allowed quantity for BTC
            test_qty = 0.001
            order_value = test_qty * current_price
            # If this is too small, adjust to minimum required
            if order_value < 100:
                test_qty = 100 / current_price
                # Round to appropriate precision
                import math
                multiplier = 10 ** qty_precision
                test_qty = math.floor(test_qty * multiplier) / multiplier
                order_value = test_qty * current_price
        else:
            # For other coins, calculate minimum required
            test_qty = 100 / current_price  # Minimum 100 USDT notional value
            # Round to appropriate precision
            import math
            multiplier = 10 ** qty_precision
            test_qty = math.floor(test_qty * multiplier) / multiplier
            order_value = test_qty * current_price
        
        logger.info(f"Final test order: {test_qty} {symbol} at ~{current_price} = {order_value:.2f} USDT")
        
        # Final safety check - if still below 100 USDT, abort
        if order_value < 100:
            logger.error(f"Cannot create order above minimum notional value. Calculated: {order_value:.2f} USDT, Required: 100 USDT")
            return False
        
        # Check if we have enough balance
        if order_value > initial_balance * 0.9:  # Use 90% max for a small account
            logger.error(f"Not enough balance for test trade. Required: {order_value:.2f} USDT, Available: {initial_balance} USDT")
            return False
            
        logger.info(f"Executing test BUY order: {test_qty} {symbol} at ~{current_price}")
        
        # Place a market BUY order
        buy_order = binance_client.place_market_order(symbol, "BUY", test_qty)
        if not buy_order:
            logger.error("Test BUY order failed")
            return False
            
        logger.info(f"Test BUY order successful: {buy_order.get('orderId', 'unknown')}")
        
        # Wait a moment for the order to fully process
        time.sleep(2)
        
        # Check position
        position = binance_client.get_position_info(symbol)
        if not position or position.get('position_amount', 0) <= 0:
            logger.error("Position not opened after test BUY")
            return False
            
        logger.info(f"Test position opened: {position.get('position_amount')} {symbol} @ {position.get('entry_price')}")
        
        # Now place a SELL order to close the position
        sell_qty = position.get('position_amount')
        logger.info(f"Executing test SELL order to close position: {sell_qty} {symbol}")
        
        sell_order = binance_client.place_market_order(symbol, "SELL", sell_qty)
        if not sell_order:
            logger.error("Test SELL order failed. You may need to manually close the position!")
            return False
            
        logger.info(f"Test SELL order successful: {sell_order.get('orderId', 'unknown')}")
        
        # Wait a moment for the order to fully process
        time.sleep(2)
        
        # Check final balance
        final_balance = binance_client.get_account_balance()
        balance_diff = final_balance - initial_balance
        
        logger.info(f"Balance after test trade: {final_balance} USDT (Change: {balance_diff} USDT)")
        logger.info(f"Test trade completed successfully")
        
        # Notify on Telegram if enabled
        notifier = TelegramNotifier()
        notifier.send_message(f"✅ *Test Trade Completed*\n\n"
                             f"Symbol: {symbol}\n"
                             f"Test Size: {test_qty} (Value: {order_value:.2f} USDT)\n"
                             f"Balance Change: {balance_diff:.6f} USDT\n\n"
                             f"Trading functionality verified successfully.")
        
        return True
    except Exception as e:
        logger.error(f"Error during test trade: {e}")
        
        # Notify on Telegram if enabled
        notifier = TelegramNotifier()
        notifier.send_message(f"❌ *Test Trade Failed*\n\n"
                             f"Symbol: {symbol}\n"
                             f"Error: {str(e)}\n\n"
                             f"Please check logs and resolve issues before starting live trading.")
        
        return False

def main():
    """Main function to start the trading bot"""
    global running, stats, websocket_manager
    
    parser = argparse.ArgumentParser(description='Binance Futures Trading Bot')
    parser.add_argument('--backtest', action='store_true', help='Run in backtest mode')
    parser.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, default=TRADING_SYMBOL, help='Trading symbol for backtest')
    parser.add_argument('--strategy', type=str, default=STRATEGY, help='Strategy for backtest')
    parser.add_argument('--report', action='store_true', help='Generate performance report only')
    parser.add_argument('--interval', type=int, default=5, help='Trading check interval in minutes')
    parser.add_argument('--skip-validation', action='store_true', help='Skip strategy validation before live trading')
    parser.add_argument('--skip-test-trade', action='store_true', help='Skip test trade before live trading')
    parser.add_argument('--small-account', action='store_true', help='Run with small account (under $100) - skips test trade and uses adjusted risk')
    parser.add_argument('--force-balance', action='store_true', help='Force initialization of balance from config file')
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # For the report command, make sure we can access the config
    if args.report:
        # Try to import INITIAL_BALANCE directly here
        try:
            from modules.config import INITIAL_BALANCE
            # Ensure it's available if needed
            if INITIAL_BALANCE > 0:
                logger.info(f"Using configured initial balance: {INITIAL_BALANCE} USDT")
        except ImportError:
            logger.warning("Could not import INITIAL_BALANCE from config")
    
    if args.backtest:
        start_date = args.start_date or "1 year ago"
        run_backtest(args.symbol, TIMEFRAME, args.strategy, start_date, args.end_date)
        return
    
    if args.report:
        setup()
        # Make sure we have a state file with correct balance before generating report
        if args.force_balance:
            initialize_state_file(force=True)
        else:
            initialize_state_file()
        generate_performance_report()
        return
    
    # For small accounts, automatically skip test trade
    if args.small_account:
        args.skip_test_trade = True
        logger.info("Small account mode: Test trade will be skipped")
    
    # Run safety backtest before starting live trading
    if BACKTEST_BEFORE_LIVE and not args.skip_validation:
        logger.info("Running strategy validation before starting live trading")
        is_valid, message = run_safety_backtest(
            symbol=args.symbol or TRADING_SYMBOL,
            timeframe=TIMEFRAME,
            strategy_name=args.strategy or STRATEGY
        )
        
        if not is_valid:
            logger.error(f"Strategy validation failed: {message}")
            logger.error("Use --skip-validation to bypass this check")
            return
    
    setup()
    
    # Force initialize state file if requested
    if args.force_balance:
        initialize_state_file(force=True)
    
    # Check balance for very small accounts
    initial_balance = binance_client.get_account_balance()
    if initial_balance < 45.0 and not args.small_account:
        logger.warning(f"Account balance is very low: {initial_balance} USDT")
        logger.warning("For accounts under $45, use --small-account flag to adjust settings")
        if initial_balance < 5.0:
            logger.error("Account balance is too low to trade effectively. Please deposit funds.")
            return
    
    # Perform a test trade before starting live trading
    if not args.skip_test_trade:
        logger.info("Running test trade before starting live trading")
        if not perform_test_trade():
            logger.error("Test trade failed. Aborting live trading.")
            logger.error("Use --skip-test-trade to bypass this check")
            logger.error("For small accounts (under $100), use --small-account flag")
            return
    
    loaded_stats = load_state()
    if loaded_stats:
        stats.update(loaded_stats)
    else:
        initialize_state_file()
    
    logger.info(f"Starting trading bot with WebSocket for real-time data")
    logger.info(f"Trading pair: {TRADING_SYMBOL}")
    logger.info(f"Strategy: {strategy.strategy_name}")
    logger.info(f"Daily reports: {'Enabled' if SEND_DAILY_REPORT else 'Disabled'}")
    logger.info(f"Press Ctrl+C to stop the bot")
    
    schedule.every().hour.do(save_state)
    schedule.every(6).hours.do(send_status_report)
    
    if SEND_DAILY_REPORT:
        schedule.every().day.at(DAILY_REPORT_TIME).do(send_daily_report)
    else:
        schedule.every(24).hours.do(generate_performance_report)
    
    while running:
        schedule.run_pending()
        time.sleep(1)
        
    logger.info("Bot stopped. Generating final report...")
    generate_performance_report()
    save_state()

if __name__ == "__main__":
    main()