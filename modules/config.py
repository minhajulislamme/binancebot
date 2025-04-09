import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configuration
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')

# Testnet configuration
API_TESTNET = os.getenv('BINANCE_API_TESTNET', 'False').lower() == 'true'

# API URLs - Automatically determined based on testnet setting
if API_TESTNET:
    # Testnet URLs
    API_URL = 'https://testnet.binancefuture.com'
    WS_BASE_URL = 'wss://stream.binancefuture.com'
else:
    # Production URLs
    API_URL = os.getenv('BINANCE_API_URL', 'https://fapi.binance.com')
    WS_BASE_URL = 'wss://fstream.binance.com'

# API request settings
RECV_WINDOW = int(os.getenv('BINANCE_RECV_WINDOW', '10000'))

# Trading parameters
TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'BTCUSDT')  # Default trading pair
TRADING_TYPE = 'FUTURES'  # Use futures trading
LEVERAGE = int(os.getenv('LEVERAGE', '5'))  # Default leverage 5x
MARGIN_TYPE = os.getenv('MARGIN_TYPE', 'ISOLATED')  # ISOLATED or CROSSED

# Position sizing
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', '50.0'))  # Starting with $50
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))  # 2% risk per trade
MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '1'))

# Auto-compounding settings
AUTO_COMPOUND = os.getenv('AUTO_COMPOUND', 'True').lower() == 'true'
COMPOUND_REINVEST_PERCENT = float(os.getenv('COMPOUND_REINVEST_PERCENT', '0.75'))  # Reinvest 75% of profits
COMPOUND_INTERVAL = os.getenv('COMPOUND_INTERVAL', 'DAILY')  # Can be TRADE, DAILY, WEEKLY

# Strategy parameters
STRATEGY = os.getenv('STRATEGY', 'RSI_EMA')  # Default strategy
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT', '70'))
RSI_OVERSOLD = int(os.getenv('RSI_OVERSOLD', '30'))
FAST_EMA = int(os.getenv('FAST_EMA', '9'))
SLOW_EMA = int(os.getenv('SLOW_EMA', '21'))
TIMEFRAME = os.getenv('TIMEFRAME', '15m')  # Default timeframe

# Risk management
USE_STOP_LOSS = os.getenv('USE_STOP_LOSS', 'True').lower() == 'true'
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.03'))  # 3% stop loss
USE_TAKE_PROFIT = os.getenv('USE_TAKE_PROFIT', 'True').lower() == 'true'
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.06'))  # 6% take profit
TRAILING_STOP = os.getenv('TRAILING_STOP', 'False').lower() == 'true'
TRAILING_STOP_PCT = float(os.getenv('TRAILING_STOP_PCT', '0.015'))  # 1.5% trailing stop

# Backtesting parameters
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2023-01-01')
BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '')  # Empty means use current date
BACKTEST_INITIAL_BALANCE = float(os.getenv('BACKTEST_INITIAL_BALANCE', '50.0'))
BACKTEST_COMMISSION = float(os.getenv('BACKTEST_COMMISSION', '0.0004'))  # 0.04% taker fee
BACKTEST_USE_AUTO_COMPOUND = os.getenv('BACKTEST_USE_AUTO_COMPOUND', 'True').lower() == 'true'

# Pre-live backtest validation
BACKTEST_BEFORE_LIVE = os.getenv('BACKTEST_BEFORE_LIVE', 'True').lower() == 'true'
BACKTEST_MIN_PROFIT_PCT = float(os.getenv('BACKTEST_MIN_PROFIT_PCT', '5.0'))
BACKTEST_MIN_WIN_RATE = float(os.getenv('BACKTEST_MIN_WIN_RATE', '50.0'))
BACKTEST_PERIOD = os.getenv('BACKTEST_PERIOD', '30 days')

# Logging and notifications
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
USE_TELEGRAM = os.getenv('USE_TELEGRAM', 'False').lower() == 'true'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
SEND_DAILY_REPORT = os.getenv('SEND_DAILY_REPORT', 'True').lower() == 'true'
DAILY_REPORT_TIME = os.getenv('DAILY_REPORT_TIME', '00:00')  # 24-hour format

# Other settings
RETRY_COUNT = int(os.getenv('RETRY_COUNT', '3'))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', '5'))  # seconds