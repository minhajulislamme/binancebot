#!/bin/bash
# Binance Bot VPS Installation Script
# This script sets up the environment for running the Binance trading bot

echo "==== Binance Trading Bot Setup ===="
echo "Setting up environment for 24/7 operation on VPS"
echo ""

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y
echo "System packages updated."
echo ""

# Install required system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y python3 python3-pip python3-dev python3-venv git htop tmux supervisor
echo "System dependencies installed."
echo ""

# Set up Python virtual environment
echo "Setting up Python virtual environment..."
cd "$(dirname "$(dirname "$0")")"
BOTDIR="$(pwd)"
echo "Bot directory: $BOTDIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Python dependencies installed."

# Create .env file template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env template file..."
    cat > .env << EOF
# Binance API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_API_TESTNET=False

# Trading Configuration
TRADING_SYMBOL=BTCUSDT
INITIAL_BALANCE=50.0
RISK_PER_TRADE=0.02
LEVERAGE=5
MARGIN_TYPE=ISOLATED

# Strategy Configuration
STRATEGY=RSI_EMA
TIMEFRAME=15m

# Risk Management
USE_STOP_LOSS=True
STOP_LOSS_PCT=0.03
USE_TAKE_PROFIT=True
TAKE_PROFIT_PCT=0.06
TRAILING_STOP=True
TRAILING_STOP_PCT=0.015

# Notifications
USE_TELEGRAM=False
TELEGRAM_BOT_TOKEN=your_telegram_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Auto-compounding
AUTO_COMPOUND=True
COMPOUND_REINVEST_PERCENT=0.75

# Logging
LOG_LEVEL=INFO
EOF
    echo ".env template created. Please edit with your actual API keys and settings."
else
    echo ".env file already exists."
fi

# Create directories if they don't exist
echo "Creating required directories..."
mkdir -p logs reports state backtest_results
echo "Directories created."

echo ""
echo "Installation complete! Next steps:"
echo "1. Edit the .env file with your Binance API keys and settings"
echo "2. Run 'sudo ./scripts/setup_service.sh' to create the system service"
echo "3. Run 'sudo systemctl start binancebot' to start the bot"
echo ""
echo "To monitor the bot logs:"
echo "- systemctl status binancebot"
echo "- journalctl -u binancebot -f"
echo ""