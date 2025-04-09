#!/bin/bash
# Binance Trading Bot Monitoring Script
# Displays bot status, recent logs, and current performance

echo "===== Binance Trading Bot Monitor ====="
echo "$(date)"
echo ""

# Check if bot service is running
echo "Checking service status..."
if systemctl is-active --quiet binancebot; then
    echo "✅ Bot service is RUNNING"
else
    echo "❌ Bot service is NOT RUNNING"
fi
echo ""

# Get bot uptime
echo "Bot service uptime:"
systemctl show binancebot -p ActiveEnterTimestamp --value | xargs -I{} date -d "{}" "+Started: %Y-%m-%d %H:%M:%S"
echo "Current: $(date "+%Y-%m-%d %H:%M:%S")"
echo ""

# Check system resources
echo "System resources:"
echo "Memory usage: $(free -m | grep Mem | awk '{print $3 " MB used / " $2 " MB total (" int($3*100/$2) "%)"}')"
echo "CPU load: $(cat /proc/loadavg | awk '{print $1 " " $2 " " $3}')"
echo "Disk space: $(df -h / | grep / | awk '{print $4 " free / " $2 " total"}')"
echo ""

# Display recent trades
echo "Recent trades (if any):"
if [ -f "../state/trades.json" ]; then
    tail -n 5 "../state/trades.json" | grep -E "realized_profit|symbol|timestamp" | sed 's/",/"/g' | sed 's/^ *//'
else
    echo "No trade history found."
fi
echo ""

# Display recent logs
echo "Recent logs (last 20 lines):"
cd "$(dirname "$(dirname "$0")")"
tail -n 20 logs/trading_bot_$(date +%Y%m%d).log
echo ""

# Report generation option
echo "Options:"
echo "1. Generate performance report"
echo "2. View current positions"
echo "3. Run backtest"
echo "4. Restart bot service"
echo "5. Exit"

read -p "Choose an option (or press Enter to exit): " option

case "$option" in
    1)
        echo "Generating performance report..."
        cd "$(dirname "$(dirname "$0")")"
        source venv/bin/activate
        python main.py --report
        ;;
    2)
        echo "Checking current positions (using bot's venv and main.py)..."
        cd "$(dirname "$(dirname "$0")")"
        source venv/bin/activate
        # This would require adding a --positions flag to main.py
        # For now we'll use journalctl to look for recent position logs
        journalctl -u binancebot -n 100 | grep -i "position"
        ;;
    3)
        echo "Running backtest..."
        cd "$(dirname "$(dirname "$0")")"
        source venv/bin/activate
        read -p "Enter period (e.g., '30 days'): " period
        python main.py --backtest --start-date "$period"
        ;;
    4)
        echo "Restarting bot service..."
        sudo systemctl restart binancebot
        sleep 2
        systemctl status binancebot
        ;;
    *)
        echo "Exiting."
        ;;
esac