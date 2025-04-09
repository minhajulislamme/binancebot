#!/bin/bash
# Setup systemd service for Binance Trading Bot
# This script creates and enables a systemd service for 24/7 operation

echo "==== Setting up Binance Bot systemd service ===="

# Get the current directory path and username
cd "$(dirname "$(dirname "$0")")"
BOTDIR="$(pwd)"
USER="$(whoami)"

# Create systemd service file
echo "Creating systemd service file..."
cat > /tmp/binancebot.service << EOF
[Unit]
Description=Binance Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$BOTDIR
ExecStart=$BOTDIR/venv/bin/python $BOTDIR/main.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
Environment="PATH=$BOTDIR/venv/bin:$PATH"
# Kill the service if it uses too much memory (adjust as needed)
MemoryMax=1G
# Shutdown timeout
TimeoutStopSec=30s

[Install]
WantedBy=multi-user.target
EOF

# Move service file to systemd directory
sudo mv /tmp/binancebot.service /etc/systemd/system/binancebot.service

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable binancebot.service

echo ""
echo "Systemd service installed successfully!"
echo "To start the bot: sudo systemctl start binancebot"
echo "To stop the bot:  sudo systemctl stop binancebot"
echo "To check status:  sudo systemctl status binancebot"
echo "To view logs:     sudo journalctl -u binancebot -f"
echo ""
echo "Your bot will now automatically start on system boot."
echo ""