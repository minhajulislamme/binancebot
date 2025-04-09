#!/bin/bash
# Binance Trading Bot Maintenance Script
# Performs system maintenance tasks to keep the VPS running smoothly

echo "===== Binance Bot VPS Maintenance ====="
echo "$(date)"
echo ""

# Make backup of important files
echo "Creating backups of important files..."
BACKUP_DIR="../backups/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup .env file
if [ -f "../.env" ]; then
    cp "../.env" "$BACKUP_DIR/.env.backup"
    echo "✅ .env file backed up"
fi

# Backup state files
if [ -d "../state" ]; then
    cp -r "../state" "$BACKUP_DIR/state_backup"
    echo "✅ State files backed up"
fi

# Backup latest log files
if [ -d "../logs" ]; then
    mkdir -p "$BACKUP_DIR/logs"
    find "../logs" -type f -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/logs/" \;
    echo "✅ Recent log files backed up"
fi

echo ""

# Check disk usage and clean up if needed
echo "Checking disk space usage..."
DISK_USAGE=$(df -h / | grep / | awk '{print $5}' | tr -d '%')

if [ "$DISK_USAGE" -gt 80 ]; then
    echo "⚠️ Disk usage is high ($DISK_USAGE%). Performing cleanup..."
    
    # Clean old log files (older than 30 days)
    echo "Removing log files older than 30 days..."
    find "../logs" -type f -name "*.log" -mtime +30 -delete
    
    # Clean old backtest results (older than 90 days)
    echo "Removing backtest results older than 90 days..."
    find "../backtest_results" -type d -mtime +90 -exec rm -rf {} \; 2>/dev/null || true
    
    # Clean old report files (older than 60 days)
    echo "Removing reports older than 60 days..."
    find "../reports" -type f -mtime +60 -delete
    
    # Clean apt cache
    echo "Cleaning apt cache..."
    sudo apt-get clean
    
    # Remove old kernel versions
    echo "Removing old kernel versions..."
    sudo apt-get autoremove -y
    
    # Check disk usage after cleanup
    DISK_USAGE_AFTER=$(df -h / | grep / | awk '{print $5}' | tr -d '%')
    echo "Disk usage after cleanup: $DISK_USAGE_AFTER%"
else
    echo "✅ Disk usage is acceptable ($DISK_USAGE%)"
fi

echo ""

# Check system updates
echo "Checking for system updates..."
sudo apt-get update -qq
UPDATES=$(sudo apt-get -s upgrade | grep -P '^\d+ upgraded' | cut -d" " -f1)

if [ "$UPDATES" -gt 0 ]; then
    echo "⚠️ $UPDATES system updates available. Consider updating with:"
    echo "sudo apt-get upgrade -y"
else
    echo "✅ System is up to date"
fi

echo ""

# Check bot version (optional - if you use git to manage your bot code)
if [ -d "../.git" ]; then
    echo "Checking for bot updates..."
    cd ..
    git remote update > /dev/null 2>&1
    UPSTREAM=${1:-'@{u}'}
    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse "$UPSTREAM")
    
    if [ "$LOCAL" != "$REMOTE" ]; then
        echo "⚠️ Bot updates available. Consider updating with:"
        echo "git pull"
    else
        echo "✅ Bot is up to date"
    fi
    cd - > /dev/null
fi

echo ""

# Check Python dependencies for updates (optional)
echo "Checking Python package updates..."
cd ..
source venv/bin/activate
OUTDATED=$(pip list --outdated | wc -l)
if [ "$OUTDATED" -gt 1 ]; then  # -1 because the header line is counted
    echo "⚠️ $(($OUTDATED-1)) Python packages can be updated. Consider updating with:"
    echo "pip install --upgrade -r requirements.txt"
else
    echo "✅ Python packages are up to date"
fi
deactivate

echo ""
echo "Maintenance tasks completed! System is ready for continued operation."