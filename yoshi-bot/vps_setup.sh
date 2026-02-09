#!/bin/bash
# Yoshi-Bot VPS Setup Script for Ubuntu 24.04

echo "🚀 Starting Yoshi-Bot Setup..."

# 1. Update System
sudo apt update && sudo apt upgrade -y

# 2. Install Python dependencies
sudo apt install -y python3-pip python3-venv git htop

# 3. Create Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 4. Install requirements
# Since we don't have a requirements.txt, we install known dependencies
pip install numpy pandas scipy requests python-dotenv aiohttp cryptography pyarrow ccxt

# 5. Setup Logging Directory
mkdir -p logs

# 6. Ensure .env exists
if [ ! -f .env ]; then
    echo "⚠️ .env file missing! Please upload your .env file to the root directory."
fi

echo "✅ Setup Complete. To start the scanner, run:"
echo "source venv/bin/activate"
echo "nohup python3 scripts/kalshi_scanner.py --symbol BTCUSDT --loop --interval 300 --threshold 0.10 > logs/scanner.log 2>&1 &"
