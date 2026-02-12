#!/bin/bash
set -euo pipefail

# Session Start Hook for Yoshi Trading System
# Installs dependencies for Python and Node.js components

echo "=== Session Start Hook: Installing Dependencies ==="

# Install Python dependencies
if [ -f "yoshi-bot/pyproject.toml" ]; then
  echo "Installing Python dependencies from yoshi-bot/pyproject.toml..."
  cd yoshi-bot
  pip install -e . --quiet
  pip install pytest --quiet
  cd ..
  echo "✓ Python dependencies installed"
else
  echo "⚠ yoshi-bot/pyproject.toml not found"
fi

# Install Node dependencies for clawdbot
if [ -f "clawdbot/package.json" ]; then
  echo "Installing Node dependencies from clawdbot/package.json..."
  cd clawdbot
  npm install --quiet
  cd ..
  echo "✓ ClawdBot dependencies installed"
else
  echo "⚠ clawdbot/package.json not found"
fi

# Install Node dependencies for kalshi-bot
if [ -f "clawdbot/kalshi-bot/package.json" ]; then
  echo "Installing Node dependencies from clawdbot/kalshi-bot/package.json..."
  cd clawdbot/kalshi-bot
  npm install --quiet
  cd ../..
  echo "✓ Kalshi-Bot dependencies installed"
else
  echo "⚠ clawdbot/kalshi-bot/package.json not found"
fi

echo "=== Dependencies installation complete ==="
