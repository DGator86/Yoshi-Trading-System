# Yoshi Trading System (Monorepo)

A unified system for cryptocurrency prediction market trading on Kalshi.

## Architecture

```
yoshi-bot/ (Signal Engine)             clawdbot/ (Telegram Interface)
=======================             ============================

kalshi_scanner.py (loop)             moltbot gateway :18789
  | Fetches OHLCV data                 | yoshi-trading skill
  | Runs PriceTimeManifold             | Reads Trading Core API
  | Finds Kalshi edge opportunities    | Formats trade suggestions
  | Writes to scanner.log              | Sends to Telegram
  v                                    v
yoshi-bridge.py                      You (Telegram)
  | Watches scanner.log                | "What's Yoshi's status?"
  | Parses signals (edge, strike, etc) | "Show me positions"
  | POSTs to Trading Core /propose     | "approve" / "pass"
  v                                    |
Trading Core API :8000  <--------------+
  /status    /positions
  /propose   /approve/{id}
  /orders    /kill-switch
  /pause     /resume
  /flatten
```

## Structure

- **`yoshi-bot/`**: Contains the core engine, scanners, and local API.
- **`clawdbot/`**: Contains the Telegram assistant, the bridge script, and AI skills.

## Quick Start

1. **Start the Trading Core**:

   ```bash
   cd yoshi-bot
   python scripts/start_trading_core.py
   ```

2. **Start the Scanner**:

   ```bash
   cd yoshi-bot
   python scripts/kalshi_scanner.py --symbol BTCUSDT --loop
   ```

3. **Start the Bridge**:

   ```bash
   cd clawdbot
   python scripts/yoshi-bridge.py
   ```

4. **Start the Telegram Bot**:

   ```bash
   cd clawdbot
   python scripts/telegram-bot.py
   ```
