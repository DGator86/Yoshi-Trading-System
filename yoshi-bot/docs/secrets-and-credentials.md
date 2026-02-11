# Secrets and Credentials Setup

Use environment variables for all credentials. Do not hardcode tokens in code,
docs, config files, commit messages, or logs.

## Immediate security steps

1. Rotate any credential that has been shared in chat/logs/screenshots.
2. Store new credentials in a local `.env` (gitignored) or secret manager.
3. Restart services after rotation.

## Environment variable map

Set only what you use:

- Core model/LLM:
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL`
  - `OPENROUTER_API_KEY`
  - `ANTHROPIC_BASE_URL`
  - `ANTHROPIC_AUTH_TOKEN`

- Market data:
  - `COINGECKO_API_KEY`
  - `COINMARKETCAP_API_KEY`
  - `COINAPI_API_KEY`
  - `CRYPTO_API_KEY`
  - `BEAMAPI_KEY`
  - `MASSIVE_API_KEY`
  - `UNUSUAL_WHALES_API_TOKEN`

- Trading/execution:
  - `ALPACA_BASE_URL`
  - `ALPACA_API_KEY`
  - `ALPACA_API_SECRET`
  - `CCXT_API_KEY`
  - `CCXT_SECRET`

- Infrastructure/automation:
  - `DIGITALOCEAN_TOKEN`
  - `N8N_API_TOKEN`
  - `GITHUB_TOKEN`
  - `VPS_IP`

- Telegram:
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_CHAT_ID`
  - `TELEGRAM_BOT_TOKENS` (comma-separated, optional)

- Kalshi:
  - `KALSHI_KEY_ID`
  - `KALSHI_PRIVATE_KEY`

## Local setup

1. Copy template:
   ```bash
   cp .env.example .env
   ```
2. Fill values in `.env`.
3. Keep file permissions strict:
   ```bash
   chmod 600 .env
   ```

## Guardrails

- Never commit `.env`.
- Never include raw tokens in scripts/tests.
- Prefer environment lookups with clear error messages when missing.
