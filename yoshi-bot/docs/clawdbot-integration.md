# ClawdBot-V1 / Ollama: AI Audit Layer Integration

This document defines the integration of **ClawdBot-V1** (Claude 3.5 Sonnet) into the **Antigravity** (Yoshi-Bot) ecosystem.

## Overview

This repo supports an AI "Audit Layer" that reviews model forecasts and produces
an actionable trade plan JSON.

You can run this layer via:
- OpenRouter + Claude (ClawdBot-V1)
- Ollama (local LLM)

## Integration Architecture

1. **AI Client**: implemented in `src/gnosis/execution/moltbot.py`
2. **Providers**:
   - `openrouter` / `clawdbot` (OpenRouter)
   - `ollama` (local)
3. **Configuration**: `configs/moltbot.yaml`
4. **Environment**:
   - OpenRouter: `OPENROUTER_API_KEY`
   - Ollama: no API key required (local endpoint)

## Functional Roles

- **Forecast Auditing**: sanity-check forecasts and regimes.
- **Risk Management**: Checking trades against `max_position` and `leverage` limits.
- **Natural Language Reporting**: Generating the "Action" strings for Telegram alerts.

## Workflow Activation

You can now manage this integration using the Antigravity workflow:

- Path: `.agent/workflows/clawdbot.md`
- command: `/clawdbot` (if supported by your interface)

## Maintenance

When updating ClawdBot-V1, ensure that the `system_prompt` in `moltbot.yaml` is kept up-to-date with the latest trading strategies of Yoshi-Bot.
