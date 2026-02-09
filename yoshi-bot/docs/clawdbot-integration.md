# ClawdBot-V1: Antigravity-Claude Integration

This document defines the integration of **ClawdBot-V1** (Claude 3.5 Sonnet) into the **Antigravity** (Yoshi-Bot) ecosystem.

## Overview

ClawdBot-V1 serves as the primary reasoning engine for Yoshi-Bot. It acts as the "Audit Layer" that reviews mathematical forecasts from the Price-Time Manifold and generates actionable trade plans.

## Integration Architecture

1. **AI Client**: Implemented as `OpenRouterClient` in `src/gnosis/execution/moltbot.py`.
2. **Provider**: OpenRouter (Model: `anthropic/claude-3.5-sonnet`).
3. **Configuration**: Managed via `configs/moltbot.yaml`.
4. **Environment**: Secured via `OPENROUTER_API_KEY` in `.env`.

## Functional Roles

- **Forecast Auditing**: Validating the physics engine signals.
- **Risk Management**: Checking trades against `max_position` and `leverage` limits.
- **Natural Language Reporting**: Generating the "Action" strings for Telegram alerts.

## Workflow Activation

You can now manage this integration using the Antigravity workflow:

- Path: `.agent/workflows/clawdbot.md`
- command: `/clawdbot` (if supported by your interface)

## Maintenance

When updating ClawdBot-V1, ensure that the `system_prompt` in `moltbot.yaml` is kept up-to-date with the latest trading strategies of Yoshi-Bot.
