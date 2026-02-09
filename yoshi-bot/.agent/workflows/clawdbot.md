---
description: How to activate and manage ClawdBot-V1 (Claude 3.5 Sonnet) integration
---

# ClawdBot-V1 Activation Workflow

This workflow ensures that the Yoshi-Bot's AI reasoning layer (ClawdBot-V1) is correctly configured and operational.

## 1. Prerequisites

- Ensure `OPENROUTER_API_KEY` is set in your `.env` file.
- Verify `configs/moltbot.yaml` is set to use `provider: openrouter` and `model: anthropic/claude-3.5-sonnet`.

## 2. Activate ClawdBot-V1

To start the trading orchestrator using ClawdBot-V1 for reasoning:

// turbo

```bash
python scripts/run_moltbot.py
```

## 3. Verify Connection

Run the following to check if the AI client can generate a trade plan:

// turbo

```bash
python scripts/run_moltbot.py --test-ai
```

## 4. Troubleshooting

- **API Key Error**: check that `OPENROUTER_API_KEY` starts with `sk-or-v1-`.
- **JSON Error**: If ClawdBot returns improperly formatted JSON, the `OpenRouterClient` in `moltbot.py` has an extraction fallback. Check the logs for "Attempting JSON extraction".
