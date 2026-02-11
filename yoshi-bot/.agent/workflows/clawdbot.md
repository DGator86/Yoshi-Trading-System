---
description: How to activate and manage the AI audit layer integration
---

# ClawdBot-V1 Activation Workflow

This workflow ensures that the Yoshi-Bot's AI reasoning/audit layer is correctly
configured and operational (OpenRouter or Ollama).

## 1. Prerequisites

- Ollama path:
  - Ensure Ollama is running locally (`ollama serve`)
  - Pull a model (e.g. `ollama pull llama3.1`)
  - Verify `configs/moltbot.yaml` uses `provider: ollama`

- OpenRouter path (optional):
  - Ensure `OPENROUTER_API_KEY` is set in your `.env` file.
  - Verify `configs/moltbot.yaml` uses `provider: openrouter`

## 2. Activate AI Audit Layer

To start the trading orchestrator using the configured provider:

// turbo

```bash
python scripts/run_moltbot.py
```

## 3. Verify Connection

Run the following to check if the AI client can generate a trade plan:

// turbo

```bash
    python scripts/run_moltbot.py --use-stub
```

## 4. Troubleshooting

- **Ollama connection error**: check `endpoint: http://localhost:11434/api/chat`
- **API Key Error** (OpenRouter): check `OPENROUTER_API_KEY` is set.
- **JSON Error**: If ClawdBot returns improperly formatted JSON, the `OpenRouterClient` in `moltbot.py` has an extraction fallback. Check the logs for "Attempting JSON extraction".
