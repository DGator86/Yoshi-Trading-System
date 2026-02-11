"""Ensure Moltbot selects the Ollama provider."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnosis.execution.moltbot import MoltbotConfig, MoltbotOrchestrator, AIProviderConfig  # noqa: E402


def test_moltbot_uses_ollama_client_by_default_when_configured():
    cfg = MoltbotConfig(ai=AIProviderConfig(provider="ollama", model="llama3.1", endpoint="http://localhost:11434/api/chat"))
    orch = MoltbotOrchestrator(cfg)
    assert orch.ai_client.__class__.__name__ == "OllamaClient"
