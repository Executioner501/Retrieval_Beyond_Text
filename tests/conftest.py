"""
Pytest configuration and shared fixtures for unit tests.
No network calls or Ollama server required.
"""

from pathlib import Path
import pytest
import numpy as np

from src.parser.parse_whatsapp import parse_whatsapp_chat

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def mini_chat_path() -> Path:
    return FIXTURES_DIR / "mini_chat.txt"


@pytest.fixture
def mini_chat_12h_path() -> Path:
    return FIXTURES_DIR / "mini_chat_12h.txt"


@pytest.fixture
def mini_chat_messages(mini_chat_path):
    return parse_whatsapp_chat(str(mini_chat_path), filter_system=True)


@pytest.fixture
def fake_embedder():
    """Returns a deterministic fake embedding function using np.random.default_rng(0)."""
    def _embed(texts):
        rng = np.random.default_rng(0)
        vecs = rng.standard_normal((len(texts), 64), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        return vecs / np.maximum(norms, 1e-12)
    return _embed
