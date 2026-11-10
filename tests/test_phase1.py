import os
import numpy as np
import pytest

from src.chunking.chunk_messages import ConversationChunk, _parse_ts
from src.parser.parse_whatsapp import parse_whatsapp_chat
from src.retrieval.hybrid import tokenize
from src.indexing.vector_store import LocalVectorStore
from src.indexing.embedders import CACHE_VERSION, CachedEmbedder


def test_parse_ts_raises_on_garbage():
    with pytest.raises(ValueError, match="Unparseable timestamp"):
        _parse_ts("garbage")


def test_parser_month_first_and_day_first(tmp_path):
    # Day-first chat (first number > 12)
    # 04/05/2024 is ambiguous, but because 25/03/2024 is day-first, it should parse as 4th of May (2024-05-04)
    day_first_chat = tmp_path / "day_first.txt"
    day_first_chat.write_text(
        "25/03/2024, 10:00 - Alice: Day first line 1\n"
        "04/05/2024, 10:05 - Bob: Ambiguous date\n",
        encoding="utf-8",
    )
    records_df = parse_whatsapp_chat(str(day_first_chat))
    assert len(records_df) == 2
    assert records_df[0].timestamp == "2024-03-25T10:00:00"
    assert records_df[1].timestamp == "2024-05-04T10:05:00"

    # Month-first chat (second number > 12)
    # 04/05/2024 is ambiguous, but because 03/25/2024 is month-first, it should parse as 5th of April (2024-04-05)
    month_first_chat = tmp_path / "month_first.txt"
    month_first_chat.write_text(
        "03/25/2024, 10:00 - Alice: Month first line 1\n"
        "04/05/2024, 10:05 - Bob: Ambiguous date\n",
        encoding="utf-8",
    )
    records_mf = parse_whatsapp_chat(str(month_first_chat))
    assert len(records_mf) == 2
    assert records_mf[0].timestamp == "2024-03-25T10:00:00"
    assert records_mf[1].timestamp == "2024-04-05T10:05:00"


def test_parser_raises_when_unparseable_gt_1_percent(tmp_path):
    # 1 valid message, 1 invalid calendar date (Feb 31) -> 50% failures > 1%
    chat_file = tmp_path / "corrupt_chat.txt"
    chat_file.write_text(
        "20/03/2024, 10:00 - Alice: Valid message\n"
        "31/02/2024, 10:05 - Bob: Invalid date\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Failed to parse timestamps"):
        parse_whatsapp_chat(str(chat_file))


def test_tokenize_multilingual_and_collapse():
    # Drops single ASCII characters ("a", "1"), keeps single non-ASCII (e.g. CJK "日", Telugu char)
    assert tokenize("a 1 日 b") == ["日"]

    # Collapses repetitions 3+ to 2
    assert tokenize("nuvvuuuu") == ["nuvvuu"]

    # Multilingual Telugu + English + slang
    tokens = tokenize("నమస్తే bro 😂 nuvvuuuu")
    assert "bro" in tokens
    assert "nuvvuu" in tokens
    # Preserves Telugu tokens (including single non-ASCII char)
    assert any("\u0c28" in t for t in tokens)
    assert "\u0c24" in tokens


def test_vector_store_raises_when_no_encoder():
    store = LocalVectorStore()
    store._get_encoder = lambda: None

    chunk = ConversationChunk(
        chunk_id="c0",
        strategy="naive_time_gap",
        start_timestamp="2024-01-01T10:00:00",
        end_timestamp="2024-01-01T10:05:00",
        message_ids=["m0"],
        participants=["Alice"],
        attributed_text="[2024-01-01] Alice: Hi",
        raw_text="Alice: Hi",
    )

    with pytest.raises(RuntimeError, match="sentence-transformers is required"):
        store.add_chunks([chunk])

    store.chunks = [chunk]
    store.embeddings = np.zeros((1, 384))
    with pytest.raises(RuntimeError, match="sentence-transformers is required"):
        store.search("test query")


class StubEncoder:
    def __init__(self, name: str = "stub_model"):
        self.name = name
        self.call_count = 0

    def encode(self, texts, **kwargs):
        self.call_count += 1
        return np.ones((len(texts), 4), dtype=np.float32)


def test_cached_embedder(tmp_path, monkeypatch):
    monkeypatch.setattr("src.indexing.embedders.CACHE_DIR", str(tmp_path))

    stub = StubEncoder("test_model")
    cached = CachedEmbedder(stub, "cache_stub")

    # Encode first time
    v1 = cached.encode(["hello"])
    assert stub.call_count == 1

    # Encode second time: vector identical, call_count unchanged
    v2 = cached.encode(["hello"])
    assert stub.call_count == 1
    assert np.allclose(v1, v2)

    # Cache key differs between models on identical text
    stub1 = StubEncoder("model_a")
    stub2 = StubEncoder("model_b")
    c1 = CachedEmbedder(stub1, "c1")
    c2 = CachedEmbedder(stub2, "c2")
    assert c1._key("hello") != c2._key("hello")

    # Caches with a different CACHE_VERSION are ignored
    # Save a cache with version = 999
    fake_path = tmp_path / "cache_vdiff_v1.npz"
    np.savez(
        fake_path,
        keys=np.array(["testkey"]),
        vectors=np.ones((1, 4), dtype=np.float32),
        version=np.array(999),
    )

    stub_v = StubEncoder("model_v")
    cached_v = CachedEmbedder(stub_v, "cache_vdiff")
    # Cache should be ignored/empty because version 999 != CACHE_VERSION
    assert len(cached_v.cache) == 0
