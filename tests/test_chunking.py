"""
Tests for Conversational Chunking Engines.
Covers:
- naive_time_gap_chunking splits at >30 min gaps
- semantic_topic_chunking respects min_chunk_size and max_chunk_size
- every message appears in exactly one chunk across strategies
"""

import numpy as np
from src.chunking.chunk_messages import naive_time_gap_chunking, context_aware_chunking
from src.chunking.semantic_topic_chunking import semantic_topic_chunking


def test_naive_time_gap_chunking_splits_at_30_mins(mini_chat_messages):
    # mini_chat has 3 clear time blocks:
    # 1. 09:31 - 09:38 (7 msgs)
    # Gap of 82 mins (> 30 min)
    # 2. 11:00 - 11:10 (3 msgs)
    # Gap of 170 mins (> 30 min)
    # 3. 14:00 - 14:10 (4 msgs)
    chunks = naive_time_gap_chunking(mini_chat_messages, threshold_minutes=30)
    assert len(chunks) == 3

    assert len(chunks[0].message_ids) == 7
    assert len(chunks[1].message_ids) == 3
    assert len(chunks[2].message_ids) == 4

    # If threshold is large (e.g. 300 mins), everything falls into 1 chunk
    single_chunk = naive_time_gap_chunking(mini_chat_messages, threshold_minutes=300)
    assert len(single_chunk) == 1
    assert len(single_chunk[0].message_ids) == len(mini_chat_messages)


def test_naive_chunking_message_coverage(mini_chat_messages):
    chunks = naive_time_gap_chunking(mini_chat_messages, threshold_minutes=30)
    all_chunked_ids = [mid for c in chunks for mid in c.message_ids]
    original_ids = [m.message_id for m in mini_chat_messages]

    # Every message appears in exactly one chunk
    assert len(all_chunked_ids) == len(set(all_chunked_ids))
    assert sorted(all_chunked_ids) == sorted(original_ids)


def test_semantic_topic_chunking_bounds_and_coverage(mini_chat_messages):
    n = len(mini_chat_messages)
    assert n == 14

    # Generate deterministic fake embeddings with default_rng(0)
    rng = np.random.default_rng(0)
    fake_embeddings = rng.standard_normal((n, 32), dtype=np.float32)
    fake_embeddings /= np.linalg.norm(fake_embeddings, axis=-1, keepdims=True)

    min_size = 3
    max_size = 6
    chunks = semantic_topic_chunking(
        mini_chat_messages,
        embeddings=fake_embeddings,
        min_chunk_size=min_size,
        max_chunk_size=max_size,
        window=2,
    )

    assert len(chunks) >= 2

    # Check bounds on every chunk
    for c in chunks:
        chunk_len = len(c.message_ids)
        assert chunk_len >= min_size, f"Chunk {c.chunk_id} size {chunk_len} < min {min_size}"
        assert chunk_len <= max_size, f"Chunk {c.chunk_id} size {chunk_len} > max {max_size}"

    # Verify complete partition: every message appears in exactly one chunk
    all_chunked_ids = [mid for c in chunks for mid in c.message_ids]
    original_ids = [m.message_id for m in mini_chat_messages]
    assert len(all_chunked_ids) == len(set(all_chunked_ids))
    assert sorted(all_chunked_ids) == sorted(original_ids)


def test_empty_messages_returns_empty_chunks():
    assert naive_time_gap_chunking([]) == []
    assert context_aware_chunking([]) == []
    assert semantic_topic_chunking([]) == []
