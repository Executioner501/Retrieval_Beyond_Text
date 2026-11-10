from datetime import datetime, timedelta
import math
import numpy as np
import pytest

from src.chunking.chunk_messages import ConversationChunk
from src.parser.parse_whatsapp import MessageRecord
from src.pipeline import ConversationQA
from src.retrieval.ranker import (
    TimeAwareReRanker,
    detect_temporal_intent,
    parse_temporal_query,
)


class DeterministicStubEncoder:
    name = "stub"

    def encode(self, texts, **kwargs):
        # All embeddings are equal on topic dimension to isolate temporal scoring
        vecs = np.ones((len(texts), 4), dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_synthetic_decision_change_latest_and_originally():
    # 3 messages where a decision changes twice
    m1 = MessageRecord(
        message_id="msg_00001",
        timestamp="2024-01-10T10:00:00",
        date="2024-01-10",
        time="10:00",
        sender="Alice",
        text="We originally agreed to use Vue for frontend.",
        is_system=False,
        has_media=False,
        char_count=45,
        word_count=8,
    )
    m2 = MessageRecord(
        message_id="msg_00002",
        timestamp="2024-04-10T10:00:00",
        date="2024-04-10",
        time="10:00",
        sender="Bob",
        text="We updated the plan and changed to React.",
        is_system=False,
        has_media=False,
        char_count=41,
        word_count=8,
    )
    m3 = MessageRecord(
        message_id="msg_00003",
        timestamp="2024-08-10T10:00:00",
        date="2024-08-10",
        time="10:00",
        sender="Alice",
        text="The final and latest decision is Svelte.",
        is_system=False,
        has_media=False,
        char_count=40,
        word_count=7,
    )

    qa = ConversationQA(
        [m1, m2, m3],
        encoder=DeterministicStubEncoder(),
        chunker="naive",
        retrieval="hybrid",
    )

    # Latest question -> latest chunk (m3) ranks first
    latest_hits = qa.retrieve("What is our latest frontend decision?")
    assert latest_hits[0]["chunk"].message_ids == ["msg_00003"]

    # Historical question -> earliest chunk (m1) ranks first
    orig_hits = qa.retrieve("What did we originally agree on?")
    assert orig_hits[0]["chunk"].message_ids == ["msg_00001"]


def test_decay_same_regardless_of_candidate_pool():
    reranker = TimeAwareReRanker(alpha=0.45, decay_lambda=1.2)
    ref_time = datetime(2024, 8, 10, 10, 0)
    chat_span = (ref_time - datetime(2024, 1, 10, 10, 0)).total_seconds()

    c1 = ConversationChunk(
        chunk_id="c1",
        strategy="naive",
        start_timestamp="2024-06-01T10:00:00",
        end_timestamp="2024-06-01T10:00:00",
        message_ids=["m1"],
        participants=["Alice"],
        attributed_text="t1",
        raw_text="t1",
    )
    c2 = ConversationChunk(
        chunk_id="c2",
        strategy="naive",
        start_timestamp="2024-08-01T10:00:00",
        end_timestamp="2024-08-01T10:00:00",
        message_ids=["m2"],
        participants=["Bob"],
        attributed_text="t2",
        raw_text="t2",
    )
    c_old = ConversationChunk(
        chunk_id="c_old",
        strategy="naive",
        start_timestamp="2024-01-10T10:00:00",
        end_timestamp="2024-01-10T10:00:00",
        message_ids=["m0"],
        participants=["Alice"],
        attributed_text="t0",
        raw_text="t0",
    )

    # Candidate pool A: [c1, c2]
    pool_a = [{"chunk": c1, "semantic_score": 0.8}, {"chunk": c2, "semantic_score": 0.8}]
    reranked_a = reranker.rerank(
        pool_a, reference_time=ref_time, query="latest update", chat_span_seconds=chat_span
    )
    decay_c1_a = next(r["time_decay"] for r in reranked_a if r["chunk"].chunk_id == "c1")

    # Candidate pool B: [c1, c2, c_old] - includes a much older candidate
    pool_b = [
        {"chunk": c1, "semantic_score": 0.8},
        {"chunk": c2, "semantic_score": 0.8},
        {"chunk": c_old, "semantic_score": 0.8},
    ]
    reranked_b = reranker.rerank(
        pool_b, reference_time=ref_time, query="latest update", chat_span_seconds=chat_span
    )
    decay_c1_b = next(r["time_decay"] for r in reranked_b if r["chunk"].chunk_id == "c1")

    # With chat_span_seconds and fixed reference_time, decay for c1 must not change
    assert math.isclose(decay_c1_a, decay_c1_b, rel_tol=1e-6)


def test_parse_temporal_query():
    chat_end = datetime(2024, 11, 20, 15, 30)

    # "in March 2024"
    t_march = parse_temporal_query("What happened in March 2024?", chat_end=chat_end)
    assert t_march.date_start == datetime(2024, 3, 1, 0, 0, 0)
    assert t_march.date_end == datetime(2024, 3, 31, 23, 59, 59, 999999)

    # "on 12/11/2024"
    t_date = parse_temporal_query("What was the plan on 12/11/2024?", chat_end=chat_end)
    assert t_date.date_start == datetime(2024, 11, 12, 0, 0, 0)
    assert t_date.date_end == datetime(2024, 11, 12, 23, 59, 59, 999999)

    # "last week" relative to given chat_end
    t_week = parse_temporal_query("What did we discuss last week?", chat_end=chat_end)
    assert t_week.date_start == (chat_end - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    assert t_week.date_end == chat_end.replace(hour=23, minute=59, second=59, microsecond=999999)

    # No-date query returns None range
    t_none = parse_temporal_query("Which framework should we use?", chat_end=chat_end)
    assert t_none.date_start is None
    assert t_none.date_end is None


def test_date_filter_fallback_when_nothing_matches():
    # Chat with messages only in March 2024
    m1 = MessageRecord(
        message_id="msg_00001",
        timestamp="2024-03-10T10:00:00",
        date="2024-03-10",
        time="10:00",
        sender="Alice",
        text="Discussing project details in March.",
        is_system=False,
        has_media=False,
        char_count=36,
        word_count=5,
    )
    qa = ConversationQA(
        [m1],
        encoder=DeterministicStubEncoder(),
        chunker="naive",
        retrieval="hybrid",
    )

    # Query with a date range in August 2025 (no messages match)
    hits = qa.retrieve("What did we discuss on 15/08/2025?")
    # Falls back to unfiltered candidates
    assert len(hits) == 1
    assert hits[0]["chunk"].chunk_id == qa.chunks[0].chunk_id


def test_supersession_chronological_ordering():
    # Test Task 5: For 'latest' query, excerpts with cosine similarity >= 0.5
    # with the top hit are ordered chronologically (oldest first) in user prompt
    m1 = MessageRecord(
        message_id="msg_00001",
        timestamp="2024-01-10T10:00:00",
        date="2024-01-10",
        time="10:00",
        sender="Alice",
        text="First choice Vue.",
        is_system=False,
        has_media=False,
        char_count=17,
        word_count=3,
    )
    m2 = MessageRecord(
        message_id="msg_00002",
        timestamp="2024-08-10T10:00:00",
        date="2024-08-10",
        time="10:00",
        sender="Alice",
        text="Latest choice Svelte.",
        is_system=False,
        has_media=False,
        char_count=21,
        word_count=3,
    )
    qa = ConversationQA(
        [m1, m2],
        encoder=DeterministicStubEncoder(),
        chunker="naive",
        retrieval="hybrid",
    )

    hits = qa.retrieve("What is our latest choice?")
    # In retrieve, m2 (latest) is top hit
    assert hits[0]["chunk"].message_ids == ["msg_00002"]

    # In _build_user_prompt, topic hits are ordered chronologically: [1] is m1, [2] is m2
    prompt = qa._build_user_prompt("What is our latest choice?", hits)
    pos_m1 = prompt.find("First choice Vue")
    pos_m2 = prompt.find("Latest choice Svelte")
    assert pos_m1 != -1 and pos_m2 != -1
    assert pos_m1 < pos_m2
