"""
Tests for Time-Aware Re-Ranker (src.retrieval.ranker).
Covers:
- Empty candidate input returns []
- Candidates returned are sorted by final_score descending
- Intent detection triggers proper temporal weighting
"""

from src.retrieval.ranker import TimeAwareReRanker, detect_temporal_intent
from src.chunking.chunk_messages import naive_time_gap_chunking


def test_empty_candidates_returns_empty_list():
    reranker = TimeAwareReRanker(alpha=0.5, decay_lambda=1.0)
    assert reranker.rerank([]) == []
    assert reranker.rerank([], query="What is our latest plan?") == []


def test_candidates_sorted_by_final_score_descending(mini_chat_messages):
    chunks = naive_time_gap_chunking(mini_chat_messages, threshold_minutes=30)
    assert len(chunks) == 3

    # Provide candidates with varying semantic scores
    candidates = [
        {"chunk": chunks[0], "semantic_score": 0.40},
        {"chunk": chunks[1], "semantic_score": 0.90},
        {"chunk": chunks[2], "semantic_score": 0.70},
    ]

    reranker = TimeAwareReRanker(alpha=0.5, decay_lambda=1.0)
    reranked = reranker.rerank(candidates, query="some neutral question")

    assert len(reranked) == 3
    # Check that output is sorted by final_score descending
    for i in range(len(reranked) - 1):
        assert reranked[i]["final_score"] >= reranked[i + 1]["final_score"]

    # Verify expected keys in returned candidate dictionaries
    for item in reranked:
        assert "chunk" in item
        assert "semantic_score" in item
        assert "time_decay" in item
        assert "final_score" in item
        assert "elapsed_seconds" in item


def test_temporal_intent_detection():
    assert detect_temporal_intent("What is our latest plan?") == "latest"
    assert detect_temporal_intent("What are we doing current now?") == "latest"
    assert detect_temporal_intent("What was the initial agreement?") == "historical"
    assert detect_temporal_intent("What did we agree earlier?") == "historical"
    assert detect_temporal_intent("Who brought the documents?") == "neutral"
