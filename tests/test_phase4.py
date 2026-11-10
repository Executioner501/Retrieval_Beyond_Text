import json
from datetime import datetime
import numpy as np
import pytest

from scripts.split_eval import split_eval
from scripts.run_qa_eval import bootstrap_ci, locate_gold
from src.parser.parse_whatsapp import MessageRecord
from src.pipeline import ConversationQA


def test_split_eval_deterministic_and_disjoint(tmp_path):
    input_file = "data/benchmarks/evaluation_set_v2.json"
    with open(input_file, encoding="utf-8") as f:
        all_items = json.load(f)

    dev_1 = str(tmp_path / "dev_1.json")
    test_1 = str(tmp_path / "test_1.json")
    dev_2 = str(tmp_path / "dev_2.json")
    test_2 = str(tmp_path / "test_2.json")

    d1, t1 = split_eval(input_file, dev_1, test_1, seed=42)
    d2, t2 = split_eval(input_file, dev_2, test_2, seed=42)

    # Determinism: identical files generated on both runs
    assert d1 == d2
    assert t1 == t2

    # Disjointness: no questions in common between dev and test
    dev_ids = {q["id"] for q in d1}
    test_ids = {q["id"] for q in t1}
    assert dev_ids.isdisjoint(test_ids)

    # Coverage: union of dev and test covers all questions exactly
    all_ids = {q["id"] for q in all_items}
    assert (dev_ids | test_ids) == all_ids


def test_bootstrap_ci_returns_valid_bounds():
    toy_ranks = [1, 2, None, 1, 4, 3, None, 1, 2, None]

    # hit@1
    mean_hit1 = sum(r == 1 for r in toy_ranks) / len(toy_ranks)
    lo, hi = bootstrap_ci(toy_ranks, metric="hit@1", n_resamples=1000, seed=42)
    assert lo <= round(mean_hit1, 3) <= hi

    # hit@k
    mean_hitk = sum(r is not None for r in toy_ranks) / len(toy_ranks)
    lo, hi = bootstrap_ci(toy_ranks, metric="hit@k", n_resamples=1000, seed=42)
    assert lo <= round(mean_hitk, 3) <= hi

    # mrr
    mean_mrr = sum(1 / r for r in toy_ranks if r) / len(toy_ranks)
    lo, hi = bootstrap_ci(toy_ranks, metric="mrr", n_resamples=1000, seed=42)
    assert lo <= round(mean_mrr, 3) <= hi


def test_message_window_hit_within_plus_minus_two():
    # 6 chronological messages: m0, m1, m2, m3, m4, m5
    messages = [
        MessageRecord(
            message_id=f"m_{i}",
            timestamp=f"2024-01-01T10:0{i}:00",
            date="2024-01-01",
            time=f"10:0{i}",
            sender=f"User{i}",
            text=f"Message turn number {i}",
            is_system=False,
            has_media=False,
            char_count=21,
            word_count=4,
        )
        for i in range(6)
    ]

    class TargetStubEncoder:
        name = "target_stub"

        def encode(self, texts, is_query=False, **kwargs):
            out = []
            for t in texts:
                # Give m_1 highest similarity with query
                if "1" in t or is_query:
                    out.append([1.0, 0.0, 0.0, 0.0])
                else:
                    out.append([0.0, 1.0, 0.0, 0.0])
            return np.array(out, dtype=np.float32)

    qa = ConversationQA(
        messages,
        encoder=TargetStubEncoder(),
        retrieval="message_window",
        top_k=1,
    )

    hits = qa.retrieve("query targeting message 1")
    retrieved_chunk = hits[0]["chunk"]

    # Since m_1 is retrieved, its +/- 2 window includes m_0, m_1, m_2, m_3
    # m_3 is within +2 distance, so it must be included in the window
    assert "m_3" in retrieved_chunk.message_ids
    assert "m_0" in retrieved_chunk.message_ids
    # m_4 is distance 3, so it must NOT be in the window
    assert "m_4" not in retrieved_chunk.message_ids

    # locate_gold checks if gold is in chunk.message_ids
    gold_set = {"m_3"}
    assert bool(gold_set & set(retrieved_chunk.message_ids)) is True
