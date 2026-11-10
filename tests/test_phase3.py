import numpy as np
import pytest

from src.retrieval.hybrid import BM25, HybridChunkRetriever, char_ngrams
from src.indexing.embedders import ME5Embedder


def test_char_ngram_bm25_ranking():
    # Query has small romanization variation: "unnavaa" vs chunk's "unnav"
    docs = [
        "Alice: em chestunnav bro",
        "Bob: we have a project discussion tomorrow at noon",
        "Charlie: please review the database table schema",
    ]
    bm_char = BM25(docs, tokenizer=char_ngrams)
    scores = bm_char.scores("unnavaa")

    # Chunk 0 containing "unnav" must score highest
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]

    # In HybridChunkRetriever, chunk 0 must rank first for "unnavaa"
    chunk_emb = np.zeros((3, 4), dtype=np.float32)
    msg_emb = np.zeros((3, 4), dtype=np.float32)
    retriever = HybridChunkRetriever(docs, chunk_emb, msg_emb, [0, 1, 2], [4, 8, 6])
    hits = retriever.search("unnavaa", np.zeros(4, dtype=np.float32))
    assert hits[0]["index"] == 0


def test_sender_match_boost():
    # Two chunks with identical semantic text & embeddings, but different senders
    docs = [
        "Nikhil: here is the assignment update",
        "Alice: here is the assignment update",
    ]
    chunk_emb = np.zeros((2, 4), dtype=np.float32)
    msg_emb = np.zeros((2, 4), dtype=np.float32)
    retriever = HybridChunkRetriever(
        docs,
        chunk_emb,
        msg_emb,
        [0, 1],
        [5, 5],
        chunk_senders=[["Nikhil"], ["Alice"]],
    )

    # Query explicitly asks about Nikhil
    hits = retriever.search("Did Nikhil share the assignment update?", np.zeros(4, dtype=np.float32))
    # Chunk with sender Nikhil (index 0) must rank first due to sender_match signal
    assert hits[0]["index"] == 0


def test_me5_embedder_prefixes():
    class StubModel:
        def __init__(self):
            self.recorded = []

        def encode(self, texts, **kwargs):
            self.recorded.extend(texts)
            return np.ones((len(texts), 4), dtype=np.float32)

    emb = ME5Embedder.__new__(ME5Embedder)
    emb.name = "me5"
    emb.model = StubModel()

    # Passages must get "passage: " prefix
    emb.encode(["first doc", "second doc"], is_query=False)
    assert emb.model.recorded == ["passage: first doc", "passage: second doc"]

    # Queries must get "query: " prefix
    emb.encode(["user question"], is_query=True)
    assert emb.model.recorded[-1] == "query: user question"


def test_default_rrf_weights_all_four_rankings():
    retriever = HybridChunkRetriever(
        ["sample doc"],
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        [0],
        [2],
    )
    # Default RRF weights dict must contain all four rankings with weight 1.0
    assert "dense_chunk" in retriever.weights
    assert "dense_message" in retriever.weights
    assert "bm25" in retriever.weights
    assert "bm25_char" in retriever.weights
    assert retriever.weights["dense_chunk"] == 1.0
    assert retriever.weights["dense_message"] == 1.0
    assert retriever.weights["bm25"] == 1.0
    assert retriever.weights["bm25_char"] == 1.0
