"""
Hybrid retrieval: keyword (BM25) + dense embeddings, merged with Reciprocal Rank Fusion.

Three ranked lists of chunks are produced for each question:
  1. dense_chunk   cosine similarity between the question and the whole chunk
  2. dense_message best cosine similarity between the question and any single message in
                   the chunk (so one relevant message is not diluted by the rest of the chunk)
  3. bm25          keyword match on the chunk text; catches exact names, terms and
                   Telugu/English slang that embedding models handle poorly

RRF score(chunk) = sum over lists of  weight / (k + rank in that list)
RRF only uses ranks, so the different score scales of BM25 and cosine similarity
do not need to be calibrated against each other.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence
import unicodedata

import numpy as np


def tokenize(text: str) -> List[str]:
    norm = unicodedata.normalize("NFKC", text).lower()
    raw = re.findall(r"\w+", norm, flags=re.UNICODE)
    tokens: List[str] = []
    for t in raw:
        t = re.sub(r"(.)\1{2,}", r"\1\1", t)
        if len(t) == 1 and t.isascii():
            continue
        if t:
            tokens.append(t)
    return tokens


def char_ngrams(
    text: str,
    n_range: Sequence[int] = (3, 4),
    boundary: str = "^",
    end_boundary: str = "$",
) -> List[str]:
    words = tokenize(text)
    ngrams: List[str] = []
    for w in words:
        padded = f"{boundary}{w}{end_boundary}"
        for n in n_range:
            if len(padded) >= n:
                for i in range(len(padded) - n + 1):
                    ngrams.append(padded[i : i + n])
    return ngrams


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75, tokenizer=None):
        self.k1, self.b = k1, b
        self.tokenizer = tokenizer or tokenize
        self.docs = [Counter(self.tokenizer(d)) for d in docs]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg_len = float(self.lengths.mean()) if len(self.docs) else 0.0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        q_terms = set(self.tokenizer(query))
        for i, d in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avg_len, 1e-9))
            s = 0.0
            for t in q_terms:
                tf = d.get(t)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            out[i] = s
        return out


def rrf_fuse(rankings: Dict[str, np.ndarray], weights: Dict[str, float], k: int = 60,
             depth: int = 100) -> Dict[int, float]:
    """rankings: name -> array of item indices, best first."""
    fused: Dict[int, float] = {}
    for name, order in rankings.items():
        w = weights.get(name, 1.0)
        for rank, idx in enumerate(order[:depth], start=1):
            fused[int(idx)] = fused.get(int(idx), 0.0) + w / (k + rank)
    return fused


class HybridChunkRetriever:
    """
    Args:
        chunk_texts:   text of each chunk used for BM25 (e.g. "Sender: message" lines)
        chunk_emb:     (n_chunks, d) normalised chunk embeddings
        msg_emb:       (n_messages, d) normalised message embeddings
        msg_chunk:     chunk index of every message
        msg_words:     word count of every message; very short messages ("ok", "same")
                       are left out of the best-message score because they match anything
        chunk_senders: optional list of list of participant names per chunk
    """

    def __init__(self, chunk_texts, chunk_emb, msg_emb, msg_chunk, msg_words,
                 min_msg_words: int = 3, weights=None, chunk_senders: Optional[List[List[str]]] = None):
        self.bm25 = BM25(chunk_texts)
        self.bm25_char = BM25(chunk_texts, tokenizer=char_ngrams)
        self.chunk_emb = chunk_emb
        keep = np.asarray(msg_words) >= min_msg_words
        self.msg_emb = msg_emb[keep]
        self.msg_chunk = np.asarray(msg_chunk)[keep]
        self.n = len(chunk_texts)
        self.weights = weights or {
            "dense_chunk": 1.0,
            "dense_message": 1.0,
            "bm25": 1.0,
            "bm25_char": 1.0,
        }
        self.chunk_senders = chunk_senders

    def search(self, query: str, q_emb: np.ndarray, top_n: int = 20) -> List[Dict]:
        chunk_sim = self.chunk_emb @ q_emb
        best_msg = np.full(self.n, -1.0)
        if len(self.msg_emb):
            np.maximum.at(best_msg, self.msg_chunk, self.msg_emb @ q_emb)
        bm25 = self.bm25.scores(query)
        bm25_char = self.bm25_char.scores(query)

        rankings = {
            "dense_chunk": np.argsort(-chunk_sim),
            "dense_message": np.argsort(-best_msg),
        }
        if bm25.max() > 0:  # no keyword overlap at all -> leave BM25 out
            rankings["bm25"] = np.argsort(-bm25)
        if bm25_char.max() > 0:  # no character n-gram overlap -> leave out
            rankings["bm25_char"] = np.argsort(-bm25_char)

        if self.chunk_senders:
            q_lower = query.lower()
            all_senders = {s for senders in self.chunk_senders for s in senders if s}
            matched_senders = set()
            for sender in all_senders:
                s_clean = sender.strip()
                if not s_clean:
                    continue
                s_lower = s_clean.lower()
                first_name = s_lower.split()[0]
                if re.search(rf"\b{re.escape(s_lower)}\b", q_lower):
                    matched_senders.add(s_clean)
                elif re.search(rf"\b{re.escape(first_name)}\b", q_lower):
                    matched_senders.add(s_clean)
                elif len(first_name) >= 4:
                    q_words = re.findall(r"\w+", q_lower)
                    for qw in q_words:
                        if len(qw) >= 4 and first_name.startswith(qw):
                            matched_senders.add(s_clean)
                            break

            if matched_senders:
                matched_idx = []
                unmatched_idx = []
                for idx, senders in enumerate(self.chunk_senders):
                    senders_set = set(senders)
                    if senders_set & matched_senders:
                        matched_idx.append(idx)
                    else:
                        unmatched_idx.append(idx)

                dense_order = np.argsort(-chunk_sim)
                dense_rank_map = {idx: r for r, idx in enumerate(dense_order)}
                matched_idx.sort(key=lambda i: dense_rank_map.get(i, 999999))
                unmatched_idx.sort(key=lambda i: dense_rank_map.get(i, 999999))

                if matched_idx:
                    rankings["sender_match"] = np.array(matched_idx + unmatched_idx, dtype=int)

        fused = rrf_fuse(rankings, self.weights)
        order = sorted(fused, key=fused.get, reverse=True)[:top_n]
        return [{
            "index": i,
            "fused_score": fused[i],
            "semantic_score": float(max(chunk_sim[i], best_msg[i])),
            "chunk_similarity": float(chunk_sim[i]),
            "best_message_similarity": float(best_msg[i]),
            "bm25": float(bm25[i]),
            "bm25_char": float(bm25_char[i]),
        } for i in order]
