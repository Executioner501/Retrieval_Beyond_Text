"""
Semantic Topic-Shift Chunking (no hardcoded topic rules).

Detects topic boundaries in a chat automatically, using embeddings instead of
regex discourse markers. It is an embedding version of TextTiling (Hearst, 1997)
adapted to short chat messages:

1. Embed every message with a sentence-embedding model.
2. At each gap between message i-1 and i, compare the context BEFORE the gap
   (last `window` messages) with the context AFTER it (next `window` messages).
   Short messages ("ok", "same") get low weight so they don't dominate.
3. Turn the similarity curve into "depth scores": how deep a dip in similarity
   is compared to the peaks around it. A deep dip = a topic shift.
4. Optionally blend in the silence before the message (log of the time gap),
   because long silences in chats usually mean a new conversation.
5. Both signals are standardised (z-scores) per chat, and the cut-off is
   derived from that chat's own score distribution, so there are no fixed
   thresholds tied to a particular dataset or vocabulary.
6. Silences that are outliers for this chat always start a new chunk.
   Size bounds are enforced: boundaries closer than `min_chunk_size` are
   suppressed (weakest first), and oversized segments are split at their
   weakest internal point rather than at an arbitrary message count.

Output uses the same ConversationChunk format as the rest of the pipeline,
so it drops into the indexing / retrieval stages unchanged.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..parser.parse_whatsapp import MessageRecord
from .chunk_messages import ConversationChunk, _build_chunk, _parse_ts

EmbedFn = Callable[[List[str]], np.ndarray]

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_CACHE = {}


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def default_embed_fn(model_name: str = DEFAULT_MODEL) -> EmbedFn:
    """Returns an embed function backed by a (cached) SentenceTransformer."""

    def _embed(texts: List[str]) -> np.ndarray:
        if model_name not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        return _MODEL_CACHE[model_name].encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )

    return _embed


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


# --------------------------------------------------------------------------- #
# Boundary detection
# --------------------------------------------------------------------------- #
@dataclass
class BoundaryResult:
    """Everything computed during segmentation, useful for plots and tuning.

    Arrays are indexed by gap: entry g describes the gap between message g and
    message g+1, so a boundary at gap g means a new chunk starts at message g+1.
    """
    similarity: np.ndarray          # cross-window cosine similarity per gap
    depth: np.ndarray               # TextTiling depth score per gap
    log_time_gap: np.ndarray        # log(1 + minutes of silence) per gap
    combined: np.ndarray            # final boundary score per gap
    threshold: float                # chat-specific cut-off applied to `combined`
    boundaries: List[int] = field(default_factory=list)  # chunk START indices (excluding 0)
    silence_threshold_minutes: Optional[float] = None    # learned "new conversation" silence


def _window_vectors(emb: np.ndarray, weights: np.ndarray, window: int):
    """Weighted mean embedding of the `window` messages left/right of each gap."""
    n = len(emb)
    left, right = [], []
    for g in range(n - 1):
        lo, hi = max(0, g + 1 - window), min(n, g + 1 + window)
        lw, rw = weights[lo:g + 1], weights[g + 1:hi]
        left.append((emb[lo:g + 1] * lw[:, None]).sum(0) / lw.sum())
        right.append((emb[g + 1:hi] * rw[:, None]).sum(0) / rw.sum())
    return _l2_normalise(np.array(left)), _l2_normalise(np.array(right))


def _depth_scores(sim: np.ndarray) -> np.ndarray:
    """Hearst depth: climb to the highest similarity on each side of a dip."""
    depth = np.zeros_like(sim)
    for i in range(len(sim)):
        lpeak = sim[i]
        for j in range(i - 1, -1, -1):
            if sim[j] < lpeak:
                break
            lpeak = sim[j]
        rpeak = sim[i]
        for j in range(i + 1, len(sim)):
            if sim[j] < rpeak:
                break
            rpeak = sim[j]
        depth[i] = (lpeak - sim[i]) + (rpeak - sim[i])
    return depth


def _otsu_threshold(x: np.ndarray, bins: int = 64) -> Optional[float]:
    """Threshold that best separates a 1-D sample into two groups (Otsu, 1979)."""
    hist, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    best, thr = -1.0, None
    for i in range(1, bins):
        w0, w1 = hist[:i].sum(), hist[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (hist[:i] * centers[:i]).sum() / w0
        m1 = (hist[i:] * centers[i:]).sum() / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best:
            best, thr = between, edges[i]
    return thr


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def _time_gaps_minutes(messages: Sequence[MessageRecord]) -> np.ndarray:
    ts = [_parse_ts(m.timestamp) for m in messages]
    gaps = [max(0.0, (ts[i + 1] - ts[i]).total_seconds() / 60.0) for i in range(len(ts) - 1)]
    return np.array(gaps, dtype=float)


def detect_topic_boundaries(
    messages: Sequence[MessageRecord],
    embed_fn: Optional[EmbedFn] = None,
    window: int = 4,
    time_weight: float = 0.5,
    sensitivity: float = 0.5,
    min_chunk_size: int = 3,
    max_chunk_size: Optional[int] = 15,
    embeddings: Optional[np.ndarray] = None,
    split_on_long_silence: bool = True,
) -> BoundaryResult:
    """
    Finds topic-shift boundaries automatically.

    Args:
        messages: parsed messages in chronological order.
        embed_fn: texts -> (n, d) array. Defaults to all-MiniLM-L6-v2.
        window: messages on each side of a gap used as context. Larger = smoother,
            fewer false splits on short replies; smaller = catches quick switches.
        time_weight: how much the silence before a message counts relative to the
            semantic signal (0 = pure semantics).
        sensitivity: cut-off = mean + (1 - 2*sensitivity) * std of the peak scores,
            computed per chat. 0.5 -> mean of peaks; higher -> more boundaries.
        min_chunk_size: minimum messages per chunk.
        max_chunk_size: oversized segments are split at their weakest gap (None = no cap).
        embeddings: precomputed message embeddings (skips embed_fn).
        split_on_long_silence: always split at silences longer than this chat's
            "new conversation" threshold (Otsu's method on log gaps), even if the chunk
            ends up below min_chunk_size.
    """
    n = len(messages)
    empty = np.zeros(max(n - 1, 0))
    if n < 2:
        return BoundaryResult(empty, empty, empty, empty, 0.0, [])

    # 1. Message embeddings
    if embeddings is None:
        embed_fn = embed_fn or default_embed_fn()
        embeddings = embed_fn([m.text for m in messages])
    emb = _l2_normalise(np.asarray(embeddings, dtype=float))

    # 2. Cross-window similarity; short messages carry less weight
    weights = np.log1p(np.array([max(m.word_count, 1) for m in messages], dtype=float))
    left, right = _window_vectors(emb, weights, max(1, window))
    sim = (left * right).sum(axis=1)

    # 3. Depth scores. Near the start/end of the chat one window is truncated,
    #    which makes similarity noisy; scale those gaps down by window coverage.
    w = max(1, window)
    coverage = np.array([min(g + 1, n - g - 1, w) / w for g in range(n - 1)])
    depth = _depth_scores(sim) * coverage

    # 4. Time signal
    log_gap = np.log1p(_time_gaps_minutes(messages))

    # 5. Combined, standardised per chat
    combined = _zscore(depth) + time_weight * _zscore(log_gap)
    # Candidate boundaries: local maxima ("peaks") of the combined score.
    # The cut-off is computed over the peaks only, so it adapts to how
    # pronounced topic shifts are in this particular chat.
    peaks = [
        g for g in range(len(combined))
        if combined[g] >= (combined[g - 1] if g > 0 else -np.inf)
        and combined[g] >= (combined[g + 1] if g < len(combined) - 1 else -np.inf)
    ]
    pv = combined[peaks]
    threshold = float(pv.mean() + (1.0 - 2.0 * sensitivity) * pv.std())
    cands = [g for g in peaks if combined[g] > threshold]

    # 6a. Silences longer than this chat's "new conversation" threshold always split,
    #     even if that leaves a small chunk. The threshold is found with Otsu's method
    #     on the log gaps: chat gaps are bimodal (replies within minutes vs. new
    #     conversations hours/days later) and Otsu picks the split between the two modes.
    #     Without this, sparse one-to-one chats glued different days into one chunk.
    forced: List[int] = []
    silence_threshold = None
    if split_on_long_silence and len(log_gap) >= 10:
        t = _otsu_threshold(log_gap)
        # Only trust the split if the two groups are far apart: the typical "long"
        # silence must be several times longer than the typical reply gap
        # (difference of means >= 2 in log space, i.e. ~7x). Chats with no clear
        # "new conversation" gaps skip this step.
        if t is not None and (log_gap > t).any() and \
                log_gap[log_gap > t].mean() - log_gap[log_gap <= t].mean() >= 2.0:
            silence_threshold = float(np.expm1(t))
            forced = [g + 1 for g in range(len(log_gap)) if log_gap[g] > t]

    # 6b. Minimum size: keep strongest boundaries first, drop ones too close
    chosen: List[int] = list(forced)  # chunk start indices
    for g in sorted(cands, key=lambda g: combined[g], reverse=True):
        start = g + 1
        if start in chosen or start < min_chunk_size or n - start < min_chunk_size:
            continue
        if all(abs(start - c) >= min_chunk_size for c in chosen):
            chosen.append(start)
    chosen.sort()

    # 6c. Maximum size: split oversized segments at their weakest internal gap
    if max_chunk_size:
        chosen = _enforce_max_size(chosen, n, combined, min_chunk_size, max_chunk_size)

    # 6d. A chunk of one message (a lone "ok", or a question nobody answered for a
    #     while) has no context. Attach it to whichever neighbour is closer in time.
    chosen = _merge_singletons(chosen, n, log_gap)

    return BoundaryResult(sim, depth, log_gap, combined, threshold, chosen, silence_threshold)


def _merge_singletons(starts: List[int], n: int, log_gap: np.ndarray) -> List[int]:
    starts = sorted(starts)
    changed = True
    while changed:
        changed = False
        edges = [0] + starts + [n]
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a != 1:
                continue
            left = log_gap[a - 1] if a > 0 else np.inf      # silence before the message
            right = log_gap[a] if b < n else np.inf          # silence after it
            drop = a if left <= right else b                 # merge into the closer side
            if drop in starts:
                starts.remove(drop)
                changed = True
                break
    return starts


def _enforce_max_size(starts, n, combined, min_size, max_size) -> List[int]:
    edges = [0] + list(starts) + [n]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out.extend(_split_segment(a, b, combined, min_size, max_size))
    return sorted(set(s for s in out if 0 < s < n))


def _split_segment(a, b, combined, min_size, max_size) -> List[int]:
    """Returns chunk start indices for segment [a, b), including a."""
    if b - a <= max_size:
        return [a]
    lo, hi = a + max(min_size, 1), b - max(min_size, 1)
    if lo > hi:  # cannot respect min size; fall back to the midpoint
        lo = hi = (a + b) // 2
    # A boundary starting at message s corresponds to gap s-1
    s = max(range(lo, hi + 1), key=lambda s: combined[s - 1])
    return _split_segment(a, s, combined, min_size, max_size) + \
        _split_segment(s, b, combined, min_size, max_size)


# --------------------------------------------------------------------------- #
# Chunking API (drop-in alongside the existing strategies)
# --------------------------------------------------------------------------- #
def semantic_topic_chunking(
    messages: List[MessageRecord],
    embed_fn: Optional[EmbedFn] = None,
    window: int = 4,
    time_weight: float = 0.5,
    sensitivity: float = 0.5,
    min_chunk_size: int = 3,
    max_chunk_size: Optional[int] = 15,
    return_diagnostics: bool = False,
    embeddings: Optional[np.ndarray] = None,
    split_on_long_silence: bool = True,
):
    """
    Chunks a conversation at automatically detected topic shifts.

    Returns List[ConversationChunk], or (chunks, BoundaryResult) if
    return_diagnostics=True.
    """
    if not messages:
        return ([], None) if return_diagnostics else []

    result = detect_topic_boundaries(
        messages, embed_fn=embed_fn, window=window, time_weight=time_weight,
        sensitivity=sensitivity, min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size,
        embeddings=embeddings, split_on_long_silence=split_on_long_silence,
    )

    edges = [0] + result.boundaries + [len(messages)]
    chunks: List[ConversationChunk] = []
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        chunk = _build_chunk(messages[a:b], f"SEM-CHUNK-{k:03d}", "semantic_topic")
        if a > 0:
            g = a - 1
            chunk.metadata.update({
                "boundary_score": float(result.combined[g]),
                "boundary_similarity": float(result.similarity[g]),
                "boundary_depth": float(result.depth[g]),
                "silence_minutes_before": float(np.expm1(result.log_time_gap[g])),
            })
        chunks.append(chunk)

    return (chunks, result) if return_diagnostics else chunks
