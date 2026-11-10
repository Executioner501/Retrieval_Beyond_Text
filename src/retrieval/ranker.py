"""
Module 4: Time-Aware Semantic Retrieval & Re-ranking.
Implements the core mathematical formulation:
Score(c) = alpha * Sim(q, c) + (1 - alpha) * exp(-lambda * delta_t_norm)
"""

import re
from typing import List, Dict, Any
from datetime import datetime
import math
from ..chunking.chunk_messages import _parse_ts


"""
Module 4: Time-Aware Semantic Retrieval & Re-ranking.
Implements the core mathematical formulation:
Score(c) = alpha * Sim(q, c) + (1 - alpha) * exp(-lambda * delta_t_norm)
"""

import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import re
from typing import Any, Dict, List, Optional

from ..chunking.chunk_messages import _parse_ts


@dataclass
class TemporalQuery:
    intent: str  # "latest" | "historical" | "neutral"
    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None


_LATEST_RE = re.compile(
    r"\b(latest|most\s+recent|current|currently|now|final|finally|changed\s+to|updated)\b",
    re.IGNORECASE,
)
_HISTORICAL_RE = re.compile(
    r"\b(originally|at\s+first|initial|initially|earlier|back\s+then)\b",
    re.IGNORECASE,
)
_DMY_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_REL_RE = re.compile(r"\b(yesterday|last\s+week|last\s+month)\b", re.IGNORECASE)
_MONTH_RE = re.compile(
    r"\b(?:in\s+)?(january|february|march|april|june|july|august|september|october|november|december|in\s+may|may\s+\d{4})(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_temporal_query(query: str, chat_end: Optional[datetime] = None) -> TemporalQuery:
    q = (query or "").strip()
    ref = chat_end or datetime.now()

    intent = "neutral"
    if _LATEST_RE.search(q):
        intent = "latest"
    elif _HISTORICAL_RE.search(q):
        intent = "historical"

    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None

    m_dmy = _DMY_RE.search(q)
    m_rel = _REL_RE.search(q)
    m_month = _MONTH_RE.search(q)

    if m_dmy:
        d, m, y = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
        if y < 100:
            y += 2000
        try:
            date_start = datetime(y, m, d, 0, 0, 0)
            date_end = datetime(y, m, d, 23, 59, 59, 999999)
        except ValueError:
            date_start, date_end = None, None
    elif m_rel:
        rel = m_rel.group(1).lower()
        if "yesterday" in rel:
            yd = ref.date() - timedelta(days=1)
            date_start = datetime(yd.year, yd.month, yd.day, 0, 0, 0)
            date_end = datetime(yd.year, yd.month, yd.day, 23, 59, 59, 999999)
        elif "week" in rel:
            date_start = (ref - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
            date_end = ref.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif "month" in rel:
            date_start = (ref - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
            date_end = ref.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif m_month:
        raw_m = m_month.group(1).lower().replace("in ", "").strip().split()[0]
        m_num = _MONTH_MAP.get(raw_m)
        if m_num:
            y_num = int(m_month.group(2)) if m_month.group(2) else ref.year
            last_d = calendar.monthrange(y_num, m_num)[1]
            date_start = datetime(y_num, m_num, 1, 0, 0, 0)
            date_end = datetime(y_num, m_num, last_d, 23, 59, 59, 999999)

    return TemporalQuery(intent=intent, date_start=date_start, date_end=date_end)


def detect_temporal_intent(query: str, chat_end: Optional[datetime] = None) -> str:
    """'historical', 'latest' or 'neutral'. Kept for backward compatibility."""
    return parse_temporal_query(query, chat_end=chat_end).intent


class TimeAwareReRanker:
    """
    Reranks candidate conversational chunks by weighting semantic similarity
    against temporal recency decay.
    """

    def __init__(self, alpha: float = 0.5, decay_lambda: float = 1.0):
        self.alpha = max(0.0, min(1.0, alpha))
        self.decay_lambda = max(0.0, decay_lambda)

    def rerank(
        self, 
        candidates: List[Dict[str, Any]], 
        reference_time: Optional[datetime] = None,
        query: str = "",
        chat_span_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidates using combined semantic and exponential time decay score.
        If query explicitly specifies historical intent (e.g. 'initial', 'originally'),
        it adjusts temporal weighting to prevent penalizing relevant historical context.
        """
        if not candidates:
            return []

        # Detect temporal intent from query
        temporal_info = parse_temporal_query(query, chat_end=reference_time)
        intent = temporal_info.intent
        is_historical_query = intent == "historical"
        is_latest_query = intent == "latest"

        # Effective alpha: if historical or neutral, rely primarily on semantic matching
        if is_historical_query:
            effective_alpha = 0.40
            decay_mult = self.decay_lambda
        elif is_latest_query:
            effective_alpha = 0.40  # Heavy recency decay for latest updates
            decay_mult = self.decay_lambda
        else:
            effective_alpha = 0.85
            decay_mult = 0.20

        chunk_dts = [_parse_ts(item["chunk"].end_timestamp) for item in candidates]

        if reference_time is not None and chat_span_seconds is not None:
            span = max(float(chat_span_seconds), 1.0)
            elapsed_seconds = [
                max(0.0, (reference_time - dt).total_seconds())
                for dt in chunk_dts
            ]
            normalized = [e / span for e in elapsed_seconds]
        else:
            ref = reference_time if reference_time is not None else max(chunk_dts)
            elapsed_seconds = [
                max(0.0, (ref - dt).total_seconds())
                for dt in chunk_dts
            ]
            max_elapsed = max(elapsed_seconds) if max(elapsed_seconds) > 0 else 1.0
            normalized = [e / max_elapsed for e in elapsed_seconds]

        reranked = []
        for i, item in enumerate(candidates):
            sim_score = float(item.get("semantic_score", 0.0))
            norm_val = normalized[i]
            if is_historical_query:
                norm_decay = max(0.0, 1.0 - norm_val)
            else:
                norm_decay = norm_val
            time_decay = math.exp(-decay_mult * norm_decay)

            final_score = (effective_alpha * sim_score) + ((1.0 - effective_alpha) * time_decay)

            reranked.append({
                "chunk": item["chunk"],
                "semantic_score": sim_score,
                "time_decay": time_decay,
                "final_score": final_score,
                "elapsed_seconds": elapsed_seconds[i],
            })

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked
