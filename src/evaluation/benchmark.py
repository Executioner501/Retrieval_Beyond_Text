"""
Module 6: Diagnostic Benchmarking & Failure Taxonomy.
Evaluates retrieval and QA across:
1. Baseline Naive Time-Gap
2. Generic RAG (document style)
3. Proposed Context-Aware Framework

Categorizes failures:
- BAD_CHUNK_BOUNDARY
- LOST_SPEAKER_CONTEXT
- WRONG_TIME_WINDOW
- IRRELEVANT_RETRIEVAL
"""

from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass


class FailureCategory(str, Enum):
    BAD_CHUNK_BOUNDARY = "BAD_CHUNK_BOUNDARY"
    LOST_SPEAKER_CONTEXT = "LOST_SPEAKER_CONTEXT"
    WRONG_TIME_WINDOW = "WRONG_TIME_WINDOW"
    IRRELEVANT_RETRIEVAL = "IRRELEVANT_RETRIEVAL"
    NONE = "NONE"


@dataclass
class BenchmarkQuery:
    query_id: str
    query_text: str
    ground_truth_answer: str
    expected_speakers: List[str]
    is_time_sensitive: bool
    gold_chunk_keywords: List[str]


class DiagnosticEvaluator:
    """
    Runs diagnostic benchmarking and computes precision, recall, MRR,
    and failure distributions.
    """

    def __init__(self, queries: List[BenchmarkQuery] = None):
        self.queries = queries or []

    def evaluate_retrieval(
        self,
        retrieved_chunk_texts: List[str],
        query: BenchmarkQuery,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluates retrieval quality against gold chunk keywords and speaker expectations.
        """
        hit = False
        reciprocal_rank = 0.0
        failure = FailureCategory.NONE

        for rank, chunk_text in enumerate(retrieved_chunk_texts[:top_k], start=1):
            has_keywords = any(kw.lower() in chunk_text.lower() for kw in query.gold_chunk_keywords)
            has_speakers = all(sp.lower() in chunk_text.lower() for sp in query.expected_speakers)

            if has_keywords:
                if not has_speakers and query.expected_speakers:
                    failure = FailureCategory.LOST_SPEAKER_CONTEXT
                else:
                    hit = True
                    if reciprocal_rank == 0.0:
                        reciprocal_rank = 1.0 / rank
                    failure = FailureCategory.NONE
                break

        if not hit and failure == FailureCategory.NONE:
            if query.is_time_sensitive:
                failure = FailureCategory.WRONG_TIME_WINDOW
            else:
                failure = FailureCategory.BAD_CHUNK_BOUNDARY

        return {
            "hit": hit,
            "reciprocal_rank": reciprocal_rank,
            "failure_mode": failure.value,
        }
