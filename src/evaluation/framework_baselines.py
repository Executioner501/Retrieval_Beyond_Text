"""
Framework Baselines for Comparative Study:
1. Naive Baseline (30-min time gap silence chunking)
2. LangChain Standard RAG (RecursiveCharacterTextSplitter document chunking)
   & LangChain Message-Split RAG (split on message boundaries)
3. LlamaIndex Standard RAG (SentenceSplitter token chunking)
   & LlamaIndex Message-Split RAG (split on message boundaries)
4. Proposed Context-Aware RAG (Speaker-turn preservation + time-decay re-ranking)
"""

import os
import json
from typing import List, Dict, Any, Optional, Sequence
from datetime import datetime
import numpy as np

from ..parser.parse_whatsapp import MessageRecord, parse_whatsapp_chat
from ..chunking.chunk_messages import (
    naive_time_gap_chunking,
    context_aware_chunking,
    ConversationChunk,
    _parse_ts
)
from ..retrieval.ranker import TimeAwareReRanker
from ..retrieval.hybrid import BM25, rrf_fuse

# Embedding encoder cache
_GLOBAL_MODEL = None


def get_embedding_model():
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _GLOBAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _GLOBAL_MODEL


def find_file(filename: str, search_dirs: Optional[List[str]] = None) -> str:
    """Finds a file across common search locations."""
    if search_dirs is None:
        search_dirs = [".", "data/benchmarks", "data/processed", "data/raw"]
    for d in search_dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return filename


def load_json(path: str) -> Any:
    """Loads a JSON file resolving relative search paths."""
    real_path = find_file(path)
    with open(real_path, "r", encoding="utf-8") as f:
        return json.load(f)


def auto_score(retrieved: List[Dict[str, Any]], ground_truth: str, snippet: str, is_neg: bool, top_k: int = 3) -> str:
    """
    Centralized heuristic scorer comparing retrieved text chunks against
    ground truth and source text snippets.
    """
    if is_neg:
        top_score = retrieved[0]["score"] if retrieved else 0.0
        return "hit" if top_score < 0.45 else "partial"

    target = (snippet or ground_truth or "").lower()
    combined_top = " ".join([(r.get("chunk_text") or r.get("text", "")) for r in retrieved[:top_k]]).lower()
    top1_text = (retrieved[0].get("chunk_text") or retrieved[0].get("text", "")).lower() if retrieved else ""

    if target and target in top1_text:
        return "hit"
    elif target and target in combined_top:
        return "hit"

    keywords = [w for w in target.replace("/", " ").replace("|", " ").split() if len(w) > 4]
    matched = sum(1 for kw in keywords if kw in combined_top)
    if keywords and (matched / len(keywords)) >= 0.5:
        return "partial"
    return "miss"


class BaseRetriever:
    """Base class providing shared embedding and vector similarity capabilities."""

    def __init__(self, name: str, hybrid: bool = False):
        self.name = name
        self.hybrid = hybrid
        self.chunk_texts: List[str] = []
        self.chunk_metas: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._bm25: Optional[BM25] = None

    def _index_texts(self, texts: List[str], metas: List[Dict[str, Any]]):
        self.chunk_texts = texts
        self.chunk_metas = metas
        if not texts:
            self.embeddings = None
            self._bm25 = None
            return

        model = get_embedding_model()
        self.embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        if self.hybrid:
            self._bm25 = BM25(texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.chunk_texts or self.embeddings is None:
            return []

        model = get_embedding_model()
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)

        if self.hybrid and self._bm25 is not None:
            bm25_scores = self._bm25.scores(query)
            rankings = {
                "dense": np.argsort(sims)[::-1],
                "bm25": np.argsort(bm25_scores)[::-1],
            }
            weights = {"dense": 0.6, "bm25": 0.4}
            fused = rrf_fuse(rankings, weights=weights, depth=len(self.chunk_texts))
            top_indices = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:top_k]
            score_fn = lambda idx: float(fused[idx])
        else:
            top_indices = np.argsort(sims)[::-1][:top_k]
            score_fn = lambda idx: float(sims[idx])

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append({
                "rank": rank,
                "text": self.chunk_texts[idx],
                "chunk_text": self.chunk_texts[idx],
                "meta": self.chunk_metas[idx],
                "chunk_id": self.chunk_metas[idx].get("chunk_id", f"CHUNK-{idx:03d}"),
                "score": score_fn(idx),
            })
        return results


# ==========================================
# 1. Naive Time-Gap Baseline
# ==========================================
class NaiveBaselineRAG(BaseRetriever):
    """
    Naive baseline using 30-minute inactivity threshold.
    Converts raw messages into unstructured text blocks.
    """

    def __init__(
        self,
        messages: Optional[List[MessageRecord]] = None,
        chunks: Optional[List[Any]] = None,
        threshold_minutes: int = 30,
        hybrid: bool = False,
    ):
        super().__init__("Naive Time-Gap Baseline", hybrid=hybrid)
        if chunks is not None:
            self.chunks = chunks
            texts = [
                c.get("text") or c.get("raw_text", "")
                if isinstance(c, dict)
                else getattr(c, "raw_text", str(c))
                for c in chunks
            ]
            metas = []
            for i, c in enumerate(chunks):
                if isinstance(c, dict):
                    meta = dict(c)
                    if "chunk_id" not in meta:
                        meta["chunk_id"] = f"NAIVE-CHUNK-{i+1:03d}"
                    metas.append(meta)
                else:
                    meta = c.to_dict()
                    metas.append(meta)
        elif messages is not None:
            c_objs = naive_time_gap_chunking(messages, threshold_minutes=threshold_minutes)
            self.chunks = c_objs
            texts = [c.raw_text for c in c_objs]
            metas = [c.to_dict() for c in c_objs]
        else:
            texts, metas = [], []

        self._index_texts(texts, metas)


# ==========================================
# 2. LangChain Framework Baselines
# ==========================================
class LangChainRAG(BaseRetriever):
    """
    LangChain RAG retriever supporting two splitting configurations:
    - standard: document-oriented RecursiveCharacterTextSplitter defaults (splits flat text on newlines and spaces)
    - message_split: fair message-boundary splitting (preserves whole messages without mid-utterance splits)
    """

    def __init__(
        self,
        raw_text: str = "",
        messages: Optional[List[Any]] = None,
        chunk_size: int = 350,
        chunk_overlap: int = 70,
        split_mode: str = "standard",
        hybrid: bool = False,
    ):
        name = (
            "LangChain Message-Split RAG"
            if split_mode == "message_split"
            else "LangChain Standard RAG (document-oriented defaults)"
        )
        super().__init__(name, hybrid=hybrid)
        self.split_mode = split_mode
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if messages is not None and not raw_text:
            lines = []
            for m in messages:
                if isinstance(m, dict):
                    lines.append(f"[{m.get('date', '')} {m.get('time', '')}] {m.get('sender', '')}: {m.get('text', '')}")
                else:
                    lines.append(f"[{m.date} {m.time}] {m.sender}: {m.text}")
            raw_text = "\n".join(lines)

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if split_mode == "message_split":
            # Message-boundary splitting: only split on newline message boundaries
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n"],
                keep_separator=True
            )
            docs = splitter.split_text(raw_text)
            metas = [
                {"chunk_id": f"LC-MSG-{i:03d}", "source": "langchain_message_split"}
                for i in range(len(docs))
            ]
        else:
            # Document-oriented defaults: split on paragraphs, newlines, and words
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            docs = splitter.split_text(raw_text)
            metas = [
                {"chunk_id": f"LC-{i:03d}", "source": "langchain_recursive"}
                for i in range(len(docs))
            ]

        self._index_texts(docs, metas)


class LangChainMessageSplitRAG(LangChainRAG):
    """Convenience subclass for fair message-split LangChain baseline."""

    def __init__(
        self,
        raw_text: str = "",
        messages: Optional[List[Any]] = None,
        chunk_size: int = 350,
        chunk_overlap: int = 70,
        hybrid: bool = False,
    ):
        super().__init__(
            raw_text=raw_text,
            messages=messages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_mode="message_split",
            hybrid=hybrid,
        )


# ==========================================
# 3. LlamaIndex Framework Baselines
# ==========================================
class LlamaIndexRAG(BaseRetriever):
    """
    LlamaIndex RAG retriever supporting two splitting configurations:
    - standard: document-oriented SentenceSplitter defaults (token-based sentence splitting)
    - message_split: fair message-boundary splitting (preserves whole messages without mid-utterance splits)
    """

    def __init__(
        self,
        raw_text: str = "",
        messages: Optional[List[Any]] = None,
        chunk_size: int = 150,
        chunk_overlap: int = 25,
        split_mode: str = "standard",
        hybrid: bool = False,
    ):
        name = (
            "LlamaIndex Message-Split RAG"
            if split_mode == "message_split"
            else "LlamaIndex Standard RAG (document-oriented defaults)"
        )
        super().__init__(name, hybrid=hybrid)
        self.split_mode = split_mode
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if messages is not None and not raw_text:
            lines = []
            for m in messages:
                if isinstance(m, dict):
                    lines.append(f"[{m.get('date', '')} {m.get('time', '')}] {m.get('sender', '')}: {m.get('text', '')}")
                else:
                    lines.append(f"[{m.date} {m.time}] {m.sender}: {m.text}")
            raw_text = "\n".join(lines)

        from llama_index.core.node_parser import SentenceSplitter

        if split_mode == "message_split":
            # Message-boundary splitting: split strictly on message lines (\n)
            splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
            nodes = splitter.split_text(raw_text)
            metas = [
                {"chunk_id": f"LLAMA-MSG-{i:03d}", "source": "llama_index_message_split"}
                for i in range(len(nodes))
            ]
        else:
            # Document-oriented defaults: token-based sentence splitting
            splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            nodes = splitter.split_text(raw_text)
            metas = [
                {"chunk_id": f"LLAMA-{i:03d}", "source": "llama_index_sentence"}
                for i in range(len(nodes))
            ]

        self._index_texts(nodes, metas)


class LlamaIndexMessageSplitRAG(LlamaIndexRAG):
    """Convenience subclass for fair message-split LlamaIndex baseline."""

    def __init__(
        self,
        raw_text: str = "",
        messages: Optional[List[Any]] = None,
        chunk_size: int = 150,
        chunk_overlap: int = 25,
        hybrid: bool = False,
    ):
        super().__init__(
            raw_text=raw_text,
            messages=messages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_mode="message_split",
            hybrid=hybrid,
        )


# ==========================================
# 4. Proposed Context-Aware Framework
# ==========================================
class ContextAwareRAG(BaseRetriever):
    """
    Proposed adapted conversational retrieval framework:
    - Explicit speaker turn attribution: [Date Time] Sender: Message
    - Topic and bounded turn chunking
    - Exponential time-decay re-ranking: Score = alpha*Sim + (1-alpha)*exp(-lambda*delta_t)
    """

    def __init__(
        self, 
        messages: List[MessageRecord], 
        alpha: float = 0.45, 
        decay_lambda: float = 1.2
    ):
        super().__init__("Proposed Context-Aware RAG")
        self.alpha = alpha
        self.decay_lambda = decay_lambda
        self.reranker = TimeAwareReRanker(alpha=alpha, decay_lambda=decay_lambda)

        chunks = context_aware_chunking(messages, max_inactivity_minutes=20, max_chunk_size=8)
        self.conv_chunks = chunks
        texts = [c.attributed_text for c in chunks]
        metas = [c.to_dict() for c in chunks]
        self._index_texts(texts, metas)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        candidates = super().retrieve(query, top_k=len(self.conv_chunks))
        
        # Prepare candidates for time-decay reranking
        formatted_candidates = []
        for c in candidates:
            idx = int(c["meta"]["chunk_id"].split("-")[-1]) - 1
            chunk_obj = self.conv_chunks[idx]
            formatted_candidates.append({
                "chunk": chunk_obj,
                "semantic_score": c["score"],
                "text": c["text"],
                "chunk_text": c["text"],
                "meta": c["meta"],
                "chunk_id": c.get("chunk_id", c["meta"].get("chunk_id"))
            })

        reranked = self.reranker.rerank(formatted_candidates, query=query)
        
        results = []
        for rank, item in enumerate(reranked[:top_k], start=1):
            results.append({
                "rank": rank,
                "text": item["chunk"].attributed_text,
                "chunk_text": item["chunk"].attributed_text,
                "meta": item["chunk"].to_dict(),
                "chunk_id": item["chunk"].chunk_id,
                "score": item["final_score"],
                "semantic_score": item["semantic_score"],
                "time_decay": item["time_decay"],
            })
        return results


# ==========================================
# Unified Batch Evaluation Runner
# ==========================================
def run_baseline_evaluation(
    framework: str,
    eval_file: str = "evaluation_set.json",
    output_file: Optional[str] = None,
    top_k: int = 3,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    split_mode: str = "standard",
    hybrid: bool = False,
) -> List[Dict[str, Any]]:
    """
    Unified batch evaluation driver executing any baseline over benchmark sets.
    Preserves exact backward-compatible JSON schema in eval_results/.
    """
    real_eval_file = find_file(eval_file, [".", "data/benchmarks"])
    print(f"[*] Loading evaluation queries from: {real_eval_file}")
    eval_set = load_json(real_eval_file)
    print(f"[+] Loaded {len(eval_set)} benchmark questions")

    # Map framework aliases
    f_lower = framework.lower().strip()
    if f_lower in ("naive", "naive_baseline"):
        fw_type = "naive"
    elif f_lower in ("langchain", "langchain_standard"):
        fw_type = "langchain"
    elif f_lower == "langchain_message_split":
        fw_type = "langchain"
        split_mode = "message_split"
    elif f_lower in ("llamaindex", "llama_index", "llamaindex_standard"):
        fw_type = "llamaindex"
    elif f_lower in ("llamaindex_message_split", "llama_index_message_split"):
        fw_type = "llamaindex"
        split_mode = "message_split"
    else:
        fw_type = f_lower

    if output_file is None:
        os.makedirs("eval_results", exist_ok=True)
        if fw_type == "naive":
            output_file = "eval_results/results_naive_baseline.json"
        elif fw_type == "langchain":
            output_file = (
                "eval_results/results_langchain_message_split.json"
                if split_mode == "message_split"
                else "eval_results/results_langchain.json"
            )
        elif fw_type == "llamaindex":
            output_file = (
                "eval_results/results_llamaindex_message_split.json"
                if split_mode == "message_split"
                else "eval_results/results_llamaindex.json"
            )
        else:
            output_file = f"eval_results/results_{fw_type}.json"

    retrievers_by_chat: Dict[str, BaseRetriever] = {}
    results = []

    for q in eval_set:
        chat = q.get("chat", "sample")
        if chat not in retrievers_by_chat:
            if fw_type == "naive":
                chunk_file = find_file(f"{chat}_parsed_chunks.json", [".", "data/processed", "data/raw"])
                if os.path.exists(chunk_file):
                    chunks = load_json(chunk_file)
                else:
                    parsed_file = find_file(f"{chat}_parsed.json", [".", "data/processed", "data/raw"])
                    msgs = load_json(parsed_file)
                    chunks = [
                        {
                            "chunk_id": f"{chat}-{i:03d}",
                            "text": f"[{m.get('date', '')} {m.get('time', '')}] {m.get('sender', '')}: {m.get('text', '')}",
                            "raw_text": f"{m.get('sender', '')}: {m.get('text', '')}"
                        }
                        for i, m in enumerate(msgs)
                    ]
                print(f"[*] Indexing {chat} ({len(chunks)} chunks)...")
                retrievers_by_chat[chat] = NaiveBaselineRAG(chunks=chunks, hybrid=hybrid)

            elif fw_type == "langchain":
                parsed_file = find_file(f"{chat}_parsed.json", [".", "data/processed", "data/raw"])
                messages = load_json(parsed_file)
                c_size = chunk_size or (400 if split_mode == "standard" else 350)
                c_overlap = chunk_overlap or (60 if split_mode == "standard" else 70)
                retrievers_by_chat[chat] = LangChainRAG(
                    messages=messages,
                    chunk_size=c_size,
                    chunk_overlap=c_overlap,
                    split_mode=split_mode,
                    hybrid=hybrid,
                )
                print(f"  [LangChain] {chat}: {len(messages)} messages -> {len(retrievers_by_chat[chat].chunk_texts)} chunks")

            elif fw_type == "llamaindex":
                parsed_file = find_file(f"{chat}_parsed.json", [".", "data/processed", "data/raw"])
                messages = load_json(parsed_file)
                c_size = chunk_size or (250 if split_mode == "standard" else 150)
                c_overlap = chunk_overlap or (40 if split_mode == "standard" else 25)
                retrievers_by_chat[chat] = LlamaIndexRAG(
                    messages=messages,
                    chunk_size=c_size,
                    chunk_overlap=c_overlap,
                    split_mode=split_mode,
                    hybrid=hybrid,
                )
                print(f"  [LlamaIndex] {chat}: {len(messages)} messages -> {len(retrievers_by_chat[chat].chunk_texts)} chunks")

        retriever = retrievers_by_chat[chat]
        retrieved = retriever.retrieve(q["question"], top_k=top_k)
        is_neg = q.get("question_type") == "negative-control"
        score = auto_score(retrieved, q.get("ground_truth_answer", ""), q.get("source_text_snippet", ""), is_neg, top_k=top_k)

        # Build exact backward-compatible item dictionary
        record: Dict[str, Any] = {
            "id": q["id"],
            "chat": chat,
            "question": q["question"],
            "question_type": q.get("question_type", "general"),
            "ground_truth_answer": q.get("ground_truth_answer"),
            "source_text_snippet": q.get("source_text_snippet"),
        }
        if fw_type in ("langchain", "llamaindex"):
            record["num_chunks_in_chat"] = len(retriever.chunk_texts)

        record["retrieved_chunks"] = [
            {
                "rank": r["rank"],
                "score": r["score"],
                "chunk_text": r.get("chunk_text") or r["text"],
                "chunk_id": r.get("chunk_id") or r.get("meta", {}).get("chunk_id", f"CHK-{idx:03d}"),
            }
            for idx, r in enumerate(retrieved)
        ]
        record["auto_score"] = score
        record["manual_score"] = ""
        record["notes"] = q.get("notes", "")
        results.append(record)

    if output_file:
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[+] Successfully saved {len(results)} evaluated queries to: {output_file}")

    return results
