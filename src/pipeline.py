"""
End-to-end Conversational QA Pipeline.

    chat export --parse--> messages --semantic_topic_chunking--> chunks
    chunks --embed--> LocalVectorStore
    question --dense search--> candidates --TimeAwareReRanker--> top-k
    top-k --prompt--> Ollama --> grounded answer + cited sources

Usage:
    qa = ConversationQA.from_file("data/raw/sample_chat.txt")
    result = qa.ask("Which frontend framework did we finally choose?")
    print(result["answer"])
"""

import re
from typing import Any, Dict, List, Optional

import numpy as np

from .chunking.chunk_messages import (ConversationChunk, _build_chunk, _parse_ts, context_aware_chunking,
                                      naive_time_gap_chunking)
from .chunking.semantic_topic_chunking import semantic_topic_chunking
from .generation.ollama_client import OllamaClient, OllamaError
from .indexing.embedders import default_min_relevance, get_embedder
from .indexing.vector_store import LocalVectorStore
from .parser.parse_whatsapp import MessageRecord, parse_whatsapp_chat
from .retrieval.hybrid import HybridChunkRetriever
from .retrieval.ranker import TimeAwareReRanker, detect_temporal_intent, parse_temporal_query
from .generation.prompts import SYSTEM_PROMPT

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # kept for backwards compatibility
NOT_FOUND = "I couldn't find anything about that in this chat."


class ConversationQA:
    def __init__(
        self,
        messages: List[MessageRecord],
        llm: Optional[Any] = None,
        encoder: Optional[Any] = None,
        top_k: int = 4,
        candidate_pool: int = 20,
        min_relevance: Optional[float] = None,
        chunking_kwargs: Optional[Dict[str, Any]] = None,
        embedder: str = "minilm",
        embed_text: str = "clean",
        chunker: str = "semantic",
        retrieval: str = "hybrid",
        expand_queries: bool = False,
    ):
        """
        Args:
            messages: parsed chat messages.
            llm: anything with .chat(messages) -> str. Defaults to OllamaClient().
            encoder: anything with .encode(texts, convert_to_numpy=True, normalize_embeddings=True).
                Overrides `embedder`. Shared by chunking and retrieval.
            embedder: "minilm", "bge-m3", or "me5" (see src/indexing/embedders.py).
            chunker: "semantic" (topic-shift detection), "naive" (30-min time gap),
                "regex" (old rule-based context-aware), or "fixed" (every 8 messages).
            retrieval: "hybrid" (BM25 + chunk embedding + best-message embedding, fused with
                RRF; time-aware reranking only for questions about recent/latest things) or
                "dense" (the original: chunk embedding + time-aware reranking for every question).
            embed_text: "clean" embeds "Sender: text" lines (no timestamps, more room for content);
                "attributed" embeds the full "[date time] Sender: text" lines. The LLM always
                receives the full attributed text either way.
            top_k: chunks passed to the LLM.
            candidate_pool: chunks kept from dense search before time-aware reranking.
            min_relevance: if the best semantic score is below this, answer NOT_FOUND
                without calling the LLM. Defaults per embedder (scores differ between models).
            expand_queries: whether to expand queries with LLM-generated paraphrases.
        """
        if encoder is None:
            encoder = get_embedder(embedder)
        self.encoder = encoder
        self.embedder_name = getattr(encoder, "name", embedder)
        if min_relevance is None:
            min_relevance = default_min_relevance(self.embedder_name)
        if embed_text not in ("clean", "attributed"):
            raise ValueError("embed_text must be 'clean' or 'attributed'")
        self.embed_text = embed_text
        self.llm = llm or OllamaClient()
        self.top_k = top_k
        self.candidate_pool = candidate_pool
        self.min_relevance = min_relevance
        self.expand_queries = expand_queries
        self.reranker = TimeAwareReRanker(alpha=0.45, decay_lambda=1.2)

        dts = [_parse_ts(m.timestamp) for m in messages if getattr(m, "timestamp", None)]
        if dts:
            self.chat_start = min(dts)
            self.chat_end = max(dts)
            self.chat_span_seconds = max(0.0, (self.chat_end - self.chat_start).total_seconds())
        else:
            self.chat_start = None
            self.chat_end = None
            self.chat_span_seconds = 0.0

        if retrieval not in ("hybrid", "dense", "bm25_only", "message_window", "cross_encoder"):
            raise ValueError(f"Unknown retrieval method '{retrieval}'")
        self.retrieval = retrieval
        self.messages = messages
        self._by_id = {m.message_id: m for m in messages}
        # Message embeddings are computed once and shared by the chunker and hybrid retrieval
        self.msg_emb = self._embed([m.text for m in messages])
        self.chunker = chunker
        self.chunks: List[ConversationChunk] = self._make_chunks(messages, chunker, chunking_kwargs or {})
        self.store = LocalVectorStore(
            self.embedder_name,
            text_field="raw_text" if embed_text == "clean" else "attributed_text",
        )
        self.store._encoder = self.encoder  # reuse the loaded model
        self.store.add_chunks(self.chunks)

        chunk_of = {mid: ci for ci, c in enumerate(self.chunks) for mid in c.message_ids}
        self.hybrid = HybridChunkRetriever(
            chunk_texts=[c.raw_text for c in self.chunks],
            chunk_emb=self.store.embeddings,
            msg_emb=self.msg_emb,
            msg_chunk=[chunk_of[m.message_id] for m in messages],
            msg_words=[m.word_count for m in messages],
            chunk_senders=[c.participants for c in self.chunks],
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path: str, **kwargs) -> "ConversationQA":
        messages = parse_whatsapp_chat(path)
        if not messages:
            raise ValueError(f"No messages could be parsed from {path}")
        return cls(messages, **kwargs)

    def _make_chunks(self, messages, chunker, kwargs) -> List[ConversationChunk]:
        if chunker == "semantic":
            return semantic_topic_chunking(messages, embeddings=self.msg_emb, **kwargs)
        if chunker == "naive":
            return naive_time_gap_chunking(messages, **kwargs)
        if chunker == "regex":
            return context_aware_chunking(messages, **kwargs)
        if chunker == "fixed":
            size = kwargs.get("size", 8)
            return [_build_chunk(messages[i:i + size], f"FIX-CHUNK-{i // size + 1:03d}", "fixed")
                    for i in range(0, len(messages), size)]
        raise ValueError(f"Unknown chunker '{chunker}'")

    def _embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        try:
            return self.encoder.encode(texts, is_query=is_query, convert_to_numpy=True, normalize_embeddings=True)
        except TypeError:
            return self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def stats(self) -> Dict[str, Any]:
        senders = sorted({m.sender for m in self.messages})
        return {
            "messages": len(self.messages),
            "chunks": len(self.chunks),
            "participants": senders,
            "embedder": self.embedder_name,
            "embed_text": self.embed_text,
            "chunker": self.chunker,
            "retrieval": self.retrieval,
            "start": self.messages[0].timestamp if self.messages else None,
            "end": self.messages[-1].timestamp if self.messages else None,
        }

    # ------------------------------------------------------------------ #
    def _retrieve_single(self, question: str) -> List[Dict[str, Any]]:
        pool = min(self.candidate_pool, len(self.chunks))
        temporal_info = parse_temporal_query(question, chat_end=self.chat_end)

        if self.retrieval == "bm25_only":
            scores = self.hybrid.bm25.scores(question)
            order = np.argsort(-scores)[: self.top_k]
            return [{
                "chunk": self.chunks[i],
                "index": int(i),
                "semantic_score": float(scores[i]),
                "final_score": float(scores[i]),
                "time_decay": 1.0,
                "time_rerank_applied": False,
            } for i in order]

        if self.retrieval == "message_window":
            q_emb = self._embed([question], is_query=True)[0]
            sims = self.msg_emb @ q_emb
            top_msg_indices = np.argsort(-sims)[: self.top_k]
            hits = []
            for m_idx in top_msg_indices:
                win_start = max(0, int(m_idx) - 2)
                win_end = min(len(self.messages), int(m_idx) + 3)
                win_msgs = self.messages[win_start:win_end]
                win_ids = [m.message_id for m in win_msgs]
                win_participants = sorted({m.sender for m in win_msgs})
                win_attributed = "\n".join(
                    f"[{m.date} {m.time}] {m.sender}: {m.text}" for m in win_msgs
                )
                win_raw = "\n".join(f"{m.sender}: {m.text}" for m in win_msgs)
                win_chunk = ConversationChunk(
                    chunk_id=f"win_{m_idx:05d}",
                    strategy="message_window",
                    start_timestamp=win_msgs[0].timestamp,
                    end_timestamp=win_msgs[-1].timestamp,
                    message_ids=win_ids,
                    participants=win_participants,
                    attributed_text=win_attributed,
                    raw_text=win_raw,
                )
                hits.append({
                    "chunk": win_chunk,
                    "index": int(m_idx),
                    "semantic_score": float(sims[m_idx]),
                    "final_score": float(sims[m_idx]),
                    "time_decay": 1.0,
                    "time_rerank_applied": False,
                })
            return hits

        if self.retrieval == "cross_encoder":
            q_emb = self._embed([question], is_query=True)[0]
            found = self.hybrid.search(question, q_emb, top_n=20)
            candidate_chunks = [self.chunks[f["index"]] for f in found]
            try:
                from sentence_transformers import CrossEncoder
                ce = getattr(self, "_cross_encoder", None)
                if ce is None:
                    ce = CrossEncoder("BAAI/bge-reranker-base")
                    self._cross_encoder = ce
                pairs = [[question, c.attributed_text] for c in candidate_chunks]
                ce_scores = ce.predict(pairs)
                ce_order = np.argsort(-ce_scores)[: self.top_k]
                return [{
                    "chunk": candidate_chunks[i],
                    "index": found[i]["index"],
                    "semantic_score": float(ce_scores[i]),
                    "final_score": float(ce_scores[i]),
                    "time_decay": 1.0,
                    "time_rerank_applied": False,
                } for i in ce_order]
            except Exception as e:
                top_fused = max((f["fused_score"] for f in found), default=1.0) or 1.0
                return [{
                    **f,
                    "chunk": self.chunks[f["index"]],
                    "final_score": f["fused_score"] / top_fused,
                    "time_decay": 1.0,
                    "time_rerank_applied": False,
                } for f in found[: self.top_k]]

        if self.retrieval == "dense":
            candidates = self.store.search(question, top_k=pool)
            if temporal_info.date_start and temporal_info.date_end:
                filtered = [
                    c for c in candidates
                    if _parse_ts(c["chunk"].start_timestamp) <= temporal_info.date_end
                    and _parse_ts(c["chunk"].end_timestamp) >= temporal_info.date_start
                ]
                if filtered:
                    candidates = filtered
            hits = self.reranker.rerank(
                candidates,
                reference_time=self.chat_end,
                query=question,
                chat_span_seconds=self.chat_span_seconds,
            )[: self.top_k]
            for h in hits:
                h["time_rerank_applied"] = True
            return hits

        q_emb = self._embed([question], is_query=True)[0]
        found = self.hybrid.search(question, q_emb, top_n=pool)
        top_fused = max((f["fused_score"] for f in found), default=1.0) or 1.0
        hits = [{**f, "chunk": self.chunks[f["index"]], "final_score": f["fused_score"] / top_fused,
                 "time_decay": 1.0, "time_rerank_applied": False} for f in found]

        # Filter candidate chunks to those overlapping the range BEFORE reranking
        if temporal_info.date_start and temporal_info.date_end:
            filtered_hits = [
                h for h in hits
                if _parse_ts(h["chunk"].start_timestamp) <= temporal_info.date_end
                and _parse_ts(h["chunk"].end_timestamp) >= temporal_info.date_start
            ]
            if filtered_hits:
                hits = filtered_hits

        # Recency only matters when the question asks about the latest/current state or historical state.
        if temporal_info.intent in ("latest", "historical") and hits:
            by_id = {h["chunk"].chunk_id: h for h in hits}
            reranked = self.reranker.rerank(
                [{"chunk": h["chunk"], "semantic_score": h["final_score"]} for h in hits],
                reference_time=self.chat_end,
                query=question,
                chat_span_seconds=self.chat_span_seconds,
            )
            hits = []
            for r in reranked:
                h = by_id[r["chunk"].chunk_id]
                hits.append({**h, "final_score": r["final_score"], "time_decay": r["time_decay"],
                             "time_rerank_applied": True})
        return hits[: self.top_k]

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        if not self.expand_queries:
            return self._retrieve_single(question)

        paraphrases = []
        if self.llm:
            try:
                if getattr(self.llm, "is_available", lambda: True)():
                    messages = [
                        {
                            "role": "user",
                            "content": (
                                f"Provide 2 short, alternative ways to ask this question about a chat. "
                                f"Return ONLY the 2 questions, one per line:\n{question}"
                            ),
                        }
                    ]
                    resp = self.llm.chat(messages)
                    for line in resp.strip().splitlines():
                        cleaned = re.sub(r"^[\d\.\-\*\s]+", "", line).strip().strip('"\'')
                        if cleaned and cleaned.lower() != question.lower():
                            paraphrases.append(cleaned)
                            if len(paraphrases) == 2:
                                break
            except Exception:
                paraphrases = []

        if not paraphrases:
            return self._retrieve_single(question)

        all_queries = [question] + paraphrases[:2]
        query_results = [self._retrieve_single(q) for q in all_queries]

        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}
        for r_list in query_results:
            for rank, hit in enumerate(r_list, start=1):
                cid = hit["chunk"].chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + rank)
                if cid not in chunk_lookup:
                    chunk_lookup[cid] = hit

        fused_order = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        top_fused = max(rrf_scores.values(), default=1.0) or 1.0
        fused_hits = []
        for cid in fused_order[: self.top_k]:
            orig_hit = chunk_lookup[cid]
            fused_hits.append({
                **orig_hit,
                "final_score": rrf_scores[cid] / top_fused,
            })
        return fused_hits

    def _order_for_supersession(self, question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not hits:
            return hits
        temporal_info = parse_temporal_query(question, chat_end=self.chat_end)
        if temporal_info.intent != "latest" or self.store.embeddings is None:
            return hits

        top_hit = hits[0]
        chunk_to_idx = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        top_idx = chunk_to_idx.get(top_hit["chunk"].chunk_id)
        if top_idx is None:
            return hits

        top_emb = self.store.embeddings[top_idx]
        topic_hits = []
        other_hits = []
        for h in hits:
            idx = chunk_to_idx.get(h["chunk"].chunk_id)
            if idx is not None:
                sim = float(np.dot(top_emb, self.store.embeddings[idx]))
                if sim >= 0.5:
                    topic_hits.append(h)
                    continue
            other_hits.append(h)

        topic_hits.sort(key=lambda h: _parse_ts(h["chunk"].start_timestamp))
        return topic_hits + other_hits

    def _build_user_prompt(self, question: str, hits: List[Dict[str, Any]]) -> str:
        ordered_hits = self._order_for_supersession(question, hits)
        parts = []
        for i, h in enumerate(ordered_hits, start=1):
            c = h["chunk"]
            parts.append(f"[{i}] ({c.start_timestamp} to {c.end_timestamp})\n{c.attributed_text}")
        evidence = "\n\n".join(parts)
        return f"Evidence excerpts:\n\n{evidence}\n\nQuestion: {question}"

    def ask(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Answers a natural-language question about the chat.

        history: optional previous turns [{'role': 'user'|'assistant', 'content': ...}]
            so follow-ups like "and who disagreed?" read naturally.
        """
        question = question.strip()
        if not question:
            raise ValueError("Question is empty")

        # Include the previous user question in the search so follow-ups retrieve well
        search_query = question
        if history:
            prev_user = [t["content"] for t in history if t.get("role") == "user"]
            if prev_user:
                search_query = f"{prev_user[-1]} {question}"

        hits = self.retrieve(search_query)
        prompt_hits = self._order_for_supersession(question, hits)
        sources = [self._source(i, h) for i, h in enumerate(prompt_hits, start=1)]
        for s, h in zip(sources, prompt_hits):
            s["messages"] = [
                {"id": mid, "sender": m.sender, "date": m.date, "time": m.time, "text": m.text}
                for mid in h["chunk"].message_ids if (m := self._by_id.get(mid))
            ]

        if not hits or max(h["semantic_score"] for h in hits) < self.min_relevance:
            return {"question": question, "answer": NOT_FOUND, "sources": [], "used_llm": False}

        llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in (history or [])[-4:]:
            llm_messages.append({"role": turn["role"], "content": turn["content"]})
        llm_messages.append({"role": "user", "content": self._build_user_prompt(question, prompt_hits)})

        try:
            answer = self.llm.chat(llm_messages)
        except OllamaError as e:
            return {"question": question, "answer": None, "error": str(e),
                    "sources": sources, "used_llm": False}

        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        for s in sources:
            s["cited"] = s["ref"] in cited
        return {"question": question, "answer": answer, "sources": sources, "used_llm": True}

    @staticmethod
    def _source(ref: int, hit: Dict[str, Any]) -> Dict[str, Any]:
        c: ConversationChunk = hit["chunk"]
        return {
            "ref": ref,
            "chunk_id": c.chunk_id,
            "start": c.start_timestamp,
            "end": c.end_timestamp,
            "participants": c.participants,
            "text": c.attributed_text,
            "semantic_score": round(hit["semantic_score"], 4),
            "time_decay": round(hit["time_decay"], 4),
            "final_score": round(hit["final_score"], 4),
            "bm25": round(hit.get("bm25", 0.0), 4),
            "best_message_similarity": round(hit.get("best_message_similarity", hit["semantic_score"]), 4),
            "time_rerank_applied": hit.get("time_rerank_applied", False),
            "cited": False,
        }
