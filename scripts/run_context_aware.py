"""
Proposed Context-Aware Conversational RAG Runner
------------------------------------------------
Evaluates the Proposed Context-Aware Conversational RAG framework:
1. Speaker-turn attribution preservation ([Date Time] Sender: Message)
2. Bounded conversational chunking with topic-shift detection
3. Time-aware exponential decay re-ranking:
   Score(c) = alpha * Sim(q,c) + (1-alpha) * exp(-lambda * delta_t_norm)

Usage:
    python scripts/run_context_aware.py
"""

import os
import sys
from typing import List, Dict, Any
import json
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
from src.parser.parse_whatsapp import MessageRecord
from src.chunking.chunk_messages import context_aware_chunking, ConversationChunk
from src.retrieval.ranker import TimeAwareReRanker

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


def find_file(filename: str, search_dirs: List[str] = None) -> str:
    if search_dirs is None:
        search_dirs = [".", "data/benchmarks", "data/processed", "data/raw"]
    for d in search_dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return filename


def load_json(path: str) -> Any:
    real_path = find_file(path)
    with open(real_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_context_aware_index(chat_name: str, model: SentenceTransformer):
    parsed_file = find_file(f"{chat_name}_parsed.json", [".", "data/processed", "data/raw"])
    raw_msgs = load_json(parsed_file)
    
    # Convert to MessageRecord objects
    records = []
    for i, m in enumerate(raw_msgs):
        d = m.get("date", "01/01/2026")
        t = m.get("time", "12:00")
        txt = m.get("text", "")
        records.append(
            MessageRecord(
                message_id=m.get("message_id", f"MSG-{i+1:04d}"),
                timestamp=f"{d} {t}",
                date=d,
                time=t,
                sender=m.get("sender", "User"),
                text=txt,
                is_system=m.get("is_system", False),
                has_media=m.get("has_media", False),
                char_count=len(txt),
                word_count=len(txt.split())
            )
        )

    chunks = context_aware_chunking(records, max_inactivity_minutes=20, max_chunk_size=8)
    print(f"  [Context-Aware] {chat_name}: {len(records)} messages -> {len(chunks)} attributed chunks")

    texts = [c.attributed_text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings, chunks


def search(query: str, model: SentenceTransformer, reranker: TimeAwareReRanker, embeddings: np.ndarray, chunks: List[ConversationChunk], top_k: int = TOP_K):
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = np.dot(embeddings, qvec)
    
    # Prepare all candidates for time-decay reranking
    candidates = []
    for idx, c in enumerate(chunks):
        candidates.append({
            "chunk": c,
            "semantic_score": float(sims[idx]),
            "index": idx
        })

    reranked = reranker.rerank(candidates, query=query)

    results = []
    for rank, item in enumerate(reranked[:top_k], start=1):
        c = item["chunk"]
        results.append({
            "rank": rank,
            "score": item["final_score"],
            "semantic_score": item["semantic_score"],
            "time_decay": item["time_decay"],
            "chunk_text": c.attributed_text,
            "chunk_id": c.chunk_id
        })
    return results


def auto_score(retrieved: List[Dict[str, Any]], ground_truth: str, snippet: str, is_neg: bool) -> str:
    if is_neg:
        top_score = retrieved[0]["score"] if retrieved else 0.0
        return "hit" if top_score < 0.50 else "partial"

    target = (snippet or ground_truth or "").lower()
    combined_top = " ".join([r["chunk_text"] for r in retrieved[:TOP_K]]).lower()
    top1_text = retrieved[0]["chunk_text"].lower() if retrieved else ""

    if target and target in top1_text:
        return "hit"
    elif target and target in combined_top:
        return "hit"

    keywords = [w for w in target.replace("/", " ").replace("|", " ").split() if len(w) > 4]
    matched = sum(1 for kw in keywords if kw in combined_top)
    if keywords and (matched / len(keywords)) >= 0.5:
        return "partial"
    return "miss"


def main():
    eval_file = find_file("evaluation_set.json", [".", "data/benchmarks"])
    print(f"[*] Loading evaluation queries from: {eval_file}")
    eval_set = load_json(eval_file)
    print(f"[+] Loaded {len(eval_set)} benchmark questions")

    print(f"[*] Initializing embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    reranker = TimeAwareReRanker(alpha=0.45, decay_lambda=1.2)

    indexes = {}
    chunks_by_chat = {}
    results = []

    for q in eval_set:
        chat = q.get("chat", "sample")
        if chat not in indexes:
            print(f"[*] Building Context-Aware index for chat: {chat}")
            embeddings, chunks = build_context_aware_index(chat, model)
            indexes[chat] = embeddings
            chunks_by_chat[chat] = chunks

        retrieved = search(q["question"], model, reranker, indexes[chat], chunks_by_chat[chat])
        is_neg = q.get("question_type") == "negative-control"
        score = auto_score(retrieved, q.get("ground_truth_answer", ""), q.get("source_text_snippet", ""), is_neg)

        results.append({
            "id": q["id"],
            "chat": chat,
            "question": q["question"],
            "question_type": q.get("question_type", "general"),
            "ground_truth_answer": q.get("ground_truth_answer"),
            "source_text_snippet": q.get("source_text_snippet"),
            "num_chunks_in_chat": len(chunks_by_chat[chat]),
            "retrieved_chunks": retrieved,
            "auto_score": score,
            "manual_score": "",  # <-- fill this in yourself by reading retrieved_chunks against ground_truth_answer
            "notes": q.get("notes", "")
        })

    os.makedirs("eval_results", exist_ok=True)
    out_path = "eval_results/results_context_aware.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Successfully saved {len(results)} Context-Aware evaluated queries to: {out_path}")


if __name__ == "__main__":
    main()
