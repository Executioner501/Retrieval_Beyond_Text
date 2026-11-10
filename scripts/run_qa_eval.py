"""
Evaluate the full QA pipeline (semantic chunking + time-aware retrieval + Ollama)
on real chat exports, and export a dataset ready for RAGAS.

Gold evidence is located by the question's source_timestamp (+ snippet), so
retrieval is scored on whether the chunk that actually contains the gold
message was retrieved - not on substring matching of abbreviated snippets.

Usage (from the project root, chats in data/raw_real/<chat>.txt):
    python scripts/run_qa_eval.py --embedder minilm --no-llm     # retrieval only
    python scripts/run_qa_eval.py --embedder bge-m3 --no-llm
    python scripts/run_qa_eval.py --embedder bge-m3              # + answers from Ollama
    python scripts/run_qa_eval.py --eval data/benchmarks/new_queries.json --embedder minilm --no-llm
    python scripts/run_qa_eval.py --embedder minilm --no-llm --chunker naive   # semantic|naive|regex|fixed
    python scripts/run_qa_eval.py --embedder minilm --no-llm --retrieval dense # original retrieval

Outputs (the tag is <eval-file>_<embedder>_<embed-text>, e.g. evaluation_set_bge-m3_clean):
    eval_results/qa_eval_<tag>.json     summary + per-question details
    eval_results/ragas_<tag>.jsonl      user_input / retrieved_contexts / response / reference
"""

import argparse
import json
import time
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.parser.parse_whatsapp import MessageRecord, parse_whatsapp_chat
from src.pipeline import NOT_FOUND, ConversationQA


def _norm(s: str) -> str:
    s = s.replace("...", " ").replace("|", " ").replace("\u2002", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def locate_gold(messages: List[MessageRecord], q: Dict[str, Any]) -> Set[str]:
    """Message ids that hold the gold evidence; empty set = nothing to retrieve."""
    ts = (q.get("source_timestamp") or "").strip()
    if not ts or q.get("question_type") == "negative-control":
        return set()
    date, _, time = ts.partition(" ")
    at_ts = [m for m in messages if m.date == date and m.time == time]
    snippet = _norm(q.get("source_text_snippet") or "")[:25]
    if snippet:
        matched = [m for m in at_ts if snippet in _norm(m.text)]
        if matched:
            return {m.message_id for m in matched}
    return {m.message_id for m in at_ts}


def _first_rank(chunks, gold: Set[str]) -> Optional[int]:
    for rank, c in enumerate(chunks, start=1):
        if gold & set(c.message_ids):
            return rank
    return None


import numpy as np

def bootstrap_ci(ranks: List[Optional[int]], metric: str = "hit@k", n_resamples: int = 1000, seed: int = 42) -> List[float]:
    """Computes a 95% bootstrap confidence interval for hit@1, hit@k, or mrr."""
    if not ranks:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    n = len(ranks)

    if metric == "hit@1":
        vals = np.array([1.0 if r == 1 else 0.0 for r in ranks], dtype=float)
    elif metric == "mrr" or metric.startswith("mrr"):
        vals = np.array([1.0 / r if (r is not None and r > 0) else 0.0 for r in ranks], dtype=float)
    else:  # hit@k
        vals = np.array([1.0 if (r is not None and r > 0) else 0.0 for r in ranks], dtype=float)

    indices = rng.choice(n, size=(n_resamples, n), replace=True)
    resamples = vals[indices]
    means = np.mean(resamples, axis=1)
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return [round(lo, 3), round(hi, 3)]


def evaluate(eval_items, chat_paths, encoder=None, llm=None, top_k=4, use_llm=True,
             embedder="minilm", embed_text="clean", chunker="semantic", retrieval="hybrid",
             expand_queries=False, judge=False):
    pipelines: Dict[str, ConversationQA] = {}
    rows = []
    for q in eval_items:
        chat = q["chat"]
        if chat not in pipelines:
            print(f"[*] Indexing {chat} ...")
            t0 = time.time()
            pipelines[chat] = ConversationQA.from_file(
                chat_paths[chat], llm=llm, encoder=encoder, top_k=top_k,
                embedder=embedder, embed_text=embed_text, chunker=chunker, retrieval=retrieval,
                expand_queries=expand_queries)
            st = pipelines[chat].stats()
            print(f"    {st['messages']} messages -> {st['chunks']} chunks ({time.time() - t0:.1f}s)")
        qa = pipelines[chat]

        gold = locate_gold(qa.messages, q)
        is_negative = not gold

        dense = qa.store.search(q["question"], top_k=len(qa.chunks))
        dense_rank = _first_rank([d["chunk"] for d in dense], gold) if gold else None
        hits = qa.retrieve(q["question"])
        rerank_rank = _first_rank([h["chunk"] for h in hits], gold) if gold else None

        retrieved_num_msgs = sum(len(h["chunk"].message_ids) for h in hits) if hits else 0
        retrieved_num_chars = sum(len(h["chunk"].attributed_text) for h in hits) if hits else 0

        row = {
            "id": q["id"], "chat": chat, "question": q["question"],
            "question_type": q.get("question_type"), "is_negative": is_negative,
            "gold_message_ids": sorted(gold),
            "dense_rank": dense_rank,              # rank in pure similarity order (all chunks)
            "rank_in_top_k": rerank_rank,          # rank after time-aware reranking, None if outside top-k
            "top_semantic_score": round(max((h["semantic_score"] for h in hits), default=0.0), 4),
            "gold_semantic_score": (round(float(next(d["semantic_score"] for d in dense
                                                       if gold & set(d["chunk"].message_ids))), 4)
                                    if gold else None),
            "retrieved_chunk_ids": [h["chunk"].chunk_id for h in hits],
            "retrieved_contexts": [h["chunk"].attributed_text for h in hits],
            "retrieved_num_messages": retrieved_num_msgs,
            "retrieved_num_chars": retrieved_num_chars,
            "reference": q.get("ground_truth_answer", ""),
            "response": None,
            "judge_verdict": None,
        }
        if use_llm:
            res = qa.ask(q["question"])
            row["response"] = res.get("answer") or f"[error] {res.get('error')}"
            row["abstained"] = row["response"].strip().startswith(NOT_FOUND[:20])

            if judge and not is_negative and row["response"] and not row["abstained"] and llm:
                judge_prompt = (
                    f"Question: {row['question']}\n"
                    f"Reference Answer: {row['reference']}\n"
                    f"Generated Answer: {row['response']}\n\n"
                    f"Does the Generated Answer match the Reference Answer?\n"
                    f"Reply with exactly one word: 'correct', 'partial', or 'wrong'."
                )
                try:
                    v_raw = llm.chat([{"role": "user", "content": judge_prompt}]).strip().lower()
                    m_v = re.search(r"\b(correct|partial|wrong)\b", v_raw)
                    row["judge_verdict"] = m_v.group(1) if m_v else "wrong"
                except Exception:
                    row["judge_verdict"] = None

        rows.append(row)
        print(f"  {q['id']:>4} {'NEG' if is_negative else 'rank=' + str(rerank_rank):>8}  {q['question'][:60]}")
    return rows


def summarise(rows, top_k, use_llm):
    pos = [r for r in rows if not r["is_negative"]]
    neg = [r for r in rows if r["is_negative"]]
    out = {"questions": len(rows), "answerable": len(pos), "negative": len(neg)}
    if pos:
        for qt in sorted({r["question_type"] for r in pos}):
            sub = [r for r in pos if r["question_type"] == qt]
            out[f"hit@{top_k}[{qt}] (n={len(sub)})"] = sum(r["rank_in_top_k"] is not None for r in sub) / len(sub)
        out["hit@1"] = sum(r["rank_in_top_k"] == 1 for r in pos) / len(pos)
        out[f"hit@{top_k}"] = sum(r["rank_in_top_k"] is not None for r in pos) / len(pos)
        out["mrr@k"] = sum(1 / r["rank_in_top_k"] for r in pos if r["rank_in_top_k"]) / len(pos)
        out[f"dense_only_hit@{top_k}"] = sum(
            r["dense_rank"] is not None and r["dense_rank"] <= top_k for r in pos) / len(pos)

        # Bootstrap 95% Confidence Intervals
        pos_ranks = [r["rank_in_top_k"] for r in pos]
        out["hit@1_ci95"] = bootstrap_ci(pos_ranks, metric="hit@1")
        out[f"hit@{top_k}_ci95"] = bootstrap_ci(pos_ranks, metric=f"hit@{top_k}")
        out["mrr@k_ci95"] = bootstrap_ci(pos_ranks, metric="mrr")

    # Average retrieved context size per question
    if rows:
        msg_counts = [r.get("retrieved_num_messages", sum(len(c.splitlines()) for c in r.get("retrieved_contexts", []))) for r in rows]
        char_counts = [r.get("retrieved_num_chars", sum(len(c) for c in r.get("retrieved_contexts", []))) for r in rows]
        out["avg_retrieved_messages"] = round(float(np.mean(msg_counts)), 1)
        out["avg_retrieved_chars"] = round(float(np.mean(char_counts)), 1)

    # Score ranges, to calibrate min_relevance: answerable questions should score
    # clearly higher than negative controls.
    if pos:
        out["mean_top_score_answerable"] = sum(r["top_semantic_score"] for r in pos) / len(pos)
    if neg:
        out["mean_top_score_negative"] = sum(r["top_semantic_score"] for r in neg) / len(neg)
    if use_llm:
        if pos:
            out["answered_when_answerable"] = sum(not r["abstained"] for r in pos) / len(pos)
        if neg:
            out["abstained_when_negative"] = sum(r["abstained"] for r in neg) / len(neg)
        judged = [r["judge_verdict"] for r in pos if r.get("judge_verdict")]
        if judged:
            n_j = len(judged)
            out["judge_correct"] = round(sum(v == "correct" for v in judged) / n_j, 3)
            out["judge_partial"] = round(sum(v == "partial" for v in judged) / n_j, 3)
            out["judge_wrong"] = round(sum(v == "wrong" for v in judged) / n_j, 3)

    return {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chats-dir", default="data/raw_real")
    ap.add_argument("--eval", default="data/benchmarks/evaluation_set_v2.json")
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--no-llm", action="store_true", help="retrieval metrics only")
    ap.add_argument("--embedder", default="minilm", choices=["minilm", "bge-m3", "me5"])
    ap.add_argument("--embed-text", default="clean", choices=["clean", "attributed"],
                    help="clean = 'Sender: text' (default), attributed = with timestamps")
    ap.add_argument("--chunker", default="semantic", choices=["semantic", "naive", "regex", "fixed"])
    ap.add_argument("--retrieval", default="hybrid", choices=["hybrid", "dense", "bm25_only", "message_window", "cross_encoder"])
    ap.add_argument("--expand-queries", action="store_true", help="expand questions using LLM paraphrases")
    ap.add_argument("--judge", action="store_true", help="enable LLM answer-level evaluation judge")
    args = ap.parse_args()
    tag = f"{os.path.splitext(os.path.basename(args.eval))[0]}_{args.embedder}_{args.embed_text}"
    if args.chunker != "semantic":
        tag += f"_{args.chunker}"
    if args.retrieval != "hybrid":
        tag += f"_{args.retrieval}"
    if args.expand_queries:
        tag += "_expanded"
    if args.judge:
        tag += "_judged"

    with open(args.eval, encoding="utf-8") as f:
        items = json.load(f)
    chat_paths = {}
    for chat in sorted({q["chat"] for q in items}):
        p = os.path.join(args.chats_dir, f"{chat}.txt")
        if not os.path.exists(p):
            sys.exit(f"Missing chat export: {p}")
        chat_paths[chat] = p

    llm = None
    if not args.no_llm:
        from src.generation.ollama_client import OllamaClient
        llm = OllamaClient(model=args.model)
        if not llm.is_available():
            sys.exit(f"Ollama model '{args.model}' not available (ollama pull {args.model}), "
                     f"or run with --no-llm")

    t0 = time.time()
    rows = evaluate(items, chat_paths, llm=llm, top_k=args.top_k, use_llm=not args.no_llm,
                    embedder=args.embedder, embed_text=args.embed_text, chunker=args.chunker,
                    retrieval=args.retrieval, expand_queries=args.expand_queries, judge=args.judge)
    summary = {"config": {"embedder": args.embedder, "embed_text": args.embed_text,
                          "chunker": args.chunker, "retrieval": args.retrieval,
                          "expand_queries": args.expand_queries,
                          "eval": args.eval, "top_k": args.top_k,
                          "llm": None if args.no_llm else args.model},
               **summarise(rows, args.top_k, not args.no_llm),
               "runtime_seconds": round(time.time() - t0, 1)}

    os.makedirs("eval_results", exist_ok=True)
    out_json = f"eval_results/qa_eval_{tag}.json"
    out_ragas = f"eval_results/ragas_{tag}.jsonl"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": rows}, f, indent=2, ensure_ascii=False)
    if not args.no_llm:
        with open(out_ragas, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "user_input": r["question"],
                    "retrieved_contexts": r["retrieved_contexts"],
                    "response": r["response"],
                    "reference": r["reference"],
                }, ensure_ascii=False) + "\n")

    print("\n=== Summary ===")
    for k, v in summary.items():
        if k != "config":
            print(f"  {k:<28} {v}")
    print(f"\nSaved {out_json}" + ("" if args.no_llm else f" and {out_ragas}"))


if __name__ == "__main__":
    main()
