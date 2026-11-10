"""
Runs extra retrieval baselines:
1. bm25_only: BM25 keyword matching over chunks only.
2. message_window: Dense retrieval over single messages, expanded with +/-2 neighbor messages.
3. cross_encoder: Reranks top-20 hybrid candidates with BAAI/bge-reranker-base (if available).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_qa_eval import evaluate, summarise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/benchmarks/eval_dev.json")
    ap.add_argument("--chats-dir", default="data/raw_real")
    ap.add_argument("--embedder", default="minilm")
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    if not os.path.exists(args.eval):
        sys.exit(f"Eval file not found: {args.eval}")

    with open(args.eval, encoding="utf-8") as f:
        items = json.load(f)

    chat_paths = {}
    for chat in sorted({q["chat"] for q in items}):
        p = os.path.join(args.chats_dir, f"{chat}.txt")
        if not os.path.exists(p):
            sys.exit(f"Missing chat export: {p}")
        chat_paths[chat] = p

    baselines = ["bm25_only", "message_window", "cross_encoder"]
    results = {}

    for b in baselines:
        print(f"\n==========================================")
        print(f"Running baseline: {b}")
        print(f"==========================================")
        try:
            rows = evaluate(
                items,
                chat_paths,
                top_k=args.top_k,
                use_llm=False,
                embedder=args.embedder,
                chunker="semantic" if b != "message_window" else "naive",
                retrieval=b,
            )
            summary = summarise(rows, top_k=args.top_k, use_llm=False)
            results[b] = summary
            print(f"[{b}] hit@1: {summary.get('hit@1')}, hit@{args.top_k}: {summary.get(f'hit@{args.top_k}')}, MRR: {summary.get('mrr@k')}")
        except Exception as e:
            print(f"Error running baseline {b}: {e}")

    out_path = "eval_results/baselines_extra.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved extra baselines summary to {out_path}")


if __name__ == "__main__":
    main()
