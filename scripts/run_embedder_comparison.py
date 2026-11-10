"""
Compare retrieval performance across embedding backends (minilm, me5, bge-m3).
Outputs results to eval_results/embedder_comparison.md.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_qa_eval import evaluate, summarise
from src.indexing.embedders import get_embedder


def main():
    eval_file = os.getenv("EVAL_FILE", "data/benchmarks/evaluation_set_v2.json")
    chats_dir = os.getenv("CHATS_DIR", "data/raw_real")

    if not os.path.exists(eval_file):
        sys.exit(f"Eval file not found: {eval_file}")

    with open(eval_file, encoding="utf-8") as f:
        items = json.load(f)

    chat_paths = {}
    for chat in sorted({q["chat"] for q in items}):
        p = os.path.join(chats_dir, f"{chat}.txt")
        if not os.path.exists(p):
            sys.exit(f"Missing chat export: {p}")
        chat_paths[chat] = p

    embedders = ["minilm", "me5", "bge-m3"]
    results = []

    for emb in embedders:
        print(f"\n==========================================")
        print(f"Testing embedder: {emb}")
        print(f"==========================================")
        try:
            # Test instantiation to verify availability
            _ = get_embedder(emb)
        except Exception as e:
            print(f"Note: Skipping {emb} - model unavailable ({e})")
            continue

        try:
            rows = evaluate(
                items,
                chat_paths,
                top_k=4,
                use_llm=False,
                embedder=emb,
                embed_text="clean",
                chunker="semantic",
                retrieval="hybrid",
            )
            summary = summarise(rows, top_k=4, use_llm=False)
            results.append({
                "embedder": emb,
                "hit@1": summary.get("hit@1", 0.0),
                "hit@4": summary.get("hit@4", 0.0),
                "mrr": summary.get("mrr@k", 0.0),
                "hit@4_reworded": summary.get("hit@4[paraphrased] (n=10)", 0.0),
                "hit@4_code_mixed": summary.get("hit@4[code-mixed] (n=3)", 0.0),
            })
        except Exception as e:
            print(f"Note: Skipping {emb} - evaluation failed ({e})")
            continue

    os.makedirs("eval_results", exist_ok=True)
    out_md = "eval_results/embedder_comparison.md"

    table_lines = [
        "# Embedder Comparison Benchmark",
        "",
        "| Embedder | hit@1 | hit@4 | MRR | hit@4 (paraphrased) | hit@4 (code-mixed) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        table_lines.append(
            f"| **{r['embedder']}** | {r['hit@1']:.3f} | {r['hit@4']:.3f} | {r['mrr']:.3f} | {r['hit@4_reworded']:.3f} | {r['hit@4_code_mixed']:.3f} |"
        )

    content = "\n".join(table_lines) + "\n"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nWrote comparison to {out_md}:")
    print(content)


if __name__ == "__main__":
    main()
