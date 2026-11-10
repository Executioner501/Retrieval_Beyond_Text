"""
Ablation Study:
Runs the 4x2 grid of chunkers (naive, fixed, regex, semantic) x retrieval (dense, hybrid).
Defaults to the dev split (--split dev|test|all).
Outputs markdown and JSON to eval_results/ablation.md and eval_results/ablation.json.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_qa_eval import evaluate, summarise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--chats-dir", default="data/raw_real")
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    split_map = {
        "dev": "data/benchmarks/eval_dev.json",
        "test": "data/benchmarks/eval_test.json",
        "all": "data/benchmarks/evaluation_set_v2.json",
    }
    eval_file = split_map[args.split]
    if not os.path.exists(eval_file):
        sys.exit(f"Eval file not found: {eval_file}")

    with open(eval_file, encoding="utf-8") as f:
        items = json.load(f)

    chat_paths = {}
    for chat in sorted({q["chat"] for q in items}):
        p = os.path.join(args.chats_dir, f"{chat}.txt")
        if not os.path.exists(p):
            sys.exit(f"Missing chat export: {p}")
        chat_paths[chat] = p

    chunkers = ["naive", "fixed", "regex", "semantic"]
    retrievals = ["dense", "hybrid"]

    results = []

    print(f"=== Starting Ablation Study on {args.split} split ({len(items)} queries) ===")
    for c in chunkers:
        for r in retrievals:
            print(f"\n--- Running: chunker={c}, retrieval={r} ---")
            rows = evaluate(
                items,
                chat_paths,
                top_k=args.top_k,
                use_llm=False,
                embedder="minilm",
                embed_text="clean",
                chunker=c,
                retrieval=r,
            )
            summary = summarise(rows, args.top_k, use_llm=False)
            res = {
                "chunker": c,
                "retrieval": r,
                "hit@1": summary.get("hit@1", 0.0),
                "hit@1_ci95": summary.get("hit@1_ci95", [0.0, 0.0]),
                f"hit@{args.top_k}": summary.get(f"hit@{args.top_k}", 0.0),
                f"hit@{args.top_k}_ci95": summary.get(f"hit@{args.top_k}_ci95", [0.0, 0.0]),
                "mrr": summary.get("mrr@k", 0.0),
                "mrr_ci95": summary.get("mrr@k_ci95", [0.0, 0.0]),
                "avg_msgs": summary.get("avg_retrieved_messages", 0.0),
                "avg_chars": summary.get("avg_retrieved_chars", 0.0),
            }
            results.append(res)
            print(f"Result: hit@1={res['hit@1']}, hit@4={res[f'hit@{args.top_k}']}, MRR={res['mrr']}")

    os.makedirs("eval_results", exist_ok=True)
    out_json = "eval_results/ablation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "results": results}, f, indent=2)

    # Markdown table
    md_lines = [
        f"# Ablation Study ({args.split.upper()} Split)",
        "",
        "| Chunker | Retrieval | hit@1 (95% CI) | hit@4 (95% CI) | MRR (95% CI) | Avg Msgs | Avg Chars |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in results:
        h1_str = f"{row['hit@1']:.3f} [{row['hit@1_ci95'][0]:.3f}, {row['hit@1_ci95'][1]:.3f}]"
        h4_str = f"{row['hit@4']:.3f} [{row['hit@4_ci95'][0]:.3f}, {row['hit@4_ci95'][1]:.3f}]"
        mrr_str = f"{row['mrr']:.3f} [{row['mrr_ci95'][0]:.3f}, {row['mrr_ci95'][1]:.3f}]"
        md_lines.append(
            f"| {row['chunker']} | {row['retrieval']} | {h1_str} | {h4_str} | {mrr_str} | {row['avg_msgs']:.1f} | {row['avg_chars']:.0f} |"
        )

    out_md = "eval_results/ablation.md"
    content = "\n".join(md_lines) + "\n"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nWrote ablation results to {out_md} and {out_json}:")
    print(content)


if __name__ == "__main__":
    main()
