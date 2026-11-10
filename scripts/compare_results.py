"""
Compare Results — Summary Table & Failure Diagnostics
-----------------------------------------------------
Aggregates and compares retrieval evaluations across all four frameworks:
1. Naive Time-Gap Baseline
2. LangChain Standard RAG
3. LlamaIndex Standard RAG
4. Proposed Context-Aware RAG

Usage:
    python scripts/compare_results.py
"""

import os
import json
from collections import defaultdict
from typing import Dict, List, Any

FILES = {
    "Naive Baseline": "eval_results/results_naive_baseline.json",
    "LangChain": "eval_results/results_langchain.json",
    "LlamaIndex": "eval_results/results_llamaindex.json",
    "Context-Aware": "eval_results/results_context_aware.json",
}

# Fallback paths if run from different working directory
ALT_FILES = {
    "Naive Baseline": "results_naive_baseline.json",
    "LangChain": "results_langchain.json",
    "LlamaIndex": "results_llamaindex.json",
    "Context-Aware": "results_context_aware.json",
}

SCORE_WEIGHTS = {"hit": 1.0, "partial": 0.5, "miss": 0.0}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    all_results = {}
    for label, path in FILES.items():
        target = path if os.path.exists(path) else ALT_FILES.get(label, "")
        if os.path.exists(target):
            all_results[label] = load_json(target)
        else:
            print(f"[-] Skipping {label}: {path} not found.")

    if not all_results:
        print("[-] No result files found. Run the pipeline scripts in scripts/ first:")
        print("    python scripts/run_naive_baseline.py")
        print("    python scripts/run_langchain.py")
        print("    python scripts/run_llamaindex.py")
        print("    python scripts/run_context_aware.py")
        return

    # Guard: refuse to summarize anything that hasn't actually been manually scored.
    # An empty "" manual_score means the human review step hasn't been done yet —
    # silently treating it as "miss" would produce a misleading report.
    any_unscored = False
    for label, results in all_results.items():
        unscored = [r["id"] for r in results if not r.get("manual_score")]
        if unscored:
            any_unscored = True
            print(f"[!] {label}: {len(unscored)} question(s) not yet manually scored: {unscored}")
    if any_unscored:
        print("\n[!] Fill in \"manual_score\" (\"hit\" / \"partial\" / \"miss\") for every question in")
        print("    every eval_results/results_*.json file before running this script.")
        print("    Each result already has an \"auto_score\" field as a starting suggestion —")
        print("    read the actual retrieved_chunks yourself and confirm or correct it.")
        return

    print("\n" + "=" * 80)
    print("EMPIRICAL RETRIEVAL BENCHMARK: OVERALL SCORES")
    print("=" * 80)
    print(f"{'Framework':<22} | {'Overall Score':<14} | {'Hits':<6} | {'Partials':<8} | {'Misses':<6} | {'Total':<5}")
    print("-" * 80)

    for label, results in all_results.items():
        total = len(results)
        score_sum = sum(SCORE_WEIGHTS.get(r.get("manual_score", "miss"), 0.0) for r in results)
        pct = (score_sum / total) * 100 if total else 0.0
        hits = sum(1 for r in results if r.get("manual_score") == "hit")
        partials = sum(1 for r in results if r.get("manual_score") == "partial")
        misses = sum(1 for r in results if r.get("manual_score") == "miss")
        print(f"{label:<22} | {pct:>12.1f}% | {hits:>6} | {partials:>8} | {misses:>6} | {total:>5}")

    print("\n" + "=" * 80)
    print("PER-QUESTION-TYPE PERFORMANCE BREAKDOWN")
    print("=" * 80)

    for label, results in all_results.items():
        print(f"\n--- {label} ---")
        by_type = defaultdict(list)
        for r in results:
            qtype = r.get("question_type", "general")
            by_type[qtype].append(r.get("manual_score", "miss"))

        for qtype, scores in sorted(by_type.items()):
            total = len(scores)
            score_sum = sum(SCORE_WEIGHTS.get(s, 0.0) for s in scores)
            pct = (score_sum / total) * 100 if total else 0.0
            h = scores.count("hit")
            p = scores.count("partial")
            m = scores.count("miss")
            print(f"  {qtype:<22} : {pct:>5.1f}%  (hit={h}, partial={p}, miss={m}, n={total})")

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE QUESTION COMPARISON MATRIX")
    print("=" * 80)
    labels = list(all_results.keys())
    base_results = all_results[labels[0]]

    header_cols = [f"{lbl[:10]:<10}" for lbl in labels]
    print(f"{'ID':<5} | {'Query Snippet':<35} | " + " | ".join(header_cols))
    print("-" * (45 + 13 * len(labels)))

    for i, r in enumerate(base_results):
        qid = r.get("id", f"q{i+1}")
        qtext = r.get("question", "")[:35]
        scores = []
        for lbl in labels:
            if i < len(all_results[lbl]):
                sc = all_results[lbl][i].get("manual_score", "?")
            else:
                sc = "N/A"
            scores.append(f"{sc:<10}")
        print(f"{qid:<5} | {qtext:<35} | " + " | ".join(scores))

    # Save summary report JSON
    summary_report = {
        "frameworks": list(all_results.keys()),
        "total_queries": len(base_results),
        "overall": {
            lbl: {
                "score_pct": round((sum(SCORE_WEIGHTS.get(r.get("manual_score", "miss"), 0.0) for r in res) / len(res)) * 100, 2),
                "hits": sum(1 for r in res if r.get("manual_score") == "hit"),
                "partials": sum(1 for r in res if r.get("manual_score") == "partial"),
                "misses": sum(1 for r in res if r.get("manual_score") == "miss"),
            }
            for lbl, res in all_results.items()
        }
    }
    with open("eval_results/benchmark_summary_report.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print("\n[+] Exported summary report to: eval_results/benchmark_summary_report.json")


if __name__ == "__main__":
    main()
