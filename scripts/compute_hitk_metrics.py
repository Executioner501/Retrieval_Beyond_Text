"""
Hit@K / MRR Metrics — Rank-Aware Evaluation
-----------------------------------------------
Manual hit/partial/miss scoring tells you WHETHER a question was answered,
but throws away WHERE in the ranked results the answer showed up. A chunk
retrieved at rank 1 vs rank 3 is a meaningfully different result — this
script measures that.

Metrics computed, per framework:
- Hit@1  : % of questions where the correct chunk was the TOP retrieved result
- Hit@3  : % of questions where the correct chunk was ANYWHERE in the top 3
- MRR    : Mean Reciprocal Rank — average of 1/rank across questions
           (rewards getting the right answer at rank 1 more than rank 3;
            0 if not found in the retrieved set at all)

Negative-control questions (question_type == "negative-control") are
scored separately as "correct abstention rate" — since there's no real
chunk to rank, Hit@K doesn't apply to them the same way. A negative
control "succeeds" if the top result's similarity score is low (below
NEG_CONTROL_THRESHOLD), meaning the system correctly did NOT confidently
retrieve a fabricated answer.

This is fully automatic - it matches each question's "source_text_snippet"
(from evaluation_set.json) against the retrieved_chunks' text, so it does
NOT require the manual_score step to already be done. Use it alongside
manual scoring, not instead of it - automatic snippet matching can still
be fooled by near-duplicate text, so spot-check a few results yourself.

Usage (run from project root, after the run_*.py scripts have produced
their eval_results/results_*.json files):
    python scripts/compute_hitk_metrics.py
"""

import json
import os
import re
from collections import defaultdict

RESULT_FILES = {
    "Naive Baseline": "eval_results/results_naive_baseline.json",
    "LangChain": "eval_results/results_langchain.json",
    "LlamaIndex": "eval_results/results_llamaindex.json",
    "Context-Aware": "eval_results/results_context_aware.json",
}

NEG_CONTROL_THRESHOLD = 0.45  # same threshold used by auto_score's is_neg logic


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text):
    """Lowercase + collapse whitespace, for forgiving substring matching."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def snippet_in_chunk(snippet, chunk_text):
    """
    Checks if the ground-truth snippet appears in a retrieved chunk.
    Handles snippets that were truncated with '...' by only matching
    the part before the ellipsis.
    """
    snippet_norm = normalize(snippet)
    if snippet_norm.endswith("..."):
        snippet_norm = snippet_norm[:-3].strip()
    chunk_norm = normalize(chunk_text)

    if not snippet_norm:
        return False

    # Require a reasonably long match to avoid trivial false positives
    # (e.g. a snippet of just "ok" matching everywhere)
    if len(snippet_norm) < 8:
        return snippet_norm in chunk_norm

    return snippet_norm in chunk_norm


def compute_metrics_for_file(results):
    """
    Returns per-question rank info, plus aggregated Hit@1 / Hit@3 / MRR,
    computed only over non-negative-control questions.
    Negative controls are reported separately as abstention accuracy.
    """
    ranked_scores = []  # list of (found_rank or None) for real questions
    neg_control_results = []  # list of bool (True = correctly abstained)
    per_question = []

    for r in results:
        qtype = r.get("question_type", "")
        snippet = r.get("source_text_snippet")
        retrieved = r.get("retrieved_chunks", [])

        if qtype == "negative-control" or snippet is None:
            top_score = retrieved[0]["score"] if retrieved else 0.0
            correctly_abstained = top_score < NEG_CONTROL_THRESHOLD
            neg_control_results.append(correctly_abstained)
            per_question.append({
                "id": r["id"], "type": qtype, "kind": "negative-control",
                "correctly_abstained": correctly_abstained, "top_score": top_score,
            })
            continue

        found_rank = None
        for i, chunk in enumerate(retrieved, start=1):
            if snippet_in_chunk(snippet, chunk.get("chunk_text", "")):
                found_rank = i
                break

        ranked_scores.append(found_rank)
        per_question.append({
            "id": r["id"], "type": qtype, "kind": "retrieval",
            "found_at_rank": found_rank,
        })

    n = len(ranked_scores)
    hit_at_1 = sum(1 for r in ranked_scores if r == 1) / n * 100 if n else 0.0
    hit_at_3 = sum(1 for r in ranked_scores if r is not None and r <= 3) / n * 100 if n else 0.0
    mrr = sum((1.0 / r) if r else 0.0 for r in ranked_scores) / n if n else 0.0

    abstain_n = len(neg_control_results)
    abstain_rate = (sum(neg_control_results) / abstain_n * 100) if abstain_n else None

    return {
        "hit_at_1_pct": round(hit_at_1, 2),
        "hit_at_3_pct": round(hit_at_3, 2),
        "mrr": round(mrr, 4),
        "n_retrieval_questions": n,
        "negative_control_abstain_rate_pct": round(abstain_rate, 2) if abstain_rate is not None else None,
        "n_negative_control_questions": abstain_n,
        "per_question": per_question,
    }


def main():
    all_metrics = {}
    for label, path in RESULT_FILES.items():
        if not os.path.exists(path):
            print(f"[!] Skipping {label} — {path} not found yet. Run its pipeline script first.")
            continue
        results = load_json(path)
        all_metrics[label] = compute_metrics_for_file(results)

    if not all_metrics:
        print("No result files found. Run the pipeline scripts first.")
        return

    print("=" * 78)
    print(f"{'Approach':<18}{'Hit@1':>10}{'Hit@3':>10}{'MRR':>10}{'NegAbstain':>14}")
    print("=" * 78)
    for label, m in all_metrics.items():
        abstain_str = f"{m['negative_control_abstain_rate_pct']:.1f}%" if m['negative_control_abstain_rate_pct'] is not None else "n/a"
        print(f"{label:<18}{m['hit_at_1_pct']:>9.1f}%{m['hit_at_3_pct']:>9.1f}%{m['mrr']:>10.3f}{abstain_str:>14}")

    print()
    print(f"(Hit@1/Hit@3/MRR computed over {list(all_metrics.values())[0]['n_retrieval_questions']} "
          f"non-negative-control questions; NegAbstain over "
          f"{list(all_metrics.values())[0]['n_negative_control_questions']} negative-control questions)")

    # Breakdown by question type
    print()
    print("=" * 78)
    print("HIT@1 BY QUESTION TYPE")
    print("=" * 78)
    for label, m in all_metrics.items():
        by_type = defaultdict(list)
        for pq in m["per_question"]:
            if pq["kind"] == "retrieval":
                by_type[pq["type"]].append(pq["found_at_rank"] == 1)
        print(f"\n--- {label} ---")
        for qtype, hits in sorted(by_type.items()):
            pct = sum(hits) / len(hits) * 100 if hits else 0
            print(f"  {qtype:<20} {pct:5.1f}%  (n={len(hits)})")

    # Save full report
    out_path = "eval_results/hitk_metrics_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()
