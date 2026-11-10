"""
Naive Baseline — Batch Evaluation Runner
----------------------------------------
Runs benchmark evaluation sets through the naive time-gap chunked pipeline.
Uses local dense embeddings (sentence-transformers) and normalized cosine similarity.
Saves results for comparison and failure diagnosis.

Usage:
    python scripts/run_naive_baseline.py
    python scripts/run_naive_baseline.py --help
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.framework_baselines import run_baseline_evaluation


def main():
    parser = argparse.ArgumentParser(description="Naive Baseline - Batch Evaluation Runner")
    parser.add_argument("--eval-set", default="evaluation_set.json", help="Path to evaluation questions JSON")
    parser.add_argument("--output", default="eval_results/results_naive_baseline.json", help="Output path for JSON results")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K chunks to retrieve")
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid retrieval (dense + BM25)")
    args = parser.parse_args()

    run_baseline_evaluation(
        framework="naive",
        eval_file=args.eval_set,
        output_file=args.output,
        top_k=args.top_k,
        hybrid=args.hybrid,
    )


if __name__ == "__main__":
    main()
