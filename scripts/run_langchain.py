"""
LangChain Pipeline — Framework Comparison Runner
------------------------------------------------
Evaluates LangChain RAG retrieval on conversational WhatsApp exports.
Supports:
- standard: document-oriented defaults (RecursiveCharacterTextSplitter)
- message_split: fair message-boundary splitting

Usage:
    python scripts/run_langchain.py
    python scripts/run_langchain.py --split-mode message_split
    python scripts/run_langchain.py --help
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.framework_baselines import run_baseline_evaluation


def main():
    parser = argparse.ArgumentParser(description="LangChain Pipeline - Framework Comparison Runner")
    parser.add_argument("--eval-set", default="evaluation_set.json", help="Path to evaluation questions JSON")
    parser.add_argument("--output", default=None, help="Output path for JSON results")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K chunks to retrieve")
    parser.add_argument(
        "--split-mode",
        choices=["standard", "message_split"],
        default="standard",
        help="Splitting strategy: 'standard' (document-oriented defaults) or 'message_split' (message boundaries)",
    )
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid retrieval (dense + BM25)")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = (
            "eval_results/results_langchain_message_split.json"
            if args.split_mode == "message_split"
            else "eval_results/results_langchain.json"
        )

    run_baseline_evaluation(
        framework="langchain",
        eval_file=args.eval_set,
        output_file=output,
        top_k=args.top_k,
        split_mode=args.split_mode,
        hybrid=args.hybrid,
    )


if __name__ == "__main__":
    main()
