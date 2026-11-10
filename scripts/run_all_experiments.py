"""
Runs every retrieval configuration on both question sets and writes one results
table for the review: eval_results/experiments_summary.md

    python scripts/run_all_experiments.py                  # MiniLM only
    python scripts/run_all_experiments.py --with-bge-m3    # also bge-m3 (needs: ollama pull bge-m3)
    python scripts/run_all_experiments.py --eval data/benchmarks/evaluation_set.json   # old set

Works on Windows, macOS and Linux. Retrieval only (no answer model needed).
"""

import argparse
import json
import os
import subprocess
import sys

DEFAULT_EVAL_SETS = ["data/benchmarks/evaluation_set_v2.json"]

# (label, chunker, retrieval) - ordered as the story for the review:
# baselines -> original proposed system -> improvements
CONFIGS = [
    ("Baseline: 30-min time-gap chunks, dense", "naive", "dense"),
    ("Baseline: fixed 8-message chunks, dense", "fixed", "dense"),
    ("Old proposed: regex topic rules, dense + time rerank", "regex", "dense"),
    ("Semantic topic chunking, dense + time rerank", "semantic", "dense"),
    ("Semantic topic chunking + hybrid retrieval (final)", "semantic", "hybrid"),
]


def run(embedder, chunker, retrieval, eval_path):
    cmd = [sys.executable, "scripts/run_qa_eval.py", "--no-llm", "--embedder", embedder,
           "--chunker", chunker, "--retrieval", retrieval, "--eval", eval_path]
    print(">>", " ".join(cmd[1:]))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-1500:], proc.stderr[-1500:])
        raise SystemExit("Run failed; see the error above.")
    tag = f"{os.path.splitext(os.path.basename(eval_path))[0]}_{embedder}_clean"
    if chunker != "semantic":
        tag += f"_{chunker}"
    if retrieval != "hybrid":
        tag += f"_{retrieval}"
    with open(f"eval_results/qa_eval_{tag}.json", encoding="utf-8") as f:
        return json.load(f)["summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-bge-m3", action="store_true")
    ap.add_argument("--eval", nargs="+", default=DEFAULT_EVAL_SETS,
                    help="question set(s); default evaluation_set_v2.json")
    args = ap.parse_args()
    embedders = ["minilm"] + (["bge-m3"] if args.with_bge_m3 else [])

    lines = ["# Retrieval experiments", "",
             "hit@k = share of answerable questions whose gold message is in the top-k retrieved chunks.", ""]
    for eval_path in args.eval:
        name = os.path.splitext(os.path.basename(eval_path))[0]
        lines += [f"## {name}", "",
                  "| embedder | configuration | hit@1 | hit@4 | MRR | hit@4 reworded | hit@4 code-mixed |",
                  "|---|---|---|---|---|---|---|"]
        for emb in embedders:
            for label, chunker, retrieval in CONFIGS:
                s = run(emb, chunker, retrieval, eval_path)
                by = lambda qt: next((f"{v:.2f}" for k, v in s.items() if k.startswith(f"hit@4[{qt}]")), "-")
                lines.append(f"| {emb} | {label} | {s['hit@1']:.2f} | {s['hit@4']:.2f} | {s['mrr@k']:.2f} "
                             f"| {by('paraphrased')} | {by('code-mixed')} |")
        lines.append("")

    out = "eval_results/experiments_summary.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
