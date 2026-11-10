"""
Abstention Calibration:
Sweeps min_relevance from 0.05 to 0.60 (step 0.01) on the dev split.
Reports:
(a) fraction of answerable questions answered (top score >= tau)
(b) fraction of negative-control questions abstained (top score < tau)
Writes eval_results/abstention_calibration.csv and eval_results/abstention_calibration.png.
Prints the threshold maximizing the sum of the two fractions.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_qa_eval import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/benchmarks/eval_dev.json")
    ap.add_argument("--chats-dir", default="data/raw_real")
    ap.add_argument("--embedder", default="minilm")
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

    print(f"Evaluating dev split ({len(items)} queries) for abstention calibration...")
    rows = evaluate(
        items,
        chat_paths,
        top_k=4,
        use_llm=False,
        embedder=args.embedder,
        embed_text="clean",
        chunker="semantic",
        retrieval="hybrid",
    )

    pos = [r for r in rows if not r["is_negative"]]
    neg = [r for r in rows if r["is_negative"]]
    n_pos = len(pos)
    n_neg = len(neg)

    print(f"Total: {len(rows)} (Answerable: {n_pos}, Negative: {n_neg})")

    thresholds = [round(0.05 + 0.01 * i, 2) for i in range(56)]
    data = []

    best_score = -1.0
    best_tau = 0.0

    for tau in thresholds:
        ans_answered = sum(r["top_semantic_score"] >= tau for r in pos) / n_pos if n_pos else 0.0
        neg_abstained = sum(r["top_semantic_score"] < tau for r in neg) / n_neg if n_neg else 0.0
        score_sum = ans_answered + neg_abstained

        data.append({
            "threshold": tau,
            "fraction_answered_when_answerable": round(ans_answered, 4),
            "fraction_abstained_when_negative": round(neg_abstained, 4),
            "sum_fractions": round(score_sum, 4),
        })

        if score_sum > best_score:
            best_score = score_sum
            best_tau = tau

    os.makedirs("eval_results", exist_ok=True)
    csv_path = "eval_results/abstention_calibration.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "fraction_answered_when_answerable",
                "fraction_abstained_when_negative",
                "sum_fractions",
            ],
        )
        writer.writeheader()
        writer.writerows(data)

    # Plot PNG
    png_path = "eval_results/abstention_calibration.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        taus = [d["threshold"] for d in data]
        ans = [d["fraction_answered_when_answerable"] for d in data]
        abst = [d["fraction_abstained_when_negative"] for d in data]
        sums = [d["sum_fractions"] for d in data]

        plt.figure(figsize=(9, 5))
        plt.plot(taus, ans, label="Answered (Answerable)", color="tab:blue", lw=2)
        plt.plot(taus, abst, label="Abstained (Negative Control)", color="tab:orange", lw=2)
        plt.plot(taus, sums, label="Sum (Objective)", color="tab:green", lw=2, linestyle="--")
        plt.axvline(best_tau, color="red", linestyle=":", label=f"Optimal threshold ({best_tau:.2f})")

        plt.xlabel("min_relevance Threshold")
        plt.ylabel("Fraction")
        plt.title(f"Abstention Calibration on Dev Split (Best threshold = {best_tau:.2f}, Sum = {best_score:.3f})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved plot to {png_path}")
    except Exception as e:
        print(f"Note: Could not generate plot ({e})")

    print(f"Saved calibration data to {csv_path}")
    print(f"\n>>> Optimal min_relevance threshold: {best_tau:.2f} (Sum: {best_score:.4f}) <<<")


if __name__ == "__main__":
    main()
