"""
Compare two or more evaluation runs side by side.

    python scripts/compare_evals.py eval_results/qa_eval_evaluation_set_minilm_clean.json \\
                                    eval_results/qa_eval_evaluation_set_bge-m3_clean.json

    # or everything in eval_results/ for one question set:
    python scripts/compare_evals.py --glob "eval_results/qa_eval_new_queries_*.json"

Prints (and saves as markdown next to the first file):
  1. summary metrics per run
  2. per-question rank of the gold chunk in each run, with differences marked
"""

import argparse
import glob
import json
import os


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cfg = data["summary"].get("config", {})
    label = f"{cfg.get('embedder', '?')}/{cfg.get('embed_text', '?')}/{cfg.get('chunker', 'semantic')}/{cfg.get('retrieval', 'dense')}"
    return label, data["summary"], {r["id"]: r for r in data["results"]}


def fmt(v):
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def rank_str(r):
    if r["is_negative"]:
        return f"NEG ({r['top_semantic_score']:.2f})"
    rk = r["rank_in_top_k"]
    dense = r["dense_rank"]
    return f"{rk}" if rk else f"miss (dense {dense})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--glob")
    args = ap.parse_args()
    files = list(args.files) + (sorted(glob.glob(args.glob)) if args.glob else [])
    if len(files) < 2:
        raise SystemExit("Give at least two result files.")

    runs = [load(f) for f in files]
    labels = [r[0] for r in runs]
    lines = []

    metrics = [k for k in runs[0][1] if k not in ("config",)]
    lines.append("## Summary\n")
    lines.append("| metric | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for m in metrics:
        lines.append(f"| {m} | " + " | ".join(fmt(r[1].get(m)) for r in runs) + " |")

    lines.append("\n## Per question (rank of gold chunk in top-k; NEG shows top score)\n")
    lines.append("| id | type | question | " + " | ".join(labels) + " |")
    lines.append("|---|---|---|" + "---|" * len(labels))
    ids = list(runs[0][2].keys())
    changed = 0
    for qid in ids:
        rows = [r[2].get(qid) for r in runs]
        if any(x is None for x in rows):
            continue
        cells = [rank_str(x) for x in rows]
        differs = len({x["rank_in_top_k"] for x in rows}) > 1 and not rows[0]["is_negative"]
        changed += differs
        q = rows[0]["question"][:55].replace("|", "/")
        mark = " **≠**" if differs else ""
        lines.append(f"| {qid}{mark} | {rows[0]['question_type']} | {q} | " + " | ".join(cells) + " |")
    lines.append(f"\n{changed} answerable questions changed rank between runs.")

    text = "\n".join(lines)
    print(text)
    eval_name = os.path.splitext(os.path.basename(runs[0][1].get("config", {}).get("eval", "eval")))[0]
    out = os.path.join(os.path.dirname(files[0]) or ".", f"comparison_{eval_name}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
