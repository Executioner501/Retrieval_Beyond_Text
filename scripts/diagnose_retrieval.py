"""
Diagnose why retrieval is failing.

    python scripts/diagnose_retrieval.py --embedder minilm
    python scripts/diagnose_retrieval.py --embedder minilm --eval data/benchmarks/new_queries.json

Checks, in order:
  1. Model sanity   - is the real model loaded, and does it rank a matching message above
                      unrelated ones? (catches broken installs / random fallback vectors)
  2. Index sanity   - are chunk embeddings distinct, or is every question getting the same chunks?
  3. Four retrievers on the same chunks, to locate the problem:
        A  pipeline      dense search on chunk text (what run_qa_eval measures, before reranking)
        B  keyword       TF-IDF on the same chunks (no neural model at all)
        C  message-level dense search on single messages, mapped back to their chunk
        D  attributed    dense search on chunk text with timestamps
     Reading the result:
        B good, A bad  -> the embedding model is the problem
        C good, A bad  -> whole-chunk embeddings dilute the match (chunks too long/mixed)
        all bad        -> chunking or the gold labels are the problem
  4. Example failures with the top retrieved chunk vs the gold chunk.

Writes eval_results/diagnosis_<eval>_<embedder>.txt
"""

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.feature_extraction.text import TfidfVectorizer

from src.indexing.embedders import get_embedder
from src.pipeline import ConversationQA
from scripts.run_qa_eval import locate_gold

K = 4


def chunk_rank(order, chunks, gold):
    for r, idx in enumerate(order, start=1):
        if gold & set(chunks[idx].message_ids):
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="minilm", choices=["minilm", "bge-m3"])
    ap.add_argument("--eval", default="data/benchmarks/new_queries.json")
    ap.add_argument("--chats-dir", default="data/raw_real")
    ap.add_argument("--chunker", default="semantic", choices=["semantic", "naive", "regex", "fixed"])
    args = ap.parse_args()

    buf = io.StringIO()

    def out(*a):
        print(*a)
        print(*a, file=buf)

    enc = get_embedder(args.embedder)
    inner = getattr(enc, "inner", enc)

    # ---------------------------------------------------------------- 1
    out("=" * 70, "\n1. MODEL SANITY\n" + "=" * 70)
    model_obj = getattr(inner, "model", None)
    from src.indexing.embedders import EMBEDDERS
    out(f"embedder: {args.embedder} | backend: {type(inner).__name__} | model: {EMBEDDERS[args.embedder]['model']}")
    if model_obj is not None and hasattr(model_obj, "max_seq_length"):
        out(f"max_seq_length: {model_obj.max_seq_length} tokens")
    q = "Did Nikhil invite me to join the hackathon?"
    cands = ["Dey will u join us for the Flipkart hackathon????",
             "Bro bring my record book and file from my shelf",
             "I will sleep again after eating",
             "Seems george russel lifted on the last corner"]
    v = enc.encode([q] + cands)
    sims = v[1:] @ v[0]
    out(f"dim={v.shape[1]}  norms={np.round(np.linalg.norm(v, axis=1), 3).tolist()}")
    for c, s in zip(cands, sims):
        out(f"  {s:+.3f}  {c}")
    ok = int(np.argmax(sims)) == 0
    out("  -> OK: matching message ranked first" if ok else
        "  -> PROBLEM: the model does not rank the obvious match first. Check the install / model download.")

    # ---------------------------------------------------------------- 2-4
    items = json.load(open(args.eval, encoding="utf-8"))
    chats = sorted({i["chat"] for i in items})
    results = {k: [] for k in "ABCD"}
    failures = []
    for chat in chats:
        path = os.path.join(args.chats_dir, f"{chat}.txt")
        with redirect_stdout(io.StringIO()):
            qa = ConversationQA.from_file(path, llm=object(), encoder=enc, embed_text="clean",
                                          chunker=args.chunker)
        chunks = qa.chunks
        E = qa.store.embeddings
        E_attr = enc.encode([c.attributed_text for c in chunks])
        msg_emb = enc.encode([m.text for m in qa.messages])
        msg_to_chunk = {mid: i for i, c in enumerate(chunks) for mid in c.message_ids}
        tf = TfidfVectorizer(sublinear_tf=True).fit([c.raw_text for c in chunks])
        T = tf.transform([c.raw_text for c in chunks])

        out("\n" + "=" * 70 + f"\n2. INDEX SANITY: {chat}\n" + "=" * 70)
        sz = [c.metadata["message_count"] for c in chunks]
        sample = E[np.random.default_rng(0).choice(len(E), min(200, len(E)), replace=False)]
        pair = sample @ sample.T
        mean_pair = (pair.sum() - len(sample)) / (len(sample) ** 2 - len(sample))
        out(f"{len(qa.messages)} messages -> {len(chunks)} chunks (median {int(np.median(sz))} msgs) | "
            f"mean cosine between random chunks: {mean_pair:.3f}")
        if mean_pair > 0.9:
            out("  -> PROBLEM: chunk embeddings are nearly identical (broken embeddings)")

        chat_items = [i for i in items if i["chat"] == chat]
        top1 = []
        for it in chat_items:
            gold = locate_gold(qa.messages, it)
            if not gold:
                continue
            qv = enc.encode([it["question"]])[0]
            order_A = np.argsort(-(E @ qv))
            order_D = np.argsort(-(E_attr @ qv))
            order_B = np.argsort(-(T @ tf.transform([it["question"]]).T).toarray().ravel())
            seen, order_C = set(), []
            for mi in np.argsort(-(msg_emb @ qv)):
                ci = msg_to_chunk.get(qa.messages[mi].message_id)
                if ci is not None and ci not in seen:
                    seen.add(ci)
                    order_C.append(ci)
                if len(order_C) >= 50:
                    break
            ranks = {k: chunk_rank(o, chunks, gold) for k, o in
                     zip("ABCD", [order_A, order_B, order_C, order_D])}
            for k in "ABCD":
                results[k].append((it["id"], it.get("question_type"), ranks[k]))
            top1.append(int(order_A[0]))
            if not ranks["A"] or ranks["A"] > K:
                gi = next(i for i, c in enumerate(chunks) if gold & set(c.message_ids))
                gold_msg = next(m for m in qa.messages if m.message_id in gold)
                failures.append((it, ranks, chunks[int(order_A[0])], float((E @ qv)[order_A[0]]),
                                 chunks[gi], float(E[gi] @ qv), gold_msg,
                                 chunks[gi].message_ids.index(gold_msg.message_id) + 1))
        if top1:
            out(f"distinct top-1 chunks across {len(top1)} questions: {len(set(top1))}"
                + ("  -> PROBLEM: questions all get the same chunk" if len(set(top1)) <= max(1, len(top1) // 3) else ""))

    out("\n" + "=" * 70 + "\n3. RETRIEVER COMPARISON (answerable questions, hit@4 / hit@1)\n" + "=" * 70)
    names = {"A": "A pipeline (dense, clean chunk)", "B": "B keyword TF-IDF on chunks",
             "C": "C dense on single messages", "D": "D dense, chunk with timestamps"}
    for k in "ABCD":
        r = results[k]
        h4 = sum(1 for _, _, x in r if x and x <= K) / max(len(r), 1)
        h1 = sum(1 for _, _, x in r if x == 1) / max(len(r), 1)
        out(f"  {names[k]:<34} hit@4={h4:.2f}  hit@1={h1:.2f}")
    out("\n  per question (gold chunk rank; - = beyond 50):")
    out(f"  {'id':<5}{'type':<17}{'A':>5}{'B':>5}{'C':>5}{'D':>5}")
    for i in range(len(results["A"])):
        qid, qt = results["A"][i][0], results["A"][i][1]
        cells = "".join(f"{(results[k][i][2] or '-'):>5}" for k in "ABCD")
        out(f"  {qid:<5}{(qt or '')[:16]:<17}{cells}")

    out("\n" + "=" * 70 + "\n4. EXAMPLE FAILURES (pipeline A)\n" + "=" * 70)
    for it, ranks, top, top_s, gold_c, gold_s, gold_msg, pos in failures[:8]:
        out(f"\nQ [{it['id']}] {it['question']}")
        out(f"  ranks A/B/C/D: {ranks['A']}/{ranks['B']}/{ranks['C']}/{ranks['D']}")
        out(f"  retrieved #1 (score {top_s:.3f}): {top.raw_text[:250]!r}")
        out(f"  gold chunk   (score {gold_s:.3f}, {gold_c.metadata['message_count']} msgs, "
            f"gold is message {pos}): {gold_c.raw_text[:200]!r}")
        out(f"  gold message: {gold_msg.sender}: {gold_msg.text[:150]!r}")

    os.makedirs("eval_results", exist_ok=True)
    name = os.path.splitext(os.path.basename(args.eval))[0]
    p = f"eval_results/diagnosis_{name}_{args.embedder}_{args.chunker}.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"\nSaved {p}")


if __name__ == "__main__":
    main()
