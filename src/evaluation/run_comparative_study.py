"""
Automated Comparative Study Runner.
Benchmarks:
1. Naive Baseline (Time-Gap 30m)
2. LangChain Standard RAG (RecursiveCharacterTextSplitter)
3. LlamaIndex Standard RAG (SentenceSplitter)
4. Proposed Context-Aware RAG (Attributed turns + Time-decay reranker)

Computes Hit@1, Hit@3, MRR, Speaker Attribution Accuracy, Temporal Validity,
and detailed failure mode categorizations.
"""

import json
import os
from typing import List, Dict, Any
from datetime import datetime

from ..parser.parse_whatsapp import parse_whatsapp_chat
from .framework_baselines import (
    NaiveBaselineRAG,
    LangChainRAG,
    LlamaIndexRAG,
    ContextAwareRAG
)


def evaluate_framework(retriever, queries: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
    query_results = []
    hit_count_1 = 0
    hit_count_3 = 0
    reciprocal_ranks = []
    correct_speaker_count = 0
    temporal_correct_count = 0
    time_sensitive_total = 0

    failure_counts = {
        "BAD_CHUNK_BOUNDARY": 0,
        "LOST_SPEAKER_CONTEXT": 0,
        "WRONG_TIME_WINDOW": 0,
        "IRRELEVANT_RETRIEVAL": 0,
        "NONE": 0
    }

    for q in queries:
        qid = q["id"]
        qtext = q["query"]
        expected_speakers = q.get("expected_speakers", [])
        target_kws = q.get("target_keywords", [])
        is_time_sensitive = q.get("is_time_sensitive", False)
        superseded_kws = q.get("superseded_keywords", [])

        if is_time_sensitive:
            time_sensitive_total += 1

        results = retriever.retrieve(qtext, top_k=top_k)
        retrieved_texts = [r["text"] for r in results]

        # 1. Evaluate Hit & MRR
        first_hit_rank = 0
        top1_hit = False
        top3_hit = False

        for rank, text in enumerate(retrieved_texts, start=1):
            has_kw = any(kw.lower() in text.lower() for kw in target_kws)
            if has_kw:
                if rank == 1:
                    top1_hit = True
                if rank <= 3:
                    top3_hit = True
                if first_hit_rank == 0:
                    first_hit_rank = rank
                break

        rr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0
        reciprocal_ranks.append(rr)
        if top1_hit:
            hit_count_1 += 1
        if top3_hit:
            hit_count_3 += 1

        # 2. Evaluate Speaker Attribution
        top_text = retrieved_texts[0] if retrieved_texts else ""
        speaker_found = False
        if expected_speakers:
            speaker_found = any(sp.lower() in top_text.lower() for sp in expected_speakers)
            if speaker_found and top3_hit:
                correct_speaker_count += 1

        # 3. Evaluate Temporal Validity
        temporal_valid = False
        superseded_found = False
        if is_time_sensitive:
            if superseded_kws:
                superseded_found = any(skw.lower() in top_text.lower() for skw in superseded_kws)
            target_found = any(kw.lower() in top_text.lower() for kw in target_kws)

            if target_found and not superseded_found:
                temporal_valid = True
                temporal_correct_count += 1

        # 4. Failure Mode Classification
        failure = "NONE"
        if not top3_hit:
            if is_time_sensitive and superseded_found:
                failure = "WRONG_TIME_WINDOW"
            else:
                failure = "BAD_CHUNK_BOUNDARY"
        elif not speaker_found and expected_speakers:
            failure = "LOST_SPEAKER_CONTEXT"
        elif is_time_sensitive and superseded_found and not temporal_valid:
            failure = "WRONG_TIME_WINDOW"

        failure_counts[failure] += 1

        query_results.append({
            "query_id": qid,
            "query": qtext,
            "category": q.get("category"),
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "mrr": rr,
            "speaker_attributed": speaker_found,
            "temporal_valid": temporal_valid if is_time_sensitive else True,
            "failure_mode": failure,
            "top_retrieved_snippet": top_text[:200] + "..." if len(top_text) > 200 else top_text
        })

    num_queries = len(queries)
    return {
        "framework": retriever.name,
        "metrics": {
            "hit_rate_at_1": round(hit_count_1 / num_queries, 4),
            "hit_rate_at_3": round(hit_count_3 / num_queries, 4),
            "mrr": round(sum(reciprocal_ranks) / num_queries, 4),
            "attribution_accuracy": round(correct_speaker_count / num_queries, 4),
            "temporal_validity_rate": round(temporal_correct_count / max(1, time_sensitive_total), 4)
        },
        "failure_counts": failure_counts,
        "query_results": query_results
    }


def main():
    chat_file = "data/raw/sample_chat.txt"
    queries_file = "data/benchmarks/benchmark_queries_25.json"

    print(f"[*] Loading chat logs from: {chat_file}")
    messages = parse_whatsapp_chat(chat_file, filter_system=True)
    with open(chat_file, "r", encoding="utf-8") as f:
        raw_chat_text = f.read()

    print(f"[*] Loading benchmark queries from: {queries_file}")
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"[+] Total parsed user messages: {len(messages)}")
    print(f"[+] Total benchmark test queries: {len(queries)}")

    print("\n" + "="*70)
    print("INITIALIZING COMPARATIVE FRAMEWORKS")
    print("="*70)

    print("1. Initializing Naive Time-Gap Baseline...")
    naive_rag = NaiveBaselineRAG(messages, threshold_minutes=30)

    print("2. Initializing LangChain Standard RAG (RecursiveTextSplitter)...")
    langchain_rag = LangChainRAG(raw_chat_text, chunk_size=350, chunk_overlap=70)

    print("3. Initializing LlamaIndex Standard RAG (SentenceSplitter)...")
    llamaindex_rag = LlamaIndexRAG(raw_chat_text, chunk_size=150, chunk_overlap=25)

    print("4. Initializing Proposed Context-Aware RAG (Attributed + Time Decay)...")
    context_rag = ContextAwareRAG(messages, alpha=0.45, decay_lambda=1.2)

    frameworks = [naive_rag, langchain_rag, llamaindex_rag, context_rag]
    all_evaluations = {}

    print("\n" + "="*70)
    print("RUNNING EMPIRICAL BENCHMARK EXPERIMENTS")
    print("="*70)

    for fw in frameworks:
        print(f"-> Benchmarking {fw.name} on {len(queries)} queries...")
        result = evaluate_framework(fw, queries, top_k=3)
        all_evaluations[fw.name] = result

    # Save to eval_results
    os.makedirs("eval_results", exist_ok=True)
    output_json = "eval_results/comparative_study_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_evaluations, f, indent=2)

    print(f"\n[+] Raw results saved to: {output_json}")

    # Print Formatted Markdown Table
    print("\n" + "="*70)
    print("SUMMARY COMPARISON TABLE")
    print("="*70)
    print(f"{'Framework':<32} | {'Hit@1':<8} | {'Hit@3':<8} | {'MRR':<8} | {'Attrib Acc':<11} | {'Temp Acc':<9}")
    print("-" * 85)
    for name, res in all_evaluations.items():
        m = res["metrics"]
        print(f"{name:<32} | {m['hit_rate_at_1']*100:>5.1f}%  | {m['hit_rate_at_3']*100:>5.1f}%  | {m['mrr']:>7.3f} | {m['attribution_accuracy']*100:>8.1f}%   | {m['temporal_validity_rate']*100:>6.1f}%")

    print("\n" + "="*70)
    print("FAILURE MODE DISTRIBUTION COUNTS")
    print("="*70)
    print(f"{'Framework':<32} | {'Bad Boundary':<13} | {'Lost Speaker':<13} | {'Wrong Time':<11} | {'Clean (None)':<12}")
    print("-" * 88)
    for name, res in all_evaluations.items():
        fc = res["failure_counts"]
        print(f"{name:<32} | {fc['BAD_CHUNK_BOUNDARY']:>12} | {fc['LOST_SPEAKER_CONTEXT']:>12} | {fc['WRONG_TIME_WINDOW']:>10} | {fc['NONE']:>12}")


if __name__ == "__main__":
    main()
