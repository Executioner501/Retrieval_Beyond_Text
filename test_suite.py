"""
Comprehensive Automated Test Suite for Context-Aware Conversational RAG Framework.
Tests:
1. WhatsApp Export Parser (MessageRecord integrity, multi-line merging, system filtering)
2. Chunking Engines (Naive vs. Context-Aware vs. Semantic Topic TextTiling)
3. Embedding & Vector Store (all-MiniLM-L6-v2 similarity computation)
4. Time-Aware Re-Ranking (Intent detection & exponential decay behavior)
5. Hybrid Retrieval (BM25 + Chunk Dense + Best Message Dense via RRF)
6. End-to-End ConversationQA Retrieval (Temporal decision reversal verification)
"""

import time
import sys
import os
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def print_header(title):
    print("\n" + "=" * 70)
    print(f" TEST STAGE: {title}")
    print("=" * 70)

def main():
    start_all = time.time()
    results = {}

    # ---------------------------------------------------------
    # Test 1: Parser Verification
    # ---------------------------------------------------------
    print_header("1. WhatsApp Parser & Normalizer")
    from src.parser.parse_whatsapp import parse_whatsapp_chat, MessageRecord
    
    sample_file = ROOT / "data" / "raw" / "sample_chat.txt"
    t0 = time.time()
    records = parse_whatsapp_chat(str(sample_file), filter_system=True)
    t_parse = time.time() - t0
    
    print(f"Parsed {len(records)} messages from sample_chat.txt in {t_parse:.3f}s")
    senders = sorted(list({r.sender for r in records}))
    print(f"Identified participants: {senders}")
    
    # Assertions
    assert len(records) > 10, f"Expected >10 messages, got {len(records)}"
    assert "System" not in senders, "System messages were not properly filtered"
    assert "Rahul" in senders and "Sai" in senders and "Dheeraj" in senders, "Missing participants"
    
    # Verify multi-line message handling
    multi_line_found = any("\n" in r.text for r in records)
    print(f"Multi-line continuation preserved: {multi_line_found}")
    assert multi_line_found, "Expected at least one merged multi-line message"
    results["1. Parser & Sanitization"] = "PASSED"

    # ---------------------------------------------------------
    # Test 2: Chunking Engines
    # ---------------------------------------------------------
    print_header("2. Chunking Strategies Comparison")
    from src.chunking.chunk_messages import naive_time_gap_chunking, context_aware_chunking
    from src.chunking.semantic_topic_chunking import semantic_topic_chunking
    
    naive_chunks = naive_time_gap_chunking(records, threshold_minutes=30)
    ctx_chunks = context_aware_chunking(records)
    sem_chunks = semantic_topic_chunking(records)
    
    print(f"Naive Time-Gap Chunks (30m): {len(naive_chunks)}")
    for c in naive_chunks:
        print(f"  [{c.chunk_id}] {c.start_timestamp} -> {c.end_timestamp} ({len(c.message_ids)} msgs, participants: {c.participants})")
    
    print(f"\nContext-Aware Chunks (Heuristic): {len(ctx_chunks)}")
    for c in ctx_chunks:
        print(f"  [{c.chunk_id}] {c.start_timestamp} -> {c.end_timestamp} ({len(c.message_ids)} msgs)")
        
    print(f"\nSemantic Topic Chunks (Hearst TextTiling + Otsu): {len(sem_chunks)}")
    for c in sem_chunks:
        print(f"  [{c.chunk_id}] {c.start_timestamp} -> {c.end_timestamp} ({len(c.message_ids)} msgs)")
        
    assert len(naive_chunks) >= 3, "Naive chunking should create multiple temporal chunks"
    assert len(sem_chunks) >= 3, "Semantic chunking should segment conversation topics"
    results["2. Chunking Engines"] = "PASSED"

    # ---------------------------------------------------------
    # Test 3: Embedding & Vector Store
    # ---------------------------------------------------------
    print_header("3. Local Vector Store & Dense Similarity")
    from src.indexing.vector_store import LocalVectorStore
    
    store = LocalVectorStore(model_name="sentence-transformers/all-MiniLM-L6-v2", text_field="attributed_text")
    t0 = time.time()
    store.add_chunks(sem_chunks)
    t_embed = time.time() - t0
    print(f"Embedded {len(sem_chunks)} chunks in {t_embed:.3f}s. Shape: {store.embeddings.shape}")
    
    query = "What framework are we using for the frontend?"
    hits = store.search(query, top_k=2)
    print(f"Dense search results for: '{query}'")
    for h in hits:
        print(f"  Score: {h['semantic_score']:.4f} | Chunk: {h['chunk'].chunk_id}")
    assert len(hits) == 2, "Expected 2 top hits"
    results["3. Vector Indexing"] = "PASSED"

    # ---------------------------------------------------------
    # Test 4: Time-Aware Intent & Re-ranking
    # ---------------------------------------------------------
    print_header("4. Time-Aware Intent Detection & Exponential Decay")
    from src.retrieval.ranker import TimeAwareReRanker, detect_temporal_intent
    
    test_queries = [
        ("What is our latest plan for the frontend?", "latest"),
        ("What was our initial decision back in January?", "historical"),
        ("Who was bringing the documents?", "neutral"),
    ]
    
    for q, expected_intent in test_queries:
        intent = detect_temporal_intent(q)
        print(f"  Query: '{q}' -> Intent: '{intent}' (Expected: '{expected_intent}')")
        assert intent == expected_intent, f"Intent mismatch for '{q}'"
        
    # Test reranking on the frontend decision
    # January discussion (React) vs August discussion (Flutter)
    reranker = TimeAwareReRanker(alpha=0.45, decay_lambda=1.2)
    candidates = store.search("What is our latest plan for the frontend?", top_k=len(sem_chunks))
    
    reranked_latest = reranker.rerank(candidates, query="What is our latest plan for the frontend?")
    top_chunk_latest = reranked_latest[0]["chunk"]
    print(f"\nTop retrieved chunk for 'latest plan': {top_chunk_latest.chunk_id} ({top_chunk_latest.start_timestamp})")
    print(f"Snippet:\n{top_chunk_latest.attributed_text[:200]}...")
    
    # Must retrieve August update (Flutter) rather than January (React)
    assert "Flutter" in top_chunk_latest.attributed_text, "Failed to prioritize latest decision over outdated one"
    results["4. Time-Aware Re-ranking"] = "PASSED"

    # ---------------------------------------------------------
    # Test 5: Hybrid Chunk Retriever (BM25 + Best Message + RRF)
    # ---------------------------------------------------------
    print_header("5. Hybrid Retrieval (BM25 + RRF)")
    from src.indexing.embedders import get_embedder
    from src.retrieval.hybrid import HybridChunkRetriever
    
    encoder = get_embedder("minilm")
    msg_emb = encoder.encode([m.text for m in records], convert_to_numpy=True, normalize_embeddings=True)
    chunk_of = {mid: ci for ci, c in enumerate(sem_chunks) for mid in c.message_ids}
    
    hybrid = HybridChunkRetriever(
        chunk_texts=[c.raw_text for c in sem_chunks],
        chunk_emb=store.embeddings,
        msg_emb=msg_emb,
        msg_chunk=[chunk_of[m.message_id] for m in records],
        msg_words=[m.word_count for m in records],
    )
    
    q_emb = encoder.encode(["Goa semester break trip tickets"], convert_to_numpy=True, normalize_embeddings=True)[0]
    hybrid_hits = hybrid.search("Goa semester break trip tickets", q_emb, top_n=3)
    print(f"Hybrid retrieval hits for 'Goa semester break trip tickets':")
    for h in hybrid_hits:
        c = sem_chunks[h["index"]]
        print(f"  Chunk: {c.chunk_id} | Fused: {h['fused_score']:.5f} | BM25: {h['bm25']:.3f} | Best Msg Sim: {h['best_message_similarity']:.3f}")
    assert "Goa" in sem_chunks[hybrid_hits[0]["index"]].attributed_text, "Hybrid retrieval failed to rank Goa trip top"
    results["5. Hybrid Retrieval & RRF"] = "PASSED"

    # ---------------------------------------------------------
    # Test 6: End-to-End Pipeline
    # ---------------------------------------------------------
    print_header("6. End-to-End ConversationQA Pipeline")
    from src.pipeline import ConversationQA
    
    qa = ConversationQA.from_file(str(sample_file), embedder="minilm", retrieval="hybrid")
    st = qa.stats()
    print(f"Pipeline Stats: {st}")
    
    test_q = "What is our latest plan for the frontend?"
    hits = qa.retrieve(test_q)
    print(f"\nRetrieved {len(hits)} evidence chunks for: '{test_q}'")
    for i, h in enumerate(hits, start=1):
        c = h["chunk"]
        print(f"  [{i}] {c.chunk_id} | Final: {h['final_score']:.4f} | Decay: {h.get('time_decay', 1.0):.4f} | Reranked: {h.get('time_rerank_applied', False)}")
        print(f"      Text: {c.attributed_text.strip().replace(chr(10), ' | ')[:120]}...")
    
    top_hit_text = hits[0]["chunk"].attributed_text
    assert "Flutter" in top_hit_text, "Top retrieved chunk must contain Flutter"
    assert "React" not in top_hit_text or "instead of React" in top_hit_text, "Should not return outdated React decision as latest"
    results["6. End-to-End Pipeline"] = "PASSED"

    # ---------------------------------------------------------
    # Summary Report
    # ---------------------------------------------------------
    total_time = time.time() - start_all
    print("\n" + "=" * 70)
    print(" ALL TESTS COMPLETED")
    print("=" * 70)
    for name, status in results.items():
        print(f"  {name:<40}: [ {status} ]")
    print(f"\nTotal test execution time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
