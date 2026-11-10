"""
Test ConversationQA pipeline on user's Dinooooo_Amrita chat export.
"""

from src.pipeline import ConversationQA

def main():
    chat_file = "data/raw_user/chat.txt"
    print(f"Loading and indexing {chat_file}...")
    qa = ConversationQA.from_file(chat_file, embedder="minilm", retrieval="hybrid")
    st = qa.stats()
    print(f"\n[+] Successfully parsed and indexed:")
    print(f"    Messages: {st['messages']}")
    print(f"    Chunks: {st['chunks']} (using {st['chunker']} chunker)")
    print(f"    Participants: {', '.join(st['participants'])}")
    print(f"    Timespan: {st['start']} to {st['end']}")
    print(f"    Retrieval mode: {st['retrieval']}")

    queries = [
        ("Why did we need to reschedule the mid review meeting?", "reschedule"),
        ("What was the 1300 rupees expense for?", "1300"),
        ("What is our latest task for the project?", "project"),
        ("Which room and hostel was mentioned for stay?", "room"),
        ("What evaluation files did Dino send for git?", "files"),
        ("What time were we supposed to come to college on 11 September?", "college"),
    ]

    print("\n" + "=" * 70)
    print(" EXECUTING RETRIEVAL TESTS ON USER CHAT")
    print("=" * 70)

    for q, tag in queries:
        hits = qa.retrieve(q)
        print(f"\n>>> QUESTION: \"{q}\"")
        if not hits:
            print("    [!] No chunks retrieved.")
            continue
        
        top = hits[0]
        c = top["chunk"]
        print(f"    Top Hit: [{c.chunk_id}] (Score: {top['final_score']:.4f}, BM25: {top.get('bm25', 0):.2f}, BestMsg: {top.get('best_message_similarity', 0):.2f})")
        print(f"    Interval: {c.start_timestamp} -> {c.end_timestamp}")
        print(f"    Snippet:")
        for line in c.attributed_text.strip().split("\n")[:4]:
            print(f"      {line[:100]}")
        if len(c.attributed_text.strip().split("\n")) > 4:
            print(f"      ... [{len(c.message_ids)} total turns in chunk]")

if __name__ == "__main__":
    main()
