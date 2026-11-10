"""
Ask questions about a WhatsApp export from the terminal.

    python scripts/ask_chat.py data/raw/sample_chat.txt
    python scripts/ask_chat.py my_chat.txt --model qwen2.5:3b --show-sources
    python scripts/ask_chat.py my_chat.txt --embedder bge-m3 --show-sources
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generation.ollama_client import OllamaClient
from src.pipeline import ConversationQA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chat", help="WhatsApp .txt export")
    ap.add_argument("--model", default="llama3.2:3b", help="Ollama model name")
    ap.add_argument("--embedder", default="minilm", choices=["minilm", "bge-m3"])
    ap.add_argument("--show-sources", action="store_true")
    args = ap.parse_args()

    llm = OllamaClient(model=args.model)
    if not llm.is_available():
        print(f"[!] Ollama model '{args.model}' not available. Run: ollama pull {args.model}")

    print("[*] Parsing, chunking and indexing...")
    qa = ConversationQA.from_file(args.chat, llm=llm, embedder=args.embedder)
    st = qa.stats()
    print(f"[+] {st['messages']} messages, {st['chunks']} chunks ({st['embedder']}), "
          f"participants: {', '.join(st['participants'])}")
    print("Ask a question (empty line to quit).\n")

    history = []
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        res = qa.ask(q, history=history)
        if res.get("error"):
            print(f"[!] {res['error']}\n")
            continue
        print(f"\nAnswer: {res['answer']}\n")
        if args.show_sources:
            for s in res["sources"]:
                mark = "*" if s["cited"] else " "
                print(f" {mark}[{s['ref']}] {s['chunk_id']}  score={s['final_score']}")
                for m in s["messages"]:
                    print(f"      [{m['date']} {m['time']}] {m['sender']}: {m['text']}")
            print()
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": res["answer"]}]


if __name__ == "__main__":
    main()
