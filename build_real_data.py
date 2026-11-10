"""
Build Real Data — one-time setup script
-------------------------------------------
Runs the repo's OWN parser and chunker modules (src/parser, src/chunking)
on EVERY .txt chat export found in data/raw_real/, and saves the output
into data/processed/ in exactly the format the evaluation scripts expect.

To add a new chat to the study: just drop its exported .txt file into
data/raw_real/ and re-run this script. No code changes needed — the chat
name used everywhere downstream (evaluation_set.json's "chat" field,
output filenames, etc.) is just the filename without ".txt".

Usage (run from the project root):
    python build_real_data.py
"""

import json
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parser.parse_whatsapp import parse_whatsapp_chat
from src.chunking.chunk_messages import naive_time_gap_chunking, context_aware_chunking

RAW_DIR = "data/raw_real"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

txt_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.txt")))

if not txt_files:
    print(f"No .txt files found in {RAW_DIR}/ — drop your WhatsApp export(s) there and re-run.")
    sys.exit(0)

print(f"Found {len(txt_files)} chat file(s) in {RAW_DIR}/: "
      f"{[os.path.basename(f) for f in txt_files]}\n")

for filepath in txt_files:
    chat_name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"=== {chat_name} ===")

    records = parse_whatsapp_chat(filepath, filter_system=True)
    print(f"  Parsed {len(records)} real, cleaned messages")

    if not records:
        print(f"  [!] No messages parsed from {filepath} — check the export format. Skipping.\n")
        continue

    parsed_out = f"data/processed/{chat_name}_parsed.json"
    with open(parsed_out, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2, ensure_ascii=False)
    print(f"  Saved: {parsed_out}")

    naive_chunks = naive_time_gap_chunking(records)
    naive_out = f"data/processed/{chat_name}_parsed_chunks.json"
    with open(naive_out, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in naive_chunks], f, indent=2, ensure_ascii=False)
    print(f"  Saved: {naive_out}  ({len(naive_chunks)} naive chunks)")

    ctx_chunks = context_aware_chunking(records)
    ctx_out = f"data/processed/{chat_name}_context_chunks.json"
    with open(ctx_out, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in ctx_chunks], f, indent=2, ensure_ascii=False)
    print(f"  Saved: {ctx_out}  ({len(ctx_chunks)} context-aware chunks)")
    print()

print(f"Done. Processed {len(txt_files)} chat(s). Real data is now in data/processed/,")
print("ready for the evaluation scripts.")

