# Running and comparing MiniLM vs bge-m3

Both embedding models run through the same pipeline (semantic chunking → time-aware retrieval → Ollama answer). Only the embedding model changes, so any difference in the results comes from the model.

| | MiniLM (`all-MiniLM-L6-v2`) | bge-m3 |
|---|---|---|
| Runs via | sentence-transformers (Python) | Ollama |
| Reads up to | 256 tokens per chunk | 8,192 tokens per chunk |
| Languages | English | Multilingual (incl. Telugu) |
| Size / speed | 22M params, seconds | ~570M params, minutes for the first run, then cached |

## 1. One-time setup

```bash
pip install -r requirements.txt          # includes sentence-transformers for MiniLM

ollama pull bge-m3                        # embedding model
ollama pull llama3.2:3b                   # answer model (or qwen2.5:3b / qwen2.5:1.5b)
ollama list                               # both should be listed
```

If bge-m3 fails with a 404 on `/api/embed`, update Ollama to a recent version.

Put the chat exports in `data/raw_real/` (this folder is git-ignored):

```
data/raw_real/diddy.txt
data/raw_real/nikhil.txt
data/raw_real/yaoi.txt
```

## 2. Retrieval only (no LLM), both models, both question sets

Run these from the project root. Retrieval-only runs are fast, need no answer model, and are the main comparison.

```bash
# original 18-question set
python scripts/run_qa_eval.py --embedder minilm --no-llm
python scripts/run_qa_eval.py --embedder bge-m3 --no-llm

# 27 new questions (includes 6 reworded questions with no shared keywords)
python scripts/run_qa_eval.py --embedder minilm --no-llm --eval data/benchmarks/new_queries.json
python scripts/run_qa_eval.py --embedder bge-m3 --no-llm --eval data/benchmarks/new_queries.json
```

The first bge-m3 run embeds every message and chunk through Ollama, which takes a few minutes. Embeddings are cached in `.cache/embeddings/`, so later runs take seconds. Delete that folder to force re-embedding.

## 3. Compare

```bash
python scripts/compare_evals.py --glob "eval_results/qa_eval_evaluation_set_*.json"
python scripts/compare_evals.py --glob "eval_results/qa_eval_new_queries_*.json"
```

Each command prints a summary table and a per-question table, and saves them as `eval_results/comparison_<set>.md`.

## 4. With answers from Ollama (and the RAGAS dataset)

```bash
python scripts/run_qa_eval.py --embedder minilm --eval data/benchmarks/new_queries.json
python scripts/run_qa_eval.py --embedder bge-m3 --eval data/benchmarks/new_queries.json
```

Each run also writes `eval_results/ragas_<set>_<model>_clean.jsonl`. Read a few answers in the `response` field of `qa_eval_*.json` to check they are correct and cite their sources.

## 5. Try it interactively

```bash
python scripts/ask_chat.py data/raw_real/nikhil.txt --embedder minilm --show-sources
python scripts/ask_chat.py data/raw_real/nikhil.txt --embedder bge-m3 --show-sources
```

For the API: `EMBEDDER=bge-m3 uvicorn src.api.server:app --reload`

## Optional: timestamps vs no timestamps

By default both models embed "clean" chunk text (`Sender: message`, no timestamps). The answer model always sees the full timestamped text. To measure how much removing timestamps helps, add `--embed-text attributed`:

```bash
python scripts/run_qa_eval.py --embedder minilm --no-llm --embed-text attributed --eval data/benchmarks/new_queries.json
```

This is expected to matter most for MiniLM, whose 256-token limit timestamps fill quickly.

## Reading the results

| Metric | Meaning |
|---|---|
| `hit@1` / `hit@4` | Share of answerable questions where the chunk with the gold message is ranked 1st / in the top 4, after time-aware reranking. |
| `mrr@k` | Mean reciprocal rank: 1 for rank 1, 0.5 for rank 2, …, 0 if missed. |
| `dense_only_hit@4` | Same as hit@4 but using similarity order only, without the time-aware reranker. If this is higher than `hit@4`, the reranker is hurting. |
| `mean_top_score_answerable` vs `mean_top_score_negative` | Best similarity score for questions that have an answer vs. negative controls. A bigger gap means the model can tell "not in this chat" apart better. |
| `answered_when_answerable` / `abstained_when_negative` | With the LLM only: how often it answered real questions, and how often it correctly said "not found". |
| `runtime_seconds` | Total time for the run. |

In the per-question table, a number is the gold chunk's rank in the top 4. `miss (dense N)` means it fell outside the top 4 and similarity alone ranked it N. `≠` marks questions where the two runs disagree.

What to look for:

1. **The `paraphrased` questions in `new_queries`.** These share almost no words with their answers, so they show whether a model understands meaning. They are the main test between the two models.
2. **The `code-mixed` questions** (q16, n15). bge-m3 is multilingual and should do better here.
3. **`dense_only_hit@4` vs `hit@4`** for each model, which shows whether the time-aware reranker needs changing.
4. **`gold_semantic_score` vs `top_semantic_score`** in `qa_eval_*.json` for a missed question. It shows how close the correct chunk came.

## Calibrating "not found"

Before calling the LLM, the pipeline answers "I couldn't find anything about that in this chat." if the best similarity score is below `min_relevance`. Scores run on different scales per model, so the defaults in `src/indexing/embedders.py` (0.20 for MiniLM, 0.40 for bge-m3) are starting guesses. After a retrieval run, set it between `mean_top_score_negative` and `mean_top_score_answerable`, closer to the negative side. If real questions are being answered "not found", lower it.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot reach Ollama` | Start it with `ollama serve`, or open the Ollama app. |
| `ollama pull bge-m3` needed | The model isn't installed; run that command. |
| `MiniLM needs sentence-transformers` | `pip install sentence-transformers` |
| bge-m3 very slow | Normal on CPU for the first run only. Make sure nothing else is using Ollama at the same time. |
| Results didn't change after editing a chat | Delete `.cache/embeddings/`. |

## All experiments in one command (for the review)

```bash
python scripts/run_all_experiments.py                  # MiniLM, all configurations, evaluation_set_v2
python scripts/run_all_experiments.py --with-bge-m3    # adds bge-m3 rows (needs: ollama pull bge-m3)
```

This writes `eval_results/experiments_summary.md`, a table that goes from the baselines through the old rule-based system to the final system:

| Option | Values |
|---|---|
| `--chunker` | `semantic` (topic-shift detection, default), `naive` (30-min gap), `regex` (old rules), `fixed` (8 messages) |
| `--retrieval` | `hybrid` (BM25 + chunk embedding + best-message embedding, fused; time reranking only for "latest/current" questions; default), `dense` (original: chunk embedding + time reranking always) |

## Evaluation set v2

`data/benchmarks/evaluation_set_v2.json` is now the default question set: 66 questions across the three chats (63 answerable, 10 negative controls, counting q12 among the negatives).

| Type | Count |
|---|---|
| factual-recall | 30 |
| paraphrased (few shared words with the answer) | 10 |
| negative-control | 10 |
| time-based | 6 |
| who-said-what | 6 |
| code-mixed (Telugu-English) | 3 |
| latest-update | 1 |

It contains the 15 usable original questions, 26 of the earlier new queries and 25 new ones. Questions whose only evidence is a URL or a raw code paste were removed: original q2, q3 and q8, and new-query n16. No question was removed because of how any system scored on it. `evaluation_set_v2_changes.json` records this.
