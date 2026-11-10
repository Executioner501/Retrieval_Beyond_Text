# Retrieval Beyond Text: Context-Aware Retrieval Framework for Multi-Party, Time-Dependent Conversations

[![Project Phase](https://img.shields.io/badge/Project-Phase%20II-blue.svg)](https://github.com/)
[![Academic Year](https://img.shields.io/badge/Academic%20Year-2026--2027-green.svg)](https://github.com/)
[![Department](https://img.shields.io/badge/Amrita%20School%20of%20Computing-CSE-orange.svg)](https://github.com/)

**Course**: 23CSE498 – Project Phase II  
**Institution**: Amrita School of Computing, Amritapuri Campus, Amrita Vishwa Vidyapeetham  
**Project Guide**: Ms. Bindhya Bhadran  
**Team Members (Team B10)**:
- K. Sai Dinesh (AM.SC.U4CSE23130)
- Satya Sri Dheeraj Motupalli (AM.SC.U4CSE23150)
- Y. Sai Nikhil (AM.SC.U4CSE23171)

---

## 📖 1. Project Overview

Conversational data generated in messaging applications like WhatsApp is fundamentally different from traditional structured documents:
- **Fragmented utterances** ("Yeah", "Okay", "Tomorrow") lacking independent semantic context
- **Multi-party interleaved dialogues** with multiple speakers and roles
- **Abrupt topic switches** within short timeframes
- **Time-varying decisions** where newer messages supersede older commitments
- **Privacy constraints** demanding local/on-device query processing rather than cloud ingestion

Generic RAG systems (built on LangChain, LlamaIndex, Haystack) and conversational memory systems (MemGPT, SECOM) fail on multi-party chats because they were designed either for structured prose or 1-on-1 human-to-AI dialogues.

This project designs, develops, and benchmarks a **Context-Aware Conversational Retrieval Pipeline** that introduces:
1. **Speaker-Aware Context Preservation**: Preserving message author attribution within each chunk.
2. **Topic-Aware Semantic Chunking**: Detecting topic shift boundaries to prevent unrelated topic bleed.
3. **Time-Weighted Re-Ranking**: Balancing semantic relevance with temporal freshness via exponential time decay normalized by total chat span $T_{\text{chat}} = t_{\text{end}} - t_{\text{start}}$:
   $$\text{Score}(c) = \alpha \cdot \text{sim}(q, c) + (1 - \alpha) \cdot \exp\left(-\lambda \cdot \frac{\Delta t(c)}{T_{\text{chat}}}\right)$$
4. **Diagnostic Failure Benchmarking**: Evaluating precision, recall, and categorizing failure modes (Bad Chunk Boundary, Lost Speaker Context, Wrong Time Window, Irrelevant Retrieval) against naive baselines and general RAG frameworks.

> **Future Work**: Conversational thread/reply reconstruction using temporal and semantic proximity is planned as future work to link asynchronous replies across interleaved multi-party dialogues.

---

## 🏛️ 2. System Architecture

```text
Raw WhatsApp Chat Export (.txt)
             │
             ▼
   [Chat Parsing Module]
   • Parse timestamps, senders, text
   • Merge multi-line continuation messages
   • Filter system notifications & media placeholders
             │
             ▼
   [Chunking Module]
   ├── Baseline: Naive Time-Gap Chunking (30-min threshold)
   └── Proposed: Context-Aware Chunking (Speaker + Topic Boundaries)
             │
             ▼
   [Embedding & Vector Store]
   • Dense embeddings (Sentence-Transformers / Multilingual)
   • Vector indexing (Local In-Memory NumPy Vector Store)
             │
             ▼
   [Retrieval & Time-Weighted Re-ranking]
   • Semantic vector similarity
   • Dynamic time-decay re-ranking (alpha, lambda)
             │
             ▼
   [LLM Generation & Grounding]
   • Context-grounded synthesis with speaker attribution & timestamps
             │
             ▼
   [Evaluation & Diagnostic Benchmark]
   • Benchmark query sets with ground truth
   • Comparative analysis vs. Baselines & Frameworks
   • Failure Mode Classification
```

---

## 📂 3. Repository Structure

```tree
├── docs/                     # System design specifications and architecture documentation
│   ├── SYSTEM_DESIGN.md      # Detailed technical architecture, schemas, and algorithms
│   └── UI_UX_SPECIFICATION.md# UI/UX design specs, layouts, interactive prototypes
├── data/
│   ├── raw/                  # Raw WhatsApp export files (.txt) [git-ignored]
│   │   └── sample_chat.txt   # Synthetic benchmark chat export for testing
│   ├── processed/            # Structured parsed message JSONs [git-ignored]
│   └── benchmarks/           # Hand-labeled query sets & ground-truth answers
├── src/                      # Source code modules
│   ├── parser/               # WhatsApp export parsing logic
│   ├── chunking/             # Naive & Context-Aware chunking implementations
│   ├── indexing/             # Vector store & embedding managers
│   ├── retrieval/            # Semantic search & time-decay re-ranker
│   ├── generation/           # LLM prompting & grounded response generator
│   └── evaluation/           # Evaluation benchmark runner & failure diagnostic
├── ui/                       # Interactive Design Prototype & Visualizer Dashboard
├── eval_results/             # Evaluation logs and comparative benchmark reports
├── Detailed_Project_Explanation.md # Comprehensive conceptual project guide
├── Design_Proposal.pdf           # Department submission: System Design Proposal
├── proposal-1.pdf                 # Department submission: Initial Project Idea
├── requirements.txt          # Python dependencies
└── .gitignore                # Privacy & environment security exclusions
```

---

## 🚀 4. Roadmap & Milestone Progress Log

| Stage | Milestone / Task | Status | Script / Artifact |
| :--- | :--- | :---: | :--- |
| **Stage 1** | Chat Parsing & Multi-line Sanitization | ✅ Completed | `src/parser/parse_whatsapp.py` |
| **Stage 2** | Baseline vs. Context-Aware Chunking Engines | ✅ Completed | `src/chunking/chunk_messages.py` |
| **Stage 3** | Embedding & Vector Storage (Local-First) | ✅ Completed | `src/indexing/vector_store.py` |
| **Stage 4** | Benchmark Query Sets (25-query & 18-query suites) | ✅ Completed | `data/benchmarks/` |
| **Stage 5** | Baseline Retrieval Test (Naive Time-Gap 30m) | ✅ Completed | `scripts/run_naive_baseline.py` |
| **Stage 6** | Framework Benchmarking (LangChain & LlamaIndex) | ✅ Completed | `scripts/run_langchain.py`, `scripts/run_llamaindex.py` |
| **Stage 7** | Adapted Retrieval Design (Attributed + Time-Decay) | ✅ Completed | `scripts/run_context_aware.py`, `src/retrieval/ranker.py` |
| **Stage 8** | Empirical Comparative Study & Failure Taxonomy | ✅ Completed | `src/evaluation/run_comparative_study.py`, `scripts/compare_results.py` |
| **Stage 9** | Interactive Prototype Visualizer Studio | ✅ Completed | `ui/index.html` (Vanilla HTML/CSS/JS) |

---

## 📊 5. Empirical Benchmark Results

<!-- RESULTS:START -->

### 5.1 Real-Chat Evaluation (Held-Out Test Split)

Evaluated on **Held-Out Test Split (33 Questions)** using `all-MiniLM-L6-v2` with hybrid retrieval and semantic chunking:

| Metric | Score | 95% Bootstrap CI | Notes |
| :--- | :---: | :---: | :--- |
| **hit@1** | 0.607 | [0.429, 0.786] | Top-1 retrieval precision |
| **hit@4** | 0.857 | [0.714, 0.964] | Primary recall target |
| **MRR** | 0.705 | [0.571, 0.839] | Mean Reciprocal Rank |
| **Dense-Only hit@4** | 0.750 | - | Pure dense retrieval without BM25 |
| **Avg Retrieved Context** | 28.4 msgs / 2512.2 chars | - | Mean context volume per query |

**Category Breakdown (hit@4):**

| Question Type | Subset Size | hit@4 |
| :--- | :---: | :---: |
| code-mixed | n=1 | 1.000 |
| factual-recall | n=15 | 0.933 |
| latest-update | n=1 | 1.000 |
| paraphrased | n=5 | 0.600 |
| time-based | n=3 | 0.667 |
| who-said-what | n=3 | 1.000 |

#### Multilingual vs. Monolingual Embedder Comparison


| Embedder | hit@1 | hit@4 | MRR | hit@4 (paraphrased) | hit@4 (code-mixed) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **minilm** | 0.607 | 0.839 | 0.701 | 0.500 | 0.667 |
| **me5** | 0.589 | 0.750 | 0.652 | 0.500 | 0.333 |

### 5.2 Ablation Study (Chunking Strategy × Retrieval Pipeline)


| Chunker | Retrieval | hit@1 (95% CI) | hit@4 (95% CI) | MRR (95% CI) | Avg Msgs | Avg Chars |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| naive | dense | 0.393 [0.214, 0.571] | 0.607 [0.429, 0.786] | 0.494 [0.333, 0.655] | 21.7 | 2145 |
| naive | hybrid | 0.643 [0.464, 0.821] | 0.857 [0.714, 0.964] | 0.732 [0.583, 0.869] | 58.3 | 5874 |
| fixed | dense | 0.393 [0.214, 0.571] | 0.750 [0.571, 0.893] | 0.539 [0.381, 0.688] | 31.7 | 4811 |
| fixed | hybrid | 0.500 [0.321, 0.679] | 0.786 [0.642, 0.929] | 0.628 [0.467, 0.768] | 32.0 | 4977 |
| regex | dense | 0.464 [0.250, 0.643] | 0.643 [0.464, 0.821] | 0.536 [0.357, 0.702] | 15.7 | 1990 |
| regex | hybrid | 0.643 [0.464, 0.821] | 0.821 [0.679, 0.964] | 0.720 [0.554, 0.857] | 28.5 | 4023 |
| semantic | dense | 0.536 [0.321, 0.714] | 0.679 [0.500, 0.857] | 0.601 [0.417, 0.768] | 23.0 | 2756 |
| semantic | hybrid | 0.607 [0.429, 0.786] | 0.821 [0.679, 0.964] | 0.696 [0.536, 0.848] | 29.4 | 4510 |

> **Insight**: While `naive + hybrid` achieves high hit@4 by returning giant multi-topic text blobs (58.3 messages / 5,874 chars), `semantic + hybrid` reaches competitive accuracy (0.821 hit@4) with half the context size (29.4 messages / 4,510 chars). Under pure dense retrieval, `semantic` significantly outperforms `naive` (0.679 vs 0.607 hit@4).

### 5.3 Synthetic 25-Query Suite (Sanity Check)

> **Sanity check on a small synthetic chat, not a headline result.**

| Framework | Hit Rate @ 1 | Hit Rate @ 3 | MRR | Speaker Attrib Acc | Temporal Validity Rate | Clean (No Failure) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Time-Gap Baseline** | **92.0%** | **100.0%** | **0.960** | **100.0%** | 66.7% | 23 / 25 |
| **LangChain Standard RAG (document-oriented defaults)** | 84.0% | 96.0% | 0.887 | 92.0% | 33.3% | 21 / 25 |
| **LlamaIndex Standard RAG (document-oriented defaults)** | 84.0% | **100.0%** | 0.913 | **100.0%** | 66.7% | 23 / 25 |
| **Proposed Context-Aware RAG** | 84.0% | 96.0% | 0.893 | 96.0% | **100.0%** | **24 / 25** |

| Framework | Bad Boundary | Lost Speaker Context | Wrong Time Window | Total Failures |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Time-Gap Baseline** | 0 | 0 | 2 | 2 |
| **LangChain Standard RAG (document-oriented defaults)** | 1 | 1 | 2 | 4 |
| **LlamaIndex Standard RAG (document-oriented defaults)** | 0 | 0 | 2 | 2 |
| **Proposed Context-Aware RAG** | 1 | 0 | **0** | **1** |

<!-- RESULTS:END -->

---

## 📚 6. Literature Review Summary

| # | Paper | Year | Methodology | Gap Relative to Conversational RAG |
|---|-------|------|-------------|-----------------------------------|
| 1 | **SECOM** (Pang et al., ICLR) | 2025 | Segment-level memory + compression-based denoising | Single-user AI dialogue only; lacks multi-speaker attribution |
| 2 | **MemGPT** (Packer et al.) | 2023 | OS-inspired hierarchical memory paging | Assumes single 1-on-1 dialogue; does not handle multi-party chats |
| 3 | **Alonso et al.** | 2024 | Time-sensitive long-term memory | Time-awareness applied only within single-user assistant memory |
| 4 | **Gutiérrez et al.** | 2025 | Non-parametric continual learning | Built for general knowledge/QA corpora, not personal chat |
| 5 | **MultiHop-RAG** (Tang & Yang) | 2024 | Multi-hop QA benchmark | Evaluates structured news/document prose, not fragmented messaging |

---

## 💻 7. Reproducing the Comparative Study

Run the benchmark suite using the self-contained evaluation scripts in `scripts/`:

```powershell
# 1. Verify Python dependencies
python scripts/check_env.py

# 2. Run individual framework benchmark evaluations
python scripts/run_naive_baseline.py
python scripts/run_langchain.py
python scripts/run_llamaindex.py
python scripts/run_context_aware.py

# 3. View the aggregated comparison table and category breakdown
python scripts/compare_results.py

# 4. Run the comprehensive automated 25-query diagnostic evaluation
python -m src.evaluation.run_comparative_study
```

---

## ⚠️ 8. Limitations

- **Small Hand-Labeled Evaluation Set**: The benchmark contains a modest number of annotated questions across uneven question categories, with certain specialized types (such as code-mixed or single decision reversals) having limited sample sizes.
- **English-Centric Default Embedder**: The default embedding model (`all-MiniLM-L6-v2`) is primarily optimized for English text, exhibiting reduced recall on romanized and code-mixed queries (e.g. Telugu-English) relative to dedicated multilingual models.
- **Simplifying Assumptions in Temporal Modeling**: The time-decay ranker presumes later messages strictly supersede earlier commitments and parses dates in common explicit formats, which does not cover indirect or unformatted relative time expressions.
- **Retrieval vs. Generation Metric Gap**: Evaluated metrics (`hit@k` and `MRR`) assess whether the ground-truth evidence chunk was retrieved into the prompt context, not whether the generative LLM produces a completely correct answer.
- **Small Local LLM Hallucinations**: Quantized local LLMs (e.g. Llama 3 8B or Mistral 7B executed via Ollama) can still hallucinate details despite strict grounding prompts and negative abstention constraints.
- **Social Domain Specificity**: Evaluated chats originate from a single university cohort and may not directly generalize to corporate, customer support, or multi-lingual group chats.
- **Local Unauthenticated API**: The interactive FastAPI backend (`src/api/server.py`) is designed solely for local personal deployment without authentication; it should always be bound to `--host 127.0.0.1`.

