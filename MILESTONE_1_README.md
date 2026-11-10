# Milestone 1 Deliverable Report: Empirical Comparative Study & Conversational RAG Framework

**Course**: 23CSE498 – Project Phase II  
**Academic Year**: 2026–2027  
**Department**: Computer Science & Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham  
**Project Guide**: Ms. Bindhya Bhadran  
**Team Members (Team B10)**:
- K. Sai Dinesh (AM.SC.U4CSE23130)
- Satya Sri Dheeraj Motupalli (AM.SC.U4CSE23150)
- Y. Sai Nikhil (AM.SC.U4CSE23171)

---

## 1. Executive Summary

Traditional Retrieval-Augmented Generation (RAG) frameworks (such as LangChain and LlamaIndex) are optimized for structured prose (PDFs, research articles, documentation). When applied to **personal multi-party messaging (e.g., WhatsApp chat exports)**, these frameworks suffer from severe failure modes:
1. **Lost Speaker Context**: Utterances (*"Agreed"*, *"Tomorrow"*, *"Flutter it is"*) lose author attribution when stripped of dialogue structure.
2. **Arbitrary Chunk Boundary Splitting**: Character-count and sentence-based splitters sever conversational questions from their answers.
3. **Temporal Inversion**: Because past discussions often share high lexical similarity with search queries, standard dense retrievers retrieve outdated commitments (e.g., a January consensus) rather than superseded latest agreements (e.g., an August revision).

**Milestone 1 Objective**: Empirically evaluate and diagnose these limitations across **Naive Time-Gap Baseline**, **LangChain Standard RAG**, and **LlamaIndex Standard RAG**, benchmarked against our **Proposed Context-Aware Conversational RAG Framework**.

---

## 2. Milestone 1 Completed Deliverables

| Deliverable | Description | Location | Status |
| :--- | :--- | :--- | :---: |
| **WhatsApp Chat Parser** | Parses exports, handles multi-line messages, filters system notifications & media placeholders | [`src/parser/parse_whatsapp.py`](file:///c:/Users/H3X4N/Desktop/Final%20Year/src/parser/parse_whatsapp.py) | ✅ Verified |
| **Chunking Engine** | Baseline 30-min time-gap chunking vs. Context-Aware speaker-attributed bounded chunking | [`src/chunking/chunk_messages.py`](file:///c:/Users/H3X4N/Desktop/Final%20Year/src/chunking/chunk_messages.py) | ✅ Verified |
| **Local Vector Storage** | Local-first embedding manager using `all-MiniLM-L6-v2` for zero-cloud data privacy | [`src/indexing/vector_store.py`](file:///c:/Users/H3X4N/Desktop/Final%20Year/src/indexing/vector_store.py) | ✅ Verified |
| **Time-Aware Re-Ranker** | Intent-modulated time-decay re-ranking: $\text{Score} = \alpha \cdot \text{Sim} + (1-\alpha) \cdot e^{-\lambda \Delta t}$ | [`src/retrieval/ranker.py`](file:///c:/Users/H3X4N/Desktop/Final%20Year/src/retrieval/ranker.py) | ✅ Verified |
| **Grounded Answer Prompts** | System prompt enforcing strict speaker attribution and temporal trace citations | [`src/generation/prompts.py`](file:///c:/Users/H3X4N/Desktop/Final%20Year/src/generation/prompts.py) | ✅ Verified |
| **Evaluation Benchmark Suites** | 25-query multi-turn suite + 18-query chat-scoped evaluation suite | [`data/benchmarks/`](file:///c:/Users/H3X4N/Desktop/Final%20Year/data/benchmarks/) | ✅ Verified |
| **Comparative Study Runner** | Automated comparative study computing Hit@1, Hit@3, MRR, and Failure Taxonomy | [`src/evaluation/run_comparative_study.py`](file:///c:/Users/H3X4N/Desktop/Final%20Year/src/evaluation/run_comparative_study.py) | ✅ Verified |
| **Standalone Runner Scripts** | Self-contained Windows-compatible benchmark scripts | [`scripts/`](file:///c:/Users/H3X4N/Desktop/Final%20Year/scripts/) | ✅ Verified |
| **Interactive Visualizer Studio** | Full visual dashboard with chat stream, chunk inspection, and benchmark charts | [`ui/index.html`](file:///c:/Users/H3X4N/Desktop/Final%20Year/ui/index.html) | ✅ Verified |

---

## 3. Comparative Framework Architectures

```text
Raw WhatsApp Chat Export (.txt)
              │
              ├──► 1. Naive Time-Gap Baseline
              │       • Chunk split on inactivity gap (Δt > 30 mins)
              │       • Raw text concatenation without speaker structure
              │
              ├──► 2. LangChain Standard RAG
              │       • RecursiveCharacterTextSplitter (350 chars, 70 overlap)
              │       • Treats conversation as flat prose; splits on newlines and spaces
              │
              ├──► 3. LlamaIndex Standard RAG
              │       • SentenceSplitter (150 tokens, 25 overlap)
              │       • Treats conversational utterances as independent grammatical sentences
              │
              └──► 4. Proposed Context-Aware Conversational RAG
                      • Explicit structural speaker turns: [Date Time] Sender: Message
                      • Topic-shift boundary detection with bounded turn capacity (3–12 turns)
                      • Query-intent modulated exponential time-decay re-ranking
```

---

## 4. Empirical Evaluation Results

> **Note**: See README.md for current results, confidence intervals, and ablation benchmarks.

### 4.1 Quantitative Performance
See README.md for current results.

### 4.2 Diagnostic Failure Mode Distribution
See README.md for current results.

### 4.3 Key Research Findings
1. **Temporal Inversion in General RAG**: General-purpose document-oriented retrievers struggle on time-sensitive queries without temporal decay, often retrieving outdated discussions because older text may have higher lexical similarity with the prompt.
2. **Temporal Decay Elimination of False Plans**: Incorporating query-intent-aware time-decay dynamic weighting prioritizes current commitments over superseded ones.
3. **Dialogue Boundary Integrity**: Character- and sentence-based splitters risk severing questions from answers across chunk boundaries, whereas conversational turn-bounded chunking preserves dialogic coherence.

---

## 5. How to Reproduce All Results

Run the benchmark suite directly in PowerShell or Terminal:

```powershell
# Step 1: Check environment dependencies
python scripts/check_env.py

# Step 2: Run all four individual benchmark framework runners
python scripts/run_naive_baseline.py
python scripts/run_langchain.py
python scripts/run_llamaindex.py
python scripts/run_context_aware.py

# Step 3: Display aggregated side-by-side comparison matrix and category breakdown
python scripts/compare_results.py

# Step 4: Run the full 25-query automated diagnostic study with failure mode metrics
python -m src.evaluation.run_comparative_study
```

All benchmark logs and JSON reports are saved in [`eval_results/`](file:///c:/Users/H3X4N/Desktop/Final%20Year/eval_results/).

---

## 6. Project Repository Map

```tree
├── MILESTONE_1_README.md       # Milestone 1 Comprehensive Summary & Deliverables Report
├── README.md                   # Main Project Overview & Setup Documentation
├── Detailed_Project_Explanation.md # Conceptual System Formulation & Literature Background
├── docs/
│   ├── COMPARATIVE_STUDY_REPORT.md # Full Academic Comparative Study Paper & Analysis
│   ├── SYSTEM_DESIGN.md        # Technical System Architecture & Mathematical Schemas
│   └── UI_UX_SPECIFICATION.md  # Interactive Studio Design Specification
├── data/
│   ├── raw/sample_chat.txt     # Multi-turn benchmark WhatsApp chat export
│   ├── processed/              # Parsed chat records (yaoi, diddy, nikhil, sample)
│   └── benchmarks/             # 25-query and 18-query labeled test suites
├── src/
│   ├── parser/                 # WhatsApp raw export parsing module
│   ├── chunking/               # Baseline & Context-Aware chunking implementations
│   ├── indexing/               # Local-first embedding & vector storage manager
│   ├── retrieval/              # Time-decay semantic re-ranker
│   ├── generation/             # Attribution-grounded LLM synthesis
│   └── evaluation/             # Comparative study suite & failure diagnostics
├── scripts/
│   ├── check_env.py            # Dependency validation script
│   ├── run_naive_baseline.py   # Standalone Naive Baseline runner
│   ├── run_langchain.py        # Standalone LangChain runner
│   ├── run_llamaindex.py       # Standalone LlamaIndex runner
│   ├── run_context_aware.py    # Standalone Context-Aware runner
│   └── compare_results.py      # Results comparison matrix & tabular report
├── eval_results/               # Exported JSON benchmark reports & evaluation logs
└── ui/                         # Interactive Design Visualizer Studio (index.html)
```

---

*Milestone 1 completed by Team B10 for Project Phase II (23CSE498), Amrita School of Computing, Amritapuri Campus.*
