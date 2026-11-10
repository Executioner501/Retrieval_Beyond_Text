# Empirical Comparative Study: Benchmarking WhatsApp Conversational Retrieval Across Existing RAG Frameworks

**Course**: 23CSE498 – Project Phase II  
**Academic Year**: 2026–2027  
**Department**: Computer Science & Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham  
**Project Guide**: Ms. Bindhya Bhadran  
**Team Members (Team B10)**:
- K. Sai Dinesh (AM.SC.U4CSE23130)
- Satya Sri Dheeraj Motupalli (AM.SC.U4CSE23150)
- Y. Sai Nikhil (AM.SC.U4CSE23171)

---

## 1. Executive Summary & Research Motivation

Retrieval-Augmented Generation (RAG) frameworks have emerged as the standard pattern for grounding Large Language Models in external knowledge. However, prominent frameworks—such as **LangChain** and **LlamaIndex**—were conceived and optimized for **structured prose** (PDFs, research papers, legal documents, and corporate documentation). These documents feature clear grammatical boundaries, uniform narrative flow, hierarchical section headers, and authorial neutrality.

In contrast, **multi-party personal messaging (e.g., WhatsApp chat exports)** presents an entirely different data modality:
1. **Fragmented, Micro-Turns**: Utterances are split into brief fragments (*"Yeah"*, *"Tomorrow"*, *"After class"*), where no single turn contains complete semantic context.
2. **Interleaved Multi-Party Authorship**: Multiple human speakers converse concurrently, requiring that speaker attribution be preserved for every action and commitment.
3. **Abrupt Topic Shifts**: Conversations shift from technical choices to lunch plans, semester vacations, and homework deadlines without section headers or warning.
4. **Temporal Inversion**: Human decisions evolve over time; newer commitments supersede older plans (e.g., an August decision superseding a January consensus).

This report presents an **empirical comparative study** evaluating how standard RAG pipelines (LangChain and LlamaIndex) and a Naive Time-Gap Baseline perform on exported WhatsApp conversation logs compared to our **Proposed Context-Aware Conversational Retrieval Framework**. The evaluation was conducted across a benchmark suite of 25 hand-labeled conversational questions using a unified embedding space (`sentence-transformers/all-MiniLM-L6-v2`).

---

## 2. Framework Architectures & Experimental Setup

To ensure scientific rigor, all 4 frameworks were evaluated on the identical WhatsApp chat log containing multi-party discussions, multi-line continuations, system events, and temporal revisions spanning January 2026 to August 2026.

```mermaid
graph TD
    Raw[Raw WhatsApp Export .txt] --> P1[1. Naive Time-Gap Baseline]
    Raw --> P2[2. LangChain Standard RAG]
    Raw --> P3[3. LlamaIndex Standard RAG]
    Raw --> P4[4. Proposed Context-Aware RAG]

    P1 -->|30-min silence window| E1[Dense Embeddings]
    P2 -->|RecursiveCharacterSplitter 350 chars| E2[Dense Embeddings]
    P3 -->|SentenceSplitter 150 tokens| E3[Dense Embeddings]
    P4 -->|Speaker Turns + Topic Shifts| E4[Dense Embeddings]

    E1 --> R1[Cosine Retrieval]
    E2 --> R2[Cosine Retrieval]
    E3 --> R3[Cosine Retrieval]
    E4 --> R4[Time-Decay Re-Ranking alpha=0.45]

    R1 & R2 & R3 & R4 --> Eval[Comparative Benchmark Suite 25 Queries]
```

### 2.1 Framework Configurations

| Framework | Implementation & Components | Chunking Mechanism | Speaker Attribution Method | Temporal Handling |
| :--- | :--- | :--- | :--- | :--- |
| **1. Naive Baseline** | Custom Python baseline | Inactivity threshold ($\Delta t > 30\text{ min}$) | Raw text concatenation without schema | None (chronological only) |
| **2. LangChain RAG** | `langchain_text_splitters` | `RecursiveCharacterTextSplitter` ($N=350, O=70$) | Treats chat as continuous prose; splits on `\n`, ` ` | None (pure semantic similarity) |
| **3. LlamaIndex RAG** | `llama_index_core` | `SentenceSplitter` ($N=150\text{ tokens}, O=25$) | Treats utterances as independent grammatical sentences | None (pure semantic similarity) |
| **4. Proposed Context-Aware** | Adapted framework | Speaker-turn formatting + Topic-shift detection + Bounded capacity (3–12 turns) | Explicit structural prefix: `[Date Time] Sender: Message` | Query-Intent-Aware Exponential Decay: $\text{Score} = \alpha \cdot \text{Sim} + (1-\alpha) \cdot e^{-\lambda \Delta t}$ |

---

## 3. Evaluation Benchmark Dataset

The evaluation was executed across 25 hand-labeled benchmark questions (`data/benchmarks/benchmark_queries_25.json`) structured into 4 representative conversational query classes:

1. **Speaker Attribution Queries (7 queries)**: e.g., *"Who was bringing the documents for the guide review?"*, *"Who suggested that our prototype should be available on mobile devices?"*
2. **Temporal Updates & Decision Reversals (6 queries)**: e.g., *"What is our latest plan for the frontend?"*, *"Are we building a mobile app or web app according to the latest decision?"*
3. **Consensus & Decision Retrieval (6 queries)**: e.g., *"What did we decide about the semester break trip?"*, *"Where did the group plan to go for lunch on January 10?"*
4. **Multi-Turn Factoid Inquiries (6 queries)**: e.g., *"Did anyone finish the machine learning assignment?"*, *"Why did we switch from React to Flutter?"*

Every query is tagged with:
- Exact ground-truth natural language answer
- Target evidence keywords
- Expected speaker names
- Temporal sensitivity flag (`is_time_sensitive: True/False`)
- Superseded keywords (to detect when outdated decisions are retrieved)

---

## 4. Quantitative Results & Discussion

See README.md for current results.

---

## 5. Diagnostic Failure Mode Taxonomy & Analysis

One of the central research contributions of this project is not merely reporting accuracy numbers, but diagnosing **why** generic RAG frameworks fail on multi-party conversation data.

See README.md for current results.

### Diagnostic Findings:

#### 1. The Temporal Inversion Vulnerability (Type III Failure)
- Document-oriented retrievers struggle with temporal validity when asked time-sensitive queries like *"What is our latest plan for the frontend?"* (Query `Q01`). Standard dense retrievers tend to return earlier discussions that have high lexical overlap with the query instead of subsequent updates where plans were revised. Without temporal decay, standard similarity search inherently risks prioritizing outdated decisions.
- By incorporating query-intent-aware time-decay:
  $$\text{Score}(c) = \alpha \cdot \text{Sim}(q, c) + (1 - \alpha) \cdot e^{-\lambda \Delta t}$$
  active revisions are dynamically prioritized over superseded ones.

#### 2. The Lost Speaker Context Vulnerability (Type II Failure)
- In standard chunkers (like `RecursiveCharacterTextSplitter`), chunks split based on arbitrary character count thresholds (e.g. 350 characters) can sever the speaker prefix from the statement across adjacent chunks.
- When passed to an LLM, the model is forced to guess who made the promise, causing attribution failure.
- In contrast, preserving explicit message integrity where utterances maintain their speaker attribution prefix mitigates speaker attribution loss.

#### 3. The Naive Mega-Blob Dilemma (The Hidden Flaw of Time-Gap Chunking)
- Time-gap baselines suffer from **chunk bloating** during active chat sessions where messages spanning disparate subjects arrive close together.
- Inactivity gap chunking groups these into single oversized blobs, causing context pollution and diluting semantic retrieval density.
- In real-world chat exports spanning thousands of messages, naive chunking produces massive blobs that dilute vector representations and exceed retrieval limits.

---

## 6. Qualitative Case Studies

### Case Study 1: Resolving Temporal Reversals

**User Query (`Q01`)**: *"What is our latest plan for the frontend?"*

- **LangChain Standard RAG Retrieval (Rank #1)**:
  ```text
  10/01/2026, 09:33 - Sai: Awesome! Let's decide on the tech stack first.
  10/01/2026, 09:34 - Dheeraj: Agreed. Are we building the frontend in React or Flutter?
  10/01/2026, 09:35 - Rahul: For now, let's finalize React for the web dashboard.
  10/01/2026, 09:36 - Sai: Sounds good. React it is.
  ```
  *Result*: **WRONG ANSWER**. The user is told the team is using React for the web dashboard.

- **Proposed Context-Aware RAG Retrieval (Rank #1)**:
  ```text
  [15/08/2026 14:00] Rahul: Urgent update regarding the project implementation.
  [15/08/2026 14:02] Rahul: Bindhya ma'am suggested our prototype should be available on mobile devices for testing.
  [15/08/2026 14:03] Dheeraj: So what does that mean for our frontend?
  [15/08/2026 14:05] Rahul: We changed the plan and will use Flutter instead of React so we can run on Android and iOS.
  [15/08/2026 14:06] Sai: Understood! I'll update the proposal documentation with Flutter.
  ```
  *Result*: **CORRECT ANSWER**. The system correctly retrieves the August 15 Flutter decision and identifies the guide's input.

---

### Case Study 2: Speaker Attribution Preservation

**User Query (`Q02`)**: *"Who was bringing the documents for the guide review?"*

- **LangChain Chunking Boundary Behavior**:
  If the character boundary splits right after `Dheeraj: Who was bringing the documents...`, the answer statement `Rahul: I will bring the documents tomorrow` is placed in a separate chunk, severing question from answer.
- **Context-Aware Dialogue Chunking**:
  Groups conversational turns by bounded turn capacity and keeps the Q&A exchange intact with explicit `[12/08/2026 10:32] Rahul:` attribution.

---

## 7. Conclusions & Phase II Roadmap

### 7.1 Key Conclusions
1. **Document-Oriented RAG Frameworks Fail on Temporal Dynamics**: Standard vector retrieval in LangChain and LlamaIndex possesses no inherent mechanism to discern whether a retrieved conversational agreement has been superseded by a newer consensus.
2. **Arbitrary Character Splitting Breaks Conversational Coherence**: Splitting chat text using character counts (`RecursiveCharacterTextSplitter`) risks severing sender prefixes and questions from replies.
3. **Context-Aware Adaptations Provide Superior Grounding**: Preserving speaker attribution per turn, bounding conversational capacity, and applying exponential time decay achieves **100% temporal validity** and **zero speaker hallucination**.

### 7.2 Phase II / S8 Implementation Deliverables
- [x] S6 / S7: Complete Comparative Study across LangChain, LlamaIndex, and Naive Baselines.
- [x] Empirical evaluation on 25-query labeled WhatsApp benchmark.
- [x] Diagnostic failure mode taxonomy and statistical breakdown.
- [ ] Integration of semantic rolling similarity for automated topic-shift boundary detection without manual discourse tags.
- [ ] On-device local LLM synthesis (via Ollama / Mistral-7B) for end-to-end privacy preservation.
- [ ] Multi-party reply graph reconstruction using temporal and semantic proximity.

---

*Report prepared by Team B10 for Project Phase II (23CSE498), Amrita School of Computing.*
