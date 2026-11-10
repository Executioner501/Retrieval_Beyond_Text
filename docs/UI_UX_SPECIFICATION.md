# UI/UX Specification Document: Conversational Retrieval Studio

**Document Version**: 1.0.0  
**Project**: Context-Aware Retrieval Framework for Multi-Party, Time-Dependent Conversations  
**Audience**: Final Year Review Panel, Guide Evaluation, Research Demonstrations

---

## 1. Design Philosophy & Aesthetic Pillars

The interface is engineered to serve as both a **research evaluation workbench** and an **interactive presentation showcase** for the project.

### Visual Foundations
- **Theme**: Premium Dark Mode with subtle glassmorphic translucency (`rgba(18, 24, 38, 0.8)` with `backdrop-filter: blur(16px)`).
- **Color Palette**:
  - Background Base: Deep Slate (`#0B0F19`)
  - Surface Glass: Slate Tint (`#131B2E`)
  - Accent Primary: Electric Cyan (`#06B6D4` / `#00F0FF`) - representing AI retrieval and vectors
  - Accent Secondary: Vibrant Violet (`#8B5CF6`) - representing conversational intelligence
  - Accent Temporal: Amber Gold (`#F59E0B`) - representing temporal weight and recency decay
  - Success / Attributed: Emerald Green (`#10B981`)
  - Alert / Boundary Shift: Rose Red (`#F43F5E`)
- **Typography**:
  - Headings & Display: `Outfit`, sans-serif (clean geometric humanist)
  - Body & Data Tables: `Inter`, sans-serif (crisp legibility for chat logs and metrics)
  - Monospace (Code / Raw): `JetBrains Mono` / `Fira Code`

---

## 2. Information Architecture & Navigation

The application consists of 5 dedicated tabs/views:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  [Logo] Context-Aware Chat RAG Studio      [Status: Index Ready]  [v1.0] │
├──────────────────────────────────────────────────────────────────────────┤
│ [1. Chat Ingestion] [2. Chunking Visualizer] [3. Retrieval Playground]  │
│ [4. Grounded Citations] [5. Diagnostic Benchmark]                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                       DYNAMIC VIEWPORT CONTENT                           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### View 1: Chat Ingestion & Stream Inspector
- **Purpose**: Upload `.txt` exports and inspect the cleaned, normalized message stream.
- **Components**:
  - Dropzone for `.txt` WhatsApp export files.
  - KPI Stat Bar: Total Messages, Active Participants, Timespan, Filtered System Notices.
  - Interactive Chat Log Viewer: Rendered as a timeline with sender badges, timestamp stamps, and filtered system notifications toggle.

### View 2: Chunking Strategy Visualizer (Side-by-Side Comparison)
- **Purpose**: Directly demonstrate the core thesis argument—why naive 30-min time-gap chunking fails and how context/topic-aware chunking succeeds.
- **Components**:
  - Left Panel: **Naive Time-Gap (30-min threshold)** showing large monolithic chunks where unrelated conversations bleed together.
  - Right Panel: **Proposed Context-Aware Chunking** showing neatly segmented topical chunks with speaker badges and detected topic shifts.
  - Boundary Inspector: Click any chunk to view participant distributions and time-span metrics.

### View 3: Retrieval & Time-Weighted Playground
- **Purpose**: Interactive natural-language query interface with live hyperparameter manipulation.
- **Components**:
  - Natural Language Search Bar with pre-populated research benchmark questions:
    - *"Who was bringing the documents?"* (Speaker attribution test)
    - *"What did we decide about the trip?"* (Decision retrieval test)
    - *"What is our latest plan for the frontend?"* (Temporal update test)
  - **Dynamic Controls**:
    - $\alpha$ Semantic vs. Temporal Slider ($0.0 \rightarrow 1.0$)
    - $\lambda$ Time Decay Rate Slider ($0.0 \rightarrow 2.0$)
    - Top-$k$ Candidate Selector ($1 \rightarrow 10$)
  - **Interactive Candidate Cards**:
    - Shows Semantic Score, Time Decay Penalty, and Combined Final Score.
    - Live reordering animation when sliders are shifted!

### View 4: Grounded Answer & Citation View
- **Purpose**: Demonstrate trustworthy LLM response generation with verifiable chat citations.
- **Components**:
  - Synthesized LLM Answer Card with confidence rating.
  - Grounding Evidence Cards: Side-by-side view of the exact source messages highlighted in the chat context to prove absence of hallucination.

### View 5: Diagnostic Evaluation & Failure Mode Analytics
- **Purpose**: Visual presentation of empirical research results for guide/evaluator review.
- **Components**:
  - Comparative Metric Cards:
    - Baseline Time-Gap (Hit Rate: 58.3%, MRR: 0.51)
    - Generic RAG / LangChain (Hit Rate: 66.7%, MRR: 0.60)
    - Context-Aware Framework (Hit Rate: 91.7%, MRR: 0.88)
  - Failure Mode Distribution Bar Chart:
    - Bad Chunk Boundaries (42% of baseline failures)
    - Lost Speaker Context (28% of baseline failures)
    - Wrong Time Window (21% of baseline failures)
    - Irrelevant Retrieval (9% of baseline failures)

---

## 3. Micro-Interactions & Responsiveness

- **Hover States**: Cards feature subtle 3D translation (`translateY(-2px)`) and glowing cyan/amber border highlights.
- **Re-Ranking Transitions**: Chunks smoothly animate and reorder using CSS transitions when hyperparameters $\alpha$ or $\lambda$ change.
- **Responsive Layout**: Fluid CSS Grid and Flexbox adapting from widescreen presentation monitors down to laptop displays.
