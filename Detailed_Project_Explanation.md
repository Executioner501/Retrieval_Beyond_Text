# Detailed Explanation of the Project

# Retrieval Beyond Text: Designing a Context-Aware Retrieval Framework for Multi-Party, Time-Dependent Conversations

## 1. Project Overview

This project focuses on building and evaluating a **context-aware retrieval system for exported WhatsApp conversations**.

The main idea is to allow a user to ask natural-language questions about their old conversations instead of manually scrolling through thousands of messages or searching for exact keywords.

In simple terms:

> The system should understand the context, participants, topics, and time information in a conversation and retrieve the information needed to answer a user's question.

The project is not simply a WhatsApp chatbot. Its major research focus is understanding why traditional Retrieval-Augmented Generation (RAG) systems struggle with human conversations and designing adaptations that improve retrieval.

---

# 2. The Problem Being Solved

People often accumulate years of WhatsApp conversations containing useful information such as:

- Trip decisions
- Meeting plans
- Addresses
- Promises and commitments
- Project discussions
- Important dates
- Responsibilities assigned to different people
- Previous decisions

Finding this information later is difficult.

## Example

A user may remember:

> "We discussed something about the Goa trip."

But they may not remember the exact words used in the conversation.

Traditional keyword search depends heavily on matching words.

For example, searching:

> "What did we decide about the trip?"

may not find a message that says:

> "Let's finalize Goa for December."

The words are different, even though the meaning is related.

Therefore, the project aims to support **semantic and context-aware retrieval** rather than simple keyword matching.

---

# 3. Why Human Conversations Are Difficult for Traditional RAG

Traditional RAG systems are usually designed for structured documents such as:

- PDFs
- Articles
- Research papers
- Web pages
- Manuals

Documents generally contain:

- Paragraphs
- Sections
- Headings
- Clearly separated topics

WhatsApp conversations are very different.

## 3.1 Fragmented Messages

People usually communicate using very short messages.

For example:

```text
Yeah
Okay
Tomorrow
After class
```

Each message alone has very little meaning.

The meaning comes from the surrounding conversation.

Therefore, retrieving individual messages is often insufficient.

---

## 3.2 Multiple Speakers

Human conversations contain multiple participants.

For example:

```text
Rahul: I will bring the documents.
Sai: Okay.
Dheeraj: Great.
```

If the user asks:

> "Who was bringing the documents?"

the system must preserve speaker information.

Without speaker context, the answer may be incorrect.

---

## 3.3 Topic Changes

Conversations can rapidly move between unrelated subjects.

For example:

```text
Trip planning
↓
Food
↓
Assignment
↓
Football
↓
Back to trip planning
```

A simple chunking method may combine unrelated topics into one chunk.

This can reduce retrieval quality.

---

## 3.4 Time Matters

Time is important in conversations.

For example:

> "What was our latest plan?"

There may be several plans discussed at different times.

Example:

### January

> We will use React.

### August

> We changed the plan and will use Flutter.

Both messages are semantically related to the project plan.

However, the August discussion is likely more relevant because it is newer.

Therefore, retrieval should consider both:

- Semantic similarity
- Temporal relevance

---

## 3.5 Reply and Thread Relationships (Future Work)

Sometimes a message responds to something discussed earlier.

A plain WhatsApp `.txt` export does not necessarily preserve the complete internal reply structure.

Therefore, full multi-party conversational thread and reply graph reconstruction is identified as **Future Work**, while current conversational chunking groups contiguous speaker turns and detects topic boundaries. Factors planned for future investigation include:

- Semantic similarity
- Timing proximity
- Surrounding context

---

# 4. The Research Gap

Existing systems generally fall into two major categories.

## 4.1 Conversational Memory Systems

Systems such as conversational memory frameworks are often designed for:

> One user ↔ AI assistant

They are not primarily designed for multiple humans communicating with each other.

Human group conversations introduce additional complexity such as:

- Multiple speakers
- Interleaved discussions
- Topic switching
- Different participant roles

---

## 4.2 Traditional RAG Systems

Frameworks such as:

- LangChain
- LlamaIndex
- Haystack

are mainly general-purpose retrieval frameworks.

They work effectively with:

- Documents
- Knowledge bases
- Structured text

However, they do not automatically understand the special structure of informal human conversations.

---

# 5. Core Research Question

The project is essentially investigating:

> Can retrieval be adapted specifically for multi-party, fragmented, time-dependent human conversations, and can these adaptations improve performance compared with generic RAG approaches?

The proposed system investigates improvements related to:

- Speaker-aware context
- Better conversational chunking
- Topic awareness
- Time-aware retrieval
- *(Future Work: Thread or reply approximation)*

---

# 6. The Complete System Architecture

The overall system pipeline can be represented as:

```text
WhatsApp Chat Export
        ↓
     Parsing
        ↓
    Chunking
        ↓
    Embedding
        ↓
  Vector Storage
        ↓
    Retrieval
        ↓
Time-aware Re-ranking
        ↓
LLM Answer Generation
        ↓
 Natural Language Answer
```

Each stage performs a specific function.

---

# 7. Module 1 — WhatsApp Chat Parsing

## Input

The system receives an exported WhatsApp `.txt` file.

Example:

```text
12/08/2026, 10:30 - Rahul: Are we meeting tomorrow?
12/08/2026, 10:31 - Sai: Yes, after class.
```

The parser converts this into structured information.

Example:

```json
{
    "date": "12/08/2026",
    "time": "10:30",
    "sender": "Rahul",
    "text": "Are we meeting tomorrow?"
}
```

---

## The Parser Handles

### Multi-line Messages

Example:

```text
Rahul: This is a long message
that continues on another line.
```

The system must understand that the second line belongs to the same message.

### System Notifications

Examples:

```text
Messages are end-to-end encrypted
Rahul added Sai
```

These are generally not useful for semantic retrieval and can be filtered.

### Media Placeholders

Example:

```text
<Media omitted>
```

These can also be filtered during preprocessing.

---

# 8. Module 2 — Conversational Chunking

Chunking is one of the most important components of the project.

A vector retrieval system usually works better when meaningful pieces of information are grouped together.

For WhatsApp conversations, individual messages are often too small to understand independently.

Therefore, messages must be grouped into meaningful conversational chunks.

---

## 8.1 Baseline: Time-Gap Chunking

The baseline approach groups messages based on the silence between them.

Example:

```text
10:00 - Let's plan the trip.
10:05 - Goa?
10:06 - Sounds good.
```

These messages belong to the same conversation.

If the next message occurs much later:

```text
15:00 - Did you complete the assignment?
```

the system starts a new chunk.

The baseline uses a **30-minute silence threshold**.

---

## Problem With Only Time-Based Chunking

Suppose people continuously chat for several hours without a large silence gap.

They may discuss:

```text
Movies → Food → Trip → Assignment → Football
```

A time-based chunking system may combine everything into one huge chunk.

This reduces retrieval quality because unrelated information is stored together.

The project documentation specifically identifies this as a limitation of naive chunking.

---

# 9. Proposed Improvement — Context-Aware Chunking

The project proposes improving chunk boundaries by considering conversational context.

## 9.1 Speaker-Aware Context

The system preserves who said each message.

Instead of:

```text
I will bring the documents.
Okay.
```

the system retains:

```text
Rahul: I will bring the documents.
Sai: Okay.
```

This enables questions such as:

> "Who said they would bring the documents?"

---

## 9.2 Topic-Aware Chunking

The system should identify when the conversation changes topics.

Example:

### Topic 1

```text
Where should we go?
Goa sounds good.
Let's decide tomorrow.
```

### Topic 2

```text
Did you finish the assignment?
Not yet.
```

Even if these messages occur continuously, they should ideally be separated into different chunks.

---

## 9.3 Thread or Reply Awareness (Future Work)

Conversational relationships across interleaved messages remain an active area of investigation.

Since complete reply metadata may not be available in exported text files, approximating non-adjacent reply chains using semantic similarity and timing proximity is planned as **Future Work**. In the current release, conversational continuity is maintained via speaker-attributed windows and semantic topic segmentation.

---

# 10. Module 3 — Embeddings

Once meaningful chunks are created, they are converted into **vector representations**, called embeddings.

Example conversation:

> "We decided to go to Goa during December."

An embedding model converts its meaning into a numerical vector.

Conceptually:

```text
Text → Embedding Vector
```

The important idea is:

> Text with similar meanings should have similar vector representations.

Therefore, a query such as:

> "What was our plan for the vacation?"

can retrieve:

> "We decided to go to Goa during December."

even if the exact words are different.

---

# 11. Module 4 — Vector Storage

The generated embeddings are stored in a vector database.

The proposed technology includes:

- Local In-Memory NumPy Vector Store (brute-force normalized cosine similarity for strict on-device privacy)

The vector store connects:

```text
Embedding
    ↕
Original Conversational Chunk
```

When a user asks a question, the system searches for vectors that are close to the query vector.

---

# 12. Module 5 — Semantic Retrieval

Suppose the user asks:

> "What did we decide about the trip?"

The system performs the following steps:

### Step 1

Convert the question into an embedding.

### Step 2

Search the vector database.

### Step 3

Retrieve semantically similar conversational chunks.

Example retrieved information:

```text
Let's finalize Goa for December.

We should book tickets next week.

Rahul said he will check hotels.
```

These chunks are then used as evidence for answering the question.

---

# 13. Module 6 — Time-Aware Re-Ranking

Traditional semantic retrieval asks:

> Which chunks are most similar to the user's question?

The proposed system additionally considers:

> Which relevant information is most appropriate based on time?

For example:

### Older Discussion

> We will use React.

### Newer Discussion

> We changed the plan and will use Flutter.

If the user asks:

> "What is our latest plan?"

the newer discussion should receive more importance.

The proposed approach combines:

- Semantic similarity
- Temporal relevance

Conceptually:

```text
Final Score =
Semantic Similarity Weight
+
Time Relevance Weight
```

A more formal representation is:

```text
Score(c) =
α × SemanticSimilarity
+
(1 - α) × TimeWeight
```

where:

- `α` controls the importance of semantic similarity
- `TimeWeight` represents temporal relevance

This allows the system to balance meaning and recency.

---

# 14. Module 7 — LLM Answer Generation

After relevant conversation chunks are retrieved, they are provided to a Large Language Model (LLM).

Example:

### User Question

> Who was bringing the documents?

### Retrieved Conversation

```text
Rahul: I will bring the documents tomorrow.
Sai: Okay.
Dheeraj: Great.
```

### Generated Answer

> Rahul said he would bring the documents.

The LLM should generate answers based on retrieved evidence rather than guessing.

This is the core principle of:

# Retrieval-Augmented Generation (RAG)

---

# 15. The Evaluation Methodology

> See README.md for current results.

A major strength of the project is that it does not simply build a prototype and claim that it works.

Instead, it compares multiple approaches.

A set of natural-language questions is created for the conversations.

The documentation proposes approximately:

> 20–30 questions per chat

Each question should have a manually verified ground-truth answer.

Example:

| Question | Ground Truth |
|---|---|
| Who was bringing the documents? | Rahul |
| What did we decide about the trip? | Goa in December |
| When did we change the plan? | August |

---

# 16. Systems Being Compared

The evaluation compares:

## 1. Naive Baseline

Using simple time-gap chunking.

## 2. Existing RAG Frameworks

Potentially including:

- LangChain
- LlamaIndex

## 3. Proposed Context-Aware System

Using improvements such as:

- Speaker awareness
- Better chunk boundaries
- Topic awareness
- Time-aware ranking
- *(Future Work: Thread/reply approximation)*

---

# 17. Failure Analysis

The project does not only measure whether an answer is correct.

It also investigates **why retrieval fails**.

Important failure categories include:

## Bad Chunk Boundaries

Relevant messages are separated into different chunks.

## Lost Speaker Context

The relevant statement is retrieved, but the system cannot correctly identify who said it.

## Wrong Time Window

The system retrieves an old discussion when the user needs the latest information.

## Irrelevant Retrieval

The system retrieves something semantically similar but not actually relevant.

This diagnostic analysis is one of the central research contributions of the project.

---

# 18. Privacy Considerations

WhatsApp conversations contain highly sensitive personal information.

The project does not focus on directly accessing a user's live WhatsApp account.

Instead:

```text
User exports their chat
        ↓
System processes the exported data
        ↓
Retrieval is performed on the conversation data
```

The long-term privacy goal is local or on-device processing.

This reduces dependence on external cloud systems for storing raw personal conversations.

---

# 19. Dataset

The primary data source consists of exported WhatsApp conversations.

The project can include:

- Personal chats
- Small group conversations
- Informal discussions
- Multi-speaker conversations

The project scope particularly focuses on manageable conversation sizes such as:

- One-to-one conversations
- Small groups with approximately 2–3 participants

Personal information should be anonymized when necessary.

---

# 20. Technology Stack

## Programming Language

- Python

## Embedding Models

Potentially:

- Sentence Transformers or equivalent embedding models

## Vector Database

- Local in-memory NumPy vector store (normalized dot product cosine similarity)

## Baseline RAG Frameworks

- LangChain
- LlamaIndex

## Answer Generation

Possible options include:

- LLM APIs
- Locally hosted open-source language models

## Development Tools

- VS Code
- PyCharm
- Jupyter Notebook

---

# 21. How the Two Project Documents Fit Together

The two documents represent different stages of the project.

## Document 1 — Initial Project Proposal

This document mainly explains:

- Why the problem is important
- The problem statement
- Project objectives
- Dataset ideas
- Literature review
- Research gap

It answers:

> Why should this project exist?

and:

> What problem are we trying to solve?

---

## Document 2 — Design Proposal

This document develops the idea into an implementation plan.

It explains:

- System architecture
- Individual modules
- Methodology
- Retrieval pipeline
- Technology stack
- Evaluation strategy
- Expected outcomes
- Project timeline

It answers:

> How are we going to build and evaluate the project?

---

# 22. The Core Contribution of the Project

The strongest contribution is not simply building an AI chatbot for WhatsApp.

The real contribution is:

> Understanding why generic RAG systems fail on fragmented, multi-party, time-dependent human conversations and evaluating whether context-aware retrieval adaptations improve performance.

The project therefore has three major layers.

## Layer 1 — Engineering

Build a working retrieval system for WhatsApp-style conversations.

## Layer 2 — Research

Study the limitations of existing RAG approaches.

## Layer 3 — Contribution

Evaluate whether the following improvements help:

- Speaker-aware context
- Better conversational chunking
- Topic awareness
- Time-aware ranking
- *(Future Work: Thread awareness)*

---

# 23. Complete Project Flow

The entire project can be understood using the following sequence:

```text
                     PROBLEM
                         ↓
     Information is buried in long chat histories
                         ↓
              Traditional keyword search fails
                         ↓
                WHY GENERIC RAG FAILS
                         ↓
      Conversations are fragmented and multi-party
                         ↓
       Topics change and time affects relevance
                         ↓
                   PROPOSED SYSTEM
                         ↓
                  Parse the chat
                         ↓
             Create meaningful chunks
                         ↓
              Preserve speaker context
                         ↓
             Detect conversational topics
                         ↓
                Generate embeddings
                         ↓
            Store in local vector index
                         ↓
               Perform semantic retrieval
                         ↓
              Apply time-aware ranking
                         ↓
            Give context to an LLM
                         ↓
              Generate final answer
                         ↓
                    EVALUATION
                         ↓
     Compare against naive and generic approaches
                         ↓
                 Analyze failures
```

---

# 24. Short Presentation Explanation

If someone asks:

## "What is your project?"

A concise explanation is:

> Our project focuses on improving information retrieval from long-term WhatsApp-style conversations. Existing RAG systems are mainly designed for structured documents or single-user AI conversations, whereas human chats are fragmented, multi-party, and often time-dependent. We are designing and evaluating a context-aware retrieval pipeline that preserves speaker information, improves conversational chunking using speaker turn and topic awareness, and incorporates time into retrieval ranking. We compare this approach against naive chunking and existing RAG frameworks to identify where generic retrieval systems fail and whether our adapted approach improves retrieval and answer quality.

---

# 25. Final Summary

The project can be remembered using five ideas:

## Problem

🔍 Important information is difficult to find in old chats.

## Challenge

💬 Human conversations are fragmented, multi-party, informal, and time-dependent.

## Solution

🧩 Context-aware chunking  
👤 Speaker awareness  
🧵 Thread awareness (Future Work)  
🧠 Semantic retrieval  
⏰ Time-aware ranking

## Output

💬 Users can ask natural-language questions about their conversation history.

## Research Contribution

📊 Compare generic RAG systems with a context-aware approach and identify exactly where traditional retrieval methods fail.

---

# The Heart of the Project

The central idea of the project is:

> **Retrieval systems designed for documents do not automatically understand human conversations. By preserving conversational structure—such as speakers, topics, and temporal decay—the project investigates whether retrieval from multi-party conversations can be made more accurate and useful.**
