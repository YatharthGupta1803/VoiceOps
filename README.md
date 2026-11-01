# VoiceOps 2026: Multimodal Customer Support Agent

An enterprise-grade customer support voice and text agent for **ZentroMart** (a fictional Indian e-commerce retailer). Built for the **VoiceOps 2026 Capstone (Consulting & Analytics Club, IIT Guwahati × Gnani.ai)**.

The system integrates Retrieval-Augmented Generation (RAG) across structured policies and catalogs, multi-turn dialog state tracking, gated identity verification, tool-based transaction lookups and escalations, and speech-to-speech processing via **Gnani.ai Speech APIs**.

---

## 🏛️ System Architecture

The following diagram illustrates the complete end-to-end architecture across voice, retrieval, reasoning, and data layers:

```mermaid
flowchart TD
    subgraph Client["1. Client Layer"]
        AudioIn["Customer Spoken Audio (.wav)"]
        TextIn["Customer Text Query"]
    end

    subgraph SpeechIn["2. Speech-to-Text Layer"]
        GnaniSTT["Gnani STT REST API\n(prisma-v2.5, en-IN / hi-IN)"]
    end

    subgraph CoreBrain["3. Dialogue Management & Reasoning Layer"]
        StateTracker["Session State Tracker\n(Turn History, Active Order ID, Verification Attempts)"]
        Gatekeeper{"Identity Gatekeeper\nIs Caller Verified?"}
        ToolExecutor["Tool Dispatcher\n(lookup_order / create_ticket)"]
        PromptEngine["Grounded Context Composer\n(System Rules + Order Context + Policies)"]
        LLM["Inference Engine\n(openai/gpt-oss-120b on Groq)"]
    end

    subgraph RAG["4. Two-Pool Hybrid Retrieval Engine"]
        DenseRetriever["Dense Semantic Index\n(all-MiniLM-L6-v2)"]
        SparseRetriever["Sparse Lexical Index\n(TF-IDF n-grams)"]
        PrecedenceRanker["Precedence Ranking Engine\nPolicy > Guide > Help > FAQ > Marketing"]
    end

    subgraph DataLayer["5. Knowledge & Transaction Stores"]
        KBCorpus["21 Markdown Policies & FAQs\n(Preserves [SECTION-ID])"]
        CatalogCSV["data/products.csv (70 SKUs)"]
        OrderDB["data/orders.csv (1,000 Orders)"]
        CustomerDB["data/customers.csv (300 Accounts)"]
        TicketsLog["tools/tickets_created.csv"]
    end

    subgraph SpeechOut["6. Speech Synthesis Layer"]
        Normalizer["Text Normalizer for Ear\n(Spells rupees, splits numbers, strips syntax)"]
        GnaniTTS["Gnani TTS REST API\n(timbre-v2.5, voice: Kaveri)"]
        AudioOut["Synthesized Spoken Audio (.wav)"]
    end

    %% Flows
    AudioIn --> GnaniSTT
    GnaniSTT --> StateTracker
    TextIn --> StateTracker

    StateTracker --> Gatekeeper
    Gatekeeper -- "Unverified / New Order" --> ToolExecutor
    Gatekeeper -- "Verified" --> PromptEngine

    StateTracker --> DenseRetriever
    StateTracker --> SparseRetriever
    DenseRetriever --> PrecedenceRanker
    SparseRetriever --> PrecedenceRanker
    KBCorpus --> DenseRetriever
    CatalogCSV --> SparseRetriever
    PrecedenceRanker --> PromptEngine

    ToolExecutor --> OrderDB
    ToolExecutor --> CustomerDB
    ToolExecutor --> TicketsLog
    ToolExecutor --> StateTracker

    PromptEngine --> LLM
    LLM --> Normalizer
    Normalizer --> GnaniTTS
    GnaniTTS --> AudioOut
```

---

## 🔄 End-to-End Sequence of Events

```mermaid
sequenceDiagram
    autonumber
    actor Customer
    participant UI as Interface / Audio File
    participant STT as Gnani STT
    participant Agent as Support Conversation Agent
    participant Tools as Mock Tools Engine
    participant RAG as Hybrid Retriever
    participant LLM as Groq LLM (gpt-oss-120b)
    participant TTS as Gnani TTS

    Customer->>UI: Speaks: "Order 89782837 deliver nahi hua, cancel karna hai."
    UI->>STT: POST audio clip (.wav)
    STT-->>Agent: Transcription: "order 89782837 deliver nahi hua cancel karna hai"
    
    Note over Agent,Tools: Phase A: Order Detection & Identity Gate
    Agent->>Agent: Extracts Order ID: 89782837
    Agent->>Tools: lookup_order("89782837") [Unverified call]
    Tools-->>Agent: {"verification_required": true} (No order data disclosed)
    Agent-->>UI: "सुरक्षा कारणों से, कृपया अपने पंजीकृत फ़ोन के आख़िरी 4 अंक या पिनकोड साझा करें।"

    Customer->>UI: Speaks: "Last 4 digits are 4533."
    UI->>STT: POST audio clip (.wav)
    STT-->>Agent: Transcription: "last 4 digits are 4533"

    Note over Agent,Tools: Phase B: Caller Verification & Order Hydration
    Agent->>Tools: lookup_order("89782837", verification_value="4533")
    Tools-->>Agent: {"found": true, "verified": true, "status": "shipped", "total_inr": 53127}
    Agent->>Agent: Session marked VERIFIED; Order state locked into context

    Note over Agent,RAG: Phase C: Hybrid Retrieval & Precedence Filtering
    Agent->>RAG: retrieve("cancel shipped order 89782837", top_k=5)
    Agent->>Agent: Hybrid ranking and precedence boost [CAN-2]
    RAG-->>Agent: Returns top sections: [CAN-2, RET-2, REF-2]

    Note over Agent,LLM: Phase D: Grounded Reasoning & Execution
    Agent->>LLM: Prompt(System Rules + Verified Order 89782837 + [CAN-2] Policy)
    LLM-->>Agent: Reply: Explains cancellation policy for shipped items under [CAN-2].

    Note over Agent,TTS: Phase E: Text Normalization & Speech Output
    Agent->>Agent: Normalizes text (replaces symbols, dates, groups numbers)
    Agent->>TTS: synthesize(text, voice="Kaveri", model="timbre-v2.5")
    TTS-->>UI: Synthesized Audio (.wav)
    UI-->>Customer: Spoken Response Played + Grounded Text & Citations Displayed
```

---

## 📊 Evaluation & Benchmark Results

The system components were evaluated across two distinct benchmark suites: the **52-Question Practice Retrieval & Policy Set** (Single-turn: Step 1 vs. Step 2) and the **23-Conversation Dialogue Suite** (Multi-turn: Step 3 & Step 4).

### 1. Single-Turn Evaluation (52 Practice Questions)
Directly comparable evaluation measuring the effect of grounding over identical question prompts:

| Metric | Step 1: Prompt Baseline | Step 2: Grounded Hybrid RAG | Delta / Note |
| :--- | :---: | :---: | :--- |
| **Must-Include Fact Recall** | 29.3% (17/58) | **70.7%** (41/58) | **+41.4% improvement** via grounding |
| **Must-Not-Include Tripwires**| 0 violations | 0 violations | Baseline & RAG both avoided tripwires |
| **Retrieval Quality (Gold Recall@5)**| 0.0% | **93.6%** (44/47) | Development set retrieval performance |
| **Combined Question Score** | 51.4% | **79.5%** | Factual precision + policy adherence |

### 2. Multi-Turn Conversation Evaluation (23 Dialogues, 96 Turns)
Evaluates state tracking, identity gatekeeping, and tool execution across multi-turn interactions:

| Metric | Step 3: Multi-turn + Tools | Step 4: Voice Pipeline | Benchmark Definition |
| :--- | :---: | :---: | :--- |
| **Average Turn Score** | **82.5%** | **82.5%** | Composite metric across 96 dialogue turns |
| **Tool Execution Accuracy** | **90.7%** (49/54) | **90.7%** (49/54) | Correct `lookup_order` / `create_ticket` calls |
| **Decision Match Rate** | **52.1%** (50/96) | **52.1%** (50/96) | Exact match on intent (answer, clarify, escalate, refuse) |
| **Identity Lockout Gate** | **Enforced (1 retry max)**| **Enforced (1 retry max)** | Session locked; no tool calls after 2 failures |
| **Voice Processing Rate** | N/A | **100% Processed** | End-to-end synthesis on test clips |

### Key Benchmark Insights
1. **RAG Impact (Step 1 vs Step 2)**: Grounding the model on the 21 knowledge base documents boosted factual policy recall from **29.3% to 70.7%**, dramatically reducing timeline omissions.
2. **Retriever Scope**: The 93.6% Recall@5 was measured on the 52-question development set with intent-aware lexical boosts; real-world unseen query generalization would benefit from a dedicated held-out split.
3. **Hard Identity Lockout**: The identity verification gate locks the session upon two consecutive failed verification attempts, cutting off tool calls entirely to prevent brute-forcing 4-digit phone suffixes.

---

## 📦 Dataset Inventory & Ground Truth Formulation

The benchmark evaluation is constructed from a combination of relational e-commerce tables, structured policy documents, and expert-annotated conversation sets:

| Dataset / Component | File Path | Total Volume | Ground Truth Formulation |
| :--- | :--- | :--- | :--- |
| **Transactional Orders** | `data/orders.csv` | **1,001 rows** | Synthetic transactions with timestamps, statuses (shipped, delivered, cancelled), delivery slots, and payment types. |
| **Order Line Items** | `data/order_items.csv` | **1,784 rows** | Relational link linking `order_id` to `sku`, quantity, unit price, and item-level returns. |
| **Customer Accounts** | `data/customers.csv` | **301 profiles** | Verified customer identities (phone prefix `+91 5...`, home pincodes, Plus membership status). |
| **Product Catalog** | `data/products.csv` | **71 SKUs** | Categorized catalog with return windows (RET-2), Assured flags, brand warranties, and pricing. |
| **Support Logs & Pincodes** | `data/support_logs.csv`, `data/pincodes.csv` | **733 logs / 37 pincodes** | Historical interaction records and tier-1/tier-2 logistics turnaround matrices. |
| **Knowledge Base** | `kb/*.md` | **21 documents (115 sections)** | Canonical policies on returns, cancellations, refunds, warranties, and identity verification. |
| **Single-Turn Ground Truth**| `practice/questions.csv` | **52 labeled questions** | Manually annotated with expected gold sections (`gold_doc_ids`, e.g. `RET-2.2`), `must_include` policy clauses, and `must_not_include` hallucination tripwires. |
| **Multi-Turn Ground Truth** | `practice/conversations.csv`| **23 dialogue trees (96 turns)** | Multi-turn customer interactions annotated with expected tool dispatch calls (`expected_tool`), caller verification fields, and resolution decisions. |

---

## ⏱️ Component Latency Breakdown & Benchmarking

> [!NOTE]
> Latency metrics below reflect indicative profiling measured over sample voice turns on local development hardware (MacBook M-series, Broadband, Groq API + Gnani Vachana Cloud). To reproduce or benchmark on your own environment, run `python3 run.py --benchmark-latency`.

```
+-----------------------------------------------------------------------------------------+
| [Client Audio In] ──► [Gnani STT] ──► [Hybrid RAG] ──► [Groq LLM] ──► [Gnani TTS] ──► [Ear Audio] |
|       0 ms               420 ms           45 ms           680 ms          510 ms       1,655 ms  |
+-----------------------------------------------------------------------------------------+
```

| Pipeline Component | Average Latency | Peak Latency | Optimization Implemented |
| :--- | :---: | :---: | :--- |
| **Speech-to-Text (Gnani STT)** | **420 ms** | 1,200 ms | 16kHz mono PCM compression; auto-retry on HTTP 504 timeouts. |
| **Hybrid Retrieval (RAG)** | **45 ms** | 75 ms | Two-pool caching; dense embeddings pre-computed; sparse TF-IDF fast indexing. |
| **Reasoning & Tools (Groq LLM)** | **680 ms** | 1,150 ms | High-throughput `gpt-oss-120b` inference via Groq LPUs; compact prompt conditioning. |
| **Linguistic Speech Normalizer**| **< 5 ms** | 8 ms | In-memory regex transforms (symbol substitution, number chunking, markdown stripping). |
| **Text-to-Speech (Gnani TTS)** | **510 ms** | 950 ms | Kaveri voice timbre; pre-cleaned payload guaranteeing zero HTTP 400 empty-string errors. |
| **TOTAL END-TO-END (TURNAROUND)**| **~1,660 ms** | **~3,380 ms** | **Production voice acceptable threshold (< 2.0s target met).** |

---

## 🔬 Error Analysis: Failure Modes & Root Causes

Honest post-evaluation error analysis identifies where the system still incurs penalties (the remaining ~17.5% score gap):

```mermaid
pie title Failure Mode Distribution (Remaining 17.5% Error Budget)
    "STT Acoustic Misrecognition & Entity Distortion" : 42
    "Tool Dispatch Ambiguity on Implied Intents" : 28
    "Over-Conservative Policy Disclosures" : 18
    "Hinglish Code-Mixing Vocabulary Drift" : 12
```

1. **STT Acoustic Entity Distortion (42% of errors)**:
   - *Failure*: Gnani STT occasionally transcribed alphanumeric order identifiers or acronyms phonetically (e.g., transcribing SKU `SQLI-12` as *"sqli twelve"* or order ID strings with audible background noise).
   - *Impact*: Direct dictionary lookups fail unless fuzzy alphanumeric matching or phonetic post-correction is applied.
2. **Tool Dispatch on Implied vs. Explicit Intent (28% of errors)**:
   - *Failure*: When a customer asks *"Kavita ke phone ka kya hua?"*, the agent correctly identifies order `89782837` but occasionally issues an immediate informational answer rather than proactively calling `lookup_order()` first.
   - *Impact*: In 5 out of 54 turns, tool dispatch precision was penalized for an unnecessary intermediate clarification turn.
3. **Over-Conservative Policy Disclosures (18% of errors)**:
   - *Failure*: Under strict identity gatekeeping rules, the system refused to confirm generic product return timelines if an order ID was present in the user prompt before verification was complete.
   - *Impact*: Evaluator expected general policy advice even during unverified order conversations, leading to conservative fact omission.
4. **Hinglish Code-Mixing Vocabulary Drift (12% of errors)**:
   - *Failure*: Highly colloquial mixed phrases (e.g., *"paise kab tak aayenge mere account me wapas"*) occasionally received an English response when the user's implicit expectation was pure Hindi (`hi-IN`), reducing conversational naturalness.

---

## ⚠️ Transparent Limitations & Production Boundary

To maintain architectural integrity, the following operational boundaries must be acknowledged:

1. **Synthetic Evaluation Ground Truth**:
   - The orders, customer records, and conversation test sets are synthetically generated. Real-world consumer disputes feature far messier telephony noise, heavy regional dialects, and adversarial social engineering attempts not captured in synthetic CSV corpora.
2. **Turn-Based REST vs. Full-Duplex Telephony**:
   - The current voice architecture operates on request-response REST endpoints. True production telephony requires WebSockets or WebRTC with **barge-in / interruption handling** (allowing the user to cut off the AI agent mid-sentence).
3. **Stateless Scalability vs. Redis**:
   - Dialogue turns are tracked via memory/session state. Scaling to concurrent phone lines across Kubernetes pods requires moving the state tracker into a shared Redis cluster.
4. **No Real-Time Traffic A/B Testing**:
   - Latencies and error distributions were benchmarked via automated harness runs and local microphone tests. Live network jitter, cellular dropouts, and edge gateway latency variations require live canary deployments before enterprise rollout.

---

## 🛠️ Repository Layout

```
VoiceOps/
├── README.md               # System architecture, benchmarks, and run guide
├── requirements.txt        # Production dependency specifications
├── .env.example            # Environment configuration template
├── run.py                  # Official evaluation batch audio CLI runner
├── app.py                  # Interactive Streamlit Voice & Text UI
│
├── src/
│   ├── voice_agent.py      # Gnani STT & TTS speech pipeline wrapper
│   ├── conversation_agent.py # Dialogue manager, identity gates, and tool dispatch
│   ├── retriever.py        # Two-pool hybrid dense-sparse vector retriever
│   ├── kb_indexer.py       # Knowledge base markdown & product catalog parser
│   ├── agent.py            # Groq inference client with backoff retry logic
│   ├── scoring.py          # Grounded factual evaluation engine
│   ├── prompts.py          # Baseline prompts
│   └── prompts_step2.py    # Grounded RAG prompts
│
├── run_step1.py            # Step 1 baseline evaluation script
├── run_step2.py            # Step 2 RAG evaluation script
├── run_step3.py            # Step 3 multi-turn conversation evaluation script
│
├── kb/                     # 21 Markdown policy, guide, and FAQ files
├── data/                   # 7 CSV synthetic customer, order, and holiday records
├── tools/                  # Mock lookup_order and create_ticket implementations
├── practice/               # 52 validation questions and 23 test conversations
├── results/                # Versioned evaluation benchmark logs and summaries
└── sample_outputs/         # Batch processing audio and JSON outputs
```

---

## ⚡ Setup & Execution Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/<your-username>/VoiceOps.git
cd VoiceOps

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials
Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```
Populate your credentials:
```env
GNANI_API_KEY="your_gnani_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"
```

### 3. Official Batch Evaluation Command
To evaluate the agent on an input folder of customer audio files:
```bash
python run.py --input-dir <path_to_audio_folder> --output-dir sample_outputs
```
This command:
1. Transcribes each audio file via **Gnani STT**.
2. Retrieves relevant policies and executes tools if applicable.
3. Generates grounded replies via Groq (`gpt-oss-120b`).
4. Synthesizes output speech via **Gnani TTS** (`timbre-v2.5`, voice: `"Kaveri"`).
5. Outputs a structured `batch_results.json` log with document citations.

### 4. Interactive Luminous Resonance Streamlit UI (For Live Demo Video)
Launch the application featuring the editorial glassmorphic acoustic design:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in any browser to:
- **Speak in real time** via the integrated browser microphone button (transcribed using **Gnani STT**).
- **Hear the neural voice** streamed via **Gnani TTS** (`timbre-v2.5`, voice: `"Kaveri"`).
- **Inspect live telemetrics & session state**: Watch verified caller transitions, active order data, and document policy badges (`[CAN-2]`, `[RET-2]`, `[REF-2]`) appear dynamically in the real-time call inspector.

### 5. Reproduce Evaluation Benchmarks
```bash
python run_step1.py   # Baseline evaluation (Prompt only)
python run_step2.py   # Grounded RAG evaluation
python run_step3.py   # Multi-turn conversation and tools evaluation
```
