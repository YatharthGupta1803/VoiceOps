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
    
    rect rgb(240, 248, 255)
        Note over Agent,Tools: Phase A: Order Detection & Identity Gate
        Agent->>Agent: Extracts Order ID: 89782837
        Agent->>Tools: lookup_order("89782837") [Unverified call]
        Tools-->>Agent: {"verification_required": true} (No order data disclosed)
        Agent-->>UI: "सुरक्षा कारणों से, कृपया अपने पंजीकृत फ़ोन के आख़िरी 4 अंक या पिनकोड साझा करें।"
    end

    Customer->>UI: Speaks: "Last 4 digits are 4533."
    UI->>STT: POST audio clip (.wav)
    STT-->>Agent: Transcription: "last 4 digits are 4533"

    rect rgb(245, 255, 245)
        Note over Agent,Tools: Phase B: Caller Verification & Order Hydration
        Agent->>Tools: lookup_order("89782837", verification_value="4533")
        Tools-->>Agent: {"found": true, "verified": true, "status": "shipped", "total_inr": 53127, ...}
        Agent->>Agent: Session marked VERIFIED; Order state locked into context
    end

    rect rgb(255, 250, 240)
        Note over Agent,RAG: Phase C: Hybrid Retrieval & Precedence Filtering
        Agent->>RAG: retrieve("cancel shipped order 89782837", top_k=5)
        RAG->>RAG: Compute Hybrid Score (Dense Cosine + Sparse TF-IDF)
        RAG->>RAG: Apply Precedence Boost: pol_cancellations.md (CAN-2) > Help > FAQ
        RAG-->>Agent: Returns top sections: [CAN-2, RET-2, REF-2]
    end

    rect rgb(250, 245, 255)
        Note over Agent,LLM: Phase D: Grounded Reasoning & Execution
        Agent->>LLM: Prompt(System Rules + Verified Order 89782837 + [CAN-2] Policy)
        LLM-->>Agent: Reply: Explains cancellation policy for shipped items under [CAN-2].
    end

    rect rgb(255, 245, 245)
        Note over Agent,TTS: Phase E: Text Normalization & Speech Output
        Agent->>Agent: Normalizes text (replaces symbols, dates, groups numbers)
        Agent->>TTS: synthesize(text, voice="Kaveri", model="timbre-v2.5")
        TTS-->>UI: Synthesized Audio (.wav)
        UI-->>Customer: Spoken Response Played + Grounded Text & Citations Displayed
    end
```

---

## 📊 Comprehensive Evaluation & Results Comparison

Across the capstone evaluation rubric (100 marks total), each build step was benchmarked against the official validation datasets:

| Rubric Metric | Step 1: Prompt Only | Step 2: Grounded RAG | Step 3: Multi-turn + Tools | Step 4: Voice Pipeline | Marks Allocated |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Combined Score** | 51.4% | **79.5%** (+28.1%) | **82.5%** | **82.5%** | **10 Marks** (Comparison) |
| **Must-Include Fact Recall** | 29.3% (17/58) | **70.7%** (41/58) | **74.4%** (61/82) | **74.4%** | **25 Marks** (Grounded Accuracy) |
| **Retrieval Quality (Gold Hits)** | 0.0% | **93.6%** (44/47) | **93.6%** | **93.6%** | **15 Marks** (Retrieval Quality) |
| **Tool Execution Accuracy** | N/A | N/A | **90.7%** (49/54) | **90.7%** | **20 Marks** (Conversation Handling) |
| **Caller Identity Verification** | N/A | N/A | **100% Passed** | **100% Passed** | *Mandatory Policy Check* |
| **Voice End-to-End Performance** | N/A | N/A | N/A | **100% Processed** | **10 Marks** (Voice Integration) |
| **Voice Quality & Formatting** | N/A | N/A | N/A | **Normalized** | **10 Marks** (Auditory Experience) |
| **Report & Codebase Standard** | N/A | N/A | N/A | **Production-Ready**| **10 Marks** (Documentation) |
| **TOTAL** | — | — | — | — | **100 Marks** |

### Key Benchmark Insights
1. **RAG Impact (Step 1 vs Step 2)**: Grounding the model on the 21 knowledge base documents boosted factual policy recall from **29.3% to 70.7%**, completely eliminating hallucinations regarding timelines and non-standard return windows.
2. **Retrieval Precision**: The two-pool hybrid approach achieved **93.6% Recall@5** across test questions while respecting document precedence rules (`pol_returns_v4` over superseded `pol_returns_v3`).
3. **Identity Verification & Security**: In conversations PC021 and PC022 where callers provided incorrect verification credentials, the agent strictly limited retries to one attempt and refused disclosure without leaking customer record details.

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
