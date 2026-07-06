# 🧠 IntelliDoc RAG — Production AI Document Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-DC143C?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F55036?style=for-the-badge)
![Jina AI](https://img.shields.io/badge/Jina_AI-Embeddings-9B59B6?style=for-the-badge)

**Upload PDFs & DOCX files. Ask questions. Get grounded answers with cited sources.**

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Tech Stack](#-tech-stack)

</div>

---
## 🎬 Demo

![](demo-video/VIDEO_e38478a9-a564-4d7d-bb65-74d0fdeb151d.mp4)

## 🎯 What is IntelliDoc RAG?

IntelliDoc RAG is a **production-ready Retrieval-Augmented Generation (RAG)** system that lets you chat with your own documents. Upload PDFs or Word files, and the AI answers your questions by retrieving the most relevant passages and generating grounded, cited responses — not hallucinations.

> Built with **FastAPI + LangChain + Qdrant + Jina AI + Groq**, deployable on any machine without a GPU.

---

## ✨ Features

### 📥 Document Ingestion
- **PDF & DOCX** upload with rich metadata extraction (title, author, page count, file size)
- **URL-based upload** — paste a link to any PDF and it downloads + indexes automatically
- **Deduplication** — already-processed files are skipped (hash-based on backend, name-based on frontend)
- **Table extraction** — tables are extracted separately and tagged with `[TABLE]` for precise citation

### 🔍 Retrieval Pipeline
- **Hybrid search** — Semantic (Qdrant vector search) + Keyword (BM25) fused with **Reciprocal Rank Fusion (RRF)**
- **Reranking** — Jina Reranker v1 reorders results by relevance
- **Metadata filtering** — filter by source file, content type, or page range
- **Query Enhancement** — rule-based rewrite + LLM query expansion + **HyDE** (Hypothetical Document Embedding)

### 🧠 LLM & Memory
- **Provider-agnostic** — swap between Groq, OpenAI, Google Gemini, or OpenRouter via `.env`
- **Conversation memory** — last 10 messages (5 exchanges) per session
- **Greeting detection** — small talk handled separately without hitting the vector DB
- **Out-of-scope refusal** — answers only from retrieved context, never hallucinates

### ⚡ Performance
- **Client-side query cache** (1-hour TTL) — repeated questions answered instantly without API calls
- **Batched embedding upload** — 50 chunks per batch to avoid Qdrant timeouts
- **Recursive chunking** (primary) + **Semantic chunking** (fallback for large docs)

### 🎨 Professional Frontend
- Pure HTML/CSS/JS — no framework, no build step, runs in any browser
- Dark glassmorphism design with micro-animations
- **Collapsible source citations** — sources shown only when the user clicks to expand
- Inline `[Source: ...]` patterns stripped from answers automatically
- Drag & drop upload, live status indicators, progress bar

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (index.html)                  │
│  Upload Zone │ URL Input │ Chat UI │ Collapsible Sources │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (FormData / JSON)
                       ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend  (port 8000)                │
│  POST /upload  │  POST /upload-url  │  POST /chat       │
│  POST /clear   │  GET  /health                          │
└──────┬───────────────────────────────────┬──────────────┘
       │                                   │
       ▼                                   ▼
┌──────────────────┐             ┌─────────────────────────┐
│  Ingestion Layer │             │      ChatEngine          │
│  PDF (PyMuPDF +  │             │  ┌─────────────────┐    │
│  pdfplumber)     │             │  │  Memory         │    │
│  DOCX (python-   │──chunks──▶  │  │  (per-session)  │    │
│  docx)           │             │  ├─────────────────┤    │
│  URL Downloader  │             │  │  RetrievalPipe  │    │
└──────────────────┘             │  │  ┌───────────┐  │    │
                                 │  │  │ HybridRet │  │    │
┌──────────────────┐             │  │  │ (Qdrant + │  │    │
│  Chunking        │             │  │  │  BM25/RRF)│  │    │
│  Recursive +     │             │  │  └─────┬─────┘  │    │
│  Semantic FBack  │             │  │  JinaRe│ranker  │    │
└──────────────────┘             │  └────────┼────────┘    │
                                 │           │              │
┌──────────────────┐             │  ┌────────▼────────┐    │
│  Jina Embeddings │             │  │  LLMFactory     │    │
│  v2-base-en      │             │  │  (Groq/OpenAI/  │    │
│  768 dimensions  │             │  │   Gemini etc.)  │    │
└──────────────────┘             │  └─────────────────┘    │
                                 └─────────────────────────┘
                                           │
                                  ┌────────▼────────┐
                                  │  Qdrant Cloud   │
                                  │  (Vector Store) │
                                  └─────────────────┘
```

---

## 📁 Project Structure

```
intellidoc-rag/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI entry point + static files
│   │   └── routes.py          # /upload, /upload-url, /chat, /clear, /health
│   ├── chat_engine.py         # Orchestrator (ChatEngine singleton)
│   ├── chunking/
│   │   └── splitters.py       # Recursive + Semantic chunking
│   ├── config/
│   │   └── settings.py        # Pydantic settings from .env
│   ├── embeddings/
│   │   └── jina_embeddings.py # Jina API or local fallback
│   ├── ingestion/
│   │   ├── extractors.py      # PDF (PyMuPDF + pdfplumber) / DOCX extractors
│   │   ├── loaders.py         # Dispatch to correct extractor
│   │   ├── metadata.py        # Build LangChain Documents with metadata
│   │   └── url_loader.py      # Download and ingest from URLs
│   ├── llm/
│   │   └── providers.py       # LLMFactory (OpenAI, Groq, Gemini, OpenRouter)
│   ├── memory/
│   │   └── chat_memory.py     # Per-session conversation memory
│   ├── prompts/
│   │   ├── system.py          # System prompts + greeting detection
│   │   └── templates.py       # ChatPromptTemplate with MessagesPlaceholder
│   ├── query/
│   │   └── rewrite.py         # Rule rewrite + Query Expansion + HyDE
│   ├── retrieval/
│   │   ├── filters.py         # Metadata filters (source, type, page range)
│   │   ├── hybrid.py          # BM25Retriever + HybridRetriever + RRF
│   │   └── reranker.py        # Jina Reranker (API or local)
│   ├── utils/
│   │   ├── cache.py           # In-memory QueryCache with TTL
│   │   ├── helpers.py         # clean_text, format_source
│   │   └── logging.py         # Structured logging setup
│   └── vectordb/
│       └── qdrant_client.py   # Qdrant Cloud: upsert (batched) + query_points
├── frontend/
│   └── index.html             # Premium single-file UI
├── tests/
│   └── test_config.py         # Sanity tests
├── uploads/                   # Auto-created, gitignored
├── logs/                      # Auto-created, gitignored
├── .env.example               # Template for required secrets
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A free [Qdrant Cloud](https://cloud.qdrant.io) cluster (takes 2 min to set up)
- A [Groq API key](https://console.groq.com) (free tier available)
- A [Jina AI API key](https://jina.ai) (free tier: 1M tokens)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/intellidoc-rag.git
cd intellidoc-rag

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your real API keys
```

**.env** (fill in your values):
```env
# Required
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=gsk_your_groq_api_key
JINA_API_KEY=jina_your_jina_api_key

# LLM Selection
DEFAULT_LLM_PROVIDER=groq
DEFAULT_LLM_MODEL=llama-3.1-8b-instant

# Optional providers
OPENAI_API_KEY=
GOOGLE_API_KEY=
OPENROUTER_API_KEY=

# App Config
QDRANT_COLLECTION_NAME=rag_documents
UPLOAD_DIR=uploads
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=10
TOP_K_RERANK=5
APP_HOST=0.0.0.0
APP_PORT=8000
```

### 3. Run the Backend

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
🚀 RAG API starting...
LLM ready: groq/llama-3.1-8b-instant
Qdrant ready: https://your-cluster.cloud.qdrant.io
✅ All systems operational
```

### 4. Open the Frontend

```
http://localhost:8000/static/index.html
```

Or serve separately:
```bash
cd frontend && python -m http.server 8502
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload PDF or DOCX (`multipart/form-data`, field: `file`) |
| `POST` | `/upload-url` | Upload from URL (`form`, field: `url`) |
| `POST` | `/chat` | Chat (`form`, fields: `message`, `session_id`) |
| `POST` | `/clear` | Clear session history (`form`, field: `session_id`) |
| `GET` | `/docs` | Interactive Swagger UI |

### Upload Example
```bash
curl -X POST -F "file=@./document.pdf" http://localhost:8000/upload
```

### Chat Example
```bash
curl -X POST \
  -F "message=What are the main topics?" \
  -F "session_id=session1" \
  http://localhost:8000/chat
```

### Chat Response
```json
{
  "answer": "The document covers three main topics...",
  "sources": [
    { "source": "document.pdf", "page": 3, "type": "text" }
  ],
  "is_greeting": false
}
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI + Uvicorn | Async HTTP server |
| **Orchestration** | LangChain 0.2 | Chains, prompt templates, output parsers |
| **Embeddings** | Jina AI `jina-embeddings-v2-base-en` | 768-dim embeddings, no GPU needed |
| **Vector DB** | Qdrant Cloud | Vector similarity search |
| **Keyword Search** | rank-bm25 + NLTK | BM25 retrieval |
| **Reranker** | Jina AI `jina-reranker-v1-base-en` | Cross-encoder reranking |
| **LLM** | Groq (LLaMA 3.1 8B) | Fast inference, free tier |
| **PDF Extraction** | PyMuPDF + pdfplumber | Text + tables with page numbers |
| **DOCX Extraction** | python-docx | Word document parsing |
| **Frontend** | Vanilla HTML/CSS/JS | Zero-dependency single-file UI |
| **Settings** | Pydantic v2 Settings | Type-safe environment variables |

---

## 🔑 LLM Provider Switching

Edit two lines in `.env` to switch providers:

```env
# Groq (default — fastest free option)
DEFAULT_LLM_PROVIDER=groq
DEFAULT_LLM_MODEL=llama-3.1-8b-instant

# OpenAI
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4o-mini

# Google Gemini
DEFAULT_LLM_PROVIDER=google
DEFAULT_LLM_MODEL=gemini-1.5-flash

# OpenRouter (100+ models)
DEFAULT_LLM_PROVIDER=openrouter
DEFAULT_LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 🔒 Recommended .gitignore

```gitignore
.env
uploads/
logs/
venv/
__pycache__/
*.pyc
*.pyo
.processed_files.json
*.egg-info/
dist/
.pytest_cache/
```

---

## 🗺️ Roadmap

- [ ] LangSmith tracing integration
- [ ] Multi-user session isolation
- [ ] Persistent chat history (Redis / SQLite)
- [ ] OCR support for scanned PDFs (Tesseract)
- [ ] Streaming responses via Server-Sent Events
- [ ] Docker + docker-compose one-command deployment
- [ ] RAG evaluation pipeline (RAGAS)

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Built with ❤️ using FastAPI, LangChain, Qdrant, and Jina AI
</div>
