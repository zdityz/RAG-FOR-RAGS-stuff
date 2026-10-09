# RAG-FOR-RAGS

A production-grade, multi-agent **Retrieval-Augmented Generation (RAG)** copilot built with semantic chunking, hybrid BM25 + dense vector retrieval, cross-encoder reranking, and self-verification.

---

## ⚡ 1-Minute Quick Start

### 1. One-Command Setup
Clone the repo and run the automated setup script:
```bash
./setup.sh
```
This script automatically:
* Creates and activates a Python virtual environment (`.venv`)
* Installs all dependencies
* Creates a default `.env` configuration file
* Populates the vector database and BM25 index with `data/sample.pdf`

---

## 🚀 Running the Project

Use the unified launcher `run.py`:

### Option A: Interactive Terminal Chat (CLI)
Chat directly with the RAG copilot in your terminal with conversation memory:
```bash
python run.py cli
```
* Type `clear` to reset chat context.
* Type `exit` to quit.

### Option B: FastAPI Web Server
Start the REST API with Swagger documentation and SSE streaming:
```bash
python run.py api
```
* **API Server:** `http://localhost:8000`
* **Interactive Docs & Swagger:** `http://localhost:8000/docs`

---

## ⚙️ Configuration (`.env`)

Copy `.env.example` to `.env` to configure your LLM provider.

### Local LLM (Ollama)
```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3
```

### Cloud LLM (Groq — Free & Ultra-Fast)
```env
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_your_groq_api_key
LLM_MODEL=llama-3.1-8b-instant
```

### Cloud LLM (OpenAI)
```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-proj-your_key
LLM_MODEL=gpt-4o-mini
```

---

## 🛠️ Management & Utilities

| Command | Description |
|---|---|
| `python run.py cli` | Start interactive conversation CLI |
| `python run.py api` | Start FastAPI backend server |
| `python run.py ingest` | Wipe ChromaDB and re-index all chunks |
| `python run.py eval` | Run ground-truth automated evaluation |
| `python run.py test` | Run pytest test suite |

---

## 📡 API Endpoints

All authenticated routes require `Authorization: Bearer <API_KEY>` (default: `super-secret-token` in `.env`).

* `POST /query`: Synchronous RAG query with session memory.
* `POST /query/stream`: Server-Sent Events (SSE) streaming progress and tokens.
* `POST /ingest`: Upload and ingest new `.pdf`, `.txt`, or `.md` files.
* `GET /documents`: List ingested documents.
* `DELETE /documents/{id}`: Delete an ingested document.
* `GET /health`: Health check (no auth required).
