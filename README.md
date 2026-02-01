# EndeeSync

A local-first RAG-based semantic memory engine built with the Endee vector database and FastAPI. EndeeSync indexes notes using sentence embeddings and performs low-latency semantic retrieval with metadata filtering, then uses an LLM API for generation.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

- **Semantic Ingestion** — Chunk text with overlap, generate embeddings via `all-MiniLM-L6-v2`, and store in a local vector index
- **Low-Latency Retrieval** — Sub-10ms vector search using Endee's in-memory index with metadata filtering (tags, date ranges)
- **RAG Q&A** — Retrieve relevant chunks, construct prompts, and generate answers via configurable LLM providers (OpenAI, Groq, Anthropic)
- **Summarization** — Topic-based synthesis across multiple chunks with controllable output length
- **Local-First** — All data persists on disk; no external databases required
- **Tag Filtering** — Filter ingestion and retrieval by user-defined tags

---

## Architecture

### System Overview

```
┌──────────┐     ┌──────────────┐     ┌─────────────────┐     ┌───────────┐
│          │     │              │     │                 │     │           │
│   User   │────▶│   Frontend   │────▶│    FastAPI      │────▶│   Endee   │
│          │     │   (Web UI)   │     │    Backend      │     │  Vector   │
│          │◀────│              │◀────│                 │◀────│   Store   │
│          │     │              │     │        │        │     │           │
└──────────┘     └──────────────┘     └────────┼────────┘     └───────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │    LLM API      │
                                      │ (OpenAI/Groq)   │
                                      └─────────────────┘
```

### Ingestion Flow

```
┌────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌───────────┐
│  Text  │───▶│ Chunker │───▶│ Embedder │───▶│ Indexer │───▶│   Endee   │
│  Note  │    │ (400t)  │    │ MiniLM   │    │         │    │  (Store)  │
└────────┘    └─────────┘    └──────────┘    └─────────┘    └───────────┘
```

### Query Flow (RAG)

```
┌──────────┐    ┌──────────┐    ┌───────────┐    ┌─────────┐    ┌──────────┐
│ Question │───▶│ Embedder │───▶│   Endee   │───▶│ Context │───▶│   LLM    │
│          │    │          │    │  Search   │    │ Builder │    │ Generate │
└──────────┘    └──────────┘    └───────────┘    └─────────┘    └──────────┘
                                     │                               │
                                     ▼                               ▼
                              ┌────────────┐                  ┌────────────┐
                              │ Top-K Docs │                  │   Answer   │
                              │ + Scores   │                  │ + Sources  │
                              └────────────┘                  └────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.11+, FastAPI, Pydantic v2, Uvicorn |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **Vector Store** | Endee (custom file-based, NumPy-backed) |
| **LLM** | OpenAI API, Groq API, or Anthropic API |
| **Frontend** | Vanilla HTML/CSS/JavaScript (no frameworks) |
| **Deployment** | Docker, Docker Compose |

---

## Installation

### Prerequisites

- Python 3.11 or higher
- LLM API key (get one from [Groq](https://console.groq.com), [OpenAI](https://platform.openai.com), or [Anthropic](https://console.anthropic.com))

### Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/sujalkumar04/Endeesync.git
cd EndeeSync/backend
```

**2. Create virtual environment**

```bash
python -m venv venv
```

**3. Activate virtual environment**

Windows (PowerShell):
```powershell
.\venv\Scripts\Activate
```

Windows (CMD):
```cmd
venv\Scripts\activate.bat
```

macOS / Linux:
```bash
source venv/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Configure environment**
```bash
cp .env.example .env
```
Then edit `.env` and add your API key:
```
LLM_API_KEY=your-api-key-here
```

---

## Running the Application

### Start the Backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Access the App

| URL | Description |
|-----|-------------|
| http://localhost:8000 | Web Dashboard |
| http://localhost:8000/docs | API Documentation (Swagger) |
| http://localhost:8000/redoc | API Documentation (ReDoc) |

### Docker (Alternative)

```bash
# Set your API key
export LLM_API_KEY=your-api-key

# Run with Docker Compose
docker compose up --build
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with service status |
| `/api/v1/ingest` | POST | Ingest text note with tags |
| `/api/v1/ingest` | GET | List all documents |
| `/api/v1/ingest/{id}` | DELETE | Delete document by ID |
| `/api/v1/search` | POST | Semantic search with filters |
| `/api/v1/query` | POST | RAG-based Q&A |
| `/api/v1/summarize` | POST | Topic-based summarization |

### Example Requests

**Ingest a note:**
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "FastAPI is a modern Python web framework for building APIs.",
    "source": "tech_notes",
    "tags": ["python", "web"]
  }'
```

**Semantic search:**
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python web frameworks",
    "top_k": 5,
    "filters": {"tags": ["python"]}
  }'
```

**RAG query:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is FastAPI used for?",
    "top_k": 5,
    "include_sources": true
  }'
```

---

## Folder Structure

```
EndeeSync/
├── backend/
│   ├── app/
│   │   ├── api/           # Dependency injection
│   │   ├── core/          # Chunker, Embedder, Indexer
│   │   ├── db/            # Vector store interface
│   │   ├── models/        # Pydantic schemas
│   │   ├── routers/       # API route handlers
│   │   ├── services/      # Business logic (RAG, Ingest, Search)
│   │   ├── static/        # CSS, JS assets
│   │   ├── templates/     # HTML templates
│   │   ├── utils/         # Helpers, timing, ID generation
│   │   ├── config.py      # Settings management
│   │   └── main.py        # Application entry point
│   ├── data/              # Persisted vector store (gitignored)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── docs/                  # Documentation
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `400` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `LLM_PROVIDER` | `groq` | LLM provider (`openai`, `groq`, `anthropic`) |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Model identifier |
| `LLM_API_KEY` | — | API key for LLM provider |
| `VECTOR_STORE_PATH` | `data/endee/memories` | Storage path |

---

## Why Endee?

**Design Reasoning:**

1. **Zero External Dependencies** — No need for Qdrant, Pinecone, or Milvus. Endee uses NumPy arrays with memory-mapped files for persistence. This simplifies deployment and eliminates container orchestration complexity.

2. **Low Latency** — In-memory cosine similarity with NumPy achieves sub-10ms search times for collections under 100K vectors. For a personal notes system, this is more than sufficient.

3. **Transparent Storage** — Vectors and metadata are stored in human-readable JSON + binary NumPy format. Easy to debug, backup, and migrate.

4. **Metadata Filtering** — Native support for filtering by tags, source, and date ranges without secondary indexes.

5. **Learning Exercise** — Building a vector store from scratch provides deep understanding of embedding spaces, similarity search, and indexing strategies.

**Trade-offs:**
- Not suitable for production-scale (millions of vectors)
- No distributed queries or replication
- Linear scan for filtering (acceptable at small scale)

---

## Future Improvements

- [ ] Add HNSW indexing for faster approximate nearest neighbor search
- [ ] Implement document chunking with semantic boundaries (paragraphs, sections)
- [ ] Add support for PDF and Markdown file ingestion
- [ ] Implement conversation memory for multi-turn Q&A
- [ ] Add embedding model selection at runtime
- [ ] Build CLI tool for terminal-based ingestion
- [ ] Add rate limiting and authentication for multi-user deployment
- [ ] Implement hybrid search (BM25 + vector similarity)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

Built as a learning project to understand RAG pipelines, vector databases, and LLM integration.
