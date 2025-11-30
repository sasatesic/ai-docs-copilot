# AI Docs Copilot 🤖📚

**AI Docs Copilot** is a production-grade **Retrieval-Augmented Generation (RAG)** system designed to chat with enterprise documents.

It goes beyond basic RAG by implementing **Hybrid Search (Dense + Sparse)** with **Reciprocal Rank Fusion (RRF)** and **Context Re-ranking**, ensuring high-precision retrieval even for complex or specific queries. The system features a modern, streaming **Next.js** frontend and a fully asynchronous **FastAPI** backend.

-----

## 🚀 Key Features

  * **Multi-Modal Ingestion**: Supports parsing and chunking for **PDF**, **Excel (.xlsx)**, **PowerPoint (.pptx)**, **Word (.docx)**, Markdown, and Text files.
  * **Advanced Retrieval Pipeline**:
      * **Hybrid Search**: Combines **Dense Vector Search** (OpenAI Embeddings) and **Sparse Keyword Search** (BM25-style) using **Reciprocal Rank Fusion (RRF)**.
      * **Context Re-ranking**: Uses **Cohere Rerank v3** to filter and order retrieved documents, significantly reducing hallucinations.
      * **Faceted Search**: Metadata filtering capabilities to scope queries to specific documents.
  * **Streaming Responses**: Real-time token streaming (Server-Sent Events) for a ChatGPT-like user experience.
  * **Robust Ingestion**:
      * **Recursive Character Splitting**: Preserves semantic structure (paragraphs/sentences) during chunking.
      * **Asynchronous Architecture**: Fully async ingestion and API handling for high throughput.
  * **Modern Stack**:
      * **Backend**: Python, FastAPI, Qdrant (Vector DB).
      * **Frontend**: Next.js 14 (App Router), TypeScript, Tailwind CSS.
      * **Infrastructure**: Fully containerized with Docker Compose.

-----

## 🛠️ Tech Stack

### Backend & AI

  * **Language**: Python 3.11+
  * **Framework**: FastAPI (Async)
  * **Vector Database**: Qdrant
  * **LLM**: OpenAI GPT-4o-mini (via `AsyncOpenAI`)
  * **Embeddings**: OpenAI `text-embedding-3-small`
  * **Re-ranking**: Cohere Rerank API
  * **Testing**: Pytest (with extensive async mocking)

### Frontend

  * **Framework**: Next.js 14 (App Router)
  * **Language**: TypeScript
  * **Styling**: Tailwind CSS
  * **Icons**: Lucide React

-----

## ⚡ Quick Start (Docker)

The easiest way to run the entire stack (Database, Backend, Frontend) is with Docker Compose.

### 1\. Prerequisites

  * Docker & Docker Compose
  * API Keys for **OpenAI** and **Cohere**

### 2\. Configuration

Create a `.env` file in the project root:

```ini
# .env
OPENAI_API_KEY="sk-..."
COHERE_API_KEY="..."   # Optional, but recommended for re-ranking
QDRANT_HOST="qdrant"   # Service name in Docker
QDRANT_PORT=6333
APP_ENV="docker"
LOG_LEVEL="INFO"
```

### 3\. Build and Run

```bash
docker compose -f infra/docker-compose.yml up --build -d
```

### 4\. Access the App

  * **Frontend (Chat UI)**: [http://localhost:3000](https://www.google.com/search?q=http://localhost:3000)
  * **Backend API Docs**: [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)
  * **Qdrant Dashboard**: [http://localhost:6333/dashboard](https://www.google.com/search?q=http://localhost:6333/dashboard)

-----

## 📂 Data Ingestion

The system includes a robust ingestion service to parse, chunk, and index your documents.

### Automatic Ingestion (Docker)

You can trigger the ingestion service via Docker. It processes all files in `data/docs/`.

```bash
# Run the ingestion profile (starts API, Qdrant, and Ingester)
docker compose -f infra/docker-compose.yml --profile ingest up
```

### Manual Ingestion (Local)

If running locally without Docker:

```bash
# 1. Install dependencies
pipenv install

# 2. Place files in data/docs/
# (Supports .pdf, .docx, .pptx, .xlsx, .md, .txt)

# 3. Run the script
python -m ingestion_service.ingest
```

-----

## 🧪 Running Tests

The project includes a comprehensive test suite covering the API endpoints, RAG logic (mocked), and chunking algorithms.

```bash
# Install test dependencies
pipenv install --dev

# Run all tests
pipenv run pytest
```

-----

## 📂 Project Structure

```text
ai-docs-copilot/
├── api_service/             # FastAPI Backend
│   ├── clients/             # Wrappers for OpenAI, Qdrant, Cohere
│   ├── models/              # Pydantic data models
│   ├── services/            # Core RAG logic (search, fusion, streaming)
│   └── main.py              # API Endpoints
├── ingestion_service/       # Data Pipeline
│   ├── ingest.py            # Main ingestion script
│   ├── parsers.py           # File parsers (PDF, Excel, etc.)
│   ├── chunking.py          # Recursive chunking logic
│   └── embeddings.py        # Embedding generation
├── frontend/                # Next.js Frontend
│   ├── src/app/page.tsx     # Chat Interface logic
│   └── next.config.mjs      # API Proxy configuration
├── infra/                   # Docker configuration
├── tests/                   # Unit tests
└── data/docs/               # Document storage
```

-----

## 🧠 How Hybrid Search (RRF) Works

1.  **Dense Retrieval**: The user query is embedded and compared against chunk vectors in Qdrant (Semantic Search).
2.  **Sparse Retrieval**: The query is treated as keywords to match exact terms in the document text (Keyword Search).
3.  **Reciprocal Rank Fusion (RRF)**: The system combines the rankings from both searches using the formula `score = 1 / (k + rank)`. This ensures that documents ranking high in *either* method (or both) bubble to the top.
4.  **Re-ranking**: The top `K` fused results are sent to the **Cohere Rerank model**, which uses a cross-encoder to score deep semantic relevance, selecting the final `N` chunks for the LLM.