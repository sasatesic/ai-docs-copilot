# AI Docs Copilot

RAG-based FastAPI service with Qdrant vector DB and OpenAI embeddings.

## Quickstart

```bash
pipenv install
pipenv shell
docker compose -f infra/docker-compose.yml up qdrant
python -m ingestion_service.ingest
uvicorn api_service.main:app --reload
