# ingestion_service/ingest.py

from pathlib import Path
from typing import List, Dict

from api_service.config import get_settings
from api_service.clients.vector_store_client import VectorStoreClient
from ingestion_service.embeddings import embed_texts, EMBEDDING_DIM
from ingestion_service.chunking import chunk_text


DOCS_DIR = Path("data/docs")


def load_documents() -> List[Dict]:
    """
    Load all .md files from data/docs and return [{"text": ..., "meta": {...}}, ...]
    """
    docs: List[Dict] = []

    for path in DOCS_DIR.glob("**/*.md"):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            docs.append(
                {
                    "text": chunk,
                    "meta": {
                        "source_file": str(path),
                        "chunk_index": i,
                        "source_id": path.name,
                    },
                }
            )

    return docs


def main() -> None:
    settings = get_settings()
    vector_store = VectorStoreClient(settings, collection_name="docs")

    docs = load_documents()
    if not docs:
        print("No documents found in data/docs")
        return

    texts = [d["text"] for d in docs]
    metadatas = [d["meta"] for d in docs]

    print(f"Loaded {len(texts)} chunks. Creating embeddings...")
    embeddings = embed_texts(texts, settings=settings)

    print("Ensuring Qdrant collection exists...")
    vector_store.ensure_collection(vector_size=EMBEDDING_DIM)

    print("Upserting embeddings into Qdrant...")
    vector_store.upsert_embeddings(embeddings, texts, metadatas)

    print("Done. Ingested chunks into Qdrant.")


if __name__ == "__main__":
    main()
