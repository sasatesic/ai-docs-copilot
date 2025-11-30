# ingestion_service/ingest.py

from pathlib import Path
from typing import List, Dict
import asyncio

from api_service.config import get_settings
from api_service.clients.vector_store_client import VectorStoreClient
from ingestion_service.embeddings import embed_texts, EMBEDDING_DIM
from ingestion_service.chunking import chunk_text
# NEW IMPORTS
from ingestion_service.parsers import parse_pdf, parse_docx, parse_pptx, parse_xlsx

DOCS_DIR = Path("data/docs")

def load_documents() -> List[Dict]:
    """
    Load .md, .txt, .pdf, .docx, .pptx, and .xlsx files from data/docs.
    """
    docs: List[Dict] = []
    
    # We'll search for all supported extensions
    extensions = ["*.md", "*.txt", "*.pdf", "*.docx", "*.pptx", "*.xlsx"]
    
    files_found = []
    for ext in extensions:
        files_found.extend(DOCS_DIR.glob(f"**/{ext}"))

    print(f"Found {len(files_found)} files to ingest.")

    for path in files_found:
        try:
            text = ""
            # Choose the correct parser based on suffix
            if path.suffix.lower() in [".md", ".txt"]:
                text = path.read_text(encoding="utf-8")
            elif path.suffix.lower() == ".pdf":
                text = parse_pdf(path)
            elif path.suffix.lower() == ".docx":
                text = parse_docx(path)
            elif path.suffix.lower() == ".pptx":
                text = parse_pptx(path)
            elif path.suffix.lower() == ".xlsx":
                text = parse_xlsx(path)
            
            if not text.strip():
                print(f"Skipping empty file: {path.name}")
                continue

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
        except Exception as e:
            print(f"Error parsing {path.name}: {e}")

    return docs

# ... (Rest of the file: main() function remains the same as previous step)
async def main() -> None:
    settings = get_settings()
    vector_store = VectorStoreClient(settings, collection_name="docs")

    docs = load_documents()
    if not docs:
        print("No documents found in data/docs")
        return

    texts = [d["text"] for d in docs]
    metadatas = [d["meta"] for d in docs]

    print(f"Loaded {len(texts)} chunks. Creating embeddings...")
    embeddings = await embed_texts(texts, settings=settings)

    print("Ensuring Qdrant collection exists...")
    await vector_store.ensure_collection(vector_size=EMBEDDING_DIM)

    print("Upserting embeddings into Qdrant...")
    await vector_store.upsert_embeddings(embeddings, texts, metadatas)

    print("Done. Ingested chunks into Qdrant.")


if __name__ == "__main__":
    asyncio.run(main())