# ingestion_service/chunking.py

from typing import List


def chunk_text(
    text: str,
    max_chars: int = 1000,
    overlap: int = 200,
) -> List[str]:
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # If we've reached the end, stop to avoid infinite loop on short texts
        if end >= length:
            break

        # Move start forward with overlap
        start = max(end - overlap, 0)

    return chunks
