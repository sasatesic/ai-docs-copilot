# ingestion_service/chunking.py

from typing import List

# Define the sequence of separators to try, from largest to smallest structural break
SEPARATORS = ["\n\n", "\n", ". ", " "] 


def chunk_text(
    text: str,
    max_chars: int = 1000,
    overlap: int = 200,
) -> List[str]:
    """
    Splits text recursively based on a list of separators to maintain logical structure.
    """
    def _split_recursively(text_to_split: str, separators: List[str]) -> List[str]:
        # --- Base Case ---
        if not separators:
            # Fallback to simple character split if no structured separators work
            chunks = []
            start = 0
            while start < len(text_to_split):
                end = min(start + max_chars, len(text_to_split))
                chunks.append(text_to_split[start:end].strip())
                # Sliding window overlap logic
                start += max_chars - overlap
            return [c for c in chunks if c]

        # --- Recursive Step ---
        separator = separators[0]
        sub_separators = separators[1:]
        
        # 1. Handle empty separator case (or fall through)
        if not separator:
            return _split_recursively(text_to_split, sub_separators)

        # 2. Split text by current separator
        parts = text_to_split.split(separator)
        
        final_chunks = []
        current_chunk = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Try to merge part into current chunk
            if current_chunk:
                # Add separator back in the calculation for accurate length
                potential_chunk = current_chunk + separator + part
            else:
                potential_chunk = part
            
            if len(potential_chunk) <= max_chars:
                current_chunk = potential_chunk
            else:
                # Current chunk is too big. Finalize the current_chunk (if exists)
                if current_chunk:
                    final_chunks.append(current_chunk)
                
                # If the current part itself is too big, recurse with the next separator
                if len(part) > max_chars:
                    recursive_chunks = _split_recursively(part, sub_separators)
                    final_chunks.extend(recursive_chunks)
                    current_chunk = ""
                else:
                    # Current part fits, but was blocked by max_chars limit previously
                    current_chunk = part # Start new chunk with the current part

        # Handle the last remaining chunk
        if current_chunk:
            final_chunks.append(current_chunk)

        # Apply cleanup and return
        return [c.strip() for c in final_chunks if c.strip()]


    return _split_recursively(text, SEPARATORS)