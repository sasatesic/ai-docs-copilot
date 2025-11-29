# ingestion_service/chunking.py

from typing import List

# Define separators from largest structure to smallest
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

def chunk_text(
    text: str,
    max_chars: int = 1000,
    overlap: int = 200,
) -> List[str]:
    """
    Recursively splits text into small meaningful pieces (preserving separators),
    then merges them into chunks with overlap.
    """
    if not text:
        return []

    # --- 1. Recursive Splitter (Produces atomic pieces with separators attached) ---
    def _split_atomic(text_segment: str, separators: List[str]) -> List[str]:
        # Base case: no more separators, or text is already small enough
        if not separators:
            return [text_segment]
        
        sep = separators[0]
        next_seps = separators[1:]
        
        if sep == "": # Character split fallback
            return [c for c in text_segment]
            
        if sep not in text_segment:
            return _split_atomic(text_segment, next_seps)
            
        # Split and keep separators attached to the previous segment
        # e.g. "Hello world" -> ["Hello ", "world"]
        parts = text_segment.split(sep)
        final_pieces = []
        
        for i, part in enumerate(parts):
            # Re-attach separator to all but the last part
            if i < len(parts) - 1:
                part_with_sep = part + sep
            else:
                part_with_sep = part
                
            if len(part_with_sep) > max_chars:
                # If this piece is still too big, recurse deeper
                final_pieces.extend(_split_atomic(part_with_sep, next_seps))
            else:
                final_pieces.append(part_with_sep)
                
        return final_pieces

    # Get all the small atomic pieces (sentences, words, etc.)
    pieces = _split_atomic(text, SEPARATORS)
    
    # --- 2. Merger with Overlap ---
    chunks = []
    current_buffer = []
    current_len = 0
    
    for piece in pieces:
        piece_len = len(piece)
        
        # If adding this piece exceeds limits, finalize the current chunk
        if current_len + piece_len > max_chars and current_buffer:
            # Emit the chunk
            full_chunk = "".join(current_buffer).strip()
            if full_chunk:
                chunks.append(full_chunk)
            
            # --- BACKTRACK FOR OVERLAP ---
            # Keep pieces from the end of the buffer until we satisfy the overlap length
            overlap_buffer = []
            overlap_len = 0
            
            # Walk backwards through the current buffer
            for p in reversed(current_buffer):
                if overlap_len < overlap:
                    overlap_buffer.insert(0, p) # Add to front to maintain order
                    overlap_len += len(p)
                else:
                    break
            
            # Start the new chunk with the overlapping content
            current_buffer = list(overlap_buffer)
            current_len = overlap_len
            
        current_buffer.append(piece)
        current_len += piece_len
        
    # Add the final chunk
    if current_buffer:
        full_chunk = "".join(current_buffer).strip()
        if full_chunk:
            chunks.append(full_chunk)
            
    return chunks