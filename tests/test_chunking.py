# tests/test_chunking.py

import pytest
from ingestion_service.chunking import chunk_text

# Define a base document with clear structural separators
TEST_DOCUMENT = """
# Header 1: Introduction

This is the first paragraph. It is relatively short, containing a key fact about RAG performance.

The goal of this test is to verify the recursive splitting strategy. This second paragraph is long and designed to force a split in a way that respects sentence boundaries. We must ensure the period is always preferred. This is critical for high-quality embedding results.

# Header 2: Conclusion
The final word.
"""

def test_chunking_honors_max_chars():
    max_chars = 50
    chunks = chunk_text(TEST_DOCUMENT, max_chars=max_chars, overlap=0)
    assert all(len(chunk) <= max_chars for chunk in chunks)
    assert len(chunks) > 5

def test_chunking_preserves_sentence_integrity():
    # Force a split in the middle of the long paragraph
    max_chars = 100 
    chunks = chunk_text(TEST_DOCUMENT, max_chars=max_chars, overlap=0)
    
    # We look for the sentence that would likely be cut by a simple splitter
    target_sentence = "This is critical for high-quality embedding results."
    found_integrity = any(target_sentence in chunk for chunk in chunks)
    assert found_integrity, "Chunking failed to preserve the full sentence intact."

def test_chunking_handles_overlap():
    # Continuous string to test pure length-based splitting + overlap behavior
    long_text = "A long string that will definitely need to be split multiple times to test overlap. " * 5
    max_chars = 50
    overlap = 15
    
    chunks = chunk_text(long_text, max_chars=max_chars, overlap=overlap)
    
    assert len(chunks) >= 2
    
    # Verify overlap
    chunk1 = chunks[0]
    chunk2 = chunks[1]
    
    # The tail of chunk1 should be present at the start of chunk2
    overlap_segment = chunk1[-overlap:].strip()
    
    # Debug print if it fails
    assert overlap_segment in chunk2, \
        f"Overlap failed.\nChunk 1 tail: '{overlap_segment}'\nChunk 2: '{chunk2}'"

def test_chunking_handles_empty_input():
    assert chunk_text("", max_chars=500, overlap=100) == []