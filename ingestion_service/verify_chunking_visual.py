from ingestion_service.chunking import chunk_text

def verify_logic():
    print("="*60)
    print("TEST 1: OVERLAP VERIFICATION")
    print("="*60)
    
    # A generic repeating string to test pure length-based overlap
    text = "A long string that will definitely need to be split multiple times to test overlap. " * 3
    max_chars = 50
    overlap = 15
    
    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
    
    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i}] (Length: {len(chunk)})")
        print(f"'{chunk}'")
        
    print("\n--- Overlap Analysis ---")
    if len(chunks) >= 2:
        c0 = chunks[0]
        c1 = chunks[1]
        
        # Extract the tail of chunk 0 that SHOULD overlap
        tail_c0 = c0[-overlap:].strip()
        # Extract the head of chunk 1 that SHOULD contain the overlap
        head_c1 = c1[:len(tail_c0) + 5].strip() # +5 buffer for context
        
        print(f"End of Chunk 0:   '...{tail_c0}'")
        print(f"Start of Chunk 1: '{head_c1}...'")
        
        if tail_c0 in c1:
            print("\n✅ SUCCESS: Overlap confirmed. The tail of Chunk 0 exists in Chunk 1.")
        else:
            print("\n❌ FAILURE: No overlap found.")

    print("\n" + "="*60)
    print("TEST 2: SEMANTIC STRUCTURE VERIFICATION")
    print("="*60)
    
    # A text with sentences that are slightly longer than the overlap but shorter than max_chars
    structured_text = (
        "This is sentence one. "
        "This is sentence two. "
        "This is sentence three. "
        "This is sentence four."
    )
    
    # max_chars=40 forces a split roughly every 2 sentences.
    # overlap=10 ensures we don't lose context.
    chunks_struct = chunk_text(structured_text, max_chars=40, overlap=10)
    
    for i, chunk in enumerate(chunks_struct):
        print(f"\n[Chunk {i}]")
        print(f"'{chunk}'")

    print("\n--- Logic Check ---")
    print("Did we split at periods? Or did we cut a word in half?")
    print("With recursive splitting, we expect splits at '. ' boundaries.")

if __name__ == "__main__":
    verify_logic()