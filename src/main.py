import os
import sys
from ingest import extract_text_from_pdf, chunk_documents
from embed import store_chunks
from retrieve import basic_vector_search

def main():
    pdf_path = "./data/sample.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: Drop a test PDF into '{pdf_path}' before running this script.")
        sys.exit(1)
        
    print("1. Ingesting & Chunking PDF...")
    pages_data = extract_text_from_pdf(pdf_path)
    chunks = chunk_documents(pages_data, doc_name="sample.pdf")
    print(f"   -> Extracted {len(chunks)} chunks.")
    
    print("\n2. Embedding & Storing Chunks in ChromaDB...")
    store_chunks(chunks)
    
    print("\n3. Testing Vector Retrieval Loop...")
    test_query = input("\nEnter a question to test retrieval: ").strip()
    if not test_query:
        test_query = "What is the summary of this document?"
        print(f"Defaulting query to: '{test_query}'")
        
    results = basic_vector_search(test_query, top_k=3)
    
    print("\n--- TOP MATCHES ---")
    for idx, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n[Match {idx+1}] Source: {meta['source']} | Page: {meta['page']}")
        print(f"Snippet: {doc[:300]}...")

if __name__ == "__main__":
    main()