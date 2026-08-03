import sys
from retrieve import advanced_search

def main():
    print("Testing Two-Stage Retrieval (Hybrid Search + Cross-Encoder Rerank)...")
    test_query = input("\nEnter a question to test retrieval: ").strip()
    
    if not test_query:
        print("Exiting...")
        sys.exit(0)
        
    results = advanced_search(test_query, top_k=3)
    
    print("\n--- TOP MATCHES (RERANKED) ---")
    for idx, res in enumerate(results):
        print(f"\n[Match {idx+1}] Score: {res['cross_score']:.2f} | Source: {res['metadata']['source']} | Page: {res['metadata']['page']}")
        print(f"Snippet: {res['text'][:300]}...")

if __name__ == "__main__":
    main()