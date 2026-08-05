import sys
from retrieve import advanced_search
from agents.planner import generate_sub_queries
from agents.synthesizer import generate_answer_with_citations
from agents.verifier import verify_answer

def main():
    print("Testing Full Multi-Agent RAG Pipeline...")
    test_query = input("\nEnter a question: ").strip()
    
    if not test_query:
        sys.exit(0)
        
    print("\n1. Planner Agent breaking down query...")
    sub_queries = generate_sub_queries(test_query)
    for i, sq in enumerate(sub_queries):
        print(f"   Sub-query {i+1}: {sq}")
        
    print("\n2. Retrieving & Reranking chunks for all queries...")
    all_results = []
    seen_ids = set()
    
    for sq in sub_queries:
        sq_results = advanced_search(sq, top_k=2)
        for res in sq_results:
            if res['id'] not in seen_ids:
                seen_ids.add(res['id'])
                all_results.append(res)
                
    print(f"   Total unique chunks retrieved: {len(all_results)}")
    
    print("\n3. Synthesizer Agent drafting answer...")
    answer = generate_answer_with_citations(test_query, all_results)
    print("\n--- DRAFT ANSWER ---")
    print(answer)
    
    print("\n4. Verifier Agent checking for hallucinations...")
    verification = verify_answer(test_query, answer, all_results)
    
    print("\n--- VERIFICATION RESULT ---")
    print(verification)

if __name__ == "__main__":
    main()