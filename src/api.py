from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retrieve import advanced_search
from agents.planner import generate_sub_queries
from agents.synthesizer import generate_answer_with_citations
from agents.verifier import verify_answer

app = FastAPI(title="RAG Copilot API", version="1.0")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    sub_queries: list[str]
    answer: str
    verification: str
    sources: list[dict]

@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    print(f"\n--- NEW REQUEST ---")
    print(f"Query: {request.query}")
    
    print("1. Planner Agent running...")
    sub_queries = generate_sub_queries(request.query)
    print(f"   Sub-queries generated: {sub_queries}")
    
    print("2. Retrieving chunks...")
    all_results = []
    seen_ids = set()
    for sq in sub_queries:
        sq_results = advanced_search(sq, top_k=2)
        for res in sq_results:
            if res['id'] not in seen_ids:
                seen_ids.add(res['id'])
                all_results.append(res)
    print(f"   Total unique chunks retrieved: {len(all_results)}")
                
    print("3. Synthesizer Agent running...")
    answer = generate_answer_with_citations(request.query, all_results)
    print("   Synthesis complete.")
    
    print("4. Verifier Agent running...")
    verification = verify_answer(request.query, answer, all_results)
    print("   Verification complete.")
    
    sources = [
        {
            "doc_id": i + 1,
            "page": res['metadata'].get('page'),
            "source": res['metadata'].get('source'),
            "score": round(res.get('cross_score', 0.0), 2)
        }
        for i, res in enumerate(all_results)
    ]
    
    print("--- REQUEST COMPLETE ---\n")
    
    return {
        "query": request.query,
        "sub_queries": sub_queries,
        "answer": answer,
        "verification": verification,
        "sources": sources
    }