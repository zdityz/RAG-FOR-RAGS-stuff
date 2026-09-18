from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retrieve import advanced_search
from agents.planner import generate_sub_queries
from agents.synthesizer import generate_answer_with_citations
from agents.verifier import verify_answer
from .config import settings
from .logger import logger
from .pipeline import RAGPipeline

app = FastAPI(title="RAG Copilot API", version="1.0")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    sub_queries: list[str]
    answer: str
    verified: bool
    verification: str
    verification_reason: str | None = None
    sources: list[dict]

@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        pipeline = RAGPipeline()
        result = pipeline.run(request.query)
        logger.info("--- REQUEST COMPLETE ---")
        return result
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=503, detail="Internal server error")