"""FastAPI application — Phase 4 complete.

Endpoints:
  POST /query              — standard synchronous RAG query
  POST /query/stream       — SSE streaming RAG query
  POST /ingest             — upload a PDF / .txt / .md and ingest it
  GET  /documents          — list ingested documents
  DELETE /documents/{id}   — remove a document from the knowledge base
  DELETE /sessions/{id}    — clear a conversation session

Authentication:
  All endpoints require  Authorization: Bearer <API_KEY>  header.
  API_KEY is read from env var (default: super-secret-token).
"""

import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Security,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .config import settings
from .logger import get_logger
from .pipeline import RAGPipeline
from .ingestion_service import ingestion_service
from . import session as session_store

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Copilot API",
    version="2.0",
    description="Retrieval-Augmented Generation API with streaming, session memory, and document management.",
)

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
_bearer_scheme = HTTPBearer(auto_error=True)


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(_bearer_scheme),
) -> str:
    if credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# Shorthand dependency alias
Auth = Depends(verify_token)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    sub_queries: List[str]
    answer: str
    verified: bool
    verification: str
    verification_reason: Optional[str] = None
    sources: List[dict]
    stage_latencies_ms: Optional[dict] = None
    session_id: Optional[str] = None


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse, dependencies=[Auth])
def handle_query(request: QueryRequest):
    """Synchronous RAG query with optional session memory."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        pipeline = RAGPipeline()
        result = pipeline.run(request.query, session_id=request.session_id)
        logger.info("Request complete — session=%s", request.session_id)
        return result
    except Exception:
        logger.exception("Error processing query")
        raise HTTPException(status_code=503, detail="Internal server error.")


@app.post("/query/stream", dependencies=[Auth])
def handle_query_stream(request: QueryRequest):
    """SSE streaming RAG query — yields progress, token, and done events."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    pipeline = RAGPipeline()

    def event_generator():
        try:
            yield from pipeline.stream(request.query, session_id=request.session_id)
        except Exception:
            logger.exception("Error during streaming query")
            import json
            yield f"event: error\ndata: {json.dumps({'detail': 'Internal server error'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------
@app.post("/ingest", response_model=DocumentInfo, dependencies=[Auth])
async def ingest_document(file: UploadFile = File(...)):
    """Upload a PDF, .txt, or .md file and ingest it into the knowledge base."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # Save to a temp file so the ingestion service can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        doc_name = Path(file.filename).stem
        entry = ingestion_service.ingest_file(tmp_path, doc_name=doc_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail="Ingestion failed.")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return entry


@app.get("/documents", response_model=List[DocumentInfo], dependencies=[Auth])
def list_documents():
    """List all documents currently in the knowledge base."""
    return ingestion_service.list_documents()


@app.delete("/documents/{doc_id}", dependencies=[Auth])
def delete_document(doc_id: str):
    """Remove a document and all its chunks from the knowledge base."""
    removed = ingestion_service.delete_document(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {"detail": f"Document '{doc_id}' deleted successfully."}


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
@app.delete("/sessions/{session_id}", dependencies=[Auth])
def clear_session(session_id: str):
    """Clear all conversation history for a session."""
    session_store.clear_session(session_id)
    return {"detail": f"Session '{session_id}' cleared."}


@app.get("/sessions", dependencies=[Auth])
def list_sessions():
    """List active session IDs (for debugging)."""
    return {"sessions": session_store.list_sessions()}


# ---------------------------------------------------------------------------
# Health check (no auth required)
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


# ---------------------------------------------------------------------------
# Frontend (Static Files)
# ---------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(str(static_dir / "index.html"))