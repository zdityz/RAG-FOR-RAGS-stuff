"""Document ingestion service — Phase 4.3.

Provides a thin service class that wires together:
  ingest.py  (PDF/text extraction + chunking)
  embed.py   (ChromaDB storage)
  retrieve.py BM25 cache rebuild

Also maintains a simple in-memory document manifest so the API can list
and delete ingested documents without extra DB queries.
"""
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings
from .embed import store_chunks
from .ingest import extract_text_from_pdf, chunk_documents
from .logger import get_logger
from .retrieve import BM25IndexCache

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Document manifest (in-memory; survives for the lifetime of the process)
# ---------------------------------------------------------------------------
# doc_id -> {filename, num_chunks, chunk_ids}
_manifest: Dict[str, Dict] = {}


def _load_text_file(path: str) -> List[Dict]:
    """Extract pages from a plain-text or markdown file."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    # Treat each 2000-char window as a pseudo-page so downstream chunking works
    page_size = 2000
    pages = []
    for i in range(0, len(text), page_size):
        pages.append({"text": text[i : i + page_size], "page": i // page_size + 1})
    return pages


class DocumentIngestionService:
    """Ingest a file, store it in ChromaDB, update BM25 cache and manifest."""

    def ingest_file(self, file_path: str, doc_name: Optional[str] = None) -> Dict:
        """Process *file_path* (PDF, .txt, .md) and persist chunks.

        Returns a manifest entry dict.
        """
        path = Path(file_path)
        doc_name = doc_name or path.stem
        suffix = path.suffix.lower()

        logger.info("Ingesting document: %s (type=%s)", doc_name, suffix)

        if suffix == ".pdf":
            pages = extract_text_from_pdf(str(path))
        elif suffix in (".txt", ".md"):
            pages = _load_text_file(str(path))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        chunks = chunk_documents(pages, doc_name=doc_name)
        if not chunks:
            raise ValueError(f"No chunks produced from {path.name}")

        store_chunks(chunks)

        # Rebuild BM25 after adding new documents
        BM25IndexCache.build()

        chunk_ids = [c["id"] for c in chunks]
        entry = {
            "doc_id": doc_name,
            "filename": path.name,
            "num_chunks": len(chunks),
            "chunk_ids": chunk_ids,
        }
        _manifest[doc_name] = entry
        logger.info("Ingested %d chunks for document '%s'", len(chunks), doc_name)
        return entry

    # ------------------------------------------------------------------
    def list_documents(self) -> List[Dict]:
        return [
            {"doc_id": k, "filename": v["filename"], "num_chunks": v["num_chunks"]}
            for k, v in _manifest.items()
        ]

    # ------------------------------------------------------------------
    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks for *doc_id* from ChromaDB and the manifest."""
        if doc_id not in _manifest:
            return False

        from .db import get_collection
        collection = get_collection()
        chunk_ids = _manifest[doc_id]["chunk_ids"]
        try:
            collection.delete(ids=chunk_ids)
            logger.info("Deleted %d chunks for document '%s'", len(chunk_ids), doc_id)
        except Exception:
            logger.exception("Failed to delete chunks for '%s'", doc_id)
            raise

        del _manifest[doc_id]
        BM25IndexCache.build()
        return True


# Singleton
ingestion_service = DocumentIngestionService()
