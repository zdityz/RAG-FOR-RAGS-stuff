import logging
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import settings
from .db import get_collection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BM25 index cache – builds once at process start and can be rebuilt after ingestion
# ---------------------------------------------------------------------------
class BM25IndexCache:
    _bm25: BM25Okapi | None = None
    _docs: List[str] = []
    _ids: List[str] = []
    _metas: List[Dict] = []

    @classmethod
    def build(cls) -> None:
        """Load all documents from ChromaDB and build the BM25 index.
        Call this after you ingest new documents.
        """
        collection = get_collection()
        data = collection.get(include=["documents", "metadatas", "ids"])
        cls._docs = data["documents"]
        cls._ids = data["ids"]
        cls._metas = data["metadatas"]
        tokenized_corpus = [doc.lower().split() for doc in cls._docs]
        cls._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 cache rebuilt with %d documents", len(cls._docs))

    @classmethod
    def ensure_built(cls) -> None:
        if cls._bm25 is None:
            cls.build()

    @classmethod
    def search(cls, query: str, top_k: int = 5) -> List[Dict]:
        cls.ensure_built()
        tokenized_query = query.lower().split()
        scores = cls._bm25.get_scores(tokenized_query)  # type: ignore[arg-type]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results: List[Dict] = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "id": cls._ids[idx],
                "text": cls._docs[idx],
                "metadata": cls._metas[idx],
                "rank": rank + 1,
                "bm25_score": float(scores[idx]),
            })
        return results

# ---------------------------------------------------------------------------
# Retrieval functions – vector, BM25, hybrid, rerank, advanced
# ---------------------------------------------------------------------------
# Embedding function for vector search (uses the BGE model)
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5"
)
_cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def vector_search(query: str, top_k: int = 5) -> List[Dict]:
    """Semantic search via ChromaDB collection."""
    try:
        collection = get_collection()
        results = collection.query(query_texts=[query], n_results=top_k)
        formatted: List[Dict] = []
        for idx, (doc_id, doc, meta) in enumerate(
            zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
        ):
            formatted.append({
                "id": doc_id,
                "text": doc,
                "metadata": meta,
                "rank": idx + 1,
            })
        return formatted
    except Exception:
        logger.exception("Vector search failed")
        raise

def bm25_search(query: str, top_k: int = 5) -> List[Dict]:
    """Keyword search using the cached BM25 index – O(1) DB access per call."""
    try:
        return BM25IndexCache.search(query, top_k=top_k)
    except Exception:
        logger.exception("BM25 search failed")
        raise

def hybrid_search(query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict]:
    """Combine vector and BM25 scores using Reciprocal Rank Fusion (RRF)."""
    vec_results = vector_search(query, top_k=top_k)
    kw_results = bm25_search(query, top_k=top_k)

    rrf_scores: Dict[str, float] = {}
    combined: Dict[str, Dict] = {}

    for res in vec_results:
        doc_id = res["id"]
        combined[doc_id] = res
        rrf_scores[doc_id] = 1.0 / (rrf_k + res["rank"])

    for res in kw_results:
        doc_id = res["id"]
        if doc_id not in combined:
            combined[doc_id] = res
            rrf_scores[doc_id] = 0.0
        rrf_scores[doc_id] += 1.0 / (rrf_k + res["rank"])

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [combined[doc_id] for doc_id in sorted_ids[:top_k]]

def rerank_results(query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
    """Cross‑encoder re‑ranking of the candidate chunks."""
    try:
        cross_inp = [[query, r["text"]] for r in results]
        scores = _cross_encoder_model.predict(cross_inp)
        for i, score in enumerate(scores):
            results[i]["cross_score"] = float(score)
        sorted_results = sorted(results, key=lambda x: x["cross_score"], reverse=True)
        return sorted_results[:top_k]
    except Exception:
        logger.exception("Rerank failed – falling back to original order")
        return results[:top_k]

def advanced_search(query: str, top_k: int = 5, retrieve_k: int = 10) -> List[Dict]:
    """Full pipeline: hybrid search → cross‑encoder re‑rank → final top‑k results."""
    candidates = hybrid_search(query, top_k=retrieve_k)
    return rerank_results(query, candidates, top_k=top_k)