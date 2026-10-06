"""RAGPipeline — single source of truth for the full RAG workflow.

Phase 4 additions:
  - session_id support → conversation history prepended to synthesiser context
  - stage_latencies_ms tracking for observability (Phase 5.3 prep)
  - streaming generator for SSE endpoint
"""
import json
import logging
import time
from typing import AsyncGenerator, Dict, Generator, List, Optional

from .config import settings
from .logger import logger
from .agents.planner import generate_sub_queries
from .agents.synthesizer import generate_answer_with_citations
from .agents.verifier import verify_answer
from .retrieve import advanced_search
from . import session as session_store


class RAGPipeline:
    """Encapsulates the full RAG workflow for a single query."""

    def __init__(self):
        self.logger = logger
        self.top_k = settings.retrieval_top_k

    # ------------------------------------------------------------------
    # Internal stage helpers
    # ------------------------------------------------------------------
    def _plan(self, query: str, history: List[Dict]) -> List[str]:
        self.logger.info("Running planner agent")
        # Provide last 3 turns as context for pronoun / reference resolution
        history_text = ""
        if history:
            turns = history[-3:]
            history_text = "\n".join(
                f"{t['role'].capitalize()}: {t['content']}" for t in turns
            )
        full_query = f"{history_text}\nUser: {query}" if history_text else query
        sub_queries = generate_sub_queries(full_query)
        self.logger.info("Sub-queries generated: %s", sub_queries)
        return sub_queries

    def _retrieve(self, sub_queries: List[str]) -> List[Dict]:
        self.logger.info("Retrieving chunks for %d sub-queries", len(sub_queries))
        all_results: List[Dict] = []
        seen_ids: set = set()
        for sq in sub_queries:
            for res in advanced_search(sq, top_k=self.top_k):
                if res["id"] not in seen_ids:
                    seen_ids.add(res["id"])
                    all_results.append(res)
        self.logger.info("Total unique chunks retrieved: %d", len(all_results))
        return all_results

    def _synthesize(self, query: str, chunks: List[Dict], history: List[Dict]) -> str:
        self.logger.info("Running synthesizer agent")
        answer = generate_answer_with_citations(query, chunks, history=history)
        self.logger.info("Synthesis complete")
        return answer

    def _verify(self, query: str, answer: str, chunks: List[Dict]) -> Dict:
        self.logger.info("Running verifier agent")
        raw = verify_answer(query, answer, chunks)
        lines = raw.strip().split("\n")
        status = lines[0].strip().upper() if lines else "FAIL"
        reason = lines[1].strip() if len(lines) > 1 else ""
        return {"raw": raw, "verified": status == "PASS", "reason": reason}

    @staticmethod
    def _build_sources(chunks: List[Dict]) -> List[Dict]:
        return [
            {
                "doc_id": i + 1,
                "page": res["metadata"].get("page"),
                "source": res["metadata"].get("source"),
                "score": round(res.get("cross_score", 0.0), 2),
            }
            for i, res in enumerate(chunks)
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def run(self, query: str, session_id: Optional[str] = None) -> Dict:
        """Run the full pipeline synchronously. Returns a result dict."""
        latencies: Dict[str, float] = {}
        history = session_store.get_turns(session_id) if session_id else []
        # Convert raw turns to role/content dicts for internal use
        history_msgs: List[Dict] = []
        for user_msg, asst_msg in history:
            history_msgs.append({"role": "user", "content": user_msg})
            history_msgs.append({"role": "assistant", "content": asst_msg})

        t0 = time.perf_counter()
        sub_queries = self._plan(query, history_msgs)
        latencies["planning"] = round((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        chunks = self._retrieve(sub_queries)
        latencies["retrieval"] = round((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        answer = self._synthesize(query, chunks, history_msgs)
        latencies["synthesis"] = round((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        verification = self._verify(query, answer, chunks)
        latencies["verification"] = round((time.perf_counter() - t0) * 1000)
        latencies["total"] = sum(latencies.values())

        self.logger.info("Stage latencies (ms): %s", latencies)

        # Persist turn in session
        if session_id:
            session_store.add_turn(session_id, query, answer)

        return {
            "query": query,
            "sub_queries": sub_queries,
            "answer": answer,
            "verified": verification["verified"],
            "verification": verification["raw"],
            "verification_reason": verification["reason"],
            "sources": self._build_sources(chunks),
            "stage_latencies_ms": latencies,
            "session_id": session_id,
        }

    def stream(self, query: str, session_id: Optional[str] = None) -> Generator[str, None, None]:
        """Run the pipeline and yield SSE-formatted event strings."""
        history = session_store.get_turns(session_id) if session_id else []
        history_msgs: List[Dict] = []
        for user_msg, asst_msg in history:
            history_msgs.append({"role": "user", "content": user_msg})
            history_msgs.append({"role": "assistant", "content": asst_msg})

        # Stage: planning
        sub_queries = self._plan(query, history_msgs)
        yield _sse("progress", {"stage": "planning", "sub_queries": sub_queries})

        # Stage: retrieval
        chunks = self._retrieve(sub_queries)
        yield _sse("progress", {"stage": "retrieving", "chunks_found": len(chunks)})

        # Stage: synthesis — stream token-by-token
        answer = self._synthesize(query, chunks, history_msgs)
        for token in answer.split(" "):
            yield _sse("token", {"token": token + " "})

        # Stage: verification
        verification = self._verify(query, answer, chunks)

        if session_id:
            session_store.add_turn(session_id, query, answer)

        yield _sse(
            "done",
            {
                "verified": verification["verified"],
                "verification_reason": verification["reason"],
                "sources": self._build_sources(chunks),
                "session_id": session_id,
            },
        )


def _sse(event: str, data: Dict) -> str:
    """Format a single Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
