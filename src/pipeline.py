import logging
from typing import List, Dict

from .config import settings
from .logger import logger
from .agents.planner import generate_sub_queries
from .agents.synthesizer import generate_answer_with_citations
from .agents.verifier import verify_answer
from .retrieve import advanced_search


class RAGPipeline:
    """Encapsulates the full RAG workflow for a single query.

    The pipeline mirrors the previous step‑by‑step logic in ``api.handle_query``
    but centralises all processing in one place so that the API layer remains thin
    and the workflow can be re‑used by other entry points (e.g. CLI, streaming).
    """

    def __init__(self):
        self.logger = logger
        self.top_k = settings.retrieval_top_k
        self.logger.info("RAGPipeline initialised with top_k=%s", self.top_k)

    def _plan(self, query: str) -> List[str]:
        self.logger.info("Running planner agent")
        sub_queries = generate_sub_queries(query)
        self.logger.info("Sub‑queries generated: %s", sub_queries)
        return sub_queries

    def _retrieve(self, sub_queries: List[str]) -> List[Dict]:
        self.logger.info("Retrieving chunks for sub‑queries")
        all_results: List[Dict] = []
        seen_ids = set()
        for sq in sub_queries:
            sq_results = advanced_search(sq, top_k=self.top_k)
            for res in sq_results:
                if res["id"] not in seen_ids:
                    seen_ids.add(res["id"]) 
                    all_results.append(res)
        self.logger.info("Total unique chunks retrieved: %d", len(all_results))
        return all_results

    def _synthesize(self, query: str, chunks: List[Dict]) -> str:
        self.logger.info("Running synthesizer agent")
        answer = generate_answer_with_citations(query, chunks)
        self.logger.info("Synthesis complete")
        return answer

    def _verify(self, query: str, answer: str, chunks: List[Dict]) -> Dict:
        self.logger.info("Running verifier agent")
        raw_verification = verify_answer(query, answer, chunks)
        self.logger.info("Verification complete")
        # Expected format: first line PASS/FAIL, second line reason
        lines = raw_verification.strip().split("\n")
        status = lines[0].strip().upper() if lines else "FAIL"
        reason = lines[1].strip() if len(lines) > 1 else ""
        verified = status == "PASS"
        return {
            "raw": raw_verification,
            "verified": verified,
            "reason": reason,
        }

    def run(self, query: str) -> Dict:
        sub_queries = self._plan(query)
        chunks = self._retrieve(sub_queries)
        answer = self._synthesize(query, chunks)
        verification = self._verify(query, answer, chunks)

        # Build sources list in the same shape as the API previously returned
        sources = [
            {
                "doc_id": i + 1,
                "page": res["metadata"].get("page"),
                "source": res["metadata"].get("source"),
                "score": round(res.get("cross_score", 0.0), 2),
            }
            for i, res in enumerate(chunks)
        ]

        return {
            "query": query,
            "sub_queries": sub_queries,
            "answer": answer,
            "verified": verification["verified"],
            "verification": verification["raw"],
            "verification_reason": verification["reason"],
            "sources": sources,
        }
