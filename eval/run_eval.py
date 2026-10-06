"""Automated Evaluation Pipeline — Phase 5.2.

Runs a set of questions through the RAG pipeline and uses an LLM-as-judge
to score the responses on Context Recall, Faithfulness, and Answer Relevance.
"""

import json
import logging
import time
from pathlib import Path

# Add project root to sys.path to import src
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import RAGPipeline
from src.llm import client, LLM_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def llm_judge(metric: str, question: str, context: str, answer: str, expected: str) -> int:
    """Uses the LLM to score a specific metric from 1 to 5."""
    if metric == "context_recall":
        prompt = f"""Given the question: "{question}"
And the expected answer: "{expected}"
Evaluate if the following retrieved context contains the necessary information to formulate the expected answer.
Context:
{context}

Score 1 to 5 (1=completely irrelevant/missing, 5=contains all necessary information).
Reply with ONLY a single integer digit."""

    elif metric == "faithfulness":
        prompt = f"""Given the context:
{context}
Evaluate if the following answer is faithful to the context (i.e., it doesn't hallucinate).
Answer: "{answer}"

Score 1 to 5 (1=completely hallucinated, 5=perfectly faithful/grounded).
Reply with ONLY a single integer digit."""

    elif metric == "answer_relevance":
        prompt = f"""Given the question: "{question}"
Evaluate how relevant and direct the following answer is to the question.
Answer: "{answer}"

Score 1 to 5 (1=completely irrelevant, 5=perfectly relevant and direct).
Reply with ONLY a single integer digit."""

    else:
        return 0

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        score_text = response.choices[0].message.content.strip()
        return int(score_text[0]) if score_text and score_text[0].isdigit() else 0
    except Exception as e:
        logger.error(f"Error scoring {metric}: {e}")
        return 0


def run_eval():
    eval_file = Path(__file__).parent / "questions.json"
    if not eval_file.exists():
        logger.error(f"Eval file not found: {eval_file}")
        return

    with open(eval_file, "r") as f:
        questions = json.load(f)

    pipeline = RAGPipeline()
    results = []

    logger.info(f"Starting evaluation of {len(questions)} questions...")
    
    total_metrics = {"context_recall": 0, "faithfulness": 0, "answer_relevance": 0, "latency_ms": 0}

    for i, q in enumerate(questions):
        logger.info(f"\n--- Evaluating Question {i+1}/{len(questions)} ---")
        logger.info(f"Q: {q['question']}")
        
        # Run pipeline
        res = pipeline.run(q["question"])
        
        answer = res["answer"]
        chunks_text = "\n".join([f"[Doc {s['doc_id']}]" for s in res["sources"]]) # Just minimal representation or actual text if we have it
        
        # Retrieve actual context text to pass to judge
        # The pipeline doesn't return the raw text in 'sources' by default, 
        # so we'll re-fetch or just extract from sources. 
        # For this script, we'll fetch the actual text by overriding/patching if needed, 
        # but let's just do a simplified eval or re-run retrieve.
        
        # Actually, let's grab the chunks directly
        sub_queries = res["sub_queries"]
        chunks = pipeline._retrieve(sub_queries)
        context_str = "\n".join([c["text"] for c in chunks])
        
        # Score
        cr = llm_judge("context_recall", q["question"], context_str, answer, q["expected_answer"])
        faith = llm_judge("faithfulness", q["question"], context_str, answer, q["expected_answer"])
        rel = llm_judge("answer_relevance", q["question"], context_str, answer, q["expected_answer"])
        latency = res["stage_latencies_ms"]["total"]
        
        logger.info(f"A: {answer}")
        logger.info(f"Scores -> Recall: {cr}/5 | Faithfulness: {faith}/5 | Relevance: {rel}/5 | Latency: {latency}ms")
        
        metrics = {
            "context_recall": cr,
            "faithfulness": faith,
            "answer_relevance": rel,
            "latency_ms": latency
        }
        
        total_metrics["context_recall"] += cr
        total_metrics["faithfulness"] += faith
        total_metrics["answer_relevance"] += rel
        total_metrics["latency_ms"] += latency
        
        results.append({
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "pipeline_answer": answer,
            "verified": res["verified"],
            "metrics": metrics
        })
        
        time.sleep(1) # Small pause for rate limits

    n = len(questions)
    summary = {
        "avg_context_recall": total_metrics["context_recall"] / n,
        "avg_faithfulness": total_metrics["faithfulness"] / n,
        "avg_answer_relevance": total_metrics["answer_relevance"] / n,
        "avg_latency_ms": total_metrics["latency_ms"] / n,
    }
    
    logger.info("\n=== Evaluation Summary ===")
    logger.info(json.dumps(summary, indent=2))
    
    out_file = Path(__file__).parent / "eval_results.json"
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    
    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    run_eval()
