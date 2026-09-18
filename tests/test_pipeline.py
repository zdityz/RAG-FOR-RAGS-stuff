import pytest
from unittest.mock import patch
from src.pipeline import RAGPipeline

def test_pipeline_run():
    pipeline = RAGPipeline()
    
    mock_chunks = [
        {"id": "doc1", "text": "RAG is a popular framework.", "metadata": {"source": "test.pdf", "page": 1}, "cross_score": 0.9}
    ]
    
    with patch.object(pipeline, "_plan", return_value=["What is RAG?"]), \
         patch.object(pipeline, "_retrieve", return_value=mock_chunks), \
         patch.object(pipeline, "_synthesize", return_value="RAG is a popular framework [Doc 1]."), \
         patch.object(pipeline, "_verify", return_value={"raw": "PASS\\nValid", "verified": True, "reason": "Valid"}):
         
        result = pipeline.run("What is RAG?")
        
        assert result["query"] == "What is RAG?"
        assert result["sub_queries"] == ["What is RAG?"]
        assert result["verified"] is True
        assert len(result["sources"]) == 1
        assert result["sources"][0]["doc_id"] == 1
        assert result["answer"] == "RAG is a popular framework [Doc 1]."
        assert result["verification_reason"] == "Valid"
