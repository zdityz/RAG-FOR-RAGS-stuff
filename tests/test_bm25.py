from src.retrieve import BM25IndexCache

def test_bm25_build_and_search(patch_get_collection):
    # Ensure cache is built from the mocked collection
    BM25IndexCache.build()
    
    assert len(BM25IndexCache._docs) == 2
    assert "id1" in BM25IndexCache._ids
    assert "id2" in BM25IndexCache._ids
    
    # Search for RAG
    results = BM25IndexCache.search("RAG", top_k=1)
    
    assert len(results) == 1
    assert results[0]["id"] == "id1"
    assert "Retrieval-Augmented" in results[0]["text"]
    
    # Search for BM25
    results_bm25 = BM25IndexCache.search("BM25", top_k=1)
    
    assert len(results_bm25) == 1
    assert results_bm25[0]["id"] == "id2"
    assert "bag-of-words" in results_bm25[0]["text"]
