import pytest
import chromadb
from unittest.mock import patch

@pytest.fixture
def mock_collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma_test_db"))
    collection = client.create_collection(name="test_collection")
    collection.add(
        documents=[
            "RAG stands for Retrieval-Augmented Generation, combining retrieval with LLMs.",
            "BM25 is a bag-of-words retrieval function that ranks a set of documents.",
            "This is a third dummy document to ensure BM25 IDF scores are strictly positive."
        ],
        metadatas=[{"source": "doc1.pdf", "page": 1}, {"source": "doc2.pdf", "page": 2}, {"source": "doc3.pdf", "page": 3}],
        ids=["id1", "id2", "id3"]
    )
    return collection

@pytest.fixture(autouse=True)
def patch_get_collection(mock_collection):
    with patch("src.retrieve.get_collection", return_value=mock_collection), \
         patch("src.db.get_collection", return_value=mock_collection):
        yield mock_collection
