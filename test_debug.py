import chromadb
from rank_bm25 import BM25Okapi

client = chromadb.Client()
collection = client.create_collection(name="test_collection")
collection.add(
    documents=[
        "RAG stands for Retrieval-Augmented Generation, combining retrieval with LLMs.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents."
    ],
    ids=["id1", "id2"]
)

data = collection.get()
tokenized_corpus = [doc.lower().split() for doc in data["documents"]]
print("Tokens:", tokenized_corpus)
bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(["bm25"])
print("Scores:", scores)
