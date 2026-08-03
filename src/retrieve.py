import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

DB_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_chunks"

def get_chroma_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

def vector_search(query: str, top_k: int = 5):
    collection = get_chroma_collection()
    results = collection.query(query_texts=[query], n_results=top_k)
    
    formatted_results = []
    for idx, (doc_id, doc, meta) in enumerate(zip(results["ids"][0], results["documents"][0], results["metadatas"][0])):
        formatted_results.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "rank": idx + 1
        })
    return formatted_results

def bm25_search(query: str, top_k: int = 5):
    collection = get_chroma_collection()
    all_data = collection.get(include=["documents", "metadatas"])
    
    docs = all_data["documents"]
    ids = all_data["ids"]
    metas = all_data["metadatas"]
    
    tokenized_corpus = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    formatted_results = []
    for rank, idx in enumerate(top_indices):
        formatted_results.append({
            "id": ids[idx],
            "text": docs[idx],
            "metadata": metas[idx],
            "rank": rank + 1
        })
    return formatted_results

def hybrid_search(query: str, top_k: int = 5, rrf_k: int = 60):
    vec_results = vector_search(query, top_k=top_k)
    kw_results = bm25_search(query, top_k=top_k)
    
    rrf_scores = {}
    combined_docs = {}
    
    for res in vec_results:
        doc_id = res["id"]
        combined_docs[doc_id] = res
        rrf_scores[doc_id] = 1.0 / (rrf_k + res["rank"])
        
    for res in kw_results:
        doc_id = res["id"]
        if doc_id not in combined_docs:
            combined_docs[doc_id] = res
            rrf_scores[doc_id] = 0.0
        rrf_scores[doc_id] += 1.0 / (rrf_k + res["rank"])
        
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    final_results = []
    for doc_id in sorted_ids[:top_k]:
        final_results.append(combined_docs[doc_id])
        
    return final_results

def rerank_results(query: str, results: list, top_k: int = 3):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    cross_inp = [[query, res["text"]] for res in results]
    scores = model.predict(cross_inp)
    
    for i in range(len(scores)):
        results[i]["cross_score"] = float(scores[i])
        
    sorted_results = sorted(results, key=lambda x: x["cross_score"], reverse=True)
    return sorted_results[:top_k]

def advanced_search(query: str, top_k: int = 3, retrieve_k: int = 10):
    candidates = hybrid_search(query, top_k=retrieve_k)
    reranked = rerank_results(query, candidates, top_k=top_k)
    return reranked