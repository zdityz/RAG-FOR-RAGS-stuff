from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from .config import settings

# Create a persistent Chroma client using the absolute DB path from settings
# Helper to get or create the collection
def get_collection():
    # Create a fresh client for the current DB path each call
    client = chromadb.PersistentClient(path=settings.db_path)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-base-en-v1.5"
    )
    # Simply get or create the collection; do not delete it here
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection
