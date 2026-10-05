from src.db import get_collection
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

def store_chunks(chunks):
    collection = get_collection()
    
    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    logger.info(f"Successfully indexed {len(chunks)} chunks into ChromaDB at '{settings.db_path}'.")