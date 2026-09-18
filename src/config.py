from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # LLM configuration
    llm_base_url: str = Field(default="http://localhost:11434/v1", env="LLM_BASE_URL")
    llm_api_key: str = Field(default="ollama", env="LLM_API_KEY")
    llm_model: str = Field(default="llama3", env="LLM_MODEL")

    # ChromaDB configuration
    db_path: str = Field(default=str(Path(__file__).parents[2] / "chroma_db"), env="CHROMA_DB_PATH")
    collection_name: str = Field(default="pdf_chunks", env="CHROMA_COLLECTION_NAME")

    # Retrieval parameters
    retrieval_top_k: int = Field(default=5, env="RETRIEVAL_TOP_K")
    rerank_top_k: int = Field(default=5, env="RERANK_TOP_K")
    retrieve_k: int = Field(default=10, env="RETRIEVE_K")

    # Authentication
    api_key: str = Field(default="super-secret-token", env="API_KEY")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Export a singleton for easy import
settings = Settings()
