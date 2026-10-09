from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # LLM configuration
    llm_base_url: str = Field(default="http://localhost:11434/v1", validation_alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="ollama", validation_alias="LLM_API_KEY")
    llm_model: str = Field(default="llama3", validation_alias="LLM_MODEL")

    # ChromaDB configuration
    db_path: str = Field(default=str(Path(__file__).resolve().parent.parent / "chroma_db"), validation_alias="CHROMA_DB_PATH")
    collection_name: str = Field(default="pdf_chunks", validation_alias="CHROMA_COLLECTION_NAME")

    # Retrieval parameters
    retrieval_top_k: int = Field(default=5, validation_alias="RETRIEVAL_TOP_K")
    rerank_top_k: int = Field(default=5, validation_alias="RERANK_TOP_K")
    retrieve_k: int = Field(default=10, validation_alias="RETRIEVE_K")

    # Authentication
    api_key: str = Field(default="super-secret-token", validation_alias="API_KEY")

    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Export a singleton for easy import
settings = Settings()

