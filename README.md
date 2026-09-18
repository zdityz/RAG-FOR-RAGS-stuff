# RAG-FOR-RAGS

A production-grade Retrieval-Augmented Generation (RAG) system with a multi-agent architecture (Planner, Synthesizer, Verifier).

## Setup

1. **Clone the repository**
2. **Install dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables**:
   Create a `.env` file or set the variables manually. Check `src/config.py` for all available configuration options (e.g., `LLM_MODEL`, `LOG_LEVEL`).
4. **Ingest data**:
   Ensure you have ingested documents into your ChromaDB instance.
5. **Run the API or CLI**:
   - API: `uvicorn src.api:app`
   - CLI: `python -m src.main`

## Testing

Run unit tests and view coverage locally using `pytest`:

```bash
pip install pytest pytest-cov
pytest --cov=src
```
