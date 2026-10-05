#!/usr/bin/env python3
"""Re‑ingest all PDF documents using the new semantic chunking pipeline.

This script:
1. Deletes the existing ChromaDB directory (defined by `settings.db_path`).
2. Walks through the `data/` directory (relative to the project root) and finds all ``.pdf`` files.
3. Extracts raw text from each PDF via :func:`extract_text_from_pdf`.
4. Splits the text into semantic chunks using the BGE embedding model.
5. Stores the chunks in ChromaDB.
6. Re‑builds the BM25 cache.

Run it from the project root:
    source .venv/bin/activate && python scripts/reingest.py
"""
import shutil
from pathlib import Path
import sys

# Ensure the project root (the directory containing `src/`) is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from src.embed import store_chunks
from src.ingest import extract_text_from_pdf, chunk_documents
from src.retrieve import BM25IndexCache

def main() -> None:
    # 1️⃣ Wipe existing vector store
    db_path = Path(settings.db_path)
    if db_path.exists():
        print(f"Removing existing ChromaDB at {db_path}…")
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    # 2️⃣ Locate PDF sources
    data_dir = PROJECT_ROOT / "data"
    pdf_files = list(data_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {data_dir}. Add PDFs and re‑run.")
        return

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}…")
        pages = extract_text_from_pdf(str(pdf_path))
        chunks = chunk_documents(pages, doc_name=pdf_path.stem)
        all_chunks.extend(chunks)

    # 3️⃣ Store in ChromaDB
    print(f"Storing {len(all_chunks)} chunks into ChromaDB…")
    store_chunks(all_chunks)

    # 4️⃣ Re‑build BM25 cache for keyword search
    print("Re‑building BM25 cache…")
    BM25IndexCache.build()
    print("Re‑ingestion complete.")

if __name__ == "__main__":
    main()
