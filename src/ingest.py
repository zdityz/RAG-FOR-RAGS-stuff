import fitz
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

def extract_text_from_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages_data = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages_data.append({
                "page": page_num + 1,
                "text": text.strip()
            })
    return pages_data

def chunk_documents(pages_data, doc_name="document"):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    splitter = SemanticChunker(embeddings)
    
    chunks = []
    for item in pages_data:
        # SemanticChunker expects a list of documents or texts. 
        # We can split individual page texts.
        split_texts = splitter.split_text(item["text"])
        
        for idx, text in enumerate(split_texts):
            chunks.append({
                "id": f"{doc_name}_p{item['page']}_c{idx}",
                "text": text,
                "metadata": {
                    "source": doc_name,
                    "page": item["page"],
                    "chunk_index": idx
                }
            })
    return chunks