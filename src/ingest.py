import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def chunk_documents(pages_data, doc_name="document", chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    for item in pages_data:
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