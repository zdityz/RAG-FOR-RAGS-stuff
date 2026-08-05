from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

def generate_answer_with_citations(query: str, retrieved_chunks: list):
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        doc_id = i + 1
        source = chunk['metadata'].get('source', 'Unknown')
        page = chunk['metadata'].get('page', 'Unknown')
        context_text += f"\n[Doc {doc_id}] (Source: {source}, Page: {page}):\n{chunk['text']}\n"
    
    system_prompt = (
        "You are an expert technical assistant. Answer the user's question based ONLY on the provided context.\n"
        "You must cite your sources inline using the [Doc X] format at the end of sentences where you use that information.\n"
        "If the answer cannot be found in the context, say 'I cannot answer this based on the provided documents.' Do not hallucinate."
    )
    
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    
    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content