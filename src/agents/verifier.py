from ..llm import client, LLM_MODEL

def verify_answer(query: str, answer: str, retrieved_chunks: list):
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        context_text += f"\n[Doc {i+1}]:\n{chunk['text']}\n"
    
    system_prompt = (
        "You are a strict fact-checking AI. Verify if the provided Answer is completely supported by the Context.\n"
        "If the Answer contains ANY claims, facts, or numbers not explicitly present in the Context, it fails.\n"
        "Respond with EXACTLY two lines.\n"
        "Line 1: PASS or FAIL\n"
        "Line 2: A brief, one-sentence explanation of why it passed or failed."
    )
    
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer to check:\n{answer}"
    
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    
    return response.choices[0].message.content