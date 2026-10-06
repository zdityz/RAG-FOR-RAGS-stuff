from typing import Dict, List, Optional
from ..llm import client, LLM_MODEL


def generate_answer_with_citations(
    query: str,
    retrieved_chunks: list,
    history: Optional[List[Dict]] = None,
):
    """Generate a grounded answer with inline citations.

    Args:
        query: The current user question.
        retrieved_chunks: Chunks returned by the retrieval stage.
        history: Optional list of ``{"role": ..., "content": ...}`` dicts
                 representing prior conversation turns, used for context.
    """
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        doc_id = i + 1
        source = chunk["metadata"].get("source", "Unknown")
        page = chunk["metadata"].get("page", "Unknown")
        context_text += f"\n[Doc {doc_id}] (Source: {source}, Page: {page}):\n{chunk['text']}\n"

    system_prompt = (
        "You are an expert technical assistant. Answer the user's question based ONLY on the provided context.\n"
        "You must cite your sources inline using the [Doc X] format at the end of sentences where you use that information.\n"
        "If the answer cannot be found in the context, say 'I cannot answer this based on the provided documents.' Do not hallucinate."
    )

    # Build message list: system → history → current user prompt
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content