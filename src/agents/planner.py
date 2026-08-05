from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

def generate_sub_queries(query: str):
    system_prompt = (
        "You are an expert search query planner. Break the user's complex question into 1 to 3 simple, distinct sub-queries "
        "optimized for a search engine. Return ONLY the sub-queries, one per line. Do not number them, bullet them, or add prefixes."
    )
    
    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.1
    )
    
    raw_output = response.choices[0].message.content.strip()
    return [q.strip() for q in raw_output.split('\n') if q.strip()]