from openai import OpenAI
from .config import settings

client = OpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key
)
LLM_MODEL = settings.llm_model