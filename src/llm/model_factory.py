from openai import AsyncOpenAI
from configs.settings import settings

class LLMFactory:
    @staticmethod
    def get_llm():
        if settings.LLM_PROVIDER == "openai":
            return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        elif settings.LLM_PROVIDER == "gemini":
            return AsyncOpenAI(
                api_key=settings.GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        elif settings.LLM_PROVIDER == "local":
            return AsyncOpenAI(base_url=settings.LOCAL_LLM_URL, api_key="dummy")
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")