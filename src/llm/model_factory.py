from openai import OpenAI
from configs.settings import settings

class LLMFactory:
    @staticmethod
    def get_llm():
        if settings.LLM_PROVIDER == "openai":
            return OpenAI(api_key=settings.OPENAI_API_KEY)
        elif settings.LLM_PROVIDER == "local":
            # Giả định local server chạy OpenAI-compatible API (vLLM, Ollama, ...)
            return OpenAI(base_url=settings.LOCAL_LLM_URL, api_key="dummy")
        else:
            raise ValueError("Unsupported LLM provider")