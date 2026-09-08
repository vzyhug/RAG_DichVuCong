from openai import AsyncOpenAI
from configs.settings import settings


class LLMFactory:
    """Create the OpenAI-compatible client selected by application settings."""

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
            return AsyncOpenAI(
                base_url=settings.LOCAL_LLM_URL.rstrip("/"),
                api_key="dummy",
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

    @staticmethod
    def get_model_name() -> str:
        """Return the model configured for the selected provider."""
        if settings.LLM_PROVIDER == "openai":
            return settings.OPENAI_MODEL
        if settings.LLM_PROVIDER == "gemini":
            return settings.GEMINI_MODEL
        if settings.LLM_PROVIDER == "local":
            return settings.LOCAL_LLM_MODEL
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

    @staticmethod
    def get_connection_info() -> dict[str, str]:
        """Return safe diagnostics for development errors, excluding secrets."""
        if settings.LLM_PROVIDER == "local":
            endpoint = settings.LOCAL_LLM_URL
        elif settings.LLM_PROVIDER == "gemini":
            endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif settings.LLM_PROVIDER == "openai":
            endpoint = "OpenAI default endpoint"
        else:
            endpoint = "unknown"
        return {
            "provider": settings.LLM_PROVIDER,
            "model": LLMFactory.get_model_name(),
            "endpoint": endpoint,
        }
