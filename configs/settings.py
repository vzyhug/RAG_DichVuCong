import os
from dotenv import load_dotenv

# Keep checked-in/local .env defaults, while allowing deployment environment
# variables to select a provider and endpoint explicitly.
load_dotenv(override=False)

class Settings:
    # Đường dẫn dữ liệu
    DATA_RAW_DIR = os.getenv("DATA_RAW_DIR", "data/raw")
    DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", "data/processed")
    CHUNKS_FILE = os.path.join(DATA_PROCESSED_DIR, "chunks.jsonl")
    VECTOR_INDEX_DIR = os.path.join(DATA_PROCESSED_DIR, "vector_index")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_FIREBASE_LOGGING = os.getenv("ENABLE_FIREBASE_LOGGING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    FIREBASE_CREDENTIALS_PATH = os.getenv(
        "FIREBASE_CREDENTIALS_PATH", "firebase-service-account.json"
    )
    FIREBASE_COLLECTION = os.getenv("FIREBASE_COLLECTION", "chat_logs")

    # LLM
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-pro")
    LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "").strip()
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "").strip()

    # Embedding
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    TOP_K = int(os.getenv("TOP_K", 10))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.70))
    RESPONSE_MAX_TOKENS = int(os.getenv("RESPONSE_MAX_TOKENS", 768))

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 400))

    def validate(self) -> None:
        """Validate settings that are required by the selected provider."""
        supported_providers = {"openai", "gemini", "local"}
        if self.LLM_PROVIDER not in supported_providers:
            raise ValueError(
                "Unsupported LLM_PROVIDER: "
                f"{self.LLM_PROVIDER!r}. Expected one of: "
                f"{', '.join(sorted(supported_providers))}"
            )

        if self.LLM_PROVIDER != "local":
            return

        missing = [
            name
            for name in ("LOCAL_LLM_URL", "LOCAL_LLM_MODEL")
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(
                "LLM_PROVIDER=local requires: " + ", ".join(missing)
            )


settings = Settings()
settings.validate()
