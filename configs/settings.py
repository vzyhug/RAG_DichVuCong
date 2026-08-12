import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings:
    # Đường dẫn dữ liệu
    DATA_RAW_DIR = os.getenv("DATA_RAW_DIR", "data/raw")
    DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", "data/processed")
    CHUNKS_FILE = os.path.join(DATA_PROCESSED_DIR, "chunks.jsonl")
    VECTOR_INDEX_DIR = os.path.join(DATA_PROCESSED_DIR, "vector_index")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # LLM
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Embedding
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    TOP_K = int(os.getenv("TOP_K", 3))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 30))

settings = Settings()
