import os
import sys
import json
from src.ingestion.document_loaders import load_all_documents
from src.ingestion.indexer import VectorIndexer
from src.utils.logger import setup_logger
from configs.settings import settings

logger = setup_logger("Ingestion")

def main():
    logger.info("Starting ingestion pipeline...")
    # 1. Load tất cả tài liệu
    documents = load_all_documents(settings.DATA_RAW_DIR)
    logger.info(f"Loaded {len(documents)} documents")

    # 2. Tạo chunks
    indexer = VectorIndexer()
    chunks = indexer.prepare_chunks(documents)
    logger.info(f"Created {len(chunks)} chunks")

    # 3. Lưu chunks thành jsonl
    os.makedirs(settings.DATA_PROCESSED_DIR, exist_ok=True)
    with open(settings.CHUNKS_FILE, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info(f"Saved chunks to {settings.CHUNKS_FILE}")

    # 4. Xây dựng vector index
    indexer.build_index(chunks)
    indexer.save_index(settings.VECTOR_INDEX_DIR)
    logger.info(f"Saved vector index to {settings.VECTOR_INDEX_DIR}")

if __name__ == "__main__":
    main()