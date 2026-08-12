import json, os, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from configs.settings import settings
from typing import List, Dict

class VectorIndexer:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device='cpu')
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.metadata = []

    def prepare_chunks(self, documents: List[Dict]):
        from .custom_chunker import chunk_text
        from .metadata_extractor import extract_metadata_from_file
        chunks = []
        for doc in documents:
            content = doc['content']
            text_chunks = chunk_text(content)
            for idx, text in enumerate(text_chunks):
                meta = {
                    'source_id': doc['source_id'],
                    'relative_path': doc['relative_path'],
                    'chunk_index': idx,
                    'text': text,
                    'metadata': extract_metadata_from_file(doc['relative_path'], content)
                }
                chunks.append(meta)
        return chunks

    def build_index(self, chunks: List[Dict]):
        # E5 models require 'passage: ' prefix for documents
        texts = [f"passage: {c['text']}" if "e5" in settings.EMBEDDING_MODEL.lower() else c['text'] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = chunks

    def save_index(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load_index(self, index_dir: str):
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "metadata.json"), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)