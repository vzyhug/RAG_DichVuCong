import numpy as np, faiss, json, os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from configs.settings import settings
import torch

class Retriever:
    def __init__(self, index_dir: str = settings.VECTOR_INDEX_DIR):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device='cpu')
        self.index = faiss.read_index(f"{index_dir}/index.faiss")
        with open(f"{index_dir}/metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def retrieve(self, query: str, top_k: int = settings.TOP_K) -> List[Dict]:
        if torch.cuda.is_available():
            self.model.to('cuda')
            
        # E5 models require 'query: ' prefix for user queries
        query_text = f"query: {query}" if "e5" in settings.EMBEDDING_MODEL.lower() else query
        query_emb = self.model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        
        distances, indices = self.index.search(query_emb.astype(np.float32), top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if dist >= settings.SIMILARITY_THRESHOLD:
                res = self.metadata[idx].copy()
                res['score'] = float(dist)
                results.append(res)
        return results