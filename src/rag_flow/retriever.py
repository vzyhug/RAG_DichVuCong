import numpy as np, faiss, json
from sentence_transformers import SentenceTransformer
from configs.settings import settings

class Retriever:
    def __init__(self, index_dir: str = settings.VECTOR_INDEX_DIR):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.index = faiss.read_index(f"{index_dir}/index.faiss")
        with open(f"{index_dir}/metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def retrieve(self, query: str, top_k: int = settings.TOP_K) -> List[Dict]:
        query_emb = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb.astype(np.float32), top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if dist >= settings.SIMILARITY_THRESHOLD:
                res = self.metadata[idx].copy()
                res['score'] = float(dist)
                results.append(res)
        return results