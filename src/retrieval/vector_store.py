from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class VectorStore(ABC):
    @abstractmethod
    def add_embeddings(self, texts: List[str], embeddings: np.ndarray):
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        pass

    @abstractmethod
    def get_text(self, index: int) -> str:
        pass

class FAISSVectorStore(VectorStore):
    def __init__(self, dimension: int):
        import faiss
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add_embeddings(self, texts: List[str], embeddings: np.ndarray):
        import faiss
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        import faiss
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        return [(idx, dist) for dist, idx in zip(distances[0], indices[0]) if idx != -1]

    def get_text(self, index: int) -> str:
        return self.texts[index]