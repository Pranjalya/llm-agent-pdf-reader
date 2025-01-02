from abc import ABC, abstractmethod
from typing import List
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        pass

class SentenceTransformersEmbedding(EmbeddingModel):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)

class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-large"):
        import openai
        openai.api_key = api_key
        self.model_name = model_name

    def _get_embedding(self, text: str):
        import openai
        response = openai.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings_list = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._get_embedding, text) for text in texts]
            for future in futures:
                embeddings_list.append(future.result())
        return np.array(embeddings_list, dtype=np.float32)