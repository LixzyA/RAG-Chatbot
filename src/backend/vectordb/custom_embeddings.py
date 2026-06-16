"""Custom embeddings wrapper using sentence_transformers directly.

Avoids segfaults caused by ``langchain_huggingface.HuggingFaceEmbeddings`` on some
PyTorch / Windows / sentence-transformers version combinations.
"""

from typing import List
from langchain_core.embeddings import Embeddings


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by ``sentence_transformers``."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True).tolist()
        return [list(e) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
