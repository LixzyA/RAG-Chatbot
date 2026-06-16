import sentence_transformers  # noqa: F401; must import before chromadb to avoid Windows segfault
import chromadb
from fastapi import Depends, Request
from .custom_embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from typing import List, Annotated
from pathlib import Path
import pickle

from exception import EmbeddingException
from .reranker import Reranker

class ChromaDB:
    def __init__(self, collection_name: str = "big_token_corpus", 
                 persist_path: str = '../../.langchain_chroma/', 
                 reranker: Reranker | None = None,
                  bm25_cache_path: str = './.bm25_cache/docs_cache.pkl') -> None:
        self.embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        chroma_path = Path(persist_path)
        chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection_name = collection_name
        self.vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        self.bm25_retriever: BM25Retriever | None = None
        self.bm25_cache_path = Path(bm25_cache_path)
        self._docs_cache: List[Document] = self._load_docs_cache()
        self.reranker = reranker
        if self._docs_cache:
            self._update_bm25()
    
    def get_or_create_collection(self, collection_name: str | None = None):
        """Get or create a collection, using instance collection_name if none provided"""
        name = collection_name or self.collection_name
        return self.client.get_or_create_collection(name)

    def vector_add_documents(self, documents: List[Document], ids=None):
        """Add documents to the vector store, persist the docs cache, and rebuild BM25."""
        result = self.vector_store.add_documents(documents, ids=ids)
        self._docs_cache.extend(documents)
        self._save_docs_cache()
        self._update_bm25()
        return result

    def add_documents_bulk(self, documents: List[Document], ids=None):
        """Add documents and persist the docs cache without rebuilding BM25.

        Call :meth:`rebuild_bm25` after bulk ingestion completes.
        """
        result = self.vector_store.add_documents(documents, ids=ids)
        self._docs_cache.extend(documents)
        self._save_docs_cache()
        return result

    def _load_docs_cache(self) -> List[Document]:
        if self.bm25_cache_path.exists():
            with open(self.bm25_cache_path, "rb") as f:
                docs = pickle.load(f)
            print(f"BM25 cache loaded: {len(docs)} docs")
            return docs
        return []

    def _save_docs_cache(self):
        self.bm25_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_cache_path, "wb") as f:
            pickle.dump(self._docs_cache, f)
    
    def similarity_search(self, query, k=4):
        """Search for similar documents"""
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def _fetch_all_documents(self) -> List[Document]:
        """Batch-fetch all documents from the collection using limit/offset
        to avoid SQLite's 999 variable limit on large collections."""
        collection = self.get_or_create_collection()
        batch_size = 500
        offset = 0
        all_docs: List[Document] = []

        while True:
            result = collection.get(
                limit=batch_size, offset=offset,
                include=["documents", "metadatas"],
            )
            ids = result.get("ids") or []
            if not ids:
                break
            docs = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(
                    result["documents"] or [],
                    result["metadatas"] or [],
                )
            ]
            all_docs.extend(docs)
            if len(ids) < batch_size:
                break
            offset += batch_size

        return all_docs

    def _update_bm25(self):
        """Build the BM25 retriever from the in-memory docs cache.

        Assumes ``_docs_cache`` is already populated.  Call
        :meth:`rebuild_bm25` when the cache may be cold.
        """
        if not self._docs_cache:
            return  # nothing to index
        self.bm25_retriever = BM25Retriever.from_documents(self._docs_cache)
        self.bm25_retriever.k = 10

    def rebuild_bm25(self):
        """Ensure the docs cache is populated and rebuild the BM25 retriever.

        If the in-memory cache is empty, batch-fetch all documents from the
        Chroma collection first.
        """
        if not self._docs_cache:
            self._docs_cache = self._fetch_all_documents()
            self._save_docs_cache()
        self._update_bm25()
    
    def hybrid_search(self, query, k: int = 10):
        if self.bm25_retriever is None:
            self.rebuild_bm25()
            if self.bm25_retriever is None:
                raise ValueError("Cannot perform hybrid search: Collection is empty.")

        # 2. Configure retrievers
        self.bm25_retriever.k = k # type: ignore
        vector_store_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        # 3. Ensemble (Using default Reciprocal Rank Fusion since standard retrievers don't return scores)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_store_retriever], # type: ignore
            weights = [0.3, 0.7]
        )

        # 4. Execute and return actual results
        return ensemble_retriever.invoke(query)

    def reranked_search(self, query, k: int = 10, candidate_multiplier: int = 4) -> List[Document]:
        """Two-stage retrieval: hybrid search followed by cross-encoder reranking.

        Parameters
        ----------
        query : str
            The user query.
        k : int
            Number of final documents to return.
        candidate_multiplier : int
            Multiply *k* by this factor to determine how many candidates
            hybrid_search should fetch before reranking.

        Returns
        -------
        List[Document]
            Top *k* documents ordered by cross-encoder relevance score.
        """
        candidate_k = k * candidate_multiplier
        candidates = self.hybrid_search(query, k=candidate_k)
        if not candidates:
            return []
        if self.reranker is None:
            return candidates[:k]
        return self.reranker.rerank(query, candidates, top_k=k)

    def heartbeat(self) -> bool:
        """
        Perform lightweight runtime health checks:
        - embeddings backend can produce an embedding
        - chroma client can list/get collections
        - vector store retriever can execute a (non-mutating) search

        Returns True if checks pass, False otherwise.
        """
        # 1) Embeddings check
        emb = None
        if hasattr(self.embeddings, "embed_query"):
            emb = self.embeddings.embed_query("health-check")
        elif hasattr(self.embeddings, "embed_documents"):
            res = self.embeddings.embed_documents(["health-check"])  # type: ignore
            if isinstance(res, list) and res:
                emb = res[0]

        if emb is None or (hasattr(emb, "__len__") and len(emb) == 0):
            raise EmbeddingException

        # 2) Chroma client / collection check
        if hasattr(self.client, "list_collections"):
            _ = self.client.list_collections()
        else:
            _ = self.get_or_create_collection()

        # 3) Vector store (non-mutating) search check
        _ = self.similarity_search(query="health check", k=1)

        return True



def get_vector_db(request: Request) -> ChromaDB:
    """Get the vector_db store from app state."""
    return request.app.state.vector_db


VectorDBClient = Annotated[ChromaDB, Depends(get_vector_db)]
