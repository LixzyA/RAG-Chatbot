from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from langchain_huggingface import HuggingFaceEmbeddings

# Pre-cache the sentence-transformers embedding model
HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model cached")

# Pre-cache the reranker model
FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False, device="cpu")
print("Reranker model cached")

print("All models cached successfully")
