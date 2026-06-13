import chromadb
from chromadb.api import ClientAPI
from fastapi import Depends
from typing import Annotated, List
from datetime import datetime

chroma_client= None

def init_chroma_client():
    """Initialize Chroma client and return it"""
    global chroma_client
    if chroma_client is None:
        chroma_client = chromadb.PersistentClient(path="../../chroma")
    return chroma_client

VectorDBClient = Annotated[ClientAPI, Depends(init_chroma_client)]

def _get_or_create_collection(client, name:str):
    collection = client.get_or_create_collection(
        name=name,
        metadata={
        "description": "collection for storing vector embeddings",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return collection

def add_data_to_collection(data, collection):
    collection.add(documents=data["documents"], metadatas=data["metadatas"], ids=data["ids"])

def query_collection(client, collection_name: str, query_text:str | List[str],top_k:int = 2):
    collection = client.get_collection(name=collection_name)
    return collection.query(query_texts=[query_text] if isinstance(query_text, str) else query_text, n_results=top_k)
    
def get_collection_by_id(ids: int | List[int], collection):
    return collection.get(ids=[ids] if isinstance(ids, int) else ids)

def get_all_collections(client):
    return client.list_collections()