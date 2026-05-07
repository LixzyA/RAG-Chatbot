import chromadb
from chromadb.api import ClientAPI
from fastapi import Depends
from typing import Annotated, List
import logging
from exception import CreateCollectionException, ChromaQueryException, ChromaInsertionException, CollectionNotFoundException
from datetime import datetime

chroma_client= None

def init_chroma_client():
    """Initialize Chroma client and return it"""
    global chroma_client
    if chroma_client is None:
        chroma_client = chromadb.Client()
    return chroma_client

type VectorDBClient = Annotated[ClientAPI, Depends(init_chroma_client)]
    
def create_collection(client: VectorDBClient, name: str):
    if name in client.list_collections():
        logging.warning(f"{name} Collection exists")
        return client.list_collections()
    
    try:
        client.create_collection(name=name, metadata={
            "description": "collection for storing vector embeddings",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        logging.info(f"{name} Collection created successfully")
        return client.list_collections()
    except Exception as e:
        logging.error(f"Failed to create collection {name}: {str(e)}")
        raise CreateCollectionException(message=str(e))

def get_collection(client: VectorDBClient, collection_name: str = "file_mgt"):
    if collection_name not in client.list_collections():
        raise CollectionNotFoundException(message=f"{collection_name} Collection does not exist")
    return client.get_collection(name=collection_name)

def add_data_to_collection(data, collection= Depends(get_collection)):    
    try:
        collection.add(
            documents=data["documents"],
            metadatas=data["metadatas"],
            ids=data["ids"]
        )
        logging.info(f"{len(data['ids'])} data added to collection successfully")
        return {"status": "200", "detail": "Data added successfully"}
    except Exception as e:
        logging.error(f"Failed to insert data into Chroma: {str(e)}")
        raise ChromaInsertionException(message=str(e))
    
def query_collection(query_text:str | List[str],top_k:int = 2, collection= Depends(get_collection)):
    try:
        logging.info(f"Query text: {query_text} and Top k: {top_k}")
        results=collection.query(
            query_texts=[query_text] if isinstance(query_text, str) else query_text, 
            n_results=top_k)
        logging.info(f"Query results: {results}")
        return results
    except Exception as e:
        logging.error(f"Failed to query collection {collection.name}: {str(e)}")
        raise ChromaQueryException(message=str(e))
    
def get_collection_by_id(ids: int | List[int], collection= Depends(get_collection)):
    try:
        logging.info(f"Get collection with id: {id}")
        results=collection.get(ids=[id] if isinstance(id, int) else id)
        logging.info(f"Get collection results: {results}")
        return results
    except Exception as e:
        logging.error(f"Failed to get collection with id {id}: {str(e)}")
        raise ChromaQueryException(message=str(e))