import time
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from config import Config

def initialize_vector_store():
    """
    Initialize and return a Pinecone vector store.
    Creates the index if it doesn't exist.
    
    Returns:
        PineconeVectorStore: The initialized vector store
    """
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)

    index_name = "chatime"

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="enabled",
        )
        # Wait for the index to be ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)
    return PineconeVectorStore(index=index, embedding=OpenAIEmbeddings()) 