from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # A good default model for general purpose embeddings

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333")

# Prepare your documents, metadata, and IDs
docs = ["Qdrant has Langchain integrations", "Qdrant also has Llama Index integrations"]

docs = [
    "Qdrant has", "Langchain integrations"
    "Qdrant also has Llama Index integrations"
]
metadata = [
    {"source": "Langchain-docs"},
    {"source": "Linkedin-docs"},
]
ids = [42, 2]

# Generate embeddings for the documents
embeddings = model.encode(docs).tolist()

# Create a collection with the appropriate vector size (384 for all-MiniLM-L6-v2)
client.recreate_collection(
    collection_name="demo_collection",
    vectors_config={
        "text": {
            "size": 384,  # Vector size for all-MiniLM-L6-v2
            "distance": "Cosine"
        }
    }
)

# Insert the documents with their embeddings
client.upsert(
    collection_name="demo_collection",
    points=[
        {
            "id": idx,
            "vector": {"text": embedding},
            "payload": {"text": doc, **meta}
        }
        for idx, doc, embedding, meta in zip(ids, docs, embeddings, metadata)
    ]
)