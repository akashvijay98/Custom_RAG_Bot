from qdrant_client import QdrantClient

# Connect to Qdrant
qdrant_client = QdrantClient("localhost", port=6333)

# Delete the collection
collection_name = "pdf_embeddings"
qdrant_client.delete_collection(collection_name)

print(f"Collection '{collection_name}' deleted.")
