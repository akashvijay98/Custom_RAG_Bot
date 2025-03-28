import os
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from sentence_transformers import SentenceTransformer
import torch

# Constants
QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "pdf_embeddings"

# Initialize the Qdrant client
qdrant_client = QdrantClient(QDRANT_URL, port=QDRANT_PORT)

# Setup Hugging Face RAG model, tokenizer, and retriever
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# Create the Qdrant vector store, use LangChain with embeddings
vector_store = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model.encode
)


def query_vector_db(query: str, top_k: int = 3):
    """Queries the Qdrant vector database and gets the most relevant documents based on the query."""

    # Perform a similarity search (embeddings will be generated internally)
    search_results = vector_store.similarity_search(query, k=top_k)

    return search_results


# Main function to query and get the response
if __name__ == "__main__":
    query = input("Enter your query: ")  # Prompt the user for input query
    search_results = query_vector_db(query)

    # Print the relevant documents with headings and spacing
    print("\n## Relevant Documents:")
    for i, result in enumerate(search_results, start=1):
        print(f"\n### Document {i}")
        print("#### Metadata")
        for key, value in result.metadata.items():
            print(f"{key}: {value}")
        print("\n#### Page Content")
        print(result.page_content)
