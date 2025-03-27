import os
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch

# Constants
QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "pdf_embeddings"

# Initialize the Qdrant client
qdrant_client = QdrantClient(QDRANT_URL, port=QDRANT_PORT)

# Setup Hugging Face RAG model, tokenizer, and retriever
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="default", use_dummy_dataset=True)

# Create the Qdrant vector store, use LangChain with embeddings
vector_store = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME
)


def generate_answer(query: str):
    """
    Generates an answer to the query using the RAG model with retrieved context from the vector database.
    """
    # Tokenize the query
    input_dict = tokenizer(query, return_tensors="pt")

    # Retrieve relevant documents using the Qdrant vector store
    context_input_ids, context_attention_mask = retriever(input_dict['input_ids'])

    # Use the RAG model to generate an answer
    generated_ids = model.generate(input_ids=input_dict['input_ids'],
                                   attention_mask=input_dict['attention_mask'],
                                   context_input_ids=context_input_ids,
                                   context_attention_mask=context_attention_mask,
                                   decoder_start_token_id=model.config.pad_token_id)

    # Decode the generated answer
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return answer


def query_vector_db(query: str, top_k: int = 3):
    """Queries the Qdrant vector database and gets the most relevant documents based on the query."""

    # Perform a similarity search (embeddings will be generated internally)
    search_results = vector_store.similarity_search(query, k=top_k)

    return search_results


# Main function to query and get the response
if __name__ == "__main__":
    query = input("Enter your query: ")  # Prompt the user for input query

    # First, perform the similarity search to retrieve relevant documents
    search_results = query_vector_db(query)

    # Now, use the RAG model to generate an answer based on the retrieved documents
    context = " ".join([result.page_content for result in search_results])
    answer = generate_answer(f"Context: {context} Question: {query}")

    print("\n## Generated Answer:")
    print(answer)

    # Optionally, print the relevant documents with headings and spacing
    print("\n## Relevant Documents:")
    for i, result in enumerate(search_results, start=1):
        print(f"\n### Document {i}")
        print("#### Metadata")
        for key, value in result.metadata.items():
            print(f"{key}: {value}")
        print("\n#### Page Content")
        print(result.page_content)
