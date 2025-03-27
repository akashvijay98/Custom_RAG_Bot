import os
import uuid
from PyPDF2 import PdfReader
from transformers import RagTokenizer, RagRetriever
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants
QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "pdf_embeddings"
MODEL_ID = "facebook/rag-sequence-nq"

# Initialize Hugging Face RAG model and tokenizer
tokenizer = RagTokenizer.from_pretrained(MODEL_ID)
retriever = RagRetriever.from_pretrained(MODEL_ID, index_name="default", use_dummy_dataset=True)

# Initialize the Qdrant client
qdrant_client = QdrantClient(QDRANT_URL, port=QDRANT_PORT)

# Flag to ensure collection is checked and created only once
collection_checked = False


def ensure_collection_exists(client, collection_name, embedding_size):
    """Checks if the collection exists and creates it if not."""
    global collection_checked
    if collection_checked:
        return  # Skip if collection has already been checked/created

    # Get the list of collections using the correct attribute
    collections = client.get_collections().collections

    # Check if the collection exists
    if collection_name not in [col.name for col in collections]:
        print(f"Collection '{collection_name}' does not exist. Creating it now...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": embedding_size, "distance": "Cosine"}
        )
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")

    # Set the flag to True to avoid repeated checks
    collection_checked = True


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


def split_text(text, chunk_size=500, chunk_overlap=50):
    """Splits text into chunks with overlap."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)


def save_to_qdrant(file_name, embedding, page_content, qdrant_client, qdrant_collection):
    """Saves embeddings to Qdrant database after ensuring the collection exists."""
    # Ensure the collection exists before inserting data
    ensure_collection_exists(qdrant_client, qdrant_collection, len(embedding))

    # Generate a UUID for the point ID
    point_id = str(uuid.uuid4())

    # Create the payload with the page content and other relevant info
    payload = {
        "file": file_name,
        "page_content": page_content  # Store the actual page content
    }

    # Upsert the embedding to the collection along with the payload
    qdrant_client.upsert(
        collection_name=qdrant_collection,
        points=[PointStruct(id=point_id, vector=embedding, payload=payload)]
    )
    print(f"Embedding for '{file_name}' inserted into the collection with page content.")


def process_pdfs(input_folder, output_folder, chunk_size, chunk_overlap):
    """Processes PDF files by extracting text, generating embeddings, and moving to output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, file)
            print(f"Processing {file}...")

            text = extract_text_from_pdf(pdf_path)
            chunks = split_text(text, chunk_size, chunk_overlap)

            # Save chunks to Qdrant
            for i, chunk in enumerate(chunks):
                # Generate embeddings using the RAG retriever
                embedding = retriever.get_embedding(chunk)  # Use RAG-specific embeddings for context
                page_content = chunk  # The chunk content is the page content

                # Save embedding to Qdrant with page content as part of the payload
                save_to_qdrant(f"{file}_chunk_{i}", embedding, page_content, qdrant_client, COLLECTION_NAME)

            # Move the original PDF to output folder
            os.rename(pdf_path, os.path.join(output_folder, file))
            print(f"Processed {file} and moved to {output_folder}")


if __name__ == "__main__":
    INPUT_FOLDER = "pdf_input"
    OUTPUT_FOLDER = "pdf_dump"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    COLLECTION_NAME = "pdf_embeddings"

    # Process PDFs
    process_pdfs(INPUT_FOLDER, OUTPUT_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP)
