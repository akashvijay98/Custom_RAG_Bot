import os
import uuid
from PyPDF2 import PdfReader
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer



# Constants
QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "pdf_embeddings"
MODEL_ID = "all-MiniLM-L6-v2"

# Initialize Qdrant client
qdrant_client = QdrantClient(QDRANT_URL, port=QDRANT_PORT)

# Initialize SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')



# Flag to avoid redundant collection checks
collection_checked = False


def ensure_collection_exists(client, collection_name, embedding_size):
    """Checks if the collection exists and creates it if not."""
    global collection_checked
    if collection_checked:
        return

    collections = client.get_collections().collections
    if collection_name not in [col.name for col in collections]:
        print(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance="Cosine")
        )
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")

    collection_checked = True


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])


def split_text(text, chunk_size=500, chunk_overlap=50):
    """Splits text into chunks with overlap."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)


def save_to_qdrant(file_name, embedding, page_content, qdrant_client, qdrant_collection):
    """Saves embeddings to Qdrant database."""
    ensure_collection_exists(qdrant_client, qdrant_collection, len(embedding))

    point_id = str(uuid.uuid4())

    payload = {
        "file": file_name,
        "page_content": page_content
    }

    qdrant_client.upsert(
        collection_name=qdrant_collection,
        points=[PointStruct(id=point_id, vector=embedding, payload=payload)]
    )
    print(f"Inserted embedding for '{file_name}' into Qdrant.")


def process_pdfs(input_folder, output_folder, chunk_size, chunk_overlap):
    """Processes PDFs: extract text, generate embeddings, and store in Qdrant."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, file)
            print(f"Processing {file}...")

            text = extract_text_from_pdf(pdf_path)
            chunks = split_text(text, chunk_size, chunk_overlap)

            for i, chunk in enumerate(chunks):
                embedding = embedding_model.encode(chunk)
                save_to_qdrant(f"{file}_chunk_{i}", embedding, chunk, qdrant_client, COLLECTION_NAME)

            os.rename(pdf_path, os.path.join(output_folder, file))
            print(f"Moved {file} to {output_folder}")


if __name__ == "__main__":
    INPUT_FOLDER = "pdf_input"
    OUTPUT_FOLDER = "pdf_dump"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    process_pdfs(INPUT_FOLDER, OUTPUT_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP)
