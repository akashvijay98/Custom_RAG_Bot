Prerequisites

Ensure you have the following installed before proceeding:

Docker

Python 3.x

Flask

Postman

Setup and Running the Project

1. Start QdrantDB

Navigate to the QdrantDBDatastore directory and start the QdrantDB container:

cd QdrantDBDatastore
docker-compose up

2. Start the Flask API Server

Navigate to the Server directory and run the Flask application:

cd Server
python app.py

3. Test with Postman

Open Postman and send a POST request to:

http://localhost:5000/similarity_search_query

with the following JSON raw payload:

{
  "query": "give me porche info"
}

Project Structure

RAGBot/
│-- QdrantDBDatastore/     # Directory containing QdrantDB setup
│-- Server/                # Flask API server
│   │-- app.py             # Main application file
│   │-- requirements.txt   # Python dependencies
│-- README.md              # Project documentation

Dependencies

To install the required dependencies for the Flask API server:
pip install -r Server/requirements.txt
