# Agent Instructions

## Project Overview

This is a Python RAG chatbot project for Porsche PDF data. It uses:

- Flask API in `server/app.py`
- Qdrant vector database via `QuadrantDBStore/docker-compose.yml`
- AWS Bedrock for embeddings and generation in `server/similarity_search_using_vector_db.py`
- LangGraph for the agentic RAG workflow
- PDF ingestion utilities under `util/`

The default Qdrant collection is `pdf_embeddings`, and the local Qdrant URL is `http://localhost:6333`.

## Repository Layout

- `server/app.py`: Flask API entrypoint.
- `server/similarity_search_using_vector_db.py`: Bedrock, Qdrant, LangChain, and LangGraph RAG logic.
- `QuadrantDBStore/docker-compose.yml`: Qdrant service definition.
- `util/seed_qdrant_bedrock.py`: Preferred Bedrock-based PDF seeding script.
- `util/pdf_to_vector_db_embeddings.py`: Older SentenceTransformer-based PDF ingestion script.
- `util/pdf_dump/`: Existing Porsche PDF source files.
- `static/RAGAIBOT.png`: High-level design image used by the README.

## Setup

Create and activate a virtual environment from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies from the plain-text server lock file:

```bash
pip install -r server/requirements.lock
```

Note: `requirements.lock` at the repo root and `server/requirements.txt` appear to be NUL-separated/UTF-16-like files. Do not reformat or replace them unless the task is specifically about dependency file cleanup. Prefer `server/requirements.lock` for installs.

## Running Locally

Start Qdrant:

```bash
cd QuadrantDBStore
docker-compose up
```

Start the Flask API:

```bash
cd server
python app.py
```

The API runs on Flask's default `http://localhost:5000`.

Useful endpoints:

- `POST /similarity_search_query` with JSON `{"query": "..."}`
- `POST /generate` with JSON `{"query": "..."}`

## Data Seeding

Use the Bedrock-based seeding script when populating Qdrant:

```bash
cd util
python seed_qdrant_bedrock.py --input-dir pdf_input --dump-dir pdf_dump
```

This script expects AWS credentials with Bedrock runtime access and uses:

- Region: `us-east-1`
- Embedding model: `amazon.titan-embed-text-v1`
- Collection: `pdf_embeddings`

The seeding script moves processed PDFs from the input directory into the dump directory. Be careful when changing this behavior because it mutates local files.

## AWS and Model Notes

`server/similarity_search_using_vector_db.py` currently hardcodes:

- `AWS_REGION = "us-east-1"`
- Embeddings: `amazon.titan-embed-text-v1`
- Chat model: `amazon.nova-pro-v1:0`

Avoid changing model IDs, region, vector dimensions, or collection names casually. Those values must stay compatible with the vectors already stored in Qdrant.

## Testing and Verification

There is no formal pytest suite in this repo at the moment. Before finishing server or RAG changes, run at least:

```bash
python -m py_compile server/app.py server/similarity_search_using_vector_db.py util/seed_qdrant_bedrock.py
```

When Qdrant and AWS credentials are available, also smoke test:

```bash
curl -X POST http://localhost:5000/similarity_search_query \
  -H 'Content-Type: application/json' \
  -d '{"query":"give me porsche info"}'
```

For `/generate`, expect live Bedrock calls.

## Code Guidelines

- Keep changes scoped. This repo is small and script-oriented, so avoid introducing large framework abstractions.
- Preserve the Flask API contract unless the user asks for an API change.
- Prefer explicit constants near the existing configuration style.
- Do not silently change Qdrant collection names or payload keys. Existing retrieval expects document content to be available as page content through LangChain/Qdrant.
- Handle external service failures clearly. Qdrant and Bedrock calls are common failure points.
- Do not commit generated caches, virtual environments, downloaded model files, or local Qdrant storage.

## Known Project Quirks

- `test.py` is a standalone dynamic-programming script and is not a test suite for the RAG app.
