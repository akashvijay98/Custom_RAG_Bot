from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Local catalog endpoints have no dependency on the legacy Bedrock RAG backend.
from catalog_api import catalog
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
app.register_blueprint(catalog)

@app.get('/')
def chat_ui():
    return render_template('chat.html')

@app.route('/health', methods=['GET'])
def health():
    try:
        from similarity_search_using_vector_db import backend_status

        backend = backend_status()
        status = "ok" if backend["collection_exists"] else "degraded"
        return jsonify({"status": status, "service": "Flask API", "backend": backend}), 200 if status == "ok" else 503
    except Exception as exc:
        return jsonify({"status": "unavailable", "service": "Flask API", "error": str(exc)}), 503


# API route for similarity search
@app.route('/similarity_search_query', methods=['POST'])
def query():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required"}), 400
    query_text = data.get("query", "")

    if not isinstance(query_text, str) or not query_text.strip():
        return jsonify({"error": "Query is required"}), 400
    query_text = query_text.strip()

    try:
        from similarity_search_using_vector_db import semantic_similarity_search
        results = semantic_similarity_search(query_text)
        # Convert LangChain Document objects to JSON-serializable format
        response_data = []
        for doc in results:
            response_data.append({
                "content": doc.page_content,  # Extract text content
                "metadata": doc.metadata  # Extract metadata (if needed)
            })

        return jsonify({"response": response_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "A JSON object is required"}), 400
        query_text = data.get("query", "")
        if not isinstance(query_text, str) or not query_text.strip():
            return jsonify({"error": "Prompt is required"}), 400
        query_text = query_text.strip()

        # CALL THE AGENT
        # The agent will perform multi-step reasoning internally
        from similarity_search_using_vector_db import agentic_response
        answer = agentic_response(query_text)

        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# @app.route("/evaluate/recall-at-k", methods=['POST'])
# def calculate_recall_at_k():
#     data = request.get_json()

#     test_set = data.get("test_set", [])
#     k = data.get("k", 3)

#     hits = 0
#     total = len(test_set)

#     async def evaluate():
#         nonlocal hits
#         for sample in test_set:
#             query = sample["query"]
#             expected_chunk = sample["expected_chunk"]

#             results = await async_query_vector_db(query, k)  # async call

#             result_texts = [
#                 f"{doc.page_content} {str(doc.metadata)}"
#                 for doc in results
#             ]

#             print("query", query)
#             print("expected", expected_chunk)
#             print("resultText:", result_texts)

#             if any(expected_chunk in result for result in result_texts):
#                 hits += 1

#     # Run the async evaluate function synchronously
#     asyncio.run(evaluate())

#     recall_at_k = hits / total if total > 0 else 0.0

#     return jsonify({
#         "recall@k": round(recall_at_k, 4),
#         "k": k,
#         "total_queries": total,
#         "hits": hits
#     })

if __name__ == '__main__':
    app.run(debug=True)
