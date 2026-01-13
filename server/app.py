from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
from similarity_search_using_vector_db import semantic_similarity_search, agentic_response

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# API route for similarity search
@app.route('/similarity_search_query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get("query", "")

    if not query_text:
        return jsonify({"error": "Query is required"}), 400

    try:
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
        data = request.get_json()
        query_text = data.get("query", "")
        if not query_text:
            return jsonify({"error": "Prompt is required"}), 400

        # CALL THE AGENT
        # The agent will perform multi-step reasoning internally
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
