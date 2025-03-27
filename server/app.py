from flask import Flask, request, jsonify
from flask_cors import CORS
from similarity_search_using_vector_db import query_vector_db, generate_answer  # Import functions from the rag_logic file

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

    # Get the similarity search results
    try:
        results = query_vector_db(query_text)
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


# API route for generating an answer based on the query using RAG
@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Get relevant documents using similarity search
        search_results = query_vector_db(prompt)

        # Use the RAG model to generate an answer based on the retrieved documents
        context = " ".join([result.page_content for result in search_results])
        answer = generate_answer(f"Context: {context} Question: {prompt}")

        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
