"""AWS-free upload/review/search API, registered by the existing Flask app."""
import json
import os
from pathlib import Path
import threading

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.utils import secure_filename

catalog = Blueprint("catalog", __name__, url_prefix="/catalog")
_service_lock = threading.Lock()


def service():
    # Lazy: importing the API does not load a model, contact AWS, or connect Qdrant.
    with _service_lock:
        instance = current_app.extensions.get("catalog_service")
        if instance is None:
            from catalog_store import CatalogStore, LocalEncoder
            from qdrant_client import QdrantClient
            root = Path(os.getenv("CATALOG_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data" / "catalog")))
            root.mkdir(parents=True, exist_ok=True)
            encoder = LocalEncoder(root)
            url = os.getenv("CATALOG_QDRANT_URL")
            client = QdrantClient(url=url) if url else QdrantClient(path=str(root / "qdrant"))
            instance = CatalogStore(root, encoder, client)
            current_app.extensions["catalog_service"] = instance
        return instance


@catalog.errorhandler(ValueError)
def invalid_request(exc):
    return jsonify(error=str(exc)), 400


@catalog.errorhandler(KeyError)
def not_found(exc):
    return jsonify(error=exc.args[0]), 404


@catalog.errorhandler(Exception)
def backend_error(exc):
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        return jsonify(error=exc.description), exc.code
    current_app.logger.exception("Catalog operation failed")
    return jsonify(error="Catalog service failed; inspect server logs and retry", detail=str(exc)), 503


@catalog.post("/documents")
def upload():
    files = request.files.getlist("file")
    if len(files) != 1 or not files[0].filename:
        raise ValueError("Upload exactly one PDF in the multipart file field")
    filename = secure_filename(files[0].filename)
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Filename must end with .pdf")
    try:
        metadata = json.loads(request.form.get("metadata", "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError("metadata must be valid JSON") from exc
    from catalog_extract import validate_metadata
    validate_metadata(metadata)
    data = files[0].read()
    if not data.startswith(b"%PDF-"):
        raise ValueError("Upload must be a PDF")
    result = service().ingest(data, filename, metadata)
    return jsonify(result), 200 if result["duplicate"] else 201


@catalog.get("/documents/<document_id>")
def document(document_id):
    return jsonify(service().document(document_id))


@catalog.get("/documents/<document_id>/source")
def source(document_id):
    store = service()
    record = store.document(document_id)
    return send_file(store.root / "sources" / f"{record['content_hash']}.pdf",
                     mimetype="application/pdf", as_attachment=False, download_name=record["filename"])


@catalog.get("/documents/<document_id>/pages/<int:number>")
def page(document_id, number):
    return jsonify(service().page(document_id, number))


@catalog.get("/vehicles")
def vehicles():
    return jsonify(vehicles=service().vehicles())


@catalog.get("/vehicles/<vehicle_id>/specifications")
def specifications(vehicle_id):
    return jsonify(facts=service().specifications(vehicle_id))


@catalog.patch("/facts/<fact_id>")
def review(fact_id):
    return jsonify(service().review_fact(fact_id, request.get_json(silent=True)))


@catalog.post("/search")
def search():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("query"), str) or not data["query"].strip():
        raise ValueError("query must be a non-empty string")
    if len(data["query"]) > 2000:
        raise ValueError("query must be at most 2000 characters")
    limit = data.get("limit", 5)
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("limit must be an integer from 1 to 50")
    vehicle_id = data.get("vehicle_id")
    if vehicle_id is not None and not isinstance(vehicle_id, str):
        raise ValueError("vehicle_id must be a string")
    return jsonify(results=service().search(data["query"].strip(), vehicle_id, limit))


@catalog.get("/overview")
def overview():
    return jsonify(service().vector_overview())


@catalog.get("/health")
def health():
    return jsonify(status="ok", service="local catalog ingestion", initialized="catalog_service" in current_app.extensions)
