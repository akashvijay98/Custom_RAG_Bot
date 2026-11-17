import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_invalid_query_type_returns_400(self):
        response = self.client.post("/generate", json={"query": 123})
        self.assertEqual(response.status_code, 400)

    def test_backend_failure_returns_500(self):
        backend = SimpleNamespace(
            semantic_similarity_search=lambda query: (_ for _ in ()).throw(
                RuntimeError("qdrant unavailable")
            )
        )
        with patch.dict(sys.modules, {"similarity_search_using_vector_db": backend}):
            response = self.client.post(
                "/similarity_search_query", json={"query": "question"}
            )
        self.assertEqual(response.status_code, 500)
        self.assertIn("qdrant unavailable", response.get_json()["error"])

    def test_health_reports_backend_status(self):
        backend = SimpleNamespace(
            backend_status=lambda: {
                "qdrant": "ok",
                "collection": "test",
                "collection_exists": True,
                "points": 12,
            }
        )
        with patch.dict(sys.modules, {"similarity_search_using_vector_db": backend}):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["backend"]["points"], 12)


if __name__ == "__main__":
    unittest.main()
