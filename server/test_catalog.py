import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qdrant_client import QdrantClient

from app import app
from catalog_extract import build_records, candidate_facts, extract_pdf, validate_metadata
from catalog_store import CatalogStore


class Encoder:
    identity = "test-encoder@v1"
    dimension = 3

    def split(self, text):
        return [text] if text.strip() else []

    def encode(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


def metadata():
    return {"vehicles": [{"make": "Example", "model": "Coupe", "year": 2025,
                          "trim": "Sport", "powertrain": "petrol", "transmission": "automatic",
                          "drivetrain": "RWD", "market": "US", "pages": [1], "table_header": "Sport"}]}


PAGES = [{"page": 1, "text": "Top speed: 150 mph\nCombined EPA: 30 mpg US\nMSRP: USD 40,000", "tables": []}]


class ExtractionTests(unittest.TestCase):
    def test_normalization_and_measurement_provenance(self):
        facts = list(candidate_facts(PAGES[0]["text"]))
        self.assertEqual(len(facts), 3)
        self.assertAlmostEqual(facts[0]["value"], 241.4016)
        self.assertEqual(facts[0]["unit"], "km/h")
        self.assertAlmostEqual(facts[1]["value"], 7.840486)
        self.assertEqual(facts[2]["value"], 40000)
        self.assertTrue(all(f["status"] == "pending" for f in facts))
        self.assertIn("EPA", facts[1]["qualifiers"])

    def test_missing_and_ambiguous_units_are_not_invented(self):
        self.assertEqual(list(candidate_facts("Top speed: 200\nPrice: 40000")), [])
        fact = next(candidate_facts("Combined: 30 mpg"))
        self.assertEqual(fact["unit"], "mpg")
        self.assertEqual(fact["value"], 30)

    def test_model_columns_and_unassigned_narrative(self):
        meta = metadata()
        meta["vehicles"].append({**meta["vehicles"][0], "trim": "Touring", "table_header": "Touring"})
        vehicles = validate_metadata(meta)
        pages = [{"page": 1, "text": "Top speed 150 mph", "tables": [
            {"rows": [["Specification", "Sport", "Touring"], ["Top speed", "150 mph", "140 mph"]], "bbox": [0, 0, 100, 100]}]}]
        chunks, facts, warnings = build_records(pages, vehicles, "document", Encoder().split)
        assigned = [f for f in facts if f["vehicle_id"]]
        self.assertEqual(len(assigned), 2)
        self.assertEqual({f["vehicle_id"]: f["original_value"] for f in assigned},
                         {vehicles[0]["id"]: 150, vehicles[1]["id"]: 140})
        self.assertIsNone(chunks[0]["metadata"]["vehicle_id"])
        self.assertTrue(warnings)
        self.assertTrue(any("Sport | Touring" in c["page_content"] for c in chunks))

    def test_invalid_metadata(self):
        for value in ([], {"vehicles": "bad"}, {"vehicles": [{}]}):
            with self.assertRaises(ValueError):
                validate_metadata(value)
        bad = metadata()
        bad["vehicles"][0]["pages"] = [True]
        with self.assertRaises(ValueError):
            validate_metadata(bad)

    def test_page_mapping_out_of_range(self):
        meta = metadata()
        meta["vehicles"][0]["pages"] = [2]
        with self.assertRaises(ValueError):
            build_records(PAGES, validate_metadata(meta), "doc", Encoder().split)

    def test_real_catalog_extraction_preserves_pages(self):
        path = Path(__file__).resolve().parents[1] / "util/pdf_input/2021-Porsche-911-Turbo-S-Brochure.pdf"
        pages = extract_pdf(path.read_bytes())
        self.assertGreater(len(pages), 1)
        self.assertEqual([p["page"] for p in pages], list(range(1, len(pages) + 1)))
        self.assertTrue(any("Porsche" in p["text"] for p in pages))
        table = next(t for t in pages[29]["tables"] if "911 Turbo S" in t["rows"][0])
        self.assertEqual(table["rows"][0][-1], "911 Turbo S Cabriolet")
        row = next(r for r in table["rows"] if r[0] == "Top track speed")
        self.assertEqual(row[1:], ["205 mph", "205 mph"])

    def test_units_in_labels(self):
        fact = next(candidate_facts("Top speed (km/h): 250"))
        self.assertEqual(fact["value"], 250)
        self.assertEqual(fact["source_excerpt"], "Top speed (km/h): 250")

    def test_conflicting_values_are_retained(self):
        pages = [{"page": 1, "text": "Top speed 150 mph; maximum speed 140 mph", "tables": []}]
        _, facts, _ = build_records(pages, [], "doc", Encoder().split)
        self.assertEqual({f["original_value"] for f in facts}, {140, 150})

    def test_malformed_pdf_rejected(self):
        for data in (b"hello", b"%PDF-not-a-document"):
            with self.assertRaises(ValueError):
                extract_pdf(data)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.client = QdrantClient(":memory:")
        self.store = CatalogStore(self.temp.name, Encoder(), self.client)
        self.extract = patch("catalog_extract.extract_pdf", return_value=PAGES)
        self.extract.start()

    def tearDown(self):
        self.extract.stop()
        self.client.close()
        self.temp.cleanup()

    def ingest(self, data=b"%PDF-test"):
        return self.store.ingest(data, "catalog.pdf", metadata())

    def test_idempotent_upload_and_search_filters(self):
        first = self.ingest()
        second = self.ingest()
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(self.client.count(self.store.collection).count, first["chunks"])
        vehicle_id = self.store.vehicles()[0]["id"]
        results = self.store.search("speed", vehicle_id)
        self.assertTrue(results)
        self.assertEqual(results[0]["metadata"]["page"], 1)
        self.assertEqual(self.store.search("speed", "other-vehicle"), [])
        self.assertEqual(self.store.specifications(vehicle_id), [])
        self.assertEqual((Path(self.temp.name) / "sources" / f"{first['content_hash']}.pdf").read_bytes(), b"%PDF-test")

    def test_vector_overview_summarizes_qdrant_payloads(self):
        self.ingest()
        overview = self.store.vector_overview()
        self.assertEqual(overview["points"], 1)
        self.assertEqual(overview["sources"][0]["source_file"], "catalog.pdf")
        self.assertEqual(overview["sources"][0]["kinds"], {"page_text": 1})
        self.assertEqual(overview["sources"][0]["unassigned_points"], 0)

    def test_partial_failure_is_not_searchable_and_can_resume(self):
        original = self.client.upsert

        def upload_then_fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("simulated timeout after write")

        with patch.object(self.client, "upsert", side_effect=upload_then_fail):
            with self.assertRaises(RuntimeError):
                self.ingest()
        self.assertEqual(self.store.search("speed"), [])
        result = self.ingest()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(self.client.count(self.store.collection).count, result["chunks"])

    def test_review_requires_conditions_and_source_vehicle(self):
        doc = self.ingest()
        fact = next(f for f in doc["facts"] if f["metric"] == "top_speed")
        with self.assertRaises(ValueError):
            self.store.review_fact(fact["id"], {"status": "approved"})
        with self.assertRaises(ValueError):
            self.store.review_fact(fact["id"], {"status": "approved", "vehicle_id": "wrong", "conditions": {"configuration": "standard"}})
        approved = self.store.review_fact(fact["id"], {"status": "approved", "conditions": {"configuration": "standard"}})
        self.assertEqual(len(self.store.specifications(approved["vehicle_id"])), 1)
        self.ingest()
        self.assertEqual(len(self.store.specifications(approved["vehicle_id"])), 1)

    def test_document_versions_and_model_isolation(self):
        first, second = self.ingest(), self.ingest(b"%PDF-version2")
        self.assertNotEqual(first["id"], second["id"])
        other = Encoder()
        other.identity = "different-model-same-dimension"
        other_store = CatalogStore(self.temp.name, other, self.client)
        self.assertNotEqual(other_store.collection, self.store.collection)
        self.assertEqual(other_store.search("speed"), [])

    def test_entity_resolution_normalizes_aliases_and_clarifies_make_only_requests(self):
        self.store.ingest(b"%PDF-honda", "Honda-Accord-2022-MY.pdf", {})
        self.store.ingest(b"%PDF-porsche", "Porsche-Cayenne-2021-INT.pdf", {})

        resolved = self.store.resolve_vehicle_query(
            "compare the horsepower of accord lx and porche cayene"
        )
        self.assertIsNone(resolved["clarification"])
        self.assertIn("Honda Accord", resolved["query"])
        self.assertIn("Porsche Cayenne", resolved["query"])

        clarification = self.store.resolve_vehicle_query("compare porsche with honda")
        self.assertIn("Honda: Accord (2022)", clarification["clarification"])
        self.assertIn("Porsche: Cayenne (2021)", clarification["clarification"])


class CatalogApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.qdrant = QdrantClient(":memory:")
        self.store = CatalogStore(self.temp.name, Encoder(), self.qdrant)
        app.extensions["catalog_service"] = self.store
        self.client = app.test_client()

    def tearDown(self):
        app.extensions.pop("catalog_service", None)
        self.qdrant.close()
        self.temp.cleanup()

    def test_upload_status_source_and_duplicate(self):
        def upload():
            return self.client.post("/catalog/documents", data={"file": (io.BytesIO(b"%PDF-test"), "../../catalog.pdf"), "metadata": json.dumps(metadata())})
        with patch("catalog_extract.extract_pdf", return_value=PAGES):
            result = upload()
            self.assertEqual(result.status_code, 201)
            document_id = result.json["id"]
            self.assertEqual(upload().status_code, 200)
        self.assertEqual(result.json["filename"], "catalog.pdf")
        self.assertEqual(self.client.get(f"/catalog/documents/{document_id}").status_code, 200)
        source = self.client.get(f"/catalog/documents/{document_id}/source")
        self.assertEqual(source.data, b"%PDF-test")
        source.close()

    def test_upload_validation(self):
        self.assertEqual(self.client.post("/catalog/documents").status_code, 400)
        for content, name, meta in ((b"bad", "test.pdf", "{}"), (b"%PDF-test", "test.txt", "{}"), (b"%PDF-test", "test.pdf", "[]")):
            result = self.client.post("/catalog/documents", data={"file": (io.BytesIO(content), name), "metadata": meta})
            self.assertEqual(result.status_code, 400)
        self.assertEqual(self.client.get("/catalog/documents/missing").status_code, 404)

    def test_search_validation(self):
        for data in ({"query": 123}, {"query": "x", "limit": True}, {"query": "x", "vehicle_id": []}):
            self.assertEqual(self.client.post("/catalog/search", json=data).status_code, 400)

    def test_request_size_limit(self):
        with patch.dict(app.config, {"MAX_CONTENT_LENGTH": 100}):
            response = self.client.post("/catalog/documents", data={"file": (io.BytesIO(b"%PDF-" + b"x" * 200), "x.pdf")})
        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
