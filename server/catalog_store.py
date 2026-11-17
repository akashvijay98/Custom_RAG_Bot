"""Local persistence and model-isolated Qdrant indexing."""
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
from difflib import SequenceMatcher

from catalog_extract import PIPELINE_VERSION


class LocalEncoder:
    def __init__(self, data_dir):
        from sentence_transformers import SentenceTransformer
        self.name = os.getenv("CATALOG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.name, device="cpu", cache_folder=str(data_dir / "models"),
                                         trust_remote_code=False)
        # Hash actual weights/tokenizer/config: model names and vector dimensions
        # alone cannot distinguish incompatible revisions or replaced local files.
        import torch
        fingerprint = hashlib.sha256(str(self.model).encode())
        fingerprint.update(str(self.model.max_seq_length).encode())
        tokenizer = self.model.tokenizer
        if getattr(tokenizer, "is_fast", False):
            fingerprint.update(tokenizer.backend_tokenizer.to_str().encode())
        else:
            fingerprint.update(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode())
        for key, tensor in sorted(self.model.state_dict().items()):
            fingerprint.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            fingerprint.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        self.identity = f"{self.name}@{fingerprint.hexdigest()}"
        dimension_method = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dimension = dimension_method()

    def split(self, text):
        tokenizer = self.model.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
        budget = min(self.model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False), 220)
        if budget < 32:
            raise ValueError("Embedding model token limit is too small")
        return [tokenizer.decode(tokens[i:i + budget], skip_special_tokens=True)
                for i in range(0, len(tokens), budget - 30)]

    def encode(self, texts):
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False).tolist()


class CatalogStore:
    def __init__(self, root, encoder, client):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.encoder, self.client = encoder, client
        self.lock = threading.RLock()
        suffix = hashlib.sha256(f"{encoder.identity}:{PIPELINE_VERSION}".encode()).hexdigest()[:16]
        self.collection = f"car_catalog_{suffix}"
        self.db_path = self.root / "catalog.sqlite3"
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, filename TEXT NOT NULL,
                    status TEXT NOT NULL, collection TEXT NOT NULL, metadata TEXT NOT NULL,
                    warnings TEXT NOT NULL, error TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS vehicles (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS pages (
                    document_id TEXT NOT NULL, page INTEGER NOT NULL, data TEXT NOT NULL,
                    PRIMARY KEY(document_id, page));
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL, vehicle_id TEXT,
                    status TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS facts_vehicle ON facts(vehicle_id, status);
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
            """)

    def db(self):
        from contextlib import contextmanager

        @contextmanager
        def connection():
            db = sqlite3.connect(self.db_path, timeout=60)
            db.row_factory = sqlite3.Row
            try:
                with db:
                    yield db
            finally:
                db.close()
        return connection()

    def ensure_collection(self):
        from qdrant_client import models
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=models.VectorParams(
                size=self.encoder.dimension, distance=models.Distance.COSINE))
            for field in ("document_id", "vehicle_id"):
                self.client.create_payload_index(self.collection, f"metadata.{field}", models.PayloadSchemaType.KEYWORD)
        params = self.client.get_collection(self.collection).config.params.vectors
        if not isinstance(params, models.VectorParams) or params.size != self.encoder.dimension or params.distance != models.Distance.COSINE:
            raise RuntimeError("Catalog collection embedding configuration is incompatible")

    def document(self, document_id):
        with self.db() as db:
            row = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            if row is None:
                raise KeyError("Document not found")
            result = dict(row)
            result["metadata"] = json.loads(result["metadata"])
            result["warnings"] = json.loads(result["warnings"])
            result["chunks"] = db.execute("SELECT COUNT(*) FROM chunks WHERE document_id=?", (document_id,)).fetchone()[0]
            result["facts"] = [json.loads(r[0]) for r in db.execute("SELECT data FROM facts WHERE document_id=?", (document_id,))]
            return result

    def ingest(self, data, filename, metadata):
        from catalog_extract import validate_metadata, extract_pdf, build_records, stable_id
        from qdrant_client import models
        vehicles = validate_metadata(metadata)
        digest = hashlib.sha256(data).hexdigest()
        document_id = stable_id(f"{digest}:{json.dumps(metadata, sort_keys=True)}:{self.collection}")
        # Serialize writers in this service; deploy one process for embedded mode.
        with self.lock:
            try:
                previous = self.document(document_id)
                if previous["status"] == "ready":
                    return {**previous, "duplicate": True}
            except KeyError:
                pass
            pages = extract_pdf(data)
            chunks, facts, warnings = build_records(pages, vehicles, document_id, self.encoder.split)
            source_dir = self.root / "sources"
            source_dir.mkdir(exist_ok=True)
            source_path = source_dir / f"{digest}.pdf"
            if not source_path.exists():
                source_path.write_bytes(data)
            # Commit staging before remote writes. Failed batches are safe to retry.
            with self.db() as db:
                db.execute("INSERT INTO documents(id,content_hash,filename,status,collection,metadata,warnings) VALUES(?,?,?,?,?,?,?) "
                           "ON CONFLICT(id) DO UPDATE SET status='indexing',error=NULL",
                           (document_id, digest, filename, "indexing", self.collection, json.dumps(metadata), json.dumps(warnings)))
                for vehicle in vehicles:
                    db.execute("INSERT OR IGNORE INTO vehicles VALUES(?,?)", (vehicle["id"], json.dumps(vehicle)))
                for page in pages:
                    db.execute("INSERT OR REPLACE INTO pages VALUES(?,?,?)", (document_id, page["page"], json.dumps(page)))
                for chunk in chunks:
                    chunk["metadata"].update(source_file=filename, content_hash=digest,
                                             embedding_model=self.encoder.identity, pipeline_version=PIPELINE_VERSION)
                    db.execute("INSERT OR REPLACE INTO chunks VALUES(?,?,?)", (chunk["id"], document_id, json.dumps(chunk)))
                for fact in facts:
                    db.execute("INSERT OR IGNORE INTO facts VALUES(?,?,?,?,?)",
                               (fact["id"], document_id, fact["vehicle_id"], fact["status"], json.dumps(fact)))
            try:
                self.ensure_collection()
                for start in range(0, len(chunks), 32):
                    batch = chunks[start:start + 32]
                    vectors = self.encoder.encode([c["page_content"] for c in batch])
                    if len(vectors) != len(batch) or any(len(v) != self.encoder.dimension for v in vectors):
                        raise RuntimeError("Embedding dimensions do not match the catalog collection")
                    self.client.upsert(self.collection, points=[models.PointStruct(
                        id=c["id"], vector=v, payload={"page_content": c["page_content"], "metadata": c["metadata"]})
                        for c, v in zip(batch, vectors)], wait=True)
                with self.db() as db:
                    db.execute("UPDATE documents SET status='ready',error=NULL WHERE id=?", (document_id,))
            except Exception as exc:
                with self.db() as db:
                    db.execute("UPDATE documents SET status='failed',error=? WHERE id=?", (str(exc), document_id))
                raise RuntimeError(f"Indexing failed for document {document_id}; upload again to retry") from exc
            return {**self.document(document_id), "duplicate": False}

    def search(self, query, vehicle_id=None, limit=5):
        from qdrant_client import models
        # Gate partial uploads using the authoritative SQLite publication state.
        with self.db() as db:
            ready = [r[0] for r in db.execute("SELECT id FROM documents WHERE status='ready' AND collection=?", (self.collection,))]
        if not ready:
            return []
        must = [models.FieldCondition(key="metadata.document_id", match=models.MatchAny(any=ready))]
        if vehicle_id:
            must.append(models.FieldCondition(key="metadata.vehicle_id", match=models.MatchValue(value=vehicle_id)))
        result = self.client.query_points(self.collection, query=self.encoder.encode([query])[0],
                                         query_filter=models.Filter(must=must), limit=limit, with_payload=True)
        return [{"id": str(p.id), "score": p.score, **p.payload} for p in result.points]

    def vector_overview(self):
        """Summarize the Qdrant payloads currently available to catalogue search."""
        if not self.client.collection_exists(self.collection):
            return {"collection": self.collection, "points": 0, "sources": []}
        points, _ = self.client.scroll(
            collection_name=self.collection,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        sources = {}
        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            source = metadata.get("source_file", "Unknown source")
            item = sources.setdefault(source, {
                "source_file": source,
                "document_ids": set(),
                "pages": set(),
                "kinds": {},
                "assigned_points": 0,
                "unassigned_points": 0,
                "sample": "",
            })
            if metadata.get("document_id"):
                item["document_ids"].add(metadata["document_id"])
            if metadata.get("page") is not None:
                item["pages"].add(metadata["page"])
            kind = metadata.get("kind", "unknown")
            item["kinds"][kind] = item["kinds"].get(kind, 0) + 1
            if metadata.get("vehicle_id"):
                item["assigned_points"] += 1
            else:
                item["unassigned_points"] += 1
            if not item["sample"]:
                item["sample"] = str(payload.get("page_content", ""))[:180]

        return {
            "collection": self.collection,
            "points": len(points),
            "sources": [{
                **item,
                "document_ids": sorted(item["document_ids"]),
                "pages": sorted(item["pages"]),
            } for item in sorted(sources.values(), key=lambda source: source["source_file"].casefold())],
        }

    @staticmethod
    def _document_label(filename):
        """Infer a searchable label from a filename without creating a vehicle fact.

        Metadata remains the authoritative source for vehicle identity. This fallback
        only lets a chatbot offer sensible clarification options for uploads that
        have not yet been mapped to a vehicle.
        """
        words = re.findall(r"[A-Za-z0-9]+", Path(filename).stem)
        if not words:
            return None
        boundary = next(
            (index for index, word in enumerate(words)
             if word.casefold() in {"features", "feature", "brochure", "specifications", "specs"}),
            len(words),
        )
        title = words[:boundary]
        years = [int(word) for word in words if re.fullmatch(r"(?:19|20)\d{2}", word)]
        title = [word for word in title if not re.fullmatch(r"(?:19|20)\d{2}", word)
                 and word.casefold() not in {"my", "int", "international"}]
        if not title:
            return None
        if len(title) == 1:
            return {"make": None, "model": title[0], "years": years}
        return {"make": title[0], "model": " ".join(title[1:]), "years": years}

    @staticmethod
    def _query_has_term(query_words, term):
        term_words = re.findall(r"[a-z0-9]+", term.casefold())
        return bool(term_words) and all(
            any(word == candidate or SequenceMatcher(None, word, candidate).ratio() >= 0.78
                for word in query_words)
            for candidate in term_words
        )

    def resolve_vehicle_query(self, query):
        """Normalize catalogue vehicle names and flag make-only comparisons.

        This is deliberately deterministic and happens before semantic retrieval.
        It prevents a vague request such as "Porsche versus Honda" from silently
        choosing a particular model or trim based on whichever chunk scored first.
        """
        with self.db() as db:
            rows = db.execute(
                "SELECT filename,metadata FROM documents WHERE status='ready' AND collection=?",
                (self.collection,),
            ).fetchall()

        entities = []
        for row in rows:
            metadata = json.loads(row["metadata"])
            vehicles = metadata.get("vehicles", [])
            if vehicles:
                for vehicle in vehicles:
                    entities.append({
                        "make": vehicle["make"],
                        "model": vehicle["model"],
                        "years": [vehicle["year"]],
                    })
            else:
                label = self._document_label(row["filename"])
                if label:
                    entities.append(label)

        # A title such as "2026 Accord Features by Trim" omits the manufacturer.
        # If another ready document names the same model with a manufacturer,
        # associate the title with that known manufacturer for query routing only.
        makes_by_model = {}
        for entity in entities:
            if entity["make"]:
                makes_by_model.setdefault(entity["model"].casefold(), set()).add(entity["make"])
        for entity in entities:
            candidates = makes_by_model.get(entity["model"].casefold(), set())
            if entity["make"] is None and len(candidates) == 1:
                entity["make"] = next(iter(candidates))

        grouped = {}
        for entity in entities:
            if not entity["make"]:
                continue
            key = (entity["make"].casefold(), entity["model"].casefold())
            grouped.setdefault(key, {"make": entity["make"], "model": entity["model"], "years": set()})
            grouped[key]["years"].update(entity["years"])
        entities = list(grouped.values())
        if not entities:
            return {"query": query, "clarification": None}

        query_words = re.findall(r"[a-z0-9]+", query.casefold())
        matched = []
        make_only = []
        for entity in entities:
            make_match = self._query_has_term(query_words, entity["make"])
            model_match = self._query_has_term(query_words, entity["model"])
            if make_match or model_match:
                matched.append(entity)
            if make_match and not model_match:
                make_only.append(entity["make"])

        if make_only:
            requested_makes = sorted({make.casefold(): make for make in make_only}.values(), key=str.casefold)
            choices = []
            for make in requested_makes:
                models = [entity for entity in entities if entity["make"].casefold() == make.casefold()]
                descriptions = []
                for entity in sorted(models, key=lambda item: item["model"].casefold()):
                    years = ", ".join(str(year) for year in sorted(entity["years"])) or "year not specified"
                    descriptions.append(f"{entity['model']} ({years})")
                choices.append(f"{make}: {', '.join(descriptions)}")
            return {
                "query": query,
                "clarification": (
                    "I found these models in your catalogues: " + "; ".join(choices)
                    + ". Please specify the model you want to compare, along with the model year and trim when horsepower varies by trim."
                ),
            }

        canonical = list(dict.fromkeys(f"{entity['make']} {entity['model']}" for entity in matched))
        if not canonical:
            return {"query": query, "clarification": None}
        return {
            "query": query + "\nResolved catalogue vehicles: " + "; ".join(canonical),
            "clarification": None,
        }

    def review_fact(self, fact_id, changes):
        if not isinstance(changes, dict) or set(changes) - {"status", "vehicle_id", "conditions"}:
            raise ValueError("Review accepts status, vehicle_id and conditions only")
        status = changes.get("status")
        if status not in {"approved", "rejected"}:
            raise ValueError("status must be approved or rejected")
        with self.lock, self.db() as db:
            row = db.execute("SELECT data FROM facts WHERE id=?", (fact_id,)).fetchone()
            if row is None:
                raise KeyError("Fact not found")
            fact = json.loads(row[0])
            vehicle_id = changes.get("vehicle_id", fact["vehicle_id"])
            if vehicle_id is not None and not isinstance(vehicle_id, str):
                raise ValueError("vehicle_id must be a string")
            conditions = changes.get("conditions", fact.get("conditions", {}))
            if not isinstance(conditions, dict) or any(not isinstance(k, str) or not isinstance(v, str) or len(v) > 500 for k, v in conditions.items()):
                raise ValueError("conditions must map names to short string values")
            if status == "approved":
                document = db.execute("SELECT status,metadata FROM documents WHERE id=?", (fact["document_id"],)).fetchone()
                if document["status"] != "ready":
                    raise ValueError("Document must finish indexing before fact approval")
                from catalog_extract import validate_metadata
                eligible = [v["id"] for v in validate_metadata(json.loads(document["metadata"])) if fact["page"] in v["pages"]]
                if not vehicle_id or vehicle_id not in eligible:
                    raise ValueError("Assign a vehicle mapped to this source page before approval")
                required = {"price": ("currency", "price_type", "effective_date", "tax_fees"),
                            "fuel_economy": ("test_standard", "driving_cycle"),
                            "acceleration": ("interval", "configuration"),
                            "top_speed": ("configuration",)}[fact["metric"]]
                if any(not conditions.get(k, "").strip() for k in required):
                    raise ValueError("Approval requires conditions: " + ", ".join(required))
                if fact["metric"] == "fuel_economy" and fact["unit"].lower() == "mpg":
                    raise ValueError("Ambiguous mpg: reject this candidate and supply a source specifying US or UK gallons")
            fact.update(status=status, vehicle_id=vehicle_id, conditions=conditions)
            db.execute("UPDATE facts SET vehicle_id=?,status=?,data=? WHERE id=?", (vehicle_id, status, json.dumps(fact), fact_id))
            return fact

    def vehicles(self):
        with self.db() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT data FROM vehicles ORDER BY id")]

    def page(self, document_id, number):
        with self.db() as db:
            row = db.execute("SELECT data FROM pages WHERE document_id=? AND page=?", (document_id, number)).fetchone()
            if row is None:
                raise KeyError("Page not found")
            return json.loads(row[0])

    def specifications(self, vehicle_id):
        with self.db() as db:
            return [json.loads(r[0]) for r in db.execute(
                "SELECT f.data FROM facts f JOIN documents d ON f.document_id=d.id "
                "WHERE f.vehicle_id=? AND f.status='approved' AND d.status='ready'", (vehicle_id,))]
