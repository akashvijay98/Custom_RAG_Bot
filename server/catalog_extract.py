"""Page-aware PDF extraction and conservative, reviewable specification candidates."""
import io
import json
import re
import uuid

PIPELINE_VERSION = "catalog-v1"
IDENTITY_FIELDS = ("make", "model", "year", "trim", "powertrain", "transmission", "drivetrain", "market")


def stable_id(value):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def validate_metadata(metadata):
    if not isinstance(metadata, dict) or set(metadata) - {"vehicles", "source_url", "published_date"}:
        raise ValueError("metadata must contain only vehicles, source_url and published_date")
    for key in ("source_url", "published_date"):
        if key in metadata and (not isinstance(metadata[key], str) or len(metadata[key]) > 2000):
            raise ValueError(f"{key} must be a string of at most 2000 characters")
    vehicles = metadata.get("vehicles", [])
    if not isinstance(vehicles, list) or len(vehicles) > 100:
        raise ValueError("vehicles must be a list of at most 100 variants")
    result = []
    for item in vehicles:
        if not isinstance(item, dict) or set(item) - {*IDENTITY_FIELDS, "pages", "table_header"}:
            raise ValueError("Invalid vehicle fields")
        if any(not isinstance(item.get(k), str) or not item[k].strip() or len(item[k]) > 200
               for k in IDENTITY_FIELDS if k != "year"):
            raise ValueError("Each vehicle requires make, model, trim, powertrain, transmission, drivetrain and market")
        if type(item.get("year")) is not int or not 1886 <= item["year"] <= 2200:
            raise ValueError("Each vehicle requires a valid integer model year")
        pages = item.get("pages")
        if not isinstance(pages, list) or not pages or any(type(p) is not int or p < 1 for p in pages):
            raise ValueError("Each vehicle requires a list of 1-based PDF pages")
        header = item.get("table_header")
        if header is not None and (not isinstance(header, str) or not header.strip() or len(header) > 200):
            raise ValueError("table_header must be a non-empty string")
        record = {k: (item[k].strip() if isinstance(item[k], str) else item[k]) for k in IDENTITY_FIELDS}
        record["id"] = stable_id(json.dumps([str(record[k]).casefold() for k in IDENTITY_FIELDS]))
        record["pages"] = sorted(set(pages))
        record["table_header"] = header.strip() if header else None
        result.append(record)
    if len({v["id"] for v in result}) != len(result):
        raise ValueError("Duplicate vehicle identities; combine their page lists")
    return result


def extract_pdf(data, max_pages=300):
    import pdfplumber
    if not data.startswith(b"%PDF-"):
        raise ValueError("Upload must be a PDF")
    pages = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > max_pages:
                raise ValueError(f"PDF exceeds the {max_pages}-page limit")
            for number, page in enumerate(pdf.pages, 1):
                # Keep full page text as evidence, including footnotes outside tables.
                text = page.extract_text(layout=False) or ""
                tables = []
                found = page.find_tables()
                strategy = "lines"
                if not found and re.search(r"technical data|specifications|top\s+(?:track\s+)?speed|fuel consumption", text, re.I):
                    strategy = "text"
                    found = page.find_tables({"vertical_strategy": "text", "horizontal_strategy": "text",
                                              "min_words_vertical": 3, "min_words_horizontal": 1})
                for table in found:
                    rows = [[re.sub(r"\s+", " ", cell or "").strip() for cell in row]
                            for row in table.extract()]
                    bbox = list(table.bbox)
                    if strategy == "text":
                        # Text-derived boundaries may clip the last column at the
                        # shortest repeated word edge. Recover it using full rows.
                        words = [w for w in page.extract_words() if w["top"] >= bbox[1] - 1 and w["bottom"] <= bbox[3] + 1]
                        right = max([bbox[2], *(w["x1"] for w in words)])
                        for row, geometry in zip(rows, table.rows):
                            if geometry.cells[-1]:
                                x0, top, _, bottom = geometry.cells[-1]
                                row[-1] = re.sub(r"\s+", " ", page.crop((x0, top, right, bottom)).extract_text() or "").strip()
                        bbox[2] = right
                    rows = [row for row in rows if any(row)]
                    if len(rows) > 1:
                        tables.append({"rows": rows, "bbox": bbox, "strategy": strategy})
                pages.append({"page": number, "text": text, "tables": tables})
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PDF cannot be read; check that it is valid and not encrypted") from exc
    if not any(p["text"].strip() for p in pages):
        raise ValueError("No extractable text. Run OCR on scanned PDFs before uploading.")
    return pages


def candidate_facts(text):
    """Never infer missing units, currency, test cycles, or feature availability."""
    specs = [
        ("top_speed", r"(?:top|max(?:imum)?)\s*(?:track\s+)?speed[^\d\n]{0,30}(\d+(?:\.\d+)?)\s*(km/h|kmh|kph|mph)"),
        ("acceleration", r"0\s*[-–]\s*(?:60\s*mph|100\s*km/h)[^\d\n]{0,30}(\d+(?:\.\d+)?)\s*(seconds|secs?|s)\b"),
        ("fuel_economy", r"(?:fuel\s*(?:economy|consumption)|mileage|combined|city|highway)[^\d\n]{0,30}(\d+(?:\.\d+)?)\s*(km/l|l/100\s*km|mpg(?:\s*\(?US\)?|\s*\(?UK\)?)?)"),
        ("price", r"(?:price|msrp|ex.showroom|on.road)[^\d\n]{0,30}?(USD|EUR|GBP|INR|\$|€|£|₹)\s*(\d[\d,]*(?:\.\d+)?)"),
    ]
    for original_line in text.splitlines():
        # Many catalog tables put the unit in the row label, before the value.
        # Reorder only the match input; retain the original evidence verbatim.
        line = re.sub(
            r"((?:top|max(?:imum)?)\s*speed)\s*\((km/h|kmh|kph|mph)\)\s*[:|]?\s*(\d+(?:\.\d+)?)",
            r"\1: \3 \2", original_line, flags=re.I)
        for metric, pattern in specs:
            for match in re.finditer(pattern, line, re.I):
                value, unit = match.groups()
                if metric == "price":
                    unit, value = value, unit
                value = float(value.replace(",", ""))
                if value <= 0:
                    continue
                original_unit = unit
                unit = unit.lower().replace(" ", "").replace("(", "").replace(")", "")
                normalized, canonical = value, original_unit
                if metric == "top_speed":
                    normalized, canonical = (value * 1.609344 if unit == "mph" else value), "km/h"
                elif metric == "acceleration":
                    canonical = "s"
                elif metric == "fuel_economy":
                    if unit == "km/l":
                        normalized, canonical = 100 / value, "L/100km"
                    elif unit in {"mpgus", "mpguk"}:
                        normalized, canonical = (235.214583 if unit == "mpgus" else 282.480936) / value, "L/100km"
                    elif unit == "l/100km":
                        canonical = "L/100km"
                yield {"metric": metric, "original_value": value, "original_unit": original_unit,
                       "value": round(normalized, 6), "unit": canonical,
                       "qualifiers": original_line.strip(), "source_excerpt": original_line.strip(), "status": "pending"}


def build_records(pages, vehicles, document_id, split):
    chunks, facts, warnings = [], [], []
    if any(max(v["pages"]) > len(pages) for v in vehicles):
        raise ValueError("Vehicle page mapping exceeds the PDF page count")
    for page in pages:
        number = page["page"]
        owners = [v for v in vehicles if number in v["pages"]]
        sole_owner = owners[0]["id"] if len(owners) == 1 else None
        if not page["text"].strip():
            warnings.append(f"Page {number} has no text; OCR may be required")
        if not sole_owner:
            warnings.append(f"Page {number}: narrative facts are unassigned; verify vehicle mapping")

        def add(content, kind, owner, location):
            for index, part in enumerate(split(content)):
                chunk_id = stable_id(f"{document_id}:{number}:{kind}:{location}:{index}:{part}")
                chunks.append({"id": chunk_id, "page_content": part, "metadata": {
                    "document_id": document_id, "page": number, "kind": kind,
                    "vehicle_id": owner, "location": location, "chunk": index}})
            # Parse complete evidence before chunking. Every candidate needs review.
            for fact in candidate_facts(content):
                fact.update(document_id=document_id, page=number, vehicle_id=owner, location=location)
                fact["id"] = stable_id(f"{document_id}:{number}:{location}:{owner}:{fact['metric']}:{fact['original_value']}:{fact['original_unit']}:{fact['source_excerpt']}")
                facts.append(fact)

        add(page["text"], "page_text", sole_owner, "text")
        for table_index, table in enumerate(page["tables"]):
            if table.get("strategy") == "text":
                warnings.append(f"Page {number}, table {table_index}: inferred column boundaries; verify against source")
            rows = table["rows"]
            header = rows[0]
            # Row chunks repeat the entire header. Never discard ambiguous columns.
            for row_index, row in enumerate(rows[1:], 1):
                row_text = "Headers: " + " | ".join(header) + "\n" + " | ".join(row)
                add(row_text, "table_row", None, f"table:{table_index}:row:{row_index}")
                for col in range(1, min(len(header), len(row))):
                    matches = [v for v in owners if v["table_header"] and
                               v["table_header"].casefold() == header[col].casefold()]
                    owner = matches[0]["id"] if len(matches) == 1 else None
                    # Only explicitly matched columns become vehicle-specific facts.
                    if owner:
                        add(f"{row[0]}: {row[col]}\nColumn: {header[col]}", "table_cell", owner,
                            f"table:{table_index}:row:{row_index}:column:{col}")
    # Overlapping narrative/table evidence can coexist; identical records cannot.
    return chunks, list({f["id"]: f for f in facts}.values()), warnings
