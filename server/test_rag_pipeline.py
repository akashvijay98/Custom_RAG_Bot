import unittest
from types import SimpleNamespace
from unittest.mock import patch

import similarity_search_using_vector_db as rag


def state(**updates):
    value = {
        "original_query": "Who recorded Rudebox?",
        "resolved_query": "Who recorded Rudebox?",
        "clarification": "",
        "query_complexity": "",
        "sub_queries": ["Rudebox", "Robbie Williams"],
        "retrieved_docs": [],
        "reranked_docs": [],
        "top_score": 0.0,
        "evidence_sufficient": False,
        "retry_count": 0,
        "bridge_expanded": False,
        "final_answer": "",
    }
    value.update(updates)
    return value


class RagPipelineTests(unittest.TestCase):
    def test_complexity_parser_accepts_punctuation_and_explanation(self):
        model = SimpleNamespace(
            invoke=lambda messages: SimpleNamespace(content="Simple. This asks one fact.")
        )
        with patch.object(rag, "llm", model):
            self.assertEqual(rag.complexity_check(state()), {"query_complexity": "simple"})

    def test_score_parser_accepts_json_surrounded_by_prose(self):
        content = "Here are the scores:\n```json\n{\"scores\":[10, 2.5, 0]}\n```"
        self.assertEqual(rag._parse_rerank_scores(content, 3), [10.0, 2.5, 0.0])

    def test_retrieval_interleaves_subqueries(self):
        class Store:
            def similarity_search(self, query, k):
                return [SimpleNamespace(page_content=f"{query}-{index}") for index in range(k)]

        with patch.object(rag, "vector_store", Store()):
            result = rag.retrieve_documents(state())

        self.assertEqual(
            result["retrieved_docs"][:4],
            ["Rudebox-0", "Robbie Williams-0", "Rudebox-1", "Robbie Williams-1"],
        )
        self.assertEqual(len(result["retrieved_docs"]), rag.MAX_RETRIEVED_DOCS)

    def test_retrieval_includes_catalog_evidence_with_source_label(self):
        class Store:
            def similarity_search(self, query, k):
                return [SimpleNamespace(page_content="legacy evidence")]

        with patch.object(rag, "vector_store", Store()), patch.object(
            rag,
            "catalog_similarity_search",
            return_value=["[Catalog source: 911.pdf, page 4]\nCatalog evidence"],
        ):
            result = rag.retrieve_documents(state(sub_queries=["Turbo S"]))

        self.assertEqual(
            result["retrieved_docs"][:2],
            ["legacy evidence", "[Catalog source: 911.pdf, page 4]\nCatalog evidence"],
        )

    def test_catalog_retrieval_survives_a_stopped_legacy_store(self):
        class BrokenStore:
            def similarity_search(self, query, k):
                raise ConnectionRefusedError("Qdrant is offline")

        catalog_passage = "[Catalog source: 911.pdf, page 4]\nCatalog evidence"
        with patch.object(rag, "vector_store", BrokenStore()), patch.object(
            rag, "catalog_similarity_search", return_value=[catalog_passage]
        ):
            result = rag.retrieve_documents(state(sub_queries=["Turbo S"]))

        self.assertEqual(result["retrieved_docs"], [catalog_passage])

    def test_bridge_retrieval_reserves_first_hop_evidence(self):
        class Store:
            def similarity_search(self, query, k):
                return [SimpleNamespace(page_content=f"new-{index}") for index in range(k)]

        first_hop = [f"first-hop-{index}" for index in range(rag.MAX_FIRST_HOP_DOCS)]
        with patch.object(rag, "vector_store", Store()):
            result = rag.retrieve_documents(
                state(
                    sub_queries=["follow-up"],
                    retrieved_docs=first_hop,
                    reranked_docs=first_hop,
                    bridge_expanded=True,
                )
            )

        self.assertEqual(result["retrieved_docs"][: rag.MAX_FIRST_HOP_DOCS], first_hop)
        self.assertEqual(len(result["retrieved_docs"]), rag.BRIDGE_RETRIEVED_DOCS)

    def test_retry_uses_larger_candidate_budget(self):
        class Store:
            def __init__(self):
                self.ks = []

            def similarity_search(self, query, k):
                self.ks.append(k)
                return [SimpleNamespace(page_content=f"{query}-{index}") for index in range(k)]

        store = Store()
        with patch.object(rag, "vector_store", store):
            result = rag.retrieve_documents(state(retry_count=1))

        self.assertTrue(all(k == rag.RETRY_RETRIEVED_DOCS for k in store.ks))
        self.assertEqual(len(result["retrieved_docs"]), rag.RETRY_RETRIEVED_DOCS)
        self.assertIn("Who recorded Rudebox?-11", result["retrieved_docs"])

    def test_bridge_expansion_uses_discovered_entity(self):
        class Model:
            def __init__(self):
                self.messages = None

            def invoke(self, messages):
                self.messages = messages
                return SimpleNamespace(content="Margaret Court current residence")

        model = Model()
        first_hop = [
            "1970 French Open\nJan Kodes and Margaret Court won the singles titles."
        ]
        with patch.object(rag, "llm", model):
            result = rag.expand_bridge_queries(
                state(
                    query_complexity="simple",
                    retrieved_docs=["dense-order distractor"],
                    reranked_docs=first_hop,
                )
            )

        self.assertEqual(
            result["sub_queries"],
            ["Who recorded Rudebox?", "Margaret Court current residence"],
        )
        self.assertTrue(result["bridge_expanded"])
        rendered_prompt = "\n".join(str(message.content) for message in model.messages)
        self.assertIn("Margaret Court", rendered_prompt)
        self.assertNotIn("dense-order distractor", rendered_prompt)

    def test_reranker_uses_one_model_call(self):
        class Model:
            def __init__(self):
                self.calls = 0

            def invoke(self, messages):
                self.calls += 1
                return SimpleNamespace(
                    content='{ "scores": [1, 10, 5, 2], "sufficient": true }'
                )

        model = Model()
        with patch.object(rag, "llm", model):
            result = rag.rerank_documents(
                state(retrieved_docs=["first", "best", "third", "fourth"])
            )

        self.assertEqual(model.calls, 1)
        self.assertEqual(result["reranked_docs"], ["best", "third", "fourth", "first"])
        self.assertEqual(result["top_score"], 10.0)
        self.assertTrue(result["evidence_sufficient"])

    def test_malformed_reranker_output_uses_dense_order(self):
        model = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="not scores"))
        with patch.object(rag, "llm", model):
            result = rag.rerank_documents(
                state(retrieved_docs=["first", "second", "third", "fourth"])
            )

        self.assertEqual(
            result["reranked_docs"], ["first", "second", "third", "fourth"]
        )
        self.assertEqual(result["top_score"], rag.CONFIDENCE_THRESHOLD)
        self.assertFalse(result["evidence_sufficient"])

    def test_simple_question_can_expand_based_on_missing_evidence(self):
        self.assertEqual(
            rag.route_after_rerank(
                state(query_complexity="simple", evidence_sufficient=False)
            ),
            "expand_bridge_queries",
        )

    def test_low_score_generates_after_retry(self):
        self.assertEqual(
            rag.route_after_rerank(
                state(
                    evidence_sufficient=False,
                    bridge_expanded=True,
                    retry_count=rag.MAX_RETRIES,
                )
            ),
            "generate_response",
        )

    def test_search_errors_are_not_hidden_as_empty_results(self):
        class BrokenStore:
            def similarity_search(self, query, k):
                raise RuntimeError("qdrant unavailable")

        with patch.object(rag, "vector_store", BrokenStore()):
            with self.assertRaisesRegex(RuntimeError, "qdrant unavailable"):
                rag.semantic_similarity_search("question")


if __name__ == "__main__":
    unittest.main()
