from typing import List, Literal, TypedDict

import boto3
from botocore.config import Config
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "pdf_embeddings"
AWS_REGION = "us-east-1"
CONFIDENCE_THRESHOLD = 6.0
MAX_RETRIES = 1


retry_config = Config(
    region_name=AWS_REGION,
    retries={
        "max_attempts": 20,
        "mode": "adaptive",
    },
)

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    config=retry_config,
)

embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v1",
    region_name=AWS_REGION,
)

qdrant_client = QdrantClient(url=QDRANT_URL, check_compatibility=False)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

llm = ChatBedrock(
    client=bedrock_client,
    model_id="amazon.nova-pro-v1:0",
    region_name=AWS_REGION,
    model_kwargs={"temperature": 0.0},
)


class RAGState(TypedDict):
    original_query: str
    query_complexity: str
    sub_queries: List[str]
    retrieved_docs: List[str]
    reranked_docs: List[str]
    top_score: float
    retry_count: int
    final_answer: str


def complexity_check(state: RAGState) -> dict:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Reply with exactly one word: simple or complex. Use simple if the "
                "question asks for one specific fact. Use complex if it compares "
                "things, asks for multiple facts, or has sub-parts.",
            ),
            ("human", "{query}"),
        ]
    )
    response = llm.invoke(prompt.format_messages(query=state["original_query"]))
    verdict = response.content.strip().lower()

    if verdict not in {"simple", "complex"}:
        verdict = "complex"

    return {"query_complexity": verdict}


def query_decomposer(state: RAGState) -> dict:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Break this question into 1-3 focused search queries, one per line. "
                "No preamble.",
            ),
            ("human", "{query}"),
        ]
    )
    response = llm.invoke(prompt.format_messages(query=state["original_query"]))

    sub_queries = []
    for line in response.content.strip().splitlines():
        cleaned = line.strip().lstrip("-* ")
        if ". " in cleaned and cleaned.split(". ", 1)[0].isdigit():
            cleaned = cleaned.split(". ", 1)[1].strip()
        if cleaned:
            sub_queries.append(cleaned)

    return {"sub_queries": sub_queries[:3] or [state["original_query"]]}


def retrieve_documents(state: RAGState) -> dict:
    queries = state["sub_queries"] or [state["original_query"]]
    if state.get("retry_count", 0) > 0:
        queries = [f"{query} overview general specifications" for query in queries]

    seen = set()
    docs = []
    for query in queries:
        for doc in vector_store.similarity_search(query, k=10):
            content = doc.page_content.strip()
            if content and content not in seen:
                seen.add(content)
                docs.append(content[:2000])

    return {"retrieved_docs": docs[:10]}


def rerank_documents(state: RAGState) -> dict:
    retrieved_docs = state["retrieved_docs"]
    if not retrieved_docs:
        return {"reranked_docs": [], "top_score": 0.0}

    if len(retrieved_docs) <= 3:
        return {"reranked_docs": retrieved_docs, "top_score": 10.0}

    score_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Score how relevant this passage is to the query from 0 to 10. "
                "Reply with only the integer score.",
            ),
            ("human", "Query: {query}\n\nPassage: {passage}"),
        ]
    )

    scored_docs = []
    for doc in retrieved_docs:
        try:
            response = llm.invoke(
                score_prompt.format_messages(
                    query=state["original_query"],
                    passage=doc[:800],
                )
            )
            score = int(response.content.strip())
        except (AttributeError, TypeError, ValueError):
            score = 0
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    top_score = float(scored_docs[0][0]) if scored_docs else 0.0
    top_docs = [doc for _, doc in scored_docs[:3]]

    return {"reranked_docs": top_docs, "top_score": top_score}


def generate_response(state: RAGState) -> dict:
    context = "\n\n---\n\n".join(state["reranked_docs"])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Porsche assistant. Answer using only the context "
                "below. Keep the answer under 120 words. If the answer is not in the "
                "context, say so.\n\nContext:\n{context}",
            ),
            ("human", "{query}"),
        ]
    )
    response = llm.invoke(
        prompt.format_messages(context=context, query=state["original_query"])
    )
    return {"final_answer": response.content}


def fallback_response(state: RAGState) -> dict:
    return {
        "final_answer": (
            f"I couldn't find relevant information about '{state['original_query']}' "
            "in the knowledge base. Please try rephrasing or ask about a related topic."
        )
    }


def increment_retry(state: RAGState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def route_by_complexity(
    state: RAGState,
) -> Literal["query_decomposer", "retrieve_documents"]:
    if state["query_complexity"] == "simple":
        return "retrieve_documents"
    return "query_decomposer"


def route_by_docs_found(
    state: RAGState,
) -> Literal["rerank_documents", "fallback_response"]:
    if state["retrieved_docs"]:
        return "rerank_documents"
    return "fallback_response"


def route_by_confidence(
    state: RAGState,
) -> Literal["generate_response", "increment_retry", "fallback_response"]:
    if state["top_score"] >= CONFIDENCE_THRESHOLD:
        return "generate_response"

    if state.get("retry_count", 0) < MAX_RETRIES:
        return "increment_retry"

    return "fallback_response"


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("complexity_check", complexity_check)
    graph.add_node("query_decomposer", query_decomposer)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("rerank_documents", rerank_documents)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("generate_response", generate_response)
    graph.add_node("fallback_response", fallback_response)

    graph.set_entry_point("complexity_check")
    graph.add_conditional_edges(
        "complexity_check",
        route_by_complexity,
        {
            "query_decomposer": "query_decomposer",
            "retrieve_documents": "retrieve_documents",
        },
    )
    graph.add_edge("query_decomposer", "retrieve_documents")
    graph.add_conditional_edges(
        "retrieve_documents",
        route_by_docs_found,
        {
            "rerank_documents": "rerank_documents",
            "fallback_response": "fallback_response",
        },
    )
    graph.add_conditional_edges(
        "rerank_documents",
        route_by_confidence,
        {
            "generate_response": "generate_response",
            "increment_retry": "increment_retry",
            "fallback_response": "fallback_response",
        },
    )
    graph.add_edge("increment_retry", "retrieve_documents")
    graph.add_edge("generate_response", END)
    graph.add_edge("fallback_response", END)

    return graph.compile()


rag_graph = build_rag_graph()


def agentic_response(query_text: str) -> str:
    initial_state: RAGState = {
        "original_query": query_text,
        "query_complexity": "",
        "sub_queries": [query_text],
        "retrieved_docs": [],
        "reranked_docs": [],
        "top_score": 0.0,
        "retry_count": 0,
        "final_answer": "",
    }

    try:
        result = rag_graph.invoke(initial_state)
        return result["final_answer"]
    except Exception as exc:
        return f"Error running LangGraph RAG workflow: {str(exc)}"


def semantic_similarity_search(query_text: str, k: int = 3):
    """
    Direct vector similarity search without graph reasoning.
    """
    try:
        return vector_store.similarity_search(query_text, k=k)
    except Exception as exc:
        print(f"Search error: {exc}")
        return []
