"""The RAG workflow as a small LangGraph graph.

    retrieve -> check_relevance -> generate   (relevant)
                                -> fallback   (not relevant)
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.exceptions import ConfigError, GenerationError
from app.rag.clients import generate_answer
from app.rag.prompts import FALLBACK_ANSWER, SYSTEM_PROMPT, build_user_prompt
from app.rag.retriever import retrieve_documents


class RAGState(TypedDict):
    question: str
    documents: list[dict]
    answer: str
    retrieval_score: float
    is_relevant: bool


def retrieve(state: RAGState) -> dict:
    return {"documents": retrieve_documents(state["question"])}


def check_relevance(state: RAGState) -> dict:
    # retrieval_score = similarity of the best chunk. Simple and easy to explain;
    # averaging would let a single strong match get dragged down by weak ones.
    scores = [doc["score"] for doc in state["documents"]]
    best = max(scores) if scores else 0.0

    return {
        "retrieval_score": best,
        "is_relevant": best >= get_settings().relevance_threshold,
    }


def route_after_check(state: RAGState) -> str:
    return "generate" if state["is_relevant"] else "fallback"


def generate(state: RAGState) -> dict:
    settings = get_settings()

    # Only chunks that individually clear the threshold go into the prompt.
    context_docs = [
        d
        for d in state["documents"]
        if d["score"] >= settings.relevance_threshold
    ]

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{build_user_prompt(state['question'], context_docs)}"
    )

    try:
        answer = generate_answer(prompt)
    except ConfigError:
        raise
    except Exception as exc:
        raise GenerationError(
            f"LLM call failed ({type(exc).__name__})"
        ) from exc

    return {"answer": answer.strip()}


def fallback(state: RAGState) -> dict:
    return {"answer": FALLBACK_ANSWER}


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("check_relevance", check_relevance)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "check_relevance")

    graph.add_conditional_edges(
        "check_relevance",
        route_after_check,
        {
            "generate": "generate",
            "fallback": "fallback",
        },
    )

    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()


rag_graph = build_graph()


def run_rag(question: str) -> RAGState:
    return rag_graph.invoke(
        {
            "question": question,
            "documents": [],
            "answer": "",
            "retrieval_score": 0.0,
            "is_relevant": False,
        }
    )