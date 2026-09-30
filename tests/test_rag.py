"""Tests for ingestion, retrieval and graph routing. No network needed."""
import hashlib
import math

from app.rag import graph, ingestion, retriever
from app.rag.prompts import FALLBACK_ANSWER


# ---- tiny in-memory stand-ins for OpenAI embeddings and Pinecone -------------

def fake_embed(texts):
    """Hashed bag-of-words vectors: similar wording -> higher cosine similarity."""
    vectors = []
    for text in texts:
        vec = [0.0] * 64
        for word in text.lower().split():
            vec[int(hashlib.md5(word.strip(".,?").encode()).hexdigest(), 16) % 64] += 1
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


class FakeIndex:
    def __init__(self):
        self.records = {}

    def upsert(self, vectors, namespace):
        for record in vectors:
            self.records[record["id"]] = record

    def query(self, vector, top_k, namespace, include_metadata):
        scored = [
            (sum(a * b for a, b in zip(vector, r["values"])), r) for r in self.records.values()
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        class Match:
            def __init__(self, score, record):
                self.id, self.score, self.metadata = record["id"], score, record["metadata"]

        class Result:
            matches = [Match(s, r) for s, r in scored[:top_k]]

        return Result


# ---- ingestion ---------------------------------------------------------------

def test_clean_text_fixes_pdf_noise():
    raw = "Agentic sys-\ntems plan and\nact.\n\n12\nSecond \u00a0paragraph."
    cleaned = ingestion.clean_text(raw)
    assert cleaned == "Agentic systems plan and act.\n\nSecond paragraph."


def test_chunks_keep_page_metadata_and_stable_ids():
    pages = [
        {"source": "Agentic AI eBook", "page": 1, "text": "Short cover"},  # too short, skipped
        {"source": "Agentic AI eBook", "page": 2, "text": "An agent perceives its environment. " * 80},
    ]
    chunks = ingestion.chunk_documents(pages)

    assert len(chunks) > 1
    assert all(c["page"] == 2 and c["source"] == "Agentic AI eBook" for c in chunks)
    assert all(len(c["text"]) <= ingestion.CHUNK_SIZE for c in chunks)
    assert chunks[0]["id"] == "page_002_chunk_001"
    assert [c["id"] for c in chunks] == [c["id"] for c in ingestion.chunk_documents(pages)]


# ---- retrieval -----------------------------------------------------------------

def test_retrieval_returns_text_page_and_score(fake_env, monkeypatch):
    index = FakeIndex()
    monkeypatch.setattr(retriever, "embed_texts", fake_embed)
    monkeypatch.setattr(retriever, "get_index", lambda: index)

    chunks = [
        {"id": "page_003_chunk_001", "source": "Agentic AI eBook", "page": 3,
         "text": "An AI agent perceives its environment and acts on it"},
        {"id": "page_009_chunk_001", "source": "Agentic AI eBook", "page": 9,
         "text": "Multi agent systems coordinate several specialised agents"},
    ]
    index.upsert(
        [{"id": c["id"], "values": v, "metadata": {"source": c["source"], "page": c["page"], "text": c["text"]}}
         for c, v in zip(chunks, fake_embed([c["text"] for c in chunks]))],
        namespace="test-ns",
    )

    results = retriever.retrieve_documents("What is an AI agent?")

    assert results[0]["id"] == "page_003_chunk_001"
    assert results[0]["page"] == 3
    assert 0 < results[0]["score"] <= 1
    assert results[0]["score"] >= results[1]["score"]


# ---- graph routing -----------------------------------------------------------------

def _docs(score):
    return [{"id": "page_001_chunk_001", "text": "some text", "page": 1, "score": score}]


def test_relevant_question_goes_to_llm(fake_env, monkeypatch):
    called = {}

    def fake_generate(state):
        called["docs"] = state["documents"]
        return {"answer": "generated answer"}

    monkeypatch.setattr(graph, "retrieve_documents", lambda q: _docs(0.8))
    monkeypatch.setattr(graph, "generate", fake_generate)
    result = graph.build_graph().invoke(
        {"question": "q", "documents": [], "answer": "", "retrieval_score": 0.0, "is_relevant": False}
    )

    assert result["answer"] == "generated answer"
    assert result["retrieval_score"] == 0.8
    assert called["docs"]


def test_irrelevant_question_uses_fallback_and_skips_llm(fake_env, monkeypatch):
    def llm_should_not_run(state):
        raise AssertionError("LLM must not be called for weak retrieval")

    monkeypatch.setattr(graph, "retrieve_documents", lambda q: _docs(0.10))
    monkeypatch.setattr(graph, "generate", llm_should_not_run)
    result = graph.build_graph().invoke(
        {"question": "q", "documents": [], "answer": "", "retrieval_score": 0.0, "is_relevant": False}
    )

    assert result["answer"] == FALLBACK_ANSWER
    assert result["is_relevant"] is False


def test_no_results_uses_fallback(fake_env, monkeypatch):
    monkeypatch.setattr(graph, "retrieve_documents", lambda q: [])
    result = graph.run_rag("anything")
    assert result["answer"] == FALLBACK_ANSWER
    assert result["retrieval_score"] == 0.0
