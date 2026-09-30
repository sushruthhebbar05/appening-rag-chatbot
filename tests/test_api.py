import pytest
from fastapi.testclient import TestClient

from app import main
from app.exceptions import ConfigError, GenerationError, RetrievalError

client = TestClient(main.app)

FAKE_STATE = {
    "question": "What is an AI agent?",
    "documents": [
        {"id": "page_004_chunk_001", "text": "An AI agent perceives, decides and acts.", "page": 4, "score": 0.87123},
        {"id": "page_005_chunk_002", "text": "Agents use tools.", "page": 5, "score": 0.61},
    ],
    "answer": "An AI agent perceives, decides and acts (page 4).",
    "retrieval_score": 0.87123,
    "is_relevant": True,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_docs_available():
    assert client.get("/docs").status_code == 200


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "   "}, {}])
def test_empty_or_missing_question_is_rejected(body):
    assert client.post("/chat", json=body).status_code == 422


def test_chat_response_structure(monkeypatch):
    monkeypatch.setattr(main, "run_rag", lambda question: FAKE_STATE)
    response = client.post("/chat", json={"question": "What is an AI agent?"})
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"answer", "retrieved_context", "retrieval_score"}
    assert body["retrieval_score"] == 0.8712
    assert body["retrieved_context"][0] == {
        "text": "An AI agent perceives, decides and acts.",
        "page": 4,
        "score": 0.8712,
        "chunk_id": "page_004_chunk_001",
    }


@pytest.mark.parametrize(
    "error, status",
    [
        (ConfigError("Missing required environment variable: LLM_API_KEY"), 503),
        (RetrievalError("boom"), 502),
        (GenerationError("boom"), 502),
    ],
)
def test_backend_errors_map_to_http_errors(monkeypatch, error, status):
    def raise_error(question):
        raise error

    monkeypatch.setattr(main, "run_rag", raise_error)
    response = client.post("/chat", json={"question": "hello"})
    assert response.status_code == status
    assert "boom" not in response.text  # internal error text is not leaked
