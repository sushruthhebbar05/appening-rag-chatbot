import pytest

from app.config import get_settings


@pytest.fixture
def fake_env(monkeypatch):
    """Dummy settings so code paths that call get_settings() work offline."""
    for name, value in {
        "PINECONE_API_KEY": "test",
        "PINECONE_INDEX_NAME": "test-index",
        "PINECONE_NAMESPACE": "test-ns",
        "LLM_API_KEY": "test",
        "TOP_K": "3",
        "RELEVANCE_THRESHOLD": "0.35",
    }.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
