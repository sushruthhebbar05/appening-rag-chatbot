"""Shared Ollama / Pinecone clients, used by both ingestion and the API."""

from functools import lru_cache

from langchain_ollama import ChatOllama, OllamaEmbeddings
from pinecone import Pinecone

from app.config import get_settings


@lru_cache
def get_embedding_model() -> OllamaEmbeddings:
    """Return the local Ollama embedding model."""
    settings = get_settings()

    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url="http://localhost:11434",
    )


@lru_cache
def get_llm() -> ChatOllama:
    """Return the local Ollama chat model."""
    settings = get_settings()

    return ChatOllama(
        model=settings.llm_model,
        base_url="http://localhost:11434",
        temperature=0,
    )


@lru_cache
def get_pinecone_client() -> Pinecone:
    """Return the Pinecone client."""
    settings = get_settings()

    return Pinecone(
        api_key=settings.pinecone_api_key
    )


def get_index():
    """Return the configured Pinecone index."""
    settings = get_settings()

    return get_pinecone_client().Index(
        settings.pinecone_index_name
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn texts into local Ollama embedding vectors."""
    return get_embedding_model().embed_documents(texts)


def generate_answer(prompt: str) -> str:
    """Generate an answer using the local Ollama LLM."""
    response = get_llm().invoke(prompt)

    return response.content