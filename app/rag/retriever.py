from app.config import get_settings
from app.exceptions import ConfigError, RetrievalError
from app.rag.clients import embed_texts, get_index


def retrieve_documents(question: str) -> list[dict]:
    """Embed the question and return the top_k closest chunks.

    Each result: {"id", "text", "page", "score"}, best match first.
    Pinecone's cosine score is the raw cosine similarity (higher = closer).
    """
    settings = get_settings()  # raises ConfigError if keys are missing
    try:
        query_vector = embed_texts([question])[0]
        result = get_index().query(
            vector=query_vector,
            top_k=settings.top_k,
            namespace=settings.pinecone_namespace,
            include_metadata=True,
        )
    except ConfigError:
        raise
    except Exception as exc:
        # Don't forward the original message; it can contain request details.
        raise RetrievalError(f"Retrieval failed ({type(exc).__name__})") from exc

    return [
        {
            "id": match.id,
            "text": match.metadata.get("text", ""),
            "page": int(match.metadata.get("page", 0)),
            "score": float(match.score),
        }
        for match in result.matches
    ]
