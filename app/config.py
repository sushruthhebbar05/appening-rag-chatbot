import os

from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

from app.exceptions import ConfigError

load_dotenv()


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str

    # Local Ollama models
    llm_model: str
    embedding_model: str
    embedding_dimension: int

    # Retrieval
    top_k: int
    relevance_threshold: float


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise ConfigError(
            f"Missing required environment variable: {name}"
        )

    return value


@lru_cache
def get_settings() -> Settings:
    return Settings(
        pinecone_api_key=_require("PINECONE_API_KEY"),

        pinecone_index_name=_require(
            "PINECONE_INDEX_NAME"
        ),

        pinecone_namespace=os.getenv(
            "PINECONE_NAMESPACE",
            "agentic-ai-ebook"
        ),

        pinecone_cloud=os.getenv(
            "PINECONE_CLOUD",
            "aws"
        ),

        pinecone_region=os.getenv(
            "PINECONE_REGION",
            "us-east-1"
        ),

        llm_model=os.getenv(
            "LLM_MODEL",
            "llama3.2:3b"
        ),

        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "nomic-embed-text"
        ),

        embedding_dimension=int(
            os.getenv(
                "EMBEDDING_DIMENSION",
                "768"
            )
        ),

        top_k=int(
            os.getenv(
                "TOP_K",
                "5"
            )
        ),

        relevance_threshold=float(
            os.getenv(
                "RELEVANCE_THRESHOLD",
                "0.35"
            )
        ),
    )