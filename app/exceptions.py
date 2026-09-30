class ConfigError(RuntimeError):
    """A required environment variable is missing."""


class RetrievalError(RuntimeError):
    """Embedding the question or querying Pinecone failed."""


class GenerationError(RuntimeError):
    """The LLM call failed."""
