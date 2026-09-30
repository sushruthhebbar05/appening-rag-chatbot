"""PDF -> cleaned pages -> chunks -> embeddings -> Pinecone."""
import re
import time
import unicodedata

import pymupdf  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.rag.clients import embed_texts, get_index, get_pinecone_client

SOURCE_NAME = "Agentic AI eBook"

# The eBook is laid out as short pages, and a chunk never crosses a page
# boundary (so every chunk has exactly one page number). ~1200 characters is
# roughly 250-300 tokens: big enough to hold a full idea, small enough that the
# similarity score isn't diluted by unrelated text on the same page.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

EMBED_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 100
MIN_PAGE_CHARS = 30  # skips blank / image-only pages


def load_pdf(path: str) -> list[dict]:
    """Return one dict per page: {"source", "page", "text"} (page is 1-based)."""
    pages = []
    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            pages.append(
                {
                    "source": SOURCE_NAME,
                    "page": page_number,
                    "text": page.get_text("text"),
                }
            )
    return pages


def clean_text(text: str) -> str:
    """Fix common PDF extraction noise but keep paragraph breaks."""
    text = unicodedata.normalize("NFKC", text)  # e.g. the "fi" ligature -> "fi"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated at line end
    # Drop lines that are just a page number.
    text = "\n".join(
        line for line in text.split("\n") if not re.fullmatch(r"\s*\d{1,3}\s*", line)
    )
    text = re.sub(r"[ \t]+", " ", text)
    # A single newline inside a paragraph is just PDF line wrapping;
    # a blank line is a real paragraph break.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def chunk_documents(pages: list[dict]) -> list[dict]:
    """Split each page into chunks. Chunk ids are stable across re-runs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        text = clean_text(page["text"])
        if len(text) < MIN_PAGE_CHARS:
            continue
        for i, piece in enumerate(splitter.split_text(text), start=1):
            chunks.append(
                {
                    "id": f"page_{page['page']:03d}_chunk_{i:03d}",
                    "source": page["source"],
                    "page": page["page"],
                    "text": piece,
                }
            )
    return chunks


def create_embeddings(chunks: list[dict]) -> list[list[float]]:
    vectors = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        vectors.extend(embed_texts([c["text"] for c in batch]))
    return vectors


def ensure_index() -> None:
    """Create the Pinecone index if needed; fail loudly if the dimension is wrong."""
    from pinecone import ServerlessSpec

    settings = get_settings()
    pc = get_pinecone_client()
    name = settings.pinecone_index_name

    if name not in pc.list_indexes().names():
        print(f"Creating index '{name}' (dimension={settings.embedding_dimension}, cosine)")
        pc.create_index(
            name=name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
        return

    existing_dimension = pc.describe_index(name).dimension
    if existing_dimension != settings.embedding_dimension:
        raise ValueError(
            f"Index '{name}' has dimension {existing_dimension} but "
            f"EMBEDDING_DIMENSION={settings.embedding_dimension}. "
            "Use a different index name or fix the setting."
        )


def upsert_documents(chunks: list[dict], vectors: list[list[float]]) -> None:
    index = get_index()
    namespace = get_settings().pinecone_namespace
    records = [
        {
            "id": chunk["id"],
            "values": vector,
            "metadata": {
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
            },
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    for start in range(0, len(records), UPSERT_BATCH_SIZE):
        index.upsert(vectors=records[start : start + UPSERT_BATCH_SIZE], namespace=namespace)


def reset_namespace() -> None:
    """Delete everything in the namespace (used when chunking settings change)."""
    try:
        get_index().delete(delete_all=True, namespace=get_settings().pinecone_namespace)
    except Exception as exc:  # namespace may not exist yet on a fresh index
        print(f"Nothing to reset ({type(exc).__name__}).")
