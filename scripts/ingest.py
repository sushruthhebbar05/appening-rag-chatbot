"""Ingest the Agentic AI eBook into Pinecone.

Usage:
    python -m scripts.ingest                      # downloads the PDF if needed
    python -m scripts.ingest --pdf path/to/book.pdf
    python -m scripts.ingest --reset              # wipe the namespace first
"""
import argparse
import os

import requests

from app.rag.ingestion import (
    chunk_documents,
    create_embeddings,
    ensure_index,
    load_pdf,
    reset_namespace,
    upsert_documents,
)

PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
DEFAULT_PDF_PATH = "data/Ebook-Agentic-AI.pdf"


def download_pdf(url: str, path: str) -> None:
    print(f"Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(response.content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH, help="local PDF path")
    parser.add_argument("--reset", action="store_true", help="delete existing vectors first")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        download_pdf(PDF_URL, args.pdf)

    pages = load_pdf(args.pdf)
    print(f"Loaded {len(pages)} pages")

    chunks = chunk_documents(pages)
    print(f"Created {len(chunks)} chunks")
    if not chunks:
        raise SystemExit("No text extracted. Is the PDF scanned/image-only?")

    ensure_index()
    if args.reset:
        reset_namespace()

    vectors = create_embeddings(chunks)
    upsert_documents(chunks, vectors)
    print(f"Upserted {len(chunks)} vectors")


if __name__ == "__main__":
    main()
