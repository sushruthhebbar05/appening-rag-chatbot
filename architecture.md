# Architecture

Two separate flows share the same Pinecone index.

## 1. Ingestion (run once, offline)

```mermaid
flowchart LR
    A[Agentic AI eBook PDF] --> B[PyMuPDF: text per page]
    B --> C[clean_text]
    C --> D[Chunking<br/>1200 chars, 200 overlap]
    D --> E[Embeddings<br/>OpenAI]
    E --> F[(Pinecone<br/>id, vector, text, page, source)]
```

## 2. Question answering (per request)

```mermaid
flowchart TD
    U[User question] --> API[FastAPI POST /chat]
    API --> R[LangGraph: retrieve<br/>embed question, query Pinecone top_k]
    R --> C{check_relevance<br/>best score >= threshold?}
    C -- yes --> G[generate<br/>LLM answers from context only]
    C -- no --> FB[fallback<br/>fixed 'not found' message]
    G --> OUT[Response: answer, retrieved_context, retrieval_score]
    FB --> OUT
    R -. vectors .-> P[(Pinecone)]
```

## Design notes

- **Why LangGraph:** the workflow has one real decision (answer or refuse). A graph makes that branch explicit and testable instead of an `if` buried inside a function.
- **Why refuse before the LLM:** when retrieval is weak, the LLM is never called, so it cannot answer from general knowledge. The prompt adds a second layer of protection for borderline cases.
- **One page per chunk:** chunks never cross page boundaries, so each chunk has exactly one page number for citation.
