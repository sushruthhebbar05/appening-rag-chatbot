import logging

from fastapi import FastAPI, HTTPException

from app.exceptions import ConfigError, GenerationError, RetrievalError
from app.rag.graph import run_rag
from app.schemas import ChatRequest, ChatResponse, RetrievedChunk

logger = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="Answers questions using only the Agentic AI eBook.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        state = run_rag(request.question)
    except ConfigError as exc:
        # Message only names the missing variable, never a secret value.
        logger.error("Configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except RetrievalError:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=502, detail="Could not query the knowledge base.")
    except GenerationError:
        logger.exception("Generation failed")
        raise HTTPException(status_code=502, detail="The language model request failed.")

    return ChatResponse(
        answer=state["answer"],
        retrieved_context=[
            RetrievedChunk(
                text=doc["text"],
                page=doc["page"],
                score=round(doc["score"], 4),
                chunk_id=doc["id"],
            )
            for doc in state["documents"]
        ],
        retrieval_score=round(state["retrieval_score"], 4),
    )
