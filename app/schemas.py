from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(..., max_length=1000, examples=["What is an AI agent?"])

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class RetrievedChunk(BaseModel):
    text: str
    page: int
    score: float
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    retrieved_context: list[RetrievedChunk]
    # Best cosine similarity among the retrieved chunks. NOT a probability
    # that the answer is correct.
    retrieval_score: float
