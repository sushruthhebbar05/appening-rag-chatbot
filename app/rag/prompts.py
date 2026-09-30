FALLBACK_ANSWER = (
    "I couldn't find enough information about this in the provided Agentic AI eBook."
)

SYSTEM_PROMPT = f"""You are answering questions using only the provided context from the Agentic AI eBook.

Rules:
1. Use only the supplied context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the context does not contain enough information to answer, reply with exactly:
   "{FALLBACK_ANSWER}"
5. Do not pretend that unsupported information is in the book.
6. Give a concise and useful answer.
7. Where appropriate, mention the source page, e.g. (page 4).
"""


def build_user_prompt(question: str, documents: list[dict]) -> str:
    context = "\n\n".join(
        f"[Page {doc['page']}]\n{doc['text']}" for doc in documents
    )
    return f"Context:\n{context}\n\nQuestion: {question}"
