"""Run the README sample questions through the RAG graph and print the results.

Usage: python -m scripts.run_samples
Requires the eBook to be ingested and the .env file to be filled in.
"""
from app.rag.graph import run_rag

SAMPLE_QUESTIONS = [
    "What is Agentic AI?",
    "What are the main components of an agentic AI system?",
    "What are multi-agent systems and why are they useful?",
    "How can an organization assess its readiness for adopting Agentic AI?",
    "What are some real-world applications of Agentic AI?",
    "What is the recipe for chocolate chip cookies?",  # out of scope on purpose
]

if __name__ == "__main__":
    for question in SAMPLE_QUESTIONS:
        state = run_rag(question)
        pages = sorted({doc["page"] for doc in state["documents"]})
        print("=" * 80)
        print(f"Q: {question}")
        print(f"retrieval_score: {state['retrieval_score']:.3f} | pages: {pages}")
        print(f"A: {state['answer']}\n")
