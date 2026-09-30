# Agentic AI RAG Chatbot

A Python-based Retrieval-Augmented Generation (RAG) chatbot that answers questions using only the provided **Agentic AI eBook**.

## Tech Stack

- Python
- FastAPI
- LangGraph
- Pinecone
- Ollama
- `nomic-embed-text` — embeddings
- `qwen2.5:1.5b` — local LLM
- PyMuPDF
- LangChain Text Splitters
- pytest

## Architecture

```text
Agentic AI PDF
      ↓
PDF Text Extraction
      ↓
Text Chunking
      ↓
Ollama Embeddings
(nomic-embed-text)
      ↓
Pinecone Vector Database
      ↓
Retrieve Relevant Chunks
      ↓
LangGraph
      ↓
Relevance Check
   ↙          ↘
Relevant     Not Relevant
   ↓              ↓
Ollama LLM      Fallback
   ↓
Final Answer
   +
Retrieved Context
   +
Retrieval Score
      ↓
FastAPI /chat
Installation
1. Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd appening-rag-chatbot
2. Create a Virtual Environment

Windows:

python -m venv .venv
.venv\Scripts\activate

Linux/macOS:

python -m venv .venv
source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Install and Run Ollama

Install Ollama and make sure it is running.

Pull the required models:

ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b

Verify the installed models:

ollama list
5. Configure Environment Variables

Copy .env.example to .env.

Windows:

copy .env.example .env

Linux/macOS:

cp .env.example .env

Add your Pinecone API key to .env.

Example configuration:

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-ebook-local
PINECONE_NAMESPACE=agentic-ai-ebook
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

LLM_MODEL=qwen2.5:1.5b
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

TOP_K=5
RELEVANCE_THRESHOLD=0.35

Do not commit .env or your Pinecone API key to GitHub.

Run the Project
1. Ingest the Agentic AI eBook
python -m scripts.ingest

This extracts the PDF, creates text chunks, generates embeddings using Ollama, and stores the vectors in Pinecone.

2. Run Sample Questions
python -m scripts.run_samples

Sample questions:

What is Agentic AI?

What are the main components of an agentic AI system?

What are multi-agent systems and why are they useful?

How can an organization assess its readiness for adopting Agentic AI?

What are some real-world applications of Agentic AI?

What is the recipe for chocolate chip cookies?

The final question is intentionally outside the knowledge base and demonstrates the fallback behavior.

3. Run Tests
pytest
4. Start the FastAPI Server
uvicorn app.main:app --reload

The API will be available at:

http://127.0.0.1:8000

Swagger API documentation:

http://127.0.0.1:8000/docs
API Usage
Health Check
GET /health
Chat
POST /chat

Example request:

{
  "question": "What is Agentic AI?"
}

The /chat endpoint returns:

Final answer
Retrieved context
Retrieval score

Example response structure:

{
  "answer": "Agentic AI is ...",
  "retrieved_context": [
    {
      "text": "...",
      "page": 3,
      "score": 0.864
    }
  ],
  "retrieval_score": 0.864
}
RAG Workflow

The system follows this workflow:

User Question
      ↓
Question Embedding
      ↓
Pinecone Similarity Search
      ↓
Retrieve Relevant Chunks
      ↓
LangGraph Relevance Check
      ↓
   ┌───────────────┐
   │               │
Relevant       Not Relevant
   ↓               ↓
Ollama LLM      Fallback
   ↓
Grounded Answer
Grounding and Fallback

A relevance threshold is used to determine whether the retrieved context is sufficient for answering the question.

RELEVANCE_THRESHOLD=0.35

If the retrieved context is not sufficiently relevant, the workflow returns a fallback response instead of intentionally generating an unsupported answer.

For example, an unrelated question such as:

What is the recipe for chocolate chip cookies?

returns a fallback response because the information is not contained in the Agentic AI eBook.

Testing

The project was tested through the complete RAG pipeline:

PDF ingestion
      ↓
Document chunking
      ↓
768-dimensional embeddings
      ↓
Pinecone retrieval
      ↓
LangGraph workflow
      ↓
Local Ollama LLM
      ↓
FastAPI /chat

The sample questions produced grounded answers for Agentic AI topics, while the unrelated cookie recipe question triggered the fallback response.

The FastAPI endpoints can also be tested through Swagger UI.

Project Structure
app/
├── main.py
├── config.py
├── schemas.py
├── exceptions.py
└── rag/
    ├── clients.py
    ├── ingestion.py
    ├── retriever.py
    ├── graph.py
    └── prompts.py

scripts/
├── ingest.py
└── run_samples.py

tests/

data/
└── Agentic-AI.pdf

.env.example
.gitignore
requirements.txt
README.md