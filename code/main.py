"""
Application entrypoint.

Starts the FastAPI server and exposes the RAG query endpoint
backed by Qdrant, Gemini, and a custom validation layer.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from code.rag import RAGEngine

rag: RAGEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = RAGEngine()
    yield


app = FastAPI(title="DocuMind RAG", lifespan=lifespan)


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Question):
    return rag.ask(q.question)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "code.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
