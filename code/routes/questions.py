"""RAG-based question answering endpoint."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from code.rag import RAGEngine

router = APIRouter(tags=["questions"])


class Question(BaseModel):
    question: str


async def get_rag_engine() -> RAGEngine:
    from code.main import app
    return app.state.rag


@router.post("/ask")
async def ask(q: Question, rag: RAGEngine = Depends(get_rag_engine)):
    return await rag.ask_async(q.question)
