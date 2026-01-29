"""FastAPI application entrypoint with RAG engine lifecycle management."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from code.rag import RAGEngine
from code.routes import health_router, questions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RAGEngine()
    yield
    await app.state.rag.close()


app = FastAPI(title="DocuMind RAG", lifespan=lifespan)

app.include_router(health_router)
app.include_router(questions_router)


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")
