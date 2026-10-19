"""FastAPI application entrypoint with RAG engine lifecycle management."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from documind.config import settings
from documind.logging_config import setup_logging
from documind.rag import RAGEngine
from documind.routes import health_router, metrics_router, questions_router

setup_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RAGEngine()
    yield
    await app.state.rag.close()


app = FastAPI(title="DocuMind RAG", lifespan=lifespan)

app.include_router(health_router)
app.include_router(questions_router)
app.include_router(metrics_router)


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")