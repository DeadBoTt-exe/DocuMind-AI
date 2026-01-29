"""FastAPI routes package."""

from code.routes.health import router as health_router
from code.routes.questions import router as questions_router

__all__ = ["health_router", "questions_router"]
