"""FastAPI routes package."""

from documind.routes.health import router as health_router
from documind.routes.metrics_route import router as metrics_router
from documind.routes.questions import router as questions_router

__all__ = ["health_router", "metrics_router", "questions_router"]