"""FastAPI application factory for MMAlign API."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import benchmark as benchmark_routes
from .routes import catalog as catalog_routes
from .routes import compare as compare_routes
from .routes import consistency as consistency_routes
from .routes import controller as controller_routes
from .routes import monitor as monitor_routes
from .routes import openai_compat as openai_compat_routes
from .routes import overview as overview_routes
from .routes import review as review_routes
from .routes import settings as settings_routes


def create_app() -> FastAPI:
    app = FastAPI(
        title="MMAlign API",
        description="Multimodal value alignment evaluation platform REST API.",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # Allow the WebUI dev server (and any other origin in dev) to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount primary /api/v1 route groups
    app.include_router(benchmark_routes.router, prefix="/api/v1")
    app.include_router(catalog_routes.router, prefix="/api/v1")
    app.include_router(compare_routes.router, prefix="/api/v1")
    app.include_router(consistency_routes.router, prefix="/api/v1")
    app.include_router(controller_routes.router, prefix="/api/v1")
    app.include_router(monitor_routes.router, prefix="/api/v1")
    app.include_router(overview_routes.router, prefix="/api/v1")
    app.include_router(review_routes.router, prefix="/api/v1")
    app.include_router(settings_routes.router, prefix="/api/v1")

    # Optional OpenAI-compatible façade: /v1/chat/completions, /v1/responses
    # These are a compatibility shim for existing OpenAI SDK clients; the
    # primary business interface remains /api/v1/* (PRD §F2).
    app.include_router(openai_compat_routes.router)

    @app.exception_handler(Exception)
    async def _generic_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "INTERNAL_ERROR", "message": str(exc)}},
        )

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    return app


# Module-level app instance used by uvicorn
app = create_app()
