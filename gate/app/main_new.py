"""Main FastAPI application for RAG Gate."""

from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.endpoints.chat import router as chat_router
from app.endpoints.documents import router as documents_router
from app.endpoints.health import router as health_router
from app.middleware import http_metrics_middleware
from app.object_interaction.exceptions import register_exception_handlers
from app.object_interaction.presign_main import lifecycle as presign_lifespan
from app.object_interaction.presign_main import protected_router, public_router
from app.state import lifespan as rag_lifespan


@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    """Combine multiple lifespan contexts."""
    async with AsyncExitStack() as stack:
        # Enter all context managers
        await stack.enter_async_context(rag_lifespan(app))
        await stack.enter_async_context(presign_lifespan(app))
        yield
        # All will be properly cleaned up even if errors occur


# Create FastAPI app
app = FastAPI(title="RAG Gate", version="0.1.0", lifespan=combined_lifespan)

# Register exception handlers
register_exception_handlers(app)

# Include routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(public_router)
app.include_router(protected_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # configured in runtime after settings load; kept permissive for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware
app.middleware("http")(http_metrics_middleware)

