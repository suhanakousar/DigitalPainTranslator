"""
FastAPI main application with router registration.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routers import infer, records, admin
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    yield
    # Shutdown


app = FastAPI(
    title="Digital Pain Translator Backend",
    description="FastAPI backend for pain assessment inference and analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(infer.router, prefix="/api", tags=["inference"])
app.include_router(infer.ws_router, tags=["websocket"])
app.include_router(records.router, prefix="/api", tags=["records"])
app.include_router(admin.router, prefix="/api", tags=["admin"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Digital Pain Translator Backend API"}