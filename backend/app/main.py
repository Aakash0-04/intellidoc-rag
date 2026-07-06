"""
backend/app/main.py
FastAPI entry point with lifespan initialization and static file serving.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.routes import router
from backend.chat_engine import get_chat_engine
from backend.config.settings import get_settings
from backend.utils.logging import setup_logging, get_logger

# Initialize logging first
setup_logging(log_level="INFO")
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: warm up the chat engine (connects to Qdrant, loads LLM).
    Fails fast if your .env is misconfigured.
    """
    logger.info("🚀 RAG API starting...")
    try:
        engine = get_chat_engine()
        logger.info(f"LLM ready: {settings.default_llm_provider}/{settings.default_llm_model}")
        logger.info(f"Qdrant ready: {settings.qdrant_url}")
        logger.info("✅ All systems operational")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    yield
    logger.info("🛑 RAG API shutting down...")


app = FastAPI(
    title="Modern RAG API",
    description="LangChain-based RAG with hybrid retrieval, reranking, and citations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# API routes
app.include_router(router, prefix="")


@app.get("/")
async def root():
    return {
        "message": "Modern RAG API is running",
        "docs": "/docs",
        "health": "/health",
        "upload": "POST /upload",
        "chat": "POST /chat",
        "frontend": "/static/index.html",
    }