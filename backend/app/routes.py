"""backend/app/routes.py
FastAPI route handlers — fully wired with URL upload and deduplication.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Form

from backend.chat_engine import get_chat_engine
from backend.chunking.splitters import chunk_documents
from backend.config.settings import get_settings
from backend.ingestion.loaders import ingest_file
from backend.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

# File to track processed files
PROCESSED_FILES_PATH = Path("uploads/.processed_files.json")


def _get_processed_files() -> dict:
    """Get dict of file hashes to metadata."""
    if PROCESSED_FILES_PATH.exists():
        return json.loads(PROCESSED_FILES_PATH.read_text())
    return {}

def _save_processed_files(files: dict):
    """Save processed files tracking."""
    PROCESSED_FILES_PATH.write_text(json.dumps(files, indent=2))

def _file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of file content."""
    return hashlib.md5(file_path.read_bytes()).hexdigest()


@router.get("/health")
async def health():
    return {"status": "healthy", "service": "rag-api"}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF or DOCX, extract, chunk, embed, and index.
    Skips re-processing if file was already uploaded.
    """
    allowed = {".pdf", ".docx"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, detail=f"Only {allowed} files supported")
    
    file_id = str(uuid.uuid4())[:8]
    safe_name = f"{file_id}_{file.filename}"
    file_path = Path(settings.upload_dir) / safe_name
    
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Saved: {safe_name}")
        
        # Check if already processed
        file_hash = _file_hash(file_path)
        processed = _get_processed_files()
        
        if file_hash in processed:
            logger.info(f"File already processed: {file.filename}")
            return {
                "success": True,
                "filename": file.filename,
                "chunks_indexed": processed[file_hash]["chunks_indexed"],
                "file_id": processed[file_hash]["file_id"],
                "already_processed": True,
            }
        
        # Ingest → Chunk → Index
        docs = ingest_file(file_path)
        chunks = chunk_documents(docs)
        get_chat_engine().index(chunks)
        
        # Track as processed
        processed[file_hash] = {
            "filename": file.filename,
            "chunks_indexed": len(chunks),
            "file_id": file_id,
            "processed_at": datetime.utcnow().isoformat(),
        }
        _save_processed_files(processed)
        
        return {
            "success": True,
            "filename": file.filename,
            "chunks_indexed": len(chunks),
            "file_id": file_id,
            "already_processed": False,
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, detail=f"Processing failed: {str(e)}")
    
    finally:
        file.file.close()


@router.post("/upload-url")
async def upload_url(url: str = Form(...)):
    """Upload and index a document from a URL."""
    from backend.ingestion.url_loader import is_valid_url, ingest_from_url
    
    if not is_valid_url(url):
        raise HTTPException(400, detail="Invalid URL. Must be http:// or https://")
    
    try:
        docs = ingest_from_url(url)
        chunks = chunk_documents(docs)
        get_chat_engine().index(chunks)
        
        return {
            "success": True,
            "url": url,
            "chunks_indexed": len(chunks),
        }
    except Exception as e:
        logger.error(f"URL upload failed: {e}")
        raise HTTPException(500, detail=f"Processing failed: {str(e)}")


@router.post("/chat")
async def chat(
    message: str = Form(...),
    session_id: str = Form("default"),
):
    """
    Chat with your documents. Returns answer + sources.
    """
    try:
        result = get_chat_engine().chat(message, session_id=session_id)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "is_greeting": result["is_greeting"],
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(500, detail=f"Chat failed: {str(e)}")


@router.post("/clear")
async def clear_session(session_id: str = Form("default")):
    """Clear chat history for a session."""
    get_chat_engine().clear_session(session_id)
    return {
        "success": True,
        "session_id": session_id,
        "message": "History cleared",
    }