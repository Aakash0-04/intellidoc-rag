"""Unified document loader dispatch."""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from backend.ingestion.extractors import PDFExtractor, DOCXExtractor, ExtractedContent
from backend.ingestion.metadata import build_documents
from backend.utils.logging import get_logger

logger = get_logger(__name__)


def load_document(file_path: str | Path) -> ExtractedContent:
    """Route to correct extractor based on file extension."""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    if ext == ".pdf":
        return PDFExtractor().extract(file_path)
    elif ext == ".docx":
        return DOCXExtractor().extract(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def ingest_file(file_path: str | Path) -> List[Document]:
    """Full ingestion: load → extract → build docs."""
    content = load_document(file_path)
    return build_documents(content)