"""Download and load documents from URLs."""
from __future__ import annotations

from pathlib import Path
from typing import List
from urllib.parse import urlparse

import httpx
from langchain_core.documents import Document

from backend.ingestion.loaders import ingest_file
from backend.utils.logging import get_logger

logger = get_logger(__name__)


def is_valid_url(url: str) -> bool:
    """Check if string is a valid HTTP URL."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def download_file(url: str, upload_dir: str = "uploads") -> Path:
    """Download file from URL to uploads directory."""
    filename = Path(urlparse(url).path).name or "downloaded_file"
    
    file_path = Path(upload_dir) / f"url_{filename}"
    
    logger.info(f"Downloading from URL: {url}")
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        
        with open(file_path, "wb") as f:
            f.write(response.content)
    
    logger.info(f"Downloaded: {file_path.name} ({file_path.stat().st_size} bytes)")
    return file_path


def ingest_from_url(url: str, upload_dir: str = "uploads") -> List[Document]:
    """Download document from URL and ingest it."""
    file_path = download_file(url, upload_dir)
    return ingest_file(file_path)