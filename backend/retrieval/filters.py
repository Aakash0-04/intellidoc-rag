"""Metadata filters for retrieval."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document


def filter_by_source(docs: List[Document], source: Optional[str] = None) -> List[Document]:
    """Filter documents by source filename."""
    if not source:
        return docs
    return [d for d in docs if d.metadata.get("source") == source]


def filter_by_content_type(docs: List[Document], content_type: Optional[str] = None) -> List[Document]:
    """Filter by text or table."""
    if not content_type:
        return docs
    return [d for d in docs if d.metadata.get("content_type") == content_type]


def filter_by_page_range(docs: List[Document], min_page: int = None, max_page: int = None) -> List[Document]:
    """Filter documents by page number range."""
    result = docs
    if min_page is not None:
        result = [d for d in result if (d.metadata.get("page_number") or 0) >= min_page]
    if max_page is not None:
        result = [d for d in result if (d.metadata.get("page_number") or 999) <= max_page]
    return result


def apply_metadata_filter(docs: List[Document], filters: dict) -> List[Document]:
    """
    Apply multiple metadata filters at once.
    filters = {"source": "file.pdf", "content_type": "text", "min_page": 1, "max_page": 10}
    """
    result = docs
    
    if "source" in filters:
        result = filter_by_source(result, filters["source"])
    if "content_type" in filters:
        result = filter_by_content_type(result, filters["content_type"])
    if "min_page" in filters or "max_page" in filters:
        result = filter_by_page_range(result, filters.get("min_page"), filters.get("max_page"))
    
    return result