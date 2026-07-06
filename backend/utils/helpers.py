"""Shared utility functions."""
from __future__ import annotations

import re
from typing import Any


def clean_text(text: str) -> str:
    """Collapse whitespace and strip text."""
    return re.sub(r"\s+", " ", text).strip()


def format_source(metadata: dict[str, Any]) -> dict[str, Any]:
    """Standardize source metadata for frontend display."""
    return {
        "source": metadata.get("source", "unknown"),
        "page": metadata.get("page_number") or metadata.get("page"),
        "type": metadata.get("content_type", "text"),
    }