"""Build LangChain Document objects with rich metadata."""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from backend.ingestion.extractors import ExtractedContent
from backend.utils.helpers import clean_text


def build_documents(content: ExtractedContent) -> List[Document]:
    """Convert extracted content into LangChain Documents."""
    documents = []
    
    # Text per page
    for page_num, text in getattr(content, "text_pages", []):
        if text.strip():
            documents.append(Document(
                page_content=clean_text(text),
                metadata={
                    **content.metadata,
                    "page_number": page_num,
                    "content_type": "text",
                }
            ))
    
    # Fallback for DOCX (no fixed pages)
    if not documents and content.text:
        documents.append(Document(
            page_content=clean_text(content.text),
            metadata={**content.metadata, "page_number": 1, "content_type": "text"}
        ))
    
    # Tables as separate docs
    for i, table in enumerate(content.tables):
        page_num = 1
        if hasattr(content, "text_pages") and content.text_pages:
            page_num = content.text_pages[min(i, len(content.text_pages) - 1)][0]
        
        documents.append(Document(
            page_content=f"[TABLE]\n{table}",
            metadata={
                **content.metadata,
                "page_number": page_num,
                "content_type": "table",
            }
        ))
    
    return documents