"""Document chunking with recursive main and semantic fallback."""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config.settings import get_settings
from backend.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents using recursive chunking as primary.
    Falls back to semantic chunking if recursive produces poor results.
    """
    # Primary: Recursive chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    
    chunks = splitter.split_documents(docs)
    logger.info(f"Recursive: {len(docs)} docs → {len(chunks)} chunks")
    
    # Fallback trigger: if recursive produced too few chunks (large docs not split well)
    avg_chunk_size = sum(len(c.page_content) for c in chunks) / max(len(chunks), 1)
    
    if avg_chunk_size > settings.chunk_size * 1.5 and len(docs) > 1:
        logger.info("Recursive produced large chunks, trying semantic fallback...")
        return _semantic_chunk(docs)
    
    return chunks


def _semantic_chunk(docs: List[Document]) -> List[Document]:
    """
    Semantic chunking using sentence boundaries.
    """
    all_chunks = []
    
    for doc in docs:
        # Split by sentences
        sentences = [s.strip() for s in doc.page_content.replace(".\n", ". ").split(". ") if s.strip()]
        
        if len(sentences) <= 3:
            all_chunks.append(doc)
            continue
        
        # Group sentences by length with overlap
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            current_text = " ".join(current_chunk)
            if len(current_text) > settings.chunk_size:
                chunks.append(Document(
                    page_content=current_text,
                    metadata={**doc.metadata, "chunk_method": "semantic"}
                ))
                # Overlap: keep last sentence
                current_chunk = [sentences[i-1], sentences[i]]
            else:
                current_chunk.append(sentences[i])
        
        if current_chunk:
            chunks.append(Document(
                page_content=" ".join(current_chunk),
                metadata={**doc.metadata, "chunk_method": "semantic"}
            ))
        
        all_chunks.extend(chunks)
    
    logger.info(f"Semantic: {len(docs)} docs → {len(all_chunks)} chunks")
    return all_chunks