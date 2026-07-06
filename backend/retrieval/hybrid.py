"""Hybrid retrieval: semantic (Qdrant) + keyword (BM25) with RRF fusion."""
from __future__ import annotations

import hashlib
from typing import List

import numpy as np
from langchain_core.documents import Document

from backend.config.settings import get_settings
from backend.vectordb.qdrant_client import VectorStore
from backend.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _doc_key(doc: Document) -> str:
    """Hash content for deduplication."""
    return hashlib.md5(doc.page_content.encode()).hexdigest()[:16]


class BM25Retriever:
    """In-memory BM25 keyword retriever."""
    
    def __init__(self):
        self.docs: List[Document] = []
        self.bm25 = None
        self.tokenized_corpus: List[List[str]] = []
    
    def add_documents(self, docs: List[Document]):
        from rank_bm25 import BM25Okapi
        import nltk
        
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        
        self.docs = docs
        self.tokenized_corpus = [
            nltk.word_tokenize(d.page_content.lower()) for d in docs
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 indexed {len(docs)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.bm25:
            return []
        
        import nltk
        tokenized_query = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.docs[i] for i in top_indices if scores[i] > 0]


class HybridRetriever:
    """Combines semantic + keyword with Reciprocal Rank Fusion."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.bm25 = BM25Retriever()
        self.k = 60
    
    def index_documents(self, docs: List[Document]):
        self.vector_store.add_documents(docs)
        self.bm25.add_documents(docs)
    
    def retrieve(self, query: str, top_k: int = None, metadata_filter: dict = None) -> List[Document]:
        """Hybrid retrieval with optional metadata filtering."""
        top_k = top_k or settings.top_k_retrieval
        
        semantic = self.vector_store.search(query, top_k=top_k * 2)
        keyword = self.bm25.search(query, top_k=top_k * 2)
        
        # Apply metadata filter if provided
        if metadata_filter:
            from backend.retrieval.filters import apply_metadata_filter
            semantic = apply_metadata_filter(semantic, metadata_filter)
            keyword = apply_metadata_filter(keyword, metadata_filter)
        
        # RRF fusion with deduplication
        final_scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(semantic):
            key = _doc_key(doc)
            final_scores[key] = 1 / (self.k + rank + 1)
            doc_map[key] = doc
        
        for rank, doc in enumerate(keyword):
            key = _doc_key(doc)
            final_scores[key] = final_scores.get(key, 0) + 1 / (self.k + rank + 1)
            doc_map[key] = doc
        
        sorted_keys = sorted(final_scores.keys(), key=lambda k: final_scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[:top_k]]