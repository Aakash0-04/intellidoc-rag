"""Qdrant Cloud vector store management."""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from backend.config.settings import get_settings
from backend.embeddings.jina_embeddings import JinaEmbeddings
from backend.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class VectorStore:
    """Manages Qdrant connection and document operations."""
    
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=120,
        )
        self.collection = settings.qdrant_collection_name
        self.embedder = JinaEmbeddings()
        self._ensure_collection()
    
    def _ensure_collection(self):
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection not in names:
            logger.info(f"Creating collection: {self.collection}")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
            )
        else:
            logger.info(f"Using collection: {self.collection}")
    
    def add_documents(self, docs: List[Document]):
        """Embed and upsert documents in batches."""
        if not docs:
            return
        
        batch_size = 50
        total = len(docs)
        
        for batch_start in range(0, total, batch_size):
            batch = docs[batch_start:batch_start + batch_size]
            texts = [d.page_content for d in batch]
            embeddings = self.embedder.embed(texts)
            
            points = [
                PointStruct(
                    id=batch_start + i,
                    vector=emb,
                    payload={"page_content": doc.page_content, **doc.metadata}
                )
                for i, (doc, emb) in enumerate(zip(batch, embeddings))
            ]
            
            self.client.upsert(collection_name=self.collection, points=points)
            logger.info(f"Upserted batch {batch_start + 1}-{batch_start + len(batch)} of {total}")
        
        logger.info(f"Total upserted: {total} documents")
    
    def search(self, query: str, top_k: int = None) -> List[Document]:
        top_k = top_k or settings.top_k_retrieval
        embedding = self.embedder.embed_query(query)
        
        # Use query_points for newer qdrant-client versions
        results = self.client.query_points(
            collection_name=self.collection,
            query=embedding,
            limit=top_k,
            with_payload=True,
        ).points
        
        return [
            Document(page_content=r.payload.pop("page_content", ""), metadata=r.payload)
            for r in results
        ]