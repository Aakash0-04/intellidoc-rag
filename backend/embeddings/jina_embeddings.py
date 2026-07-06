"""Jina Embeddings wrapper: API or local fallback."""
from __future__ import annotations

from typing import List

from backend.config.settings import get_settings
from backend.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class JinaEmbeddings:
    """Embed texts using Jina API or local sentence-transformers."""
    
    def __init__(self):
        self.api_key = settings.jina_api_key
        self.dim = 768
        self._local_model = None
        self._api_client = None
        
        if self.api_key:
            import httpx
            self._api_client = httpx.Client(
                base_url="https://api.jina.ai/v1",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
            logger.info("Using Jina Embeddings API")
        else:
            logger.info("Loading local Jina model (may take 1-2 min on first run)...")
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en")
            logger.info("Local Jina model loaded")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        if self._api_client:
            resp = self._api_client.post("/embeddings", json={
                "model": "jina-embeddings-v2-base-en",
                "input": texts,
            })
            resp.raise_for_status()
            data = resp.json()["data"]
            return [item["embedding"] for item in data]
        else:
            embeddings = self._local_model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]