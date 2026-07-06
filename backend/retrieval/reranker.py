"""Jina Reranker for reordering retrieved documents."""
from __future__ import annotations

from typing import List

import numpy as np
from langchain_core.documents import Document

from backend.config.settings import get_settings
from backend.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class JinaReranker:
    """Rerank documents by relevance to query."""
    
    def __init__(self):
        self.api_key = settings.jina_api_key
        self._local_model = None
        
        if self.api_key:
            import httpx
            self._client = httpx.Client(
                base_url="https://api.jina.ai/v1",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-reranker-v1-base-en")
            self._local_model = AutoModelForSequenceClassification.from_pretrained(
                "jinaai/jina-reranker-v1-base-en"
            )
            self._local_model.eval()
    
    def rerank(self, query: str, docs: List[Document], top_k: int = None) -> List[Document]:
        top_k = top_k or settings.top_k_rerank
        if not docs:
            return []
        
        if self.api_key:
            resp = self._client.post("/rerank", json={
                "model": "jina-reranker-v1-base-en",
                "query": query,
                "documents": [d.page_content for d in docs],
                "top_n": top_k,
            })
            resp.raise_for_status()
            results = resp.json()["results"]
            indexed = {i: d for i, d in enumerate(docs)}
            return [indexed[r["index"]] for r in results]
        else:
            import torch
            pairs = [[query, doc.page_content] for doc in docs]
            inputs = self._tokenizer(
                pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
            )
            
            with torch.no_grad():
                scores = self._local_model(**inputs).logits.squeeze()
                if scores.dim() == 0:
                    scores = scores.unsqueeze(0)
            
            top_indices = np.argsort(scores.numpy())[::-1][:top_k]
            return [docs[i] for i in top_indices]