"""backend/chat_engine.py
Orchestrates the full RAG chat flow with caching, HyDE, and query expansion.
"""
from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from backend.config.settings import get_settings
from backend.llm.providers import LLMFactory
from backend.memory.chat_memory import ConversationMemory
from backend.prompts.system import is_greeting_or_smalltalk
from backend.prompts.templates import build_chat_prompt
from backend.query.rewrite import QueryEnhancer
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import JinaReranker
from backend.utils.cache import QueryCache
from backend.utils.helpers import format_source
from backend.utils.logging import get_logger
from backend.vectordb.qdrant_client import VectorStore

logger = get_logger(__name__)
settings = get_settings()


class RetrievalPipeline:
    """
    Full retrieval pipeline: query enhancement → hybrid search → rerank.
    """
    
    def __init__(self, llm=None):
        self.vector_store = VectorStore()
        self.hybrid = HybridRetriever(self.vector_store)
        self.reranker = JinaReranker()
        self.enhancer = QueryEnhancer(llm=llm)
    
    def index(self, docs: List[Document]):
        """Index documents into vector store and BM25."""
        self.hybrid.index_documents(docs)
        logger.info(f"Indexed {len(docs)} chunks")
    
    def search(self, query: str) -> List[Document]:
        """Enhance query with expansion + HyDE, retrieve, rerank."""
        enhanced_queries = self.enhancer.enhance(query, use_hyde=True, use_expansion=True)
        logger.info(f"Query: '{query}' → Enhanced to {len(enhanced_queries)} queries")
        
        # Retrieve from all query variations
        all_docs = []
        seen_content = set()
        for q in enhanced_queries:
            retrieved = self.hybrid.retrieve(q, top_k=settings.top_k_retrieval)
            for doc in retrieved:
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)
        
        logger.info(f"Retrieved {len(all_docs)} unique documents from {len(enhanced_queries)} queries")
        
        # Rerank all unique docs against original query
        reranked = self.reranker.rerank(query, all_docs)
        logger.info(f"Reranked to top {len(reranked)}")
        
        return reranked


class ChatEngine:
    """
    Full chat orchestrator: cache + memory + retrieval + LLM + clean citations.
    """
    
    def __init__(self):
        self.llm = LLMFactory.create()
        self.memory = ConversationMemory()
        self.retrieval = RetrievalPipeline(llm=self.llm)
        self.parser = StrOutputParser()
        self.cache = QueryCache(ttl_seconds=3600, max_size=100)  # 1 hour TTL
    
    def index(self, docs: List[Document]):
        """Index documents for retrieval."""
        self.retrieval.index(docs)
    
    def chat(self, query: str, session_id: str = "default") -> dict:
        """
        Process a user message and return answer + sources.
        Uses cache for frequent queries.
        """
        # Check cache first
        cached = self.cache.get(query, session_id)
        if cached:
            logger.info(f"Cache hit for query: '{query[:50]}...'")
            return cached
        
        # Handle greetings / small talk
        if is_greeting_or_smalltalk(query):
            prompt = build_chat_prompt(is_greeting=True)
            chain = prompt | self.llm | self.parser
            response = chain.invoke({
                "input": query,
                "history": [],
                "context": "",
            })
            result = {
                "answer": response,
                "sources": [],
                "is_greeting": True,
            }
            self.cache.set(query, result, session_id)
            return result
        
        # Retrieve relevant chunks
        docs = self.retrieval.search(query)
        
        # Build context block
        context_blocks = []
        for i, doc in enumerate(docs, 1):
            src = format_source(doc.metadata)
            context_blocks.append(
                f"[Document {i}]\n{doc.page_content}\n"
                f"Source: {src['source']}, Page: {src.get('page', 'N/A')}\n"
            )
        context_text = "\n---\n".join(context_blocks)
        
        # Get conversation history
        history_msgs = self.memory.get_history(session_id)
        
        # Generate answer
        prompt = build_chat_prompt(is_greeting=False)
        chain = prompt | self.llm | self.parser
        response = chain.invoke({
            "input": query,
            "history": history_msgs,
            "context": context_text,
        })
        
        # Clean any accidental source citations from response
        response = re.sub(r'\[Source: [^\]]+\]', '', response)
        response = re.sub(r'Source: [^,\n]+, Page: \d+', '', response)
        response = response.strip()
        
        # Update memory
        self.memory.add_message(session_id, "human", query)
        self.memory.add_message(session_id, "ai", response)
        
        # Format sources for response (separate from answer)
        sources = [format_source(d.metadata) for d in docs]
        
        result = {
            "answer": response,
            "sources": sources,
            "is_greeting": False,
        }
        
        # Cache the result
        self.cache.set(query, result, session_id)
        
        return result
    
    def clear_session(self, session_id: str = "default"):
        """Clear chat history."""
        self.memory.clear(session_id)


# ═══════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════

_chat_engine_instance: ChatEngine | None = None

def get_chat_engine() -> ChatEngine:
    """Lazy singleton: creates on first call."""
    global _chat_engine_instance
    if _chat_engine_instance is None:
        _chat_engine_instance = ChatEngine()
    return _chat_engine_instance