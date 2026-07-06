"""Query enhancement: expansion, rewrite, and HyDE."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.prompts import PromptTemplate


class QueryEnhancer:
    """Rewrite queries, generate variations, and HyDE."""
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def rewrite(self, query: str) -> str:
        """Basic rule-based rewrite."""
        cleaned = query.strip()
        if len(cleaned) < 10:
            cleaned = f"information about: {cleaned}"
        return cleaned
    
    def expand(self, query: str, n: int = 3) -> List[str]:
        """
        Generate query variations using LLM for better retrieval coverage.
        """
        if not self.llm:
            return [query]
        
        prompt = PromptTemplate.from_template(
            """Generate {n} different versions of the user question to retrieve relevant documents.
Be concise. Output one per line.

Question: {query}

Variations:
1."""
        )
        
        chain = prompt | self.llm
        result = chain.invoke({"query": query, "n": n})
        text = result.content if hasattr(result, "content") else str(result)
        
        variations = [line.strip("1234567890. ") for line in text.strip().split("\n") if line.strip()]
        variations = [v for v in variations if v]
        
        if query not in variations:
            variations.insert(0, query)
        
        return variations[:n]
    
    def hyde(self, query: str) -> str:
        """
        HyDE: Hypothetical Document Embedding.
        Generate a fake document that would answer the query.
        """
        if not self.llm:
            return query
        
        prompt = PromptTemplate.from_template(
            """Write a short paragraph that would answer the following question.
This will be used to find relevant documents, so include key terms.

Question: {query}

Answer paragraph:"""
        )
        
        chain = prompt | self.llm
        result = chain.invoke({"query": query})
        hypothetical_doc = result.content if hasattr(result, "content") else str(result)
        
        return hypothetical_doc.strip()
    
    def enhance(self, query: str, use_hyde: bool = True, use_expansion: bool = True) -> List[str]:
        """
        Full enhancement pipeline.
        Returns list of queries to search (original + expanded + HyDE).
        """
        queries = [self.rewrite(query)]
        
        if use_expansion and self.llm:
            expanded = self.expand(query)
            queries.extend([q for q in expanded if q not in queries])
        
        if use_hyde and self.llm:
            hypothetical = self.hyde(query)
            if hypothetical and hypothetical not in queries:
                queries.append(hypothetical)
        
        return queries[:5]  # Max 5 queries